"""Обучение дерева типов вопроса индуктором (Delta LM, раздел 4).

Признаки фразы те же, что у ручного дерева `data/trees/intent.json`: множество лемм,
число найденных понятий и вид пары (объект + класс или иное). Индуктор открывает
вилки по корпусу `data/corpora/intent_examples.json`. Открытые вилки — гипотезы
(knowledge = "hypothesis") до приёмки человеком.

    python -m delta_text.learn_intent            # оценка и сохранение дерева
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from delta_kernel import Tree
from delta_text.annotate import annotate
from delta_text.knowledge import DATA, KnowledgeBase, default_knowledge
from delta_text.understand import _pair_kind, find_mentions
from delta_tools.induction import Budget, Inductor, predict

CORPUS = DATA / "corpora" / "intent_examples.json"
LEARNED = DATA / "trees" / "intent.learned.json"


def features(text: str, kb: KnowledgeBase) -> dict[str, Any]:
    tokens = annotate(text)
    mentions, _ = find_mentions(tokens, kb)
    concepts = list(dict.fromkeys(m.candidates[0].concept for m in mentions))
    return {
        "lemmas": sorted({lemma for t in tokens if t.kind == "word" for lemma in t.lemmas}),
        "n_concepts": len(concepts),
        "pair": _pair_kind([kb.concepts[k] for k in concepts]),
    }


def load_corpus(kb: KnowledgeBase, path: Path = CORPUS) -> list[tuple[dict[str, Any], str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(features(text, kb), label, text) for text, label in data["examples"]]


def hand_label(tree: Tree, record: dict[str, Any]) -> str | None:
    result = tree.classify(record, certify=False)
    if result.leaf is None:
        return None
    leaf = tree.nodes[result.leaf]
    label = leaf.meta.get("intent", leaf.id)
    return None if label == "NONE" else label


def evaluate(kb: KnowledgeBase, budget: Budget | None = None) -> dict[str, Any]:
    corpus = load_corpus(kb)
    schema = kb.intent.schema.to_dict()
    budget = budget or Budget(max_depth=10)
    learned = {"correct": 0, "unknown": 0, "wrong": 0, "errors": []}
    hand = {"correct": 0, "unknown": 0, "wrong": 0, "errors": []}
    for i, (record, label, text) in enumerate(corpus):
        train = [(r, l) for j, (r, l, _) in enumerate(corpus) if j != i]
        tree = Inductor(schema, budget, tree_id="intent.learned").fit(train)
        for name, stats, guess in (("learned", learned, predict(tree, record)[0]),
                                   ("hand", hand, hand_label(kb.intent, record))):
            if guess == label:
                stats["correct"] += 1
            elif guess is None:
                stats["unknown"] += 1
                stats["errors"].append((text, label, "UNKNOWN"))
            else:
                stats["wrong"] += 1
                stats["errors"].append((text, label, guess))
    full = Inductor(schema, budget, tree_id="intent.learned")
    tree = full.fit([(r, l) for r, l, _ in corpus])
    return {"n": len(corpus), "learned": learned, "hand": hand, "tree": full.tree_dict(),
            "forks": full.report.forks, "leaves": full.report.leaves, "check": tree.check()}


def main() -> int:
    kb = default_knowledge()
    result = evaluate(kb)
    n = result["n"]
    for name in ("learned", "hand"):
        s = result[name]
        print(f"{name:8s}: верно {s['correct']}/{n} = {s['correct'] / n:.3f}, "
              f"UNKNOWN {s['unknown']}, ошибок {s['wrong']}")
    print(f"обученное дерево: {result['forks']} вилок, {result['leaves']} листьев, проверки ядра: {result['check'] or 'ok'}")
    LEARNED.write_text(json.dumps(result["tree"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"сохранено: {LEARNED.relative_to(DATA.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
