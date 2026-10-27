"""Обучение деревьев понимания индуктором (Delta LM, раздел 4): тип вопроса и смысл слова.

Корпуса:
- `data/corpora/intent_examples.json` — стартовые фразы, составленные ассистентом;
- `data/corpora/mailru_questions.json` — настоящие вопросы «Ответов Mail.ru» с разметкой
  типа вопроса (в том числе OTHER: «почему», «как», «где») и смысла найденного слова;
- `data/corpora/forum_questions.json` — вопросы с форума forum.razum.wiki, та же разметка.

Деревья:
- тип вопроса — признаки: леммы, первое слово, знак вопроса, число понятий, вид пары;
- смысл слова — для каждого упоминания понятия: само слово, вид понятия и леммы
  остальных слов; метка — употреблено ли слово в физико-химическом смысле.

Открытые вилки — гипотезы (knowledge = "hypothesis") до приёмки человеком.
Оценка — 10-кратная перекрёстная проверка с детерминированным разбиением.

    python -m delta_text.learn_intent
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from delta_kernel import Tree
from delta_text.annotate import annotate
from delta_text.features import mention_record, utterance_record
from delta_text.knowledge import DATA, KnowledgeBase, default_knowledge
from delta_text.understand import find_mentions
from delta_tools.induction import Budget, Inductor, predict

SEED = DATA / "corpora" / "intent_examples.json"
REAL = {
    "mailru": DATA / "corpora" / "mailru_questions.json",
    "forum": DATA / "corpora" / "forum_questions.json",
}
LEARNED = DATA / "trees" / "intent.learned.json"
SENSE = DATA / "trees" / "sense.learned.json"
FOLDS = 10
# Глубина 30: обученные деревья — длинные цепочки «это слово не наше?»; при глубине 12
# остаток в конце цепочки становится UNKNOWN (см. docs/report-induction.md).
BUDGET = Budget(max_depth=30, min_support=2)

INTENT_SCHEMA = {
    "name": "Utterance",
    "version": "0.2.0",
    "attributes": [
        {"name": "lemmas", "type": {"kind": "set", "element": "str"}, "doc": "Леммы всех слов фразы"},
        {"name": "first", "type": {"kind": "str"}, "doc": "Лемма первого слова (самый вероятный разбор)"},
        {"name": "qmark", "type": {"kind": "bool"}, "doc": "Во фразе есть знак вопроса"},
        {"name": "n_concepts", "type": {"kind": "int"}, "doc": "Число найденных понятий"},
        {"name": "pair", "type": {"kind": "enum", "values": ["none", "instance_class", "other"]},
         "doc": "Для двух понятий: объект и класс (instance_class) или иное"},
    ],
}
SENSE_SCHEMA = {
    "name": "Mention",
    "version": "0.1.0",
    "attributes": [
        {"name": "word", "type": {"kind": "str"}, "doc": "Упоминание в тексте (строчными, ё → е)"},
        {"name": "kind", "type": {"kind": "enum", "values": ["class", "element", "compound", "term"]},
         "doc": "Вид понятия: класс дерева, химический элемент, соединение, термин"},
        {"name": "context", "type": {"kind": "set", "element": "str"}, "doc": "Леммы остальных слов фразы"},
    ],
}


# ---------------------------------------------------------------- признаки


def intent_features(text: str, kb: KnowledgeBase) -> dict[str, Any]:
    """Признаки фразы при первом кандидате каждого упоминания, без отсева по смыслу."""
    tokens = annotate(text)
    mentions, _ = find_mentions(tokens, kb)
    concepts = list(dict.fromkeys(m.candidates[0].concept for m in mentions))
    return utterance_record(tokens, [kb.concepts[k] for k in concepts])


def hand_record(record: dict[str, Any]) -> dict[str, Any]:
    """Проекция на схему ручного дерева data/trees/intent.json."""
    return {k: record[k] for k in ("lemmas", "n_concepts", "pair")}


# ---------------------------------------------------------------- корпуса


def intent_corpus(kb: KnowledgeBase) -> list[tuple[dict[str, Any], str, str, str]]:
    """(признаки, метка, текст, источник)."""
    seed = json.loads(SEED.read_text(encoding="utf-8"))["examples"]
    out = [(intent_features(text, kb), label, text, "seed") for text, label in seed]
    for source, path in REAL.items():
        for e in json.loads(path.read_text(encoding="utf-8"))["examples"]:
            out.append((intent_features(e["text"], kb), e["intent"], e["text"], source))
    return out


def sense_corpus(kb: KnowledgeBase) -> list[tuple[dict[str, Any], str, str, str]]:
    """Упоминания понятий в настоящих вопросах с меткой смысла всего вопроса."""
    real = [(source, e) for source, path in REAL.items()
            for e in json.loads(path.read_text(encoding="utf-8"))["examples"]]
    out = []
    for source, e in real:
        tokens = annotate(e["text"])
        mentions, _ = find_mentions(tokens, kb)
        for m in mentions:
            if m.candidates[0].typo:  # догадки об опечатках не размечены
                continue
            features = mention_record(tokens, m.start, m.end, m.candidates[0].concept, kb)
            out.append((features, "domain" if e["domain_sense"] else "other", e["text"], source))
    return out


# ---------------------------------------------------------------- оценка


def cross_validate(schema: dict[str, Any], corpus, budget: Budget, baseline=None) -> dict[str, Any]:
    """10-кратная проверка; фолд примера — его номер по модулю FOLDS."""
    stats = {"learned": Counter(), "baseline": Counter(), "errors": []}
    for fold in range(FOLDS):
        train = [(r, label) for i, (r, label, _, _) in enumerate(corpus) if i % FOLDS != fold]
        tree = Inductor(schema, budget).fit(train)
        for i, (record, label, text, source) in enumerate(corpus):
            if i % FOLDS != fold:
                continue
            guess = predict(tree, record)[0]
            outcome = "correct" if guess == label else "unknown" if guess is None else "wrong"
            stats["learned"][outcome] += 1
            stats["learned"][f"{source}:{outcome}"] += 1
            if outcome != "correct":
                stats["errors"].append((text, label, guess))
            if baseline is not None:
                b = baseline(record)
                b_outcome = "correct" if b == label else "unknown" if b is None else "wrong"
                stats["baseline"][b_outcome] += 1
                stats["baseline"][f"{source}:{b_outcome}"] += 1
    return stats


def hand_label(tree: Tree, record: dict[str, Any]) -> str | None:
    result = tree.classify(hand_record(record), certify=False)
    if result.leaf is None:
        return None
    leaf = tree.nodes[result.leaf]
    label = leaf.meta.get("intent", leaf.id)
    return None if label == "NONE" else label


def train(schema: dict[str, Any], corpus, budget: Budget, tree_id: str) -> tuple[Inductor, Tree]:
    inductor = Inductor(schema, budget, tree_id=tree_id)
    tree = inductor.fit([(r, label) for r, label, _, _ in corpus])
    return inductor, tree


def _line(name: str, stats: Counter, n: int, source: str = "") -> str:
    p = f"{source}:" if source else ""
    total = sum(stats[f"{p}{k}"] for k in ("correct", "unknown", "wrong"))
    return (f"  {name:10s} верно {stats[p + 'correct']}/{total} = {stats[p + 'correct'] / max(total, 1):.3f}, "
            f"UNKNOWN {stats[p + 'unknown']}, ошибок {stats[p + 'wrong']}")


def main() -> int:
    kb = default_knowledge()
    budget = BUDGET

    corpus = intent_corpus(kb)
    stats = cross_validate(INTENT_SCHEMA, corpus, budget, lambda r: hand_label(kb.hand_intent, r))
    print(f"Тип вопроса: {len(corpus)} фраз ({Counter(l for _, l, _, _ in corpus)})")
    for source in ("", "mailru", "forum", "seed"):
        print(f" {source or 'все'}:")
        print(_line("обученное", stats["learned"], len(corpus), source))
        print(_line("ручное", stats["baseline"], len(corpus), source))
    inductor, tree = train(INTENT_SCHEMA, corpus, budget, "intent.learned")
    print(f"  дерево: {inductor.report.forks} вилок, {inductor.report.leaves} листьев, проверки ядра: {tree.check() or 'ok'}")
    LEARNED.write_text(json.dumps(inductor.tree_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    corpus = sense_corpus(kb)
    majority = Counter(label for _, label, _, _ in corpus).most_common(1)[0][0]
    stats = cross_validate(SENSE_SCHEMA, corpus, budget, lambda r: majority)
    print(f"Смысл слова: {len(corpus)} упоминаний ({Counter(l for _, l, _, _ in corpus)})")
    for source in ("", "mailru", "forum"):
        print(f" {source or 'все'}:")
        print(_line("обученное", stats["learned"], len(corpus), source))
        print(_line(f"всегда {majority}", stats["baseline"], len(corpus), source))
    inductor, tree = train(SENSE_SCHEMA, corpus, budget, "sense.learned")
    print(f"  дерево: {inductor.report.forks} вилок, {inductor.report.leaves} листьев, проверки ядра: {tree.check() or 'ok'}")
    SENSE.write_text(json.dumps(inductor.tree_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
