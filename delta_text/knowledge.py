"""База знаний языкового слоя: деревья ядра, понятия, объекты и их имена.

- Понятие-класс — несинтетический узел дерева (`matter`, `elements`).
- Объект — химический элемент или соединение из `data/confirmed-concepts.json`.
  У объекта есть записи для деревьев: элемент классифицируется деревом
  `elements` (класс PubChem) и деревом `matter` (агрегатное состояние).
- Допущения данных: если значение не проверено источником (например, «ожидаемое»
  состояние или локальные метаданные прототипа), это записывается в `data_assumptions`
  и ответ не получает статус `proven`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from delta_kernel import Tree
from delta_kernel.tree import Fork
from delta_text.morph import name_pattern, normalize
from delta_tools.bridge_matter import matter_tree_from_deltas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Агрегатное состояние при стандартных условиях → признаки дерева материи.
STATE_TRAITS = {
    "solid": {"continuous": False, "grouped": True, "planetary": False, "preservesShape": True},
    "liquid": {"continuous": False, "grouped": True, "planetary": False, "preservesShape": False,
               "preservesVolume": True},
    "gas": {"continuous": False, "grouped": True, "planetary": False, "preservesShape": False,
            "preservesVolume": False, "ionized": False},
}
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


@dataclass
class Concept:
    key: str
    kind: str  # class | instance
    label: str
    names: list[str]
    tree: str | None = None  # для класса
    node: str | None = None  # для класса
    category: str = ""  # для объекта: «химический элемент», «вещество (соединение)»
    latin: list[str] = field(default_factory=list)  # символы, формулы, английские имена
    records: dict[str, dict[str, Any]] = field(default_factory=dict)  # дерево → запись
    data_assumptions: dict[str, list[str]] = field(default_factory=dict)  # дерево → оговорки
    facts: list[tuple[str, str]] = field(default_factory=list)  # описательные данные из источника

    @property
    def name(self) -> str:
        return self.names[0] if self.names else self.label.lower()


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA, intent_tree: Path | None = None):
        self.trees: dict[str, Tree] = {
            "matter": Tree.from_dict(matter_tree_from_deltas(_load(data_dir / "matter-deltas.json"))),
            "elements": Tree.from_dict(_load(data_dir / "trees" / "elements.json")),
        }
        self.intent = Tree.from_dict(_load(intent_tree or data_dir / "trees" / "intent.json"))
        self.lexicon = _load(data_dir / "lexicon" / "ru.json")
        self.concepts: dict[str, Concept] = {}
        self._load_classes()
        self._load_instances(_load(data_dir / "confirmed-concepts.json"))
        self.patterns: list[tuple[tuple[frozenset[str], ...], str]] = []
        self.latin_index: dict[str, list[str]] = {}
        self.single_words: dict[str, list[str]] = {}
        self._index_names()
        self.intent_words = _string_literals(self.intent)

    # ------------------------------------------------------------ загрузка

    def _load_classes(self) -> None:
        lex = self.lexicon.get("concepts", {})
        for tree_id, tree in self.trees.items():
            for node in tree.nodes.values():
                if isinstance(node, Fork) and node.synthetic:
                    continue
                key = f"{tree_id}:{node.id}"
                label = node.label or node.id
                names = lex.get(key, {}).get("names") or [label.lower()]
                self.concepts[key] = Concept(key, "class", label, names, tree=tree_id, node=node.id)

    def _load_instances(self, data: dict[str, Any]) -> None:
        kinds = self.lexicon.get("instance_kinds", {})
        for e in data["elements"]:
            ru = [a for a in e["aliases"] if _CYRILLIC.search(a)]
            state = e["standard_state"]
            phase = state.removeprefix("Expected to be a ").lower()
            concept = Concept(
                key=f"element:{e['symbol']}",
                kind="instance",
                label=ru[0] if ru else e["name_en"],
                names=ru,
                category=kinds.get("element", "element"),
                latin=[e["symbol"], e["name_en"]],
                records={"elements": {"z": e["atomic_number"], "group_block": e["group_block"]}},
                facts=[("символ", e["symbol"]), ("атомный номер", str(e["atomic_number"])),
                       ("атомная масса", e["atomic_mass"])],
            )
            if phase in STATE_TRAITS:
                concept.records["matter"] = dict(STATE_TRAITS[phase])
                if state.startswith("Expected"):
                    concept.data_assumptions["matter"] = [
                        f"агрегатное состояние «{state}» в PubChem — ожидаемое, а не измеренное"
                    ]
            self.concepts[concept.key] = concept
        for c in data["compounds"]:
            ru = [a for a in c["aliases"] if _CYRILLIC.search(a)]
            key = f"compound:{c['cid']}"
            concept = Concept(
                key=key,
                kind="instance",
                label=ru[0] if ru else c["query"],
                names=ru,
                category=kinds.get("compound", "compound"),
                latin=[c["formula"], c["query"]],
                facts=[("формула", c["formula"]), ("название IUPAC", c["iupac_name"]), ("PubChem CID", str(c["cid"]))],
            )
            phase = c.get("model", {}).get("phase")
            if phase in STATE_TRAITS:
                concept.records["matter"] = dict(STATE_TRAITS[phase])
                concept.data_assumptions["matter"] = [
                    "агрегатное состояние взято из локальных метаданных прототипа, а не из проверенного источника"
                ]
            self.concepts[key] = concept

    def _index_names(self) -> None:
        for key, concept in self.concepts.items():
            for name in concept.names:
                pattern = name_pattern(name)
                if pattern:
                    self.patterns.append((pattern, key))
                if len(pattern) == 1:
                    self.single_words.setdefault(normalize(name), []).append(key)
            for alias in concept.latin:
                self.latin_index.setdefault(alias, []).append(key)
                if len(alias) > 3:
                    self.latin_index.setdefault(alias.lower(), []).append(key)
        for k in self.latin_index:
            self.latin_index[k] = list(dict.fromkeys(self.latin_index[k]))
        for k in self.single_words:
            self.single_words[k] = list(dict.fromkeys(self.single_words[k]))
        # Длинные имена проверяются первыми: «благородный газ» раньше «газ».
        self.patterns.sort(key=lambda item: -len(item[0]))

    # ------------------------------------------------------------ сведения о понятиях

    def tree_of(self, concept: Concept) -> Tree:
        assert concept.tree is not None
        return self.trees[concept.tree]

    def class_key(self, tree_id: str, node_id: str) -> str | None:
        """Ключ понятия для узла; для синтетического узла — None."""
        key = f"{tree_id}:{node_id}"
        return key if key in self.concepts else None

    def say(self, tree_id: str, fork_id: str, branch: bool) -> str:
        """Фраза для ветви вилки; без записи в словаре — запись предиката."""
        phrases = self.lexicon.get("forks", {}).get(f"{tree_id}:{fork_id}")
        if phrases:
            return phrases["true" if branch else "false"]
        fork = self.trees[tree_id].nodes[fork_id]
        assert isinstance(fork, Fork)
        return fork.predicate.text if branch else f"не ({fork.predicate.text})"

    def genus(self, tree_id: str, node_id: str) -> tuple[str | None, list[tuple[str, bool]]]:
        """Ближайший несинтетический предок (род) и шаги от него до узла (видовые отличия)."""
        tree = self.trees[tree_id]
        steps: list[tuple[str, bool]] = []
        current = tree.parent[node_id]
        while current is not None:
            fork_id, branch = current
            steps.append((fork_id, branch))
            fork = tree.nodes[fork_id]
            if not (isinstance(fork, Fork) and fork.synthetic):
                return fork_id, list(reversed(steps))
            current = tree.parent[fork_id]
        return None, []


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _string_literals(tree: Tree) -> frozenset[str]:
    """Все строковые литералы в предикатах дерева (ключевые слова типов вопроса)."""
    out: set[str] = set()

    def walk(node: Any) -> None:
        if node.kind == "in":
            out.update(v for v in node.value if isinstance(v, str))
        if node.kind == "lit" and isinstance(node.value, str):
            out.add(node.value)
        for a in node.args:
            walk(a)

    for fork in tree.forks():
        walk(fork.predicate.node)
    return frozenset(out)


@lru_cache(maxsize=1)
def default_knowledge() -> KnowledgeBase:
    return KnowledgeBase()
