"""База знаний языкового слоя: деревья ядра, понятия, объекты и их имена.

- Понятие-класс — несинтетический узел дерева (`matter`, `elements`, `bonds`).
- Термин — определение учебной физики и химии (`data/knowledge/terms.json`).
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
HAND_INTENT = DATA / "trees" / "intent.json"

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
    kind: str  # class | instance | term
    label: str
    names: list[str]
    tree: str | None = None  # для класса
    node: str | None = None  # для класса
    category: str = ""  # для объекта: «химический элемент», «вещество (соединение)»
    latin: list[str] = field(default_factory=list)  # символы, формулы, английские имена
    records: dict[str, dict[str, Any]] = field(default_factory=dict)  # дерево → запись
    data_assumptions: dict[str, list[str]] = field(default_factory=dict)  # дерево → оговорки
    facts: list[tuple[str, str]] = field(default_factory=list)  # описательные данные из источника
    z: int | None = None  # элемент: заряд ядра (строение — delta_science.atoms)
    smiles: str | None = None  # соединение: граф связей атомов (PubChem)
    phase: str | None = None  # соединение: агрегатное состояние (локальные метаданные)
    definition: str = ""  # термин: определение

    @property
    def name(self) -> str:
        return self.names[0] if self.names else self.label.lower()


class KnowledgeBase:
    """По умолчанию тип вопроса и смысл слова определяют деревья, обученные индуктором
    на настоящих вопросах (`intent.learned.json`, `sense.learned.json`). Ручное дерево
    типа вопроса: `KnowledgeBase(intent_tree=HAND_INTENT, sense_tree=False)`.
    """

    def __init__(self, data_dir: Path = DATA, intent_tree: Path | None = None,
                 sense_tree: Path | None | bool = None):
        self.trees: dict[str, Tree] = {
            "matter": Tree.from_dict(matter_tree_from_deltas(_load(data_dir / "matter-deltas.json"))),
            "elements": Tree.from_dict(_load(data_dir / "trees" / "elements.json")),
            "bonds": Tree.from_dict(_load(data_dir / "trees" / "bonds.json")),
        }
        hand = Tree.from_dict(_load(data_dir / "trees" / "intent.json"))
        learned_intent = data_dir / "trees" / "intent.learned.json"
        if intent_tree is None:
            intent_tree = learned_intent if learned_intent.exists() else data_dir / "trees" / "intent.json"
        self.intent = Tree.from_dict(_load(intent_tree))
        self.hand_intent = hand  # запасное дерево, когда обученное отвечает UNKNOWN
        # программные правила: проверяются раньше обученного дерева, вилки — не гипотезы
        rules = data_dir / "trees" / "intent.rules.json"
        self.rules: Tree | None = Tree.from_dict(_load(rules)) if rules.exists() else None
        if sense_tree is None:
            sense_tree = data_dir / "trees" / "sense.learned.json"
        self.sense: Tree | None = (
            Tree.from_dict(_load(sense_tree)) if isinstance(sense_tree, Path) and sense_tree.exists() else None
        )
        self.lexicon = _load(data_dir / "lexicon" / "ru.json")
        self.concepts: dict[str, Concept] = {}
        self._load_classes()
        self._load_instances(_load(data_dir / "confirmed-concepts.json"),
                             _load(data_dir / "sources" / "pubchem" / "compound-structures.json"))
        self._load_terms(_load(data_dir / "knowledge" / "terms.json"))
        # слова причинных правил («ток», «плавиться») — не «незнакомые» в вопросах «почему»
        self.science_words = {normalize(w) for r in _load(data_dir / "knowledge" / "causes.json")["rules"]
                              for w in r["keywords"]}
        self.patterns: list[tuple[tuple[frozenset[str], ...], str]] = []
        self.latin_index: dict[str, list[str]] = {}
        self.single_words: dict[str, list[str]] = {}
        self._index_names()
        # служебные слова типов вопроса не проверяются на опечатки
        self.intent_words = _string_literals(hand)
        self.wiki: dict[str, list[dict[str, Any]]] = {}
        wiki_path = data_dir / "sources" / "razum-wiki" / "terms.json"
        if wiki_path.exists():
            for entry in _load(wiki_path)["entries"]:
                self.wiki.setdefault(normalize(entry["term"]), []).append(entry)

    def definition(self, *names: str) -> dict[str, Any] | None:
        """Определение из словаря RazumWiki по одному из имён; ссылки «см. X» разворачиваются."""
        for name in names:
            seen = set()
            key = normalize(name)
            while key in self.wiki and key not in seen:
                seen.add(key)
                entry = self.wiki[key][0]
                ref = re.fullmatch(r"см\.\s*(.+?)\.?", entry["definition"].strip())
                if ref is None:
                    return entry
                key = normalize(ref.group(1))
        return None

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

    def _load_terms(self, data: dict[str, Any]) -> None:
        for t in data["terms"]:
            key = f"term:{t['id']}"
            self.concepts[key] = Concept(key, "term", t["names"][0], t["names"], definition=t["definition"])

    def _load_instances(self, data: dict[str, Any], structures: dict[str, Any]) -> None:
        kinds = self.lexicon.get("instance_kinds", {})
        smiles = {r["CID"]: r.get("SMILES") for r in structures["compounds"]}
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
                z=e["atomic_number"],
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
                smiles=smiles.get(c["cid"]),
                phase=c.get("model", {}).get("phase"),
            )
            phase = concept.phase
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
