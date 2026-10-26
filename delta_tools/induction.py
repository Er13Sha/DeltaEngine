"""Индуктор дельт (спецификация, раздел 7): открытие вилок по размеченным примерам.

Обучение = поиск вилок, а не подбор весов.

- **Минимальное различение (7.2).** Вилка создаётся только там, где в одном листе
  оказались примеры с разными метками.
- **Кандидаты (7.3).** bool `a`; `a == v`, `a in {v1, v2}` для enum и строк;
  `a <= t` для чисел, где `t` — точная середина между соседними значениями
  (`Fraction`, без float); `any w in a: w == v` и `any w in a: w in {v1, v2}`
  для множеств; `has(a)` для необязательных атрибутов; конъюнкции двух лучших
  кандидатов и их отрицаний.
- **Критерий (7.4).** Взвешенная неоднородность Джини в точных дробях плюс штраф
  за размер предиката. При равенстве — лексикографический порядок канонической
  формы, поэтому результат детерминирован.
- **Ворота (7.5).** G1 тотальность — статически (`unguarded` ядра); G2, G3 — по
  построению языка; G4 обе ветви непусты; G5 — разбиение пересчитывается ядром и
  сверяется с быстрым вычислением кандидата, а регрессия проверяется при
  инкрементальном добавлении; G6 — бюджет глубины и размера; G7 — открытая вилка
  получает knowledge = "hypothesis" и статус «candidate», пока её не примет человек.
- **Противоречия (7.6).** Одинаковые записи с разными метками и исчерпание бюджета
  дают лист UNKNOWN (`meta.unknown`), а не «самый частый ответ».
- **Инкрементальность (7.7).** `Inductor.add` перестраивает только лист, в который
  попал противоречащий пример.

Индуктор вне доверенного ядра: он предлагает дерево, ядро его проверяет.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

from delta_kernel import Language, Schema, Tree
from delta_kernel.dsl import OUT_OF_DOMAIN, Node, to_text
from delta_kernel.tree import TREE_FORMAT

Record = dict[str, Any]
Example = tuple[Record, str]


@dataclass
class Budget:
    max_depth: int = 12
    max_set: int = 2  # размер множеств в `in {...}`
    top_k: int = 10  # сколько лучших кандидатов комбинировать в конъюнкции
    conjunctions: bool = True
    size_penalty: Fraction = Fraction(1, 2)  # штраф за каждый атом сверх первого
    min_support: int = 1  # лист с меньшим числом примеров — UNKNOWN


@dataclass
class Candidate:
    text: str
    size: int
    fn: Callable[[Record], bool]
    score: Fraction = Fraction(0)


@dataclass
class InductionReport:
    forks: int = 0
    leaves: int = 0
    unknown_leaves: int = 0
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)  # причина отказа ворот → число


class Inductor:
    def __init__(self, schema: dict[str, Any] | Schema, budget: Budget | None = None,
                 tree_id: str = "induced", version: str = "0.1.0", auto_accept: bool = False):
        self.schema_dict = schema if isinstance(schema, dict) else schema.to_dict()
        self.schema = Schema.from_dict(self.schema_dict)
        self.language = Language(self.schema)
        self.budget = budget or Budget()
        self.tree_id = tree_id
        self.version = version
        self.auto_accept = auto_accept  # только для тестовых пакетов (G7)
        self.examples: list[tuple[Record, str]] = []
        # запись + значения производных атрибутов (для быстрых кандидатов; ядро считает их само)
        self.views: list[Record] = []
        self.root: dict[str, Any] | None = None
        self.report = InductionReport()
        self._ids: Counter = Counter()

    # ------------------------------------------------------------ обучение

    def fit(self, examples: list[Example]) -> Tree:
        self.examples = [(self.schema.coerce_record(r), label) for r, label in examples]
        self.views = [self._view(r) for r, _ in self.examples]
        self.report = InductionReport()
        self._ids = Counter()
        self.root = self._build(list(range(len(self.examples))), 0, frozenset(), "root")
        return self.tree()

    def add(self, record: Record, label: str) -> Tree:
        """Инкрементально: перестраивается только лист, куда попал противоречащий пример."""
        if self.root is None:
            return self.fit([(record, label)])
        before = self.tree()
        self.examples.append((self.schema.coerce_record(record), label))
        self.views.append(self._view(self.examples[-1][0]))
        new_index = len(self.examples) - 1
        leaf_id = before.classify(self.examples[new_index][0], certify=False).leaf
        holder, key, leaf = _find(self.root, leaf_id)
        members = [i for i in range(len(self.examples)) if _reaches(before, self.examples[i][0], leaf_id)]
        if leaf.get("class") == label and len(members) >= self.budget.min_support:
            leaf["support"] = len(members)
            return self.tree()
        depth = len(before.path_to(leaf_id))
        present = frozenset().union(*(
            before.nodes[s.node].predicate.present_if(s.branch) for s in before.path_to(leaf_id)
        )) if depth else frozenset()
        subtree = self._build(members, depth, present, leaf_id.split("#")[0])
        if holder is None:
            self.root = subtree
        else:
            holder[key] = subtree
        after = self.tree()
        # G5: примеры вне перестроенного листа классифицируются как раньше
        for i, (rec, _) in enumerate(self.examples[:-1]):
            if i not in members and after.classify(rec, certify=False).leaf != before.classify(rec, certify=False).leaf:
                raise AssertionError("G5: перестройка листа изменила классификацию вне его")
        return after

    def tree(self) -> Tree:
        return Tree.from_dict(self.tree_dict())

    def _view(self, record: Record) -> Record:
        view = dict(record)
        for d in self.schema.derived:
            value = self.language.derived_value(d.name, record)
            if value is not OUT_OF_DOMAIN:
                view[d.name] = value
        return view

    def tree_dict(self) -> dict[str, Any]:
        assert self.root is not None
        return {"format": TREE_FORMAT, "id": self.tree_id, "version": self.version,
                "schema": self.schema_dict, "root": self.root}

    # ------------------------------------------------------------ построение

    def _build(self, members: list[int], depth: int, present: frozenset[str], hint: str) -> dict[str, Any]:
        labels = Counter(self.examples[i][1] for i in members)
        if len(labels) == 1:
            label = next(iter(labels))
            if len(members) < self.budget.min_support:
                return self._unknown_leaf(labels, "мало примеров")
            return self._leaf(label, len(members))
        if depth >= self.budget.max_depth:
            return self._unknown_leaf(labels, "исчерпан бюджет глубины")
        best = self._best_candidate(members, present)
        if best is None:
            records = {repr(sorted(self.examples[i][0].items(), key=lambda kv: kv[0])) for i in members}
            reason = "одинаковые записи с разными метками" if len(records) == 1 else "нет различающей вилки"
            self.report.quarantined.append({"labels": dict(labels), "reason": reason, "examples": len(members)})
            return self._unknown_leaf(labels, reason)
        predicate = self.language.compile(best.text)
        yes, no = [], []
        for i in members:
            value = predicate.evaluate(self.examples[i][0]).value
            if value is OUT_OF_DOMAIN or value != best.fn(self.views[i]):
                raise AssertionError(f"G5: ядро и кандидат расходятся на {best.text}")
            (yes if value else no).append(i)
        self.report.forks += 1
        fork_id = self._new_id(f"fork.{hint}")
        return {
            "id": fork_id,
            "predicate": predicate.text,
            "knowledge": "constatation" if self.auto_accept else "hypothesis",
            "evidence": [{
                "source": "induction",
                "status": "accepted" if self.auto_accept else "candidate",
                "examples": len(members),
                "labels": dict(sorted(labels.items())),
                "score": str(best.score),
            }],
            "true": self._build(yes, depth + 1, present | predicate.present_if(True), "t"),
            "false": self._build(no, depth + 1, present | predicate.present_if(False), "f"),
        }

    def _leaf(self, label: str, support: int) -> dict[str, Any]:
        self.report.leaves += 1
        return {"id": self._new_id(label), "label": label, "class": label, "support": support}

    def _unknown_leaf(self, labels: Counter, reason: str) -> dict[str, Any]:
        self.report.leaves += 1
        self.report.unknown_leaves += 1
        return {"id": self._new_id("UNKNOWN"), "label": "UNKNOWN", "class": None, "unknown": reason,
                "counts": dict(sorted(labels.items()))}

    def _new_id(self, base: str) -> str:
        self._ids[base] += 1
        return f"{base}#{self._ids[base]}"

    # ------------------------------------------------------------ кандидаты

    def _best_candidate(self, members: list[int], present: frozenset[str]) -> Candidate | None:
        records = [self.views[i] for i in members]
        labels = [self.examples[i][1] for i in members]
        firsts = []
        for cand in self._first_order(records, present):
            if self._score(cand, records, labels):
                firsts.append(cand)
        pool = list(firsts)
        if self.budget.conjunctions and firsts:
            top = sorted(firsts, key=lambda c: (c.score, c.text))[: self.budget.top_k]
            literals = top + [_negate(c) for c in top]
            for a, b in itertools.combinations(literals, 2):
                if a.text.removeprefix("not (").removesuffix(")") == b.text.removeprefix("not (").removesuffix(")"):
                    continue
                cand = Candidate(f"({a.text}) and ({b.text})", a.size + b.size,
                                 (lambda fa, fb: lambda r: fa(r) and fb(r))(a.fn, b.fn))
                if cand.size <= 2 and self._score(cand, records, labels):
                    pool.append(cand)
        if not pool:
            return None
        return min(pool, key=lambda c: (c.score, c.size, c.text))

    def _score(self, cand: Candidate, records: list[Record], labels: list[str]) -> bool:
        """Считает оценку; False — кандидат отклонён воротами G4 (вырожденное разбиение)."""
        yes: Counter = Counter()
        no: Counter = Counter()
        for r, label in zip(records, labels):
            (yes if cand.fn(r) else no)[label] += 1
        if not yes or not no:
            self.report.rejected["G4: одна из ветвей пуста"] += 1
            return False
        cand.score = _impurity(yes) + _impurity(no) + self.budget.size_penalty * (cand.size - 1)
        return True

    def _first_order(self, records: list[Record], present: frozenset[str]):
        required = {a.name for a in self.schema.attributes.values() if a.required}
        features = [(a.name, a.type, a.name in required or a.name in present, True)
                    for a in self.schema.attributes.values()]
        # производные атрибуты (в том числе построенные `delta_tools.featurize`)
        features += [(d.name, self.language.derived[d.name].type,
                      self.language.derived_unguarded[d.name] <= required | present, False)
                     for d in self.schema.derived]
        for name, attr_type, safe, base in features:
            have = [r for r in records if name in r]
            if not safe:
                if not base:
                    continue
                # G1: без гарантии присутствия атрибут можно только проверить на наличие
                if have and len(have) < len(records):
                    yield Candidate(f"has({name})", 1, (lambda n: lambda r: n in r)(name))
                else:
                    self.report.rejected["G1: атрибут не гарантирован на пути"] += 1
                continue
            kind = attr_type.kind
            values = sorted({r[name] for r in records}, key=_order_key) if kind != "set" else []
            if kind == "bool":
                yield Candidate(name, 1, (lambda n: lambda r: r[n])(name))
            elif kind in ("enum", "str"):
                for v in values:
                    yield Candidate(f"{name} == {_lit(v)}", 1, (lambda n, v: lambda r: r[n] == v)(name, v))
                for size in range(2, min(self.budget.max_set, len(values) - 1) + 1):
                    for combo in itertools.combinations(values, size):
                        yield Candidate(f"{name} in {{{', '.join(_lit(v) for v in combo)}}}", 1,
                                        (lambda n, s: lambda r: r[n] in s)(name, frozenset(combo)))
            elif kind in ("int", "rat"):
                for lo, hi in zip(values, values[1:]):
                    t = Fraction(lo + hi, 2)
                    yield Candidate(f"{name} <= {_lit(t)}", 1, (lambda n, t: lambda r: r[n] <= t)(name, t))
            elif kind == "set":
                items = sorted({v for r in records for v in r[name]}, key=_order_key)
                for v in items:
                    yield Candidate(f"any w in {name}: w == {_lit(v)}", 1,
                                    (lambda n, v: lambda r: v in r[n])(name, v))


def _negate(c: Candidate) -> Candidate:
    return Candidate(f"not ({c.text})", c.size, (lambda f: lambda r: not f(r))(c.fn), c.score)


def _impurity(counts: Counter) -> Fraction:
    """n · Gini = n − Σc²/n (точно, в дробях)."""
    n = sum(counts.values())
    return Fraction(n) - Fraction(sum(c * c for c in counts.values()), n)


def _order_key(v: Any) -> tuple:
    return (0, v) if isinstance(v, (int, Fraction)) and not isinstance(v, bool) else (1, str(v))


def _lit(v: Any) -> str:
    return to_text(Node("lit", v))


def _find(node: dict[str, Any], node_id: str, holder=None, key=None):
    if node["id"] == node_id:
        return holder, key, node
    for k in ("true", "false"):
        if k in node:
            found = _find(node[k], node_id, node, k)
            if found is not None:
                return found
    return None


def _reaches(tree: Tree, record: Record, leaf_id: str) -> bool:
    return tree.classify(record, certify=False).leaf == leaf_id


def predict(tree: Tree, record: Record) -> tuple[str | None, Any]:
    """Метка листа или None (UNKNOWN) и полный результат классификации ядра."""
    result = tree.classify(record)
    if result.leaf is None:
        return None, result
    leaf = tree.nodes[result.leaf]
    return leaf.meta.get("class"), result
