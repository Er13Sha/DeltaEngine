"""Дерево вилок (дельт), классификация с путём и сертификатом (спецификация, раздел 5).

Формат файла `delta-tree@0.1`:

    {
      "format": "delta-tree@0.1",
      "id": "matter", "version": "0.1.0",
      "schema": { ...схема Σ... },
      "root": {
        "id": "matter.continuous", "predicate": "continuous",
        "knowledge": "constatation",          // или "hypothesis" (S-11)
        "evidence": [...], "reserve": false, "synthetic": false,
        "true":  { ...узел... },
        "false": { "id": "Field", "label": "Поле" }   // лист
      }
    }

Инварианты I1–I3 выполняются по построению: из каждой вилки ровно две ветви,
ветви — это `P` и `not P`, вычисление трёхзначно. Тотальность вилок проверяется
статически (`check`): каждый атрибут, отсутствие которого даёт OUT_OF_DOMAIN,
должен быть обязательным в схеме либо гарантирован вилками выше по пути.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from delta_kernel.canon import digest
from delta_kernel.dsl import OUT_OF_DOMAIN, DslError, Language, Predicate, to_text
from delta_kernel.schema import Schema, SchemaError

TREE_FORMAT = "delta-tree@0.1"
CERT_FORMAT = "delta-certificate@0.1"
KNOWLEDGE = ("constatation", "hypothesis")

PROVEN = "proven"
UNKNOWN = "unknown"
INVALID = "invalid"


class TreeError(ValueError):
    pass


@dataclass
class Leaf:
    id: str
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    is_leaf = True


@dataclass
class Fork:
    id: str
    predicate: Predicate
    if_true: "Fork | Leaf"
    if_false: "Fork | Leaf"
    knowledge: str = "constatation"
    label: str = ""
    evidence: list[Any] = field(default_factory=list)
    reserve: bool = False  # ветвь может быть пустой на выборке (спецификация, «невырожденность»)
    synthetic: bool = False  # промежуточный узел бинаризации, не понятие

    is_leaf = False


Node = Fork | Leaf


@dataclass(frozen=True)
class Step:
    node: str
    predicate: str
    branch: bool


@dataclass(frozen=True)
class Issue:
    level: str  # error | warning
    code: str
    node: str
    message: str


@dataclass
class Classification:
    status: str  # proven | unknown | invalid
    leaf: str | None
    path: tuple[Step, ...]
    missing: tuple[str, ...] = ()
    stopped_at: str | None = None
    assumptions: tuple[str, ...] = ()  # вилки-гипотезы на пути
    errors: tuple[str, ...] = ()
    certificate: dict[str, Any] | None = None


class Tree:
    def __init__(self, tree_id: str, version: str, schema: Schema, root: Node, raw: dict[str, Any]):
        self.id = tree_id
        self.version = version
        self.schema = schema
        self.language = Language(schema)
        self.root = root
        self.raw = raw
        self.nodes: dict[str, Node] = {}
        self.parent: dict[str, tuple[str, bool] | None] = {}
        self._index(root, None)
        self.hash = digest(self.to_dict())

    # ------------------------------------------------------------ загрузка

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Tree":
        if data.get("format") != TREE_FORMAT:
            raise TreeError(f"Ожидался формат {TREE_FORMAT}, получен {data.get('format')!r}")
        schema = Schema.from_dict(data["schema"])
        language = Language(schema)
        seen: set[str] = set()

        def build(raw: dict[str, Any]) -> Node:
            node_id = raw.get("id")
            if not node_id:
                raise TreeError("У узла нет id")
            if node_id in seen:
                raise TreeError(f"Повторяющийся id узла: {node_id}")
            seen.add(node_id)
            has_branches = "true" in raw or "false" in raw
            if "predicate" not in raw:
                if has_branches:
                    raise TreeError(f"Узел {node_id}: есть ветви, но нет предиката")
                meta = {k: v for k, v in raw.items() if k not in ("id", "label")}
                return Leaf(node_id, raw.get("label", ""), meta)
            if "true" not in raw or "false" not in raw:
                raise TreeError(f"Вилка {node_id}: нужны обе ветви true и false")
            knowledge = raw.get("knowledge", "constatation")
            if knowledge not in KNOWLEDGE:
                raise TreeError(f"Вилка {node_id}: knowledge должно быть одним из {KNOWLEDGE}")
            try:
                predicate = language.compile(raw["predicate"])
            except DslError as exc:
                raise TreeError(f"Вилка {node_id}: {exc}") from None
            return Fork(
                id=node_id,
                predicate=predicate,
                if_true=build(raw["true"]),
                if_false=build(raw["false"]),
                knowledge=knowledge,
                label=raw.get("label", ""),
                evidence=list(raw.get("evidence", [])),
                reserve=bool(raw.get("reserve", False)),
                synthetic=bool(raw.get("synthetic", False)),
            )

        root = build(data["root"])
        return Tree(data["id"], data["version"], schema, root, data)

    def _index(self, node: Node, parent: tuple[str, bool] | None) -> None:
        self.nodes[node.id] = node
        self.parent[node.id] = parent
        if isinstance(node, Fork):
            self._index(node.if_true, (node.id, True))
            self._index(node.if_false, (node.id, False))

    def to_dict(self) -> dict[str, Any]:
        def dump(node: Node) -> dict[str, Any]:
            if isinstance(node, Leaf):
                return {"id": node.id, "label": node.label, **node.meta}
            return {
                "id": node.id,
                "label": node.label,
                "predicate": node.predicate.text,
                "knowledge": node.knowledge,
                "evidence": node.evidence,
                "reserve": node.reserve,
                "synthetic": node.synthetic,
                "true": dump(node.if_true),
                "false": dump(node.if_false),
            }

        return {
            "format": TREE_FORMAT,
            "id": self.id,
            "version": self.version,
            "schema": self.schema.to_dict(),
            "root": dump(self.root),
        }

    # ------------------------------------------------------------ навигация

    def forks(self) -> Iterator[Fork]:
        return (n for n in self.nodes.values() if isinstance(n, Fork))

    def leaves(self) -> Iterator[Leaf]:
        return (n for n in self.nodes.values() if isinstance(n, Leaf))

    def path_to(self, node_id: str) -> tuple[Step, ...]:
        """Путь от корня до узла: какие вилки и в какую сторону."""
        if node_id not in self.nodes:
            raise TreeError(f"Нет узла {node_id}")
        steps: list[Step] = []
        current = self.parent[node_id]
        while current is not None:
            fork_id, branch = current
            fork = self.nodes[fork_id]
            assert isinstance(fork, Fork)
            steps.append(Step(fork_id, fork.predicate.text, branch))
            current = self.parent[fork_id]
        return tuple(reversed(steps))

    def characteristic(self, node_id: str) -> Predicate:
        """Функция-характеристика узла χ (спецификация, раздел 5): конъюнкция вилок пути.

        χ_b(x) = ∏ b_i·δ_i(x) + (1 − b_i)·(1 − δ_i(x)) — в DSL это `δ_1 and not δ_2 and …`.
        Для корня χ ≡ true. Ядро компилирует χ как обычный предикат, поэтому её можно
        вычислить на записи и сверить с проходом по дереву.
        """
        parts = [f"({s.predicate})" if s.branch else f"not ({s.predicate})" for s in self.path_to(node_id)]
        return self.language.compile(" and ".join(parts) if parts else "true")

    def separating_fork(self, a: str, b: str) -> tuple[Step, Step] | None:
        """Вилка, на которой расходятся пути к узлам a и b (их ближайший общий предок).

        Возвращает (шаг для a, шаг для b) или None, если один узел лежит на пути к другому.
        """
        pa, pb = self.path_to(a), self.path_to(b)
        for sa, sb in zip(pa, pb):
            if sa.node != sb.node:
                break
            if sa.branch != sb.branch:
                return sa, sb
        return None

    # ------------------------------------------------------------ проверки

    def check(self, samples: Iterable[dict[str, Any]] = ()) -> list[Issue]:
        """Статические проверки и проверки на выборке. Ошибки делают дерево недопустимым."""
        issues: list[Issue] = []
        required = {a.name for a in self.schema.attributes.values() if a.required}
        descriptive = {a.name for a in self.schema.attributes.values() if not a.defining}

        def walk(node: Node, present: frozenset[str], steps: tuple[Step, ...]) -> None:
            if isinstance(node, Leaf):
                return
            present = present | self._implied_by_conditions(steps)
            risky = node.predicate.unguarded() - required - present
            if risky:
                issues.append(Issue(
                    "error", "non_total", node.id,
                    f"Вилка не тотальна: атрибуты {sorted(risky)} могут отсутствовать. "
                    f"Сделайте их обязательными или поставьте выше вилку has(...)",
                ))
            used_descriptive = sorted(_attrs(node.predicate.node, self.language) & descriptive)
            if used_descriptive:
                issues.append(Issue(
                    "warning", "descriptive_attr", node.id,
                    f"Вилка опирается на описательные атрибуты {used_descriptive} (S-14)",
                ))
            text = node.predicate.text
            walk(node.if_true, present | node.predicate.present_if(True), steps + (Step(node.id, text, True),))
            walk(node.if_false, present | node.predicate.present_if(False), steps + (Step(node.id, text, False),))

        walk(self.root, frozenset(), ())

        samples = list(samples)
        if samples:
            counts = {f.id: [0, 0] for f in self.forks()}
            for i, raw in enumerate(samples):
                result = self.classify(raw, certify=False)
                if result.status == INVALID:
                    issues.append(Issue("error", "invalid_sample", self.root.id, f"Пример {i}: {'; '.join(result.errors)}"))
                    continue
                if result.status == UNKNOWN and result.missing:
                    issues.append(Issue(
                        "error", "out_of_domain", result.stopped_at or self.root.id,
                        f"Пример {i}: нет атрибутов {list(result.missing)}",
                    ))
                for step in result.path:
                    counts[step.node][0 if step.branch else 1] += 1
            for fork in self.forks():
                t, f = counts[fork.id]
                if (t == 0 or f == 0) and not fork.reserve and (t + f) > 0:
                    empty = "true" if t == 0 else "false"
                    issues.append(Issue(
                        "warning", "degenerate", fork.id,
                        f"Ветвь {empty} пуста на выборке ({t + f} примеров дошли до вилки)",
                    ))
        return issues

    def _implied_by_conditions(self, steps: tuple[Step, ...]) -> frozenset[str]:
        """Атрибуты, обязательные по required_when, условие которых прямо выполнено на пути.

        Проверка намеренно синтаксическая (без решателя): условие C выполнено, если на пути
        есть шаг «C = true», или C имеет вид `not X` и на пути есть шаг «X = false».
        """
        facts = {(s.predicate, s.branch) for s in steps}
        out = set()
        for name, cond in self.language.conditions.items():
            node = cond.node
            if (cond.text, True) in facts:
                out.add(name)
            elif node.kind == "not" and (to_text(node.args[0]), False) in facts:
                out.add(name)
        return frozenset(out)

    # ------------------------------------------------------------ классификация

    def classify(self, record: dict[str, Any], certify: bool = True) -> Classification:
        try:
            coerced = self.schema.coerce_record(record)
        except SchemaError as exc:
            return Classification(INVALID, None, (), errors=tuple(str(exc).split("; ")))
        cond_errors = self.language.condition_errors(coerced)
        if cond_errors:
            return Classification(INVALID, None, (), errors=tuple(cond_errors))
        path: list[Step] = []
        assumptions: list[str] = []
        node: Node = self.root
        while isinstance(node, Fork):
            ev = node.predicate.evaluate(coerced)
            if ev.value is OUT_OF_DOMAIN:
                result = Classification(
                    UNKNOWN, None, tuple(path), missing=ev.missing, stopped_at=node.id,
                    assumptions=tuple(assumptions),
                )
                break
            path.append(Step(node.id, node.predicate.text, ev.value))
            if node.knowledge == "hypothesis":
                assumptions.append(node.id)
            node = node.if_true if ev.value else node.if_false
        else:
            # Ответ, опирающийся на гипотезу, не может быть proven (S-11).
            status = UNKNOWN if assumptions else PROVEN
            result = Classification(status, node.id, tuple(path), assumptions=tuple(assumptions))
        if certify:
            result.certificate = self._certificate(coerced, result)
        return result

    def _certificate(self, record: dict[str, Any], result: Classification) -> dict[str, Any]:
        body = {
            "format": CERT_FORMAT,
            "tree": {"id": self.id, "version": self.version, "hash": self.hash},
            "schema_hash": self.schema.hash(),
            "record_hash": digest(record),
            "status": result.status,
            "leaf": result.leaf,
            "path": [{"node": s.node, "predicate": s.predicate, "branch": s.branch} for s in result.path],
            "missing": list(result.missing),
            "stopped_at": result.stopped_at,
            "assumptions": list(result.assumptions),
        }
        return {**body, "hash": digest(body)}

    def verify_certificate(self, record: dict[str, Any], certificate: dict[str, Any]) -> tuple[bool, str]:
        """Независимо пересчитывает ответ и сверяет его с сертификатом."""
        body = {k: v for k, v in certificate.items() if k != "hash"}
        if digest(body) != certificate.get("hash"):
            return False, "хэш сертификата не совпадает с его содержимым"
        if body.get("tree", {}).get("hash") != self.hash:
            return False, "сертификат выдан другой версией дерева"
        fresh = self.classify(record)
        if fresh.certificate != certificate:
            return False, "пересчёт дал другой результат"
        return True, "ok"


def _attrs(node: Any, language: Language) -> set[str]:
    """Все атрибуты схемы, на которые опирается выражение, включая производные."""
    out: set[str] = set()
    if node.kind in ("attr", "has"):
        out.add(node.value)
    if node.kind == "derived":
        out |= _attrs(language.derived[node.value], language)
    if node.kind in ("all", "any"):
        out.add(node.value[1])
    for a in node.args:
        out |= _attrs(a, language)
    return out
