"""Самопостановка и выполнение задачи (Delta LM, разделы 3.3, 3.4, 3.6).

Гипотеза понимания превращается в задачу по её типу (DEFINE, DIFFERENCE, IS_A,
CLASSIFY). Каждый шаг выполняет ядро: классификация с сертификатом, путь к узлу,
разделяющая вилка. Текст собирается только из результатов ядра и словаря:
названия понятий, фразы ветвей вилок, описательные данные источника.
Если задача к понятиям неприменима, гипотеза отбрасывается с объяснением.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from delta_kernel import PROVEN, digest
from delta_kernel.tree import Fork
from delta_text.knowledge import Concept, KnowledgeBase

TREE_TITLES = {"matter": "дерево материи", "elements": "дерево химических элементов"}


class NotApplicable(Exception):
    pass


@dataclass
class TaskResult:
    lines: list[str]
    restatement: str
    proven: bool  # все выводы ядра доказаны
    caveats: list[str] = field(default_factory=list)  # оговорки данных
    certificates: list[dict[str, Any]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)


def execute(intent: str, keys: list[str], kb: KnowledgeBase) -> TaskResult:
    concepts = [kb.concepts[k] for k in keys]
    if intent == "DEFINE":
        if len(concepts) != 1:
            raise NotApplicable("для вопроса «что это» нужно одно понятие")
        return _define(concepts[0], kb)
    if intent == "CLASSIFY":
        if len(concepts) != 1:
            raise NotApplicable("для вопроса о классе нужно одно понятие")
        return _classify_only(concepts[0], kb)
    if intent == "IS_A":
        if len(concepts) != 2 or concepts[1].kind != "class":
            raise NotApplicable("для вопроса о принадлежности нужны объект или понятие и класс")
        return _is_a(concepts[0], concepts[1], kb)
    if intent == "DIFFERENCE":
        if len(concepts) != 2:
            raise NotApplicable("для сравнения нужны ровно два понятия")
        return _difference(concepts[0], concepts[1], kb)
    raise NotApplicable("тип вопроса не определён")


# ---------------------------------------------------------------- «что это»


def _define(c: Concept, kb: KnowledgeBase) -> TaskResult:
    if c.kind == "instance":
        return _describe_instance(c, kb, restatement=f"что такое «{c.name}»")
    tree = kb.tree_of(c)
    genus, steps = kb.genus(c.tree, c.node)
    lines = []
    if genus is None:
        lines.append(f"«{_cap(c.name)}» — исходное понятие ({TREE_TITLES.get(c.tree, c.tree)}).")
    else:
        genus_name = kb.concepts[f"{c.tree}:{genus}"].name
        differentia = ", ".join(kb.say(c.tree, f, b) for f, b in steps)
        lines.append(f"«{_cap(c.name)}» — вид понятия «{genus_name}»; отличительный признак: {differentia}.")
    node = tree.nodes[c.node]
    if isinstance(node, Fork):
        parts = [_division_part(kb, c.tree, node, branch) for branch in (True, False)]
        lines.append(f"Делится на: {parts[0]} и {parts[1]}.")
    lines.append(_path_line(kb, c.tree, c.node))
    cert = _structural_certificate(kb, c)
    return TaskResult(lines, f"что такое «{c.name}»", True, certificates=[cert],
                      plan=["найти узел понятия в дереве", "взять род и видовое отличие", "описать деление узла"])


def _division_part(kb: KnowledgeBase, tree_id: str, fork: Fork, branch: bool) -> str:
    child = fork.if_true if branch else fork.if_false
    names = []
    # синтетические узлы раскрываются до настоящих понятий
    stack = [child]
    while stack:
        n = stack.pop(0)
        key = kb.class_key(tree_id, n.id)
        if key is not None:
            names.append(f"«{kb.concepts[key].name}»")
        elif isinstance(n, Fork):
            stack.extend([n.if_true, n.if_false])
    return f"{', '.join(names)} ({kb.say(tree_id, fork.id, branch)})"


# ---------------------------------------------------------------- объекты


def _describe_instance(c: Concept, kb: KnowledgeBase, restatement: str) -> TaskResult:
    facts = ", ".join(f"{k} {v}" for k, v in c.facts)
    lines = [f"«{_cap(c.name)}» — {c.category}; {facts}."]
    result = TaskResult(lines, restatement, True, plan=["взять данные объекта из источника"])
    _add_classifications(c, kb, result)
    return result


def _classify_only(c: Concept, kb: KnowledgeBase) -> TaskResult:
    if c.kind == "class":
        genus, _ = kb.genus(c.tree, c.node)
        if genus is None:
            line = f"«{_cap(c.name)}» — корневое понятие ({TREE_TITLES.get(c.tree, c.tree)}), выше классов нет."
        else:
            line = f"«{_cap(c.name)}» относится к классу «{kb.concepts[f'{c.tree}:{genus}'].name}»."
        return TaskResult([line, _path_line(kb, c.tree, c.node)], f"к какому классу относится «{c.name}»",
                          True, certificates=[_structural_certificate(kb, c)])
    result = TaskResult([], f"к какому классу относится «{c.name}»", True)
    _add_classifications(c, kb, result)
    if not result.lines:
        raise NotApplicable(f"для «{c.name}» нет записей ни в одном дереве")
    return result


def _add_classifications(c: Concept, kb: KnowledgeBase, result: TaskResult) -> None:
    for tree_id in ("elements", "matter"):
        if tree_id not in c.records:
            continue
        tree = kb.trees[tree_id]
        cls = tree.classify(c.records[tree_id])
        result.certificates.append(cls.certificate)
        result.plan.append(f"классифицировать объект: {TREE_TITLES[tree_id]}")
        caveats = c.data_assumptions.get(tree_id, [])
        result.caveats.extend(caveats)
        if cls.status != PROVEN or cls.leaf is None:
            result.proven = False
            missing = ", ".join(cls.missing) or "—"
            result.lines.append(f"{_tree_prefix(tree_id)}: не определено (нет данных: {missing}).")
            continue
        leaf = kb.concepts[f"{tree_id}:{cls.leaf}"].name
        steps = " → ".join(kb.say(tree_id, s.node, s.branch) for s in cls.path)
        result.lines.append(f"{_tree_prefix(tree_id)}: {leaf}. Путь: {steps}.")


def _tree_prefix(tree_id: str) -> str:
    return {"elements": "Класс элемента", "matter": "Агрегатное состояние при стандартных условиях"}[tree_id]


# ---------------------------------------------------------------- принадлежность


def _is_a(x: Concept, cls: Concept, kb: KnowledgeBase) -> TaskResult:
    restatement = f"относится ли «{x.name}» к «{cls.name}»"
    tree = kb.tree_of(cls)
    if x.kind == "instance":
        record = x.records.get(cls.tree)
        if record is None:
            raise NotApplicable(f"«{x.name}» не описан в дереве, где лежит «{cls.name}»")
        res = tree.classify(record)
        caveats = list(x.data_assumptions.get(cls.tree, []))
        if res.status != PROVEN or res.leaf is None:
            return TaskResult([f"Не определено: нет данных ({', '.join(res.missing)})."], restatement, False,
                              caveats, [res.certificate])
        on_path = {s.node for s in res.path} | {res.leaf}
        target_path = [s.node for s in res.path] + [res.leaf]
        leaf_name = kb.concepts[f"{cls.tree}:{res.leaf}"].name
        certs = [res.certificate]
    else:
        if x.tree != cls.tree:
            raise NotApplicable(f"«{x.name}» и «{cls.name}» лежат в разных деревьях")
        target_path = [s.node for s in tree.path_to(x.node)] + [x.node]
        on_path = set(target_path)
        leaf_name = x.name
        caveats = []
        certs = [_structural_certificate(kb, x)]
    if cls.node in on_path:
        line = f"Да: «{x.name}» относится к «{cls.name}»" + (f" (класс: {leaf_name})." if leaf_name != cls.name else ".")
    else:
        # вилка, на которой путь объекта уходит от класса
        split = tree.separating_fork(target_path[-1], cls.node)
        if split is not None:
            own, _ = split
            line = (f"Нет: «{x.name}» — {kb.say(cls.tree, own.node, own.branch)}, "
                    f"поэтому к «{cls.name}» не относится (класс: {leaf_name}).")
        else:
            line = f"Нет: «{x.name}» — более общее понятие, чем «{cls.name}»."
    return TaskResult([line], restatement, True, caveats, certs,
                      plan=["классифицировать объект в дереве класса", "проверить, лежит ли класс на пути"])


# ---------------------------------------------------------------- различие


def _difference(a: Concept, b: Concept, kb: KnowledgeBase) -> TaskResult:
    restatement = f"чем отличаются «{a.name}» и «{b.name}»"
    if a.kind == "class" and b.kind == "class":
        if a.tree != b.tree:
            raise NotApplicable(f"«{a.name}» и «{b.name}» лежат в разных деревьях, общей вилки нет")
        tree = kb.tree_of(a)
        split = tree.separating_fork(a.node, b.node)
        certs = [_structural_certificate(kb, a), _structural_certificate(kb, b)]
        if split is None:
            general, special = (a, b) if a.node in {s.node for s in tree.path_to(b.node)} else (b, a)
            genus, steps = kb.genus(special.tree, special.node)
            diff = ", ".join(kb.say(special.tree, f, br) for f, br in steps)
            return TaskResult(
                [f"«{_cap(special.name)}» — частный случай понятия «{general.name}»: всякое "
                 f"«{special.name}» есть «{general.name}». Отличительный признак внутри рода: {diff}."],
                restatement, True, certificates=certs)
        sa, sb = split
        common = _named_ancestor(kb, a.tree, sa.node)
        lines = [
            f"«{_cap(a.name)}» и «{b.name}» расходятся на вилке рода «{common}»:",
            f"• {a.name} — {kb.say(a.tree, sa.node, sa.branch)};",
            f"• {b.name} — {kb.say(b.tree, sb.node, sb.branch)}.",
        ]
        return TaskResult(lines, restatement, True, certificates=certs,
                          plan=["найти пути к понятиям", "найти вилку, где пути расходятся"])
    if a.kind == "instance" and b.kind == "instance":
        result = TaskResult([], restatement, True,
                            plan=["классифицировать оба объекта", "сравнить пути в каждом дереве"])
        for tree_id in ("elements", "matter"):
            if tree_id not in a.records or tree_id not in b.records:
                continue
            tree = kb.trees[tree_id]
            ra, rb = tree.classify(a.records[tree_id]), tree.classify(b.records[tree_id])
            result.certificates += [ra.certificate, rb.certificate]
            result.caveats += a.data_assumptions.get(tree_id, []) + b.data_assumptions.get(tree_id, [])
            if ra.status != PROVEN or rb.status != PROVEN:
                result.proven = False
                result.lines.append(f"{_tree_prefix(tree_id)}: не определено.")
                continue
            la, lb = (kb.concepts[f"{tree_id}:{r.leaf}"].name for r in (ra, rb))
            if ra.leaf == rb.leaf:
                result.lines.append(f"{_tree_prefix(tree_id)}: у обоих — {la}, здесь различия нет.")
                continue
            sa, sb = tree.separating_fork(ra.leaf, rb.leaf)
            result.lines.append(
                f"{_tree_prefix(tree_id)}: {a.name} — {la}, {b.name} — {lb}. "
                f"Расходятся на вилке: {kb.say(tree_id, sa.node, sa.branch)} / {kb.say(tree_id, sb.node, sb.branch)}."
            )
        fa, fb = dict(a.facts), dict(b.facts)
        shared = [k for k in fa if k in fb and fa[k] != fb[k]]
        if shared:
            result.lines.append("Описательные данные: " + "; ".join(f"{k}: {fa[k]} и {fb[k]}" for k in shared) + ".")
        if not result.lines:
            raise NotApplicable(f"«{a.name}» и «{b.name}» не описаны в общих деревьях")
        return result
    instance, cls = (a, b) if a.kind == "instance" else (b, a)
    result = _is_a(instance, cls, kb)
    result.restatement = restatement
    result.lines.insert(0, f"«{_cap(instance.name)}» — объект, «{cls.name}» — класс; сравниваю по принадлежности.")
    return result


# ---------------------------------------------------------------- общее


def _path_line(kb: KnowledgeBase, tree_id: str, node_id: str) -> str:
    tree = kb.trees[tree_id]
    root_name = kb.concepts[f"{tree_id}:{tree.root.id}"].name
    steps = [kb.say(tree_id, s.node, s.branch) for s in tree.path_to(node_id)]
    if not steps:
        return f"Путь: корень ({TREE_TITLES.get(tree_id, tree_id)})."
    return f"Путь от «{root_name}»: " + " → ".join(steps) + "."


def _named_ancestor(kb: KnowledgeBase, tree_id: str, fork_id: str) -> str:
    """Название рода для вилки: сама вилка или ближайший несинтетический предок."""
    key = kb.class_key(tree_id, fork_id)
    if key is not None:
        return kb.concepts[key].name
    genus, _ = kb.genus(tree_id, fork_id)
    return kb.concepts[f"{tree_id}:{genus}"].name if genus else tree_id


def _structural_certificate(kb: KnowledgeBase, c: Concept) -> dict[str, Any]:
    """Сертификат положения понятия в дереве: дерево, его хэш и путь к узлу."""
    tree = kb.tree_of(c)
    body = {
        "format": "delta-node-certificate@0.1",
        "tree": {"id": tree.id, "version": tree.version, "hash": tree.hash},
        "node": c.node,
        "path": [{"node": s.node, "predicate": s.predicate, "branch": s.branch} for s in tree.path_to(c.node)],
    }
    return {**body, "hash": digest(body)}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]
