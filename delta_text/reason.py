"""Самопостановка и выполнение задачи (Delta LM, разделы 3.3, 3.4, 3.6).

Гипотеза понимания превращается в задачу по её типу (DEFINE, DIFFERENCE, IS_A,
CLASSIFY, WHY). Каждый шаг выполняет ядро: классификация с сертификатом, путь к узлу,
разделяющая вилка. Текст собирается только из результатов ядра и словаря:
названия понятий, фразы ветвей вилок, описательные данные источника.
Если задача к понятиям неприменима, гипотеза отбрасывается с объяснением.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from delta_kernel import PROVEN, digest
from delta_kernel.tree import Fork
from fractions import Fraction

from delta_science.atoms import elements, format_configuration
from delta_science.atoms import structure as atom_structure
from delta_science.bonds import ORDER_NAMES, compound_structure, fmt, simple_substance
from delta_science.causes import (BOND_NAMES, SubstanceRecord, attributes_of, class_record, compound_record,
                                  element_record, evaluate)
from delta_science.causes import rules as causal_rules
from delta_science.pyramid import unfold_compound, unfold_element
from delta_science.smiles import SmilesError
from delta_text.knowledge import Concept, KnowledgeBase
from delta_text.morph import SERVICE_POS, normalize, parses

TREE_TITLES = {"matter": "дерево материи", "elements": "дерево химических элементов",
               "bonds": "дерево химической связи", "particles": "дерево элементарных частиц",
               "substances": "дерево веществ"}
TERMS_CAVEAT = "определение термина сформулировано по школьным учебникам и ещё не проверено человеком"


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


def execute(intent: str, keys: list[str], kb: KnowledgeBase, lemmas: frozenset[str] = frozenset()) -> TaskResult:
    """lemmas — леммы вопроса: нужны «почему», чтобы найти признак, о котором спрашивают."""
    concepts = [kb.concepts[k] for k in keys]
    terms = [c for c in concepts if c.kind == "term"]
    if terms:
        if intent in ("DEFINE", "DIFFERENCE", "CLASSIFY") and len(terms) == len(concepts):
            return _define_terms(terms, kb)
        raise NotApplicable("для терминов могу дать только определения")
    if intent == "WHY":
        return _why(concepts, kb, lemmas)
    if intent == "UNFOLD":
        if len(concepts) != 1 or concepts[0].kind != "instance":
            raise NotApplicable("развёртку от частиц до вещества могу построить для одного элемента или вещества")
        return _unfold(concepts[0])
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
    if intent == "OTHER":
        names = " или ".join(f"«{c.name}»" for c in concepts)
        raise NotApplicable(
            "это вопрос вида «как», «где», «сколько» и т. п.; такие я пока не умею. "
            f"Могу объяснить, что такое {names}, сравнить понятия или проверить принадлежность классу"
        )
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
    lines.append(f"Функция-характеристика: χ = {tree.characteristic(c.node).text}")
    caveats = []
    if c.definition:
        lines.append(f"Определение по учебнику: {c.definition}.")
        caveats.append(TERMS_CAVEAT)
    lines += wiki_lines(kb, c.name, *c.names, c.label)
    cert = _structural_certificate(kb, c)
    return TaskResult(lines, f"что такое «{c.name}»", True, caveats, certificates=[cert],
                      plan=["найти узел понятия в дереве", "взять род и видовое отличие", "описать деление узла"])


def wiki_lines(kb: KnowledgeBase, *names: str) -> list[str]:
    """Цитата из словаря RazumWiki с указанием источника; ядром не проверяется."""
    entry = kb.definition(*names)
    if entry is None:
        return []
    return [f"Словарь RazumWiki (цитата, ядром не проверялась): «{entry['term']}» — {entry['definition']} "
            f"[{entry['url']}]"]


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
    result.lines += structure_lines(c)
    result.lines += wiki_lines(kb, *c.names)
    return result


def _define_terms(terms: list[Concept], kb: KnowledgeBase) -> TaskResult:
    lines = [f"«{_cap(t.name)}» — {t.definition}." for t in terms]
    for t in terms:
        lines += wiki_lines(kb, *t.names)
    names = " и ".join(f"«{t.name}»" for t in terms)
    return TaskResult(lines, f"что такое {names}", True, [TERMS_CAVEAT], plan=["взять определение термина"])


# ---------------------------------------------------------------- развёртка


def _unfold(c: Concept) -> TaskResult:
    """Пирамида от элементарных частиц до агрегатного состояния (delta_science.pyramid)."""
    if c.z is not None:
        steps = unfold_element(c.z)
    elif c.smiles:
        steps = unfold_compound(c.smiles, parse_formula_hill(c.smiles), c.phase)
    else:
        raise NotApplicable(f"для «{c.name}» нет данных о строении")
    lines = [f"Развёртка «{c.name}» по пирамиде уровней, сверху вниз:"]
    proven = True
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step.title}: " + " ".join(step.lines))
        if step.characteristic:
            lines.append(f"   χ = {step.characteristic}")
        proven = proven and step.proven
    caveats = [] if proven else ["часть шагов опирается на вилки индуктора или непроверенные данные (отмечено в шагах)"]
    return TaskResult(lines, f"из чего и как построено «{c.name}»", True, caveats,
                      plan=["частицы", "ядро", "атом", "связь", "вещество", "состояние"])


def parse_formula_hill(smiles: str) -> str:
    """Формула по графу SMILES (для развёртки соединения)."""
    from delta_science.smiles import parse

    counts = parse(smiles).formula()
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))


# ---------------------------------------------------------------- строение


def structure_lines(c: Concept) -> list[str]:
    """Строение атома и простого вещества (элемент) или граф связей (соединение)."""
    if c.z is not None:
        return _atom_lines(c.z)
    if c.smiles:
        return _compound_lines(c.smiles)
    return []


def _atom_lines(z: int) -> list[str]:
    e = elements()[z]
    s = atom_structure(z)
    group = f"группа {s.group}" if s.group else "f-элемент"
    lines = [
        f"Строение атома: ядро — {plural(s.protons, 'протон')} и {plural(s.neutrons, 'нейтрон')} (школьное правило: "
        f"округлённая атомная масса {round(e.atomic_mass)} минус {z}); {plural(s.electrons, 'электрон')} по слоям "
        f"{', '.join(map(str, s.shells))}: {format_configuration(s.configuration)}. На внешнем слое "
        f"{plural(s.outer_electrons, 'электрон')}{' (слой завершён)' if s.outer_full else ''}; период {s.period}, {group}.",
    ]
    if s.anomaly or s.notes:
        lines.append("Особенность: " + "; ".join(s.notes) + ".")
    props = [("электроотрицательность", e.electronegativity, ""), ("энергия ионизации", e.ionization_energy, " эВ"),
             ("радиус атома", e.atomic_radius, " пм"), ("температура плавления", e.melting_point, " K"),
             ("температура кипения", e.boiling_point, " K"), ("плотность", e.density, " г/см³")]
    known = [f"{name} {fmt(v, 3)}{unit}" for name, v, unit in props if v is not None]
    if known:
        lines.append("Данные PubChem: " + "; ".join(known) + ".")
    bond = simple_substance(z)
    if bond is None:
        lines.append("Простое вещество: отдельные атомы, химических связей между ними нет (внешний слой завершён).")
    elif bond.kind is None:
        lines.append(f"Простое вещество: тип связи не определён ({bond.note}).")
    else:
        why = "атомы металла" if bond.kind == "metallic" else "одинаковые атомы неметалла, разность электроотрицательностей 0"
        lines.append(f"Простое вещество: {BOND_NAMES[bond.kind]} ({why}; дерево химической связи).")
    return lines


PLURAL_ORDERS = {"одинарная": "одинарные", "двойная": "двойные", "тройная": "тройные", "ароматическая": "ароматические"}


def _order_name(order: Fraction, n: int) -> str:
    name = ORDER_NAMES.get(order, str(order))
    return name if n == 1 else PLURAL_ORDERS.get(name, name)


def plural(n: int, word: str) -> str:
    """«1 электрон», «2 электрона», «5 электронов» (для слов на согласную)."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {word}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {word}а"
    return f"{n} {word}ов"


def _compound_lines(smiles: str) -> list[str]:
    try:
        mol, bonds = compound_structure(smiles)
    except SmilesError as exc:
        return [f"Строение: запись SMILES {smiles} не разобрана ({exc})."]
    parts = []
    for b in bonds:
        if b.kind == "ionic" and Fraction(0) in b.orders:
            text = f"{b.a} и {b.b} — ионная связь между ионами"
        else:
            orders = ", ".join(f"{n} {_order_name(o, n)}" for o, n in sorted(b.orders.items()))
            kind = BOND_NAMES.get(b.kind, "тип не определён")
            text = f"{b.a}–{b.b} ×{b.count} ({orders}) — {kind}"
        if b.delta_en is not None:
            text += f", ΔЭО {fmt(b.delta_en)}"
        if b.note:
            text += f"; {b.note}"
        parts.append(text)
    n_atoms = len(mol.atoms) + sum(a.hydrogens for a in mol.atoms)
    return [f"Строение (SMILES {smiles}, PubChem): атомов {n_atoms}; связи: " + "; ".join(parts) + "."]


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
    for tree_id in ("elements", "matter", "substances"):
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
    return {"elements": "Класс элемента", "matter": "Агрегатное состояние при стандартных условиях",
            "substances": "Вещество по составу и строению"}[tree_id]


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


# ---------------------------------------------------------------- «почему»
#
# Объяснение не выучивается, а выводится: путь в дереве и есть основание, по которому
# объект отнесён к классу. Причинных знаний («почему так устроена природа») в базе нет,
# и ответ говорит об этом прямо.


def _why(concepts: list[Concept], kb: KnowledgeBase, lemmas: frozenset[str]) -> TaskResult:
    if len(concepts) == 2 and concepts[1].kind == "class":
        return _why_member(concepts[0], concepts[1], kb)
    if len(concepts) == 1:
        causal = _why_causal(concepts[0], kb, lemmas)
        return causal if causal is not None else _why_property(concepts[0], kb, lemmas)
    raise NotApplicable("для вопроса «почему» нужно одно понятие или объект и класс")


def substance_of(c: Concept) -> SubstanceRecord | None:
    if c.z is not None:
        return element_record(c.z)
    if c.smiles:
        return compound_record(c.smiles, c.phase, phase_verified=False)
    return class_record(c.key)


def _why_causal(c: Concept, kb: KnowledgeBase, lemmas: frozenset[str]) -> TaskResult | None:
    """Причинное правило учебника, следствие которого совпадает с вопросом."""
    sub = substance_of(c)
    if sub is None:
        return None
    subject = set().union(*(_content_lemmas(n) for n in c.names)) if c.names else set()
    asked = {w for w in lemmas if w not in WHY_STOP} - subject
    scored = []
    for rule in causal_rules():
        overlap = len(_content_lemmas(" ".join(rule.keywords)) & asked)
        if overlap:
            scored.append((overlap, rule, evaluate(rule, sub)))
    if not scored:
        return None
    best = max(o for o, _, ev in scored if ev.value is True) if any(ev.value is True for _, _, ev in scored) else None
    fired = [(r, ev) for o, r, ev in scored if ev.value is True and o == best]
    if fired:
        lines, caveats, certs = [], list(sub.caveats), []
        for rule, ev in fired:
            reasons = "; ".join(sub.provenance[a] for a in attributes_of(rule) if a in sub.provenance)
            lines += [f"«{_cap(c.name)}»: {rule.effect}.",
                      f"Механизм: {rule.mechanism}",
                      f"Основания для «{c.name}»: {reasons}.",
                      f"Правило «{rule.id}», источник: {rule.source}."]
            if rule.status != "accepted":
                caveats.append(f"причинное правило «{rule.id}» сформулировано по учебнику и ещё не проверено человеком")
            certs.append(_rule_certificate(rule, sub, ev))
        return TaskResult(lines, f"почему «{c.name}» {fired[0][0].effect}", True, caveats, certs,
                          plan=["построить запись вещества из данных", "найти причинное правило по вопросу",
                                "проверить условие правила ядром"])
    top = max(o for o, _, _ in scored)
    unmet = []
    for o, rule, ev in scored:
        if o != top:
            continue
        if ev.value is False:
            values = "; ".join(sub.provenance[a] for a in attributes_of(rule) if a in sub.provenance)
            unmet.append(f"«{rule.effect}» — условие не выполнено ({values})")
        else:
            unmet.append(f"«{rule.effect}» — не хватает данных: {', '.join(ev.missing)}")
    raise NotApplicable(
        f"ни одно из подходящих причинных правил к «{c.name}» не применимо: " + "; ".join(unmet)
        + ". Других правил об этом у меня нет, поэтому вывода в ту или другую сторону я не делаю"
    )


def _rule_certificate(rule: Any, sub: SubstanceRecord, ev: Any) -> dict[str, Any]:
    body = {"format": "delta-rule-certificate@0.1", "rule": rule.id, "status": rule.status,
            "predicate": rule.predicate.text, "record": sub.record, "value": ev.value}
    return {**body, "hash": digest(body)}


def _why_member(x: Concept, cls: Concept, kb: KnowledgeBase) -> TaskResult:
    restatement = f"почему «{x.name}» относится к «{cls.name}»"
    tree = kb.tree_of(cls)
    base = _is_a(x, cls, kb)
    base.restatement = restatement
    if not base.lines[0].startswith("Да"):
        base.lines.insert(0, "Посылка вопроса неверна.")
        return base
    if x.kind == "instance":
        record = x.records[cls.tree]
        res = tree.classify(record)
        nodes = [s.node for s in res.path] + [res.leaf]
        reasons = list(res.path[: nodes.index(cls.node)])
        lines = [f"«{_cap(x.name)}» относится к «{cls.name}», потому что на пути выполнены условия вилок:"]
        for s in reasons:
            values = ", ".join(f"{a} = {_value(v)}" for a, v in sorted(record.items()) if _mentions(s.predicate, a))
            lines.append(f"• {kb.say(cls.tree, s.node, s.branch)} — условие «{s.predicate}» "
                         f"{'выполнено' if s.branch else 'не выполнено'}; у «{x.name}»: {values}.")
        if not reasons:
            lines.append(f"• «{cls.name}» — корень дерева, к нему относится всё, что в нём классифицировано.")
        source = {"elements": "PubChem (класс элемента)",
                  "matter": "агрегатное состояние при стандартных условиях"}.get(cls.tree, cls.tree)
        lines.append(f"Значения признаков взяты из данных: {source}.")
        return TaskResult(lines, restatement, True, base.caveats, base.certificates,
                          plan=["классифицировать объект", "взять вилки пути до класса", "показать значения признаков"])
    path = tree.path_to(x.node)
    start = [s.node for s in path].index(cls.node) if cls.node != x.node else len(path)
    steps = [kb.say(cls.tree, s.node, s.branch) for s in path[start:]]
    if not steps:
        line = f"«{_cap(x.name)}» и «{cls.name}» — одно и то же понятие."
    else:
        line = (f"Так задано определениями: понятие «{x.name}» выделено внутри «{cls.name}» вилками: "
                f"{' → '.join(steps)}. Поэтому всё, что относится к «{x.name}», относится и к «{cls.name}».")
    return TaskResult([line], restatement, True, certificates=base.certificates,
                      plan=["найти путь от класса к понятию"])


def _why_property(c: Concept, kb: KnowledgeBase, lemmas: frozenset[str]) -> TaskResult:
    """«Почему газ не сохраняет объём?» — если признак есть на пути понятия, это определение."""
    asked = {w for w in lemmas if w not in WHY_STOP}
    paths: list[tuple[str, list[Any]]] = []
    if c.kind == "class":
        paths.append((c.tree, list(kb.tree_of(c).path_to(c.node))))
    else:
        for tree_id, record in c.records.items():
            res = kb.trees[tree_id].classify(record)
            if res.status == PROVEN:
                paths.append((tree_id, list(res.path)))
    # Шаг пути, который лучше всего совпадает с вопросом: по фразе ветви и по названию
    # понятия, в которое ветвь ведёт («твёрдый» → «твёрдое вещество»).
    best = None
    for tree_id, steps in paths:
        tree = kb.trees[tree_id]
        for depth, s in enumerate(steps):
            phrase = kb.say(tree_id, s.node, s.branch)
            fork = tree.nodes[s.node]
            child = fork.if_true if s.branch else fork.if_false
            child_key = kb.class_key(tree_id, child.id)
            child_name = kb.concepts[child_key].name if child_key else ""
            overlap = len((_content_lemmas(phrase) | _content_lemmas(child_name)) & asked)
            if overlap and (best is None or (overlap, depth) > (best[0], best[1])):
                best = (overlap, depth, tree_id, s, phrase, child_name)
    if best is None:
        raise NotApplicable(
            f"признака, о котором спрашивают, нет в моих деревьях, а причинных знаний нет совсем. "
            f"Могу объяснить, что такое «{c.name}», и показать классификацию"
        )
    _, _, tree_id, s, phrase, child_name = best
    genus = _named_ancestor(kb, tree_id, s.node)
    result_name = f" («{child_name}»)" if child_name and child_name != c.name else ""
    if c.kind == "class":
        line = (f"Так задано определением: понятие «{c.name}» отделено внутри «{genus}» вилкой "
                f"«{phrase}». Это определение, а не вывод из причин; причинных знаний о том, "
                "почему природа устроена так, в моей базе нет.")
        certs = [_structural_certificate(kb, c)]
    else:
        line = (f"Для «{c.name}» это следует из классификации: на вилке рода «{genus}» — "
                f"{phrase}{result_name}; условие «{s.predicate}» выполнено по данным. "
                "Причинных знаний в моей базе нет.")
        certs = [kb.trees[tree_id].classify(c.records[tree_id]).certificate]
    return TaskResult([line], f"почему «{c.name}» — {phrase}", True,
                      list(c.data_assumptions.get(tree_id, [])), certs,
                      plan=["найти признак на пути понятия", "показать вилку-определение"])


WHY_STOP = {"почему", "не", "ли", "так", "такой", "это", "быть", "являться"}


def _content_lemmas(text: str) -> set[str]:
    out = set()
    for word in text.replace("(", " ").replace(")", " ").split():
        variants = parses(word)
        if variants and variants[0].pos in SERVICE_POS:
            continue
        out |= {p.lemma for p in variants} | {normalize(word)}
    return out - WHY_STOP


def _mentions(predicate: str, attr: str) -> bool:
    return re.search(rf"\b{re.escape(attr)}\b", predicate) is not None


def _value(v: Any) -> str:
    return f"«{v}»" if isinstance(v, str) else ("да" if v is True else "нет" if v is False else str(v))


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
