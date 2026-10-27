"""Пирамида уровней материи: от элементарных частиц до химических связей и веществ.

Каждый уровень — дерево логических дельт ядра; уровни связаны функциями развёртки
«состоит из», которые строят объект следующего уровня из объектов предыдущего:

    Элементарные частицы (particles.json: нуклон / электрон, протон / нейтрон)
      ↓ Z протонов + N нейтронов
    Атомное ядро (заряд +Z, масса нуклонов)
      ↓ ядро + Z электронов, слои по правилу Клечковского
    Атом (atoms.json: класс элемента по строению, вилки открыты индуктором)
      ↓ пары атомов
    Химическая связь (bonds.json)
      ↓ атомы и связи
    Вещество (substances.json: простое / сложное, металл / ионное / ковалентное)
      ↓ агрегатное состояние при стандартных условиях
    Агрегатное состояние (дерево материи проекта: твёрдое, жидкость, газ)

Числа точные (Fraction): погрешности вычисления нет. Погрешность есть только у
измеренных констант (CODATA, PubChem), и она не скрывается.

Функция-характеристика каждого класса — `Tree.characteristic(node)`: конъюнкция
вилок пути от корня уровня.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from delta_kernel import PROVEN, Tree
from delta_science.atoms import elements, format_configuration, madelung_order, structure
from delta_science.bonds import fmt, simple_substance
from delta_science.causes import BOND_NAMES

ROOT = Path(__file__).resolve().parents[1]
TREES = ROOT / "data" / "trees"

# CODATA 2018 (рекомендованные значения), массы в а. е. м.
PROTON = {"charge": 1, "mass_u": Fraction("1.007276466621")}
NEUTRON = {"charge": 0, "mass_u": Fraction("1.00866491595")}
ELECTRON = {"charge": -1, "mass_u": Fraction("0.000548579909065")}
MEV_PER_U = Fraction("931.49410242")  # энергия покоя 1 а. е. м., МэВ
CODATA = "CODATA 2018"

# Моноизотопные элементы: стандартная атомная масса — масса единственного (практически)
# стабильного изотопа, поэтому для них дефект массы атома вычисляется честно.
MONONUCLIDIC = {"Be", "F", "Na", "Al", "P", "Sc", "Mn", "Co", "As", "Y", "Nb", "Rh", "I", "Cs", "Pr",
                "Tb", "Ho", "Tm", "Au", "Bi"}

CLASS_NAMES = {  # класс PubChem → название класса (дерево химических элементов)
    "Alkali metal": "щелочной металл", "Alkaline earth metal": "щёлочноземельный металл",
    "Transition metal": "переходный металл", "Lanthanide": "лантаноид", "Actinide": "актиноид",
    "Post-transition metal": "постпереходный металл", "Metalloid": "полуметалл", "Noble gas": "благородный газ",
    "Halogen": "галоген", "Nonmetal": "прочий неметалл",
}
STATE_NODES = {"solid": "Solid", "liquid": "Liquid", "gas": "NeutralGas"}

ATOM_SCHEMA = {
    "name": "AtomStructure",
    "version": "0.1.0",
    "attributes": [
        {"name": "outer", "type": {"kind": "int"}, "doc": "электронов на внешнем слое"},
        {"name": "period", "type": {"kind": "int"}, "doc": "номер внешнего слоя (период)"},
        {"name": "block", "type": {"kind": "enum", "values": ["s", "p", "d", "f"]},
         "doc": "подуровень, заполняемый последним"},
        {"name": "full", "type": {"kind": "bool"}, "doc": "внешний слой завершён"},
        {"name": "d_next", "type": {"kind": "int"}, "doc": "d-электронов предвнешнего слоя"},
        {"name": "f_inner", "type": {"kind": "int"}, "doc": "f-электронов третьего снаружи слоя"},
        {"name": "outer_minus_period", "type": {"kind": "int"}, "doc": "внешние электроны минус период"},
    ],
}


def atom_features(z: int) -> dict[str, Any]:
    """Признаки строения атома, выведенные только из Z (правило Клечковского)."""
    config = structure(z).madelung
    period = max(n for n, _, _ in config)
    outer = sum(k for n, _, k in config if n == period)
    letters = {letter for n, letter, k in config if n == period}
    full = (period == 1 and outer == 2) or (outer == 8 and letters == {"s", "p"})
    order = madelung_order()
    last = max(config, key=lambda t: order.index((t[0], t[1])))
    return {
        "outer": outer,
        "period": period,
        "block": last[1],
        "full": full,
        "d_next": sum(k for n, letter, k in config if n == period - 1 and letter == "d"),
        "f_inner": sum(k for n, letter, k in config if n == period - 2 and letter == "f"),
        "outer_minus_period": outer - period,
    }


@lru_cache(maxsize=None)
def tree(name: str) -> Tree:
    return Tree.from_dict(json.loads((TREES / f"{name}.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def matter_tree() -> Tree:
    from delta_tools.bridge_matter import matter_tree_from_deltas

    return Tree.from_dict(matter_tree_from_deltas(json.loads((ROOT / "data" / "matter-deltas.json").read_text(encoding="utf-8"))))


@dataclass
class Step:
    level: str
    title: str
    lines: list[str]
    tree: str | None = None
    node: str | None = None  # узел, в который пришёл объект
    characteristic: str = ""  # χ этого узла
    proven: bool = True
    numbers: dict[str, Any] = field(default_factory=dict)


def _classify(tree_name: str, record: dict[str, Any], t: Tree | None = None) -> tuple[Any, str]:
    t = t or tree(tree_name)
    result = t.classify(record)
    if result.leaf is None:
        return result, ""
    return result, t.characteristic(result.leaf).text


def unfold_element(z: int) -> list[Step]:
    """Развёртка элемента сверху донизу: частицы → ядро → атом → связь → вещество → состояние."""
    e = elements()[z]
    s = structure(z)
    n = s.neutrons
    steps: list[Step] = []

    particle_lines = []
    for name, rec, count in (("протон", PROTON, z), ("нейтрон", NEUTRON, n), ("электрон", ELECTRON, z)):
        res, chi = _classify("particles", rec)
        label = tree("particles").nodes[res.leaf].label
        charge = f"{rec['charge']:+d}" if rec["charge"] else "0"
        particle_lines.append(f"{label}: заряд {charge} e, масса {fmt(rec['mass_u'], 15)} а. е. м.; "
                              f"в атоме: {count}; χ = {chi}")
    steps.append(Step("particles", "Элементарные частицы", particle_lines, "particles",
                      numbers={"protons": z, "neutrons": n, "electrons": z}))

    nucleon_mass = z * PROTON["mass_u"] + n * NEUTRON["mass_u"]
    steps.append(Step("nucleus", "Атомное ядро", [
        f"Состав: {z} протонов + {n} нейтронов (нейтроны — школьное правило: округлённая масса {round(e.atomic_mass)} − Z).",
        f"Заряд ядра: +{z} e (точно: {z} · (+1)).",
        f"Масса нуклонов по отдельности: {z} · {fmt(PROTON['mass_u'], 12)} + {n} · {fmt(NEUTRON['mass_u'], 11)} "
        f"= {fmt(nucleon_mass, 12)} а. е. м. (точная арифметика, константы {CODATA}).",
    ], numbers={"charge": z, "nucleon_mass_u": nucleon_mass}))

    parts_mass = nucleon_mass + z * ELECTRON["mass_u"]
    atom_lines = [
        f"Ядро + {z} электронов: заряд {z} − {z} = 0, атом электронейтрален.",
        f"Электроны по слоям {', '.join(map(str, s.shells))}: {format_configuration(s.configuration)}"
        + (f" (по правилу Клечковского {format_configuration(s.madelung)}; провал электрона — из данных PubChem)"
           if s.anomaly else " (выведено из Z по правилу Клечковского, совпадает с PubChem)") + ".",
        f"Масса частей атома: {fmt(parts_mass, 12)} а. е. м.",
    ]
    numbers = {"parts_mass_u": parts_mass}
    if e.symbol in MONONUCLIDIC:
        defect = parts_mass - e.atomic_mass
        energy = defect * MEV_PER_U
        atom_lines.append(
            f"Масса атома (PubChem, единственный стабильный изотоп): {fmt(e.atomic_mass, 9)} а. е. м. "
            f"Дефект массы: {fmt(defect, 9)} а. е. м. = энергия связи ядра {fmt(energy, 3)} МэВ "
            f"(Δm · {fmt(MEV_PER_U, 8)} МэВ/а. е. м.). Атом легче суммы своих частиц — это энергия связи."
        )
        numbers.update(defect_u=defect, binding_mev=energy)
    else:
        atom_lines.append("Дефект массы не вычисляется: у элемента несколько изотопов, а PubChem даёт среднюю массу.")
    res, chi = _classify("atoms", atom_features(z))
    leaf = tree("atoms").nodes[res.leaf] if res.leaf else None
    cls = CLASS_NAMES.get(leaf.meta.get("class"), "?") if leaf else "не определён"
    atom_lines.append(f"Класс по строению атома (вилки индуктора): {cls}; совпадает с классом PubChem: "
                      f"{'да' if leaf and leaf.meta.get('class') == e.group_block else 'НЕТ'}.")
    steps.append(Step("atoms", "Атом", atom_lines, "atoms", res.leaf, chi, res.status == PROVEN, numbers))

    bond = simple_substance(z)
    if bond is None:
        bond_kind, bond_lines, bchi, bnode = "none", ["Внешний слой завершён: атомы не образуют связей."], "", None
    else:
        bond_kind = bond.kind or "none"
        record = {"both_metals": bond_kind == "metallic", "ions": False}
        if bond.delta_en is not None:
            record["delta_en"] = bond.delta_en
        bres, bchi = _classify("bonds", record)
        bnode = bres.leaf
        why = ("оба атома — металлы" if bond_kind == "metallic"
               else f"разность электроотрицательностей {fmt(bond.delta_en)}" if bond.delta_en is not None else "")
        bond_lines = [f"Связь между атомами {e.symbol}–{e.symbol}: {BOND_NAMES.get(bond_kind, 'не определена')}"
                      + (f" ({why})" if why else "") + "."]
    steps.append(Step("bonds", "Химическая связь", bond_lines, "bonds", bnode, bchi))

    sres, schi = _classify("substances", {"n_elements": 1, "bond": bond_kind})
    slabel = tree("substances").nodes[sres.leaf].label
    steps.append(Step("substances", "Вещество", [f"Простое вещество {e.symbol}: {slabel.lower()}."],
                      "substances", sres.leaf, schi))

    state = e.standard_state.removeprefix("Expected to be a ").lower()
    node = STATE_NODES.get(state)
    if node:
        mt = matter_tree()
        chi = mt.characteristic(node).text
        steps.append(Step("states", "Агрегатное состояние", [
            f"При стандартных условиях (PubChem «{e.standard_state}»): {mt.nodes[node].label.lower()}."
            + (" Состояние ожидаемое, не измеренное." if e.standard_state.startswith("Expected") else "")
        ], "matter", node, chi, not e.standard_state.startswith("Expected")))
    return steps


def unfold_compound(smiles: str, formula: str, phase: str | None) -> list[Step]:
    """Развёртка соединения: частицы всех атомов → атомы → связи → вещество → состояние."""
    from delta_science.atoms import by_symbol
    from delta_science.bonds import compound_structure, main_bond
    from delta_science.smiles import parse_formula

    counts = parse_formula(formula)
    protons = sum(by_symbol(sym).z * k for sym, k in counts.items())
    neutrons = sum(structure(by_symbol(sym).z).neutrons * k for sym, k in counts.items())
    steps = [Step("particles", "Элементарные частицы", [
        f"Протонов {protons}, нейтронов {neutrons} (школьное правило для каждого атома), электронов {protons}: "
        + ", ".join(f"{sym}×{k}: {by_symbol(sym).z * k} p" for sym, k in sorted(counts.items())) + "."
    ], numbers={"protons": protons, "neutrons": neutrons, "electrons": protons})]
    atom_lines = []
    for sym, k in sorted(counts.items()):
        z = by_symbol(sym).z
        res, _ = _classify("atoms", atom_features(z))
        leaf = tree("atoms").nodes[res.leaf]
        atom_lines.append(f"{sym} ×{k}: Z = {z}, слои {', '.join(map(str, structure(z).shells))}, "
                          f"класс по строению — {CLASS_NAMES.get(leaf.meta.get('class'), '?')}")
    steps.append(Step("atoms", "Атомы", [", ".join(atom_lines) + "."], proven=False))
    _, bonds = compound_structure(smiles)
    bond_lines = []
    for b in bonds:
        name = BOND_NAMES.get(b.kind, "тип не определён")
        between_ions = Fraction(0) in b.orders  # пара ионов, а не связь двух атомов
        bond_lines.append(f"{b.a} и {b.b}: {name}" if between_ions else f"{b.a}–{b.b} ×{b.count}: {name}")
    steps.append(Step("bonds", "Химическая связь", ["; ".join(bond_lines) + "."]))
    kind = main_bond(bonds) or "none"
    sres, schi = _classify("substances", {"n_elements": len(counts), "bond": kind})
    steps.append(Step("substances", "Вещество", [f"{tree('substances').nodes[sres.leaf].label}."],
                      "substances", sres.leaf, schi))
    node = STATE_NODES.get(phase or "")
    if node:
        mt = matter_tree()
        steps.append(Step("states", "Агрегатное состояние", [
            f"{mt.nodes[node].label} (по локальным метаданным прототипа, не проверено)."
        ], "matter", node, mt.characteristic(node).text, proven=False))
    return steps


def pyramid_data() -> dict[str, Any]:
    """Данные пирамиды для визуализации: уровни, деревья, χ узлов, объекты и развёртки."""
    levels = [
        ("matter", "Материя", "дерево материи проекта (поле / частицы / … / вещество)", matter_tree()),
        ("particles", "Элементарные частицы", "протон, нейтрон, электрон (модель школьного курса)", tree("particles")),
        ("atoms", "Атом", "класс элемента по строению атома, выведенному из Z", tree("atoms")),
        ("bonds", "Химическая связь", "пара атомов: металлическая, ионная, ковалентная", tree("bonds")),
        ("substances", "Вещество", "простое / сложное, по типу связи", tree("substances")),
    ]
    out_levels = []
    for level_id, title, doc, t in levels:
        nodes = []
        for node in t.nodes.values():
            parent = t.parent[node.id]
            nodes.append({
                "id": node.id,
                "label": node.label or (CLASS_NAMES.get(node.meta.get("class"), "") if node.is_leaf else "") or node.id,
                "parent": parent[0] if parent else None,
                "branch": parent[1] if parent else None,
                "predicate": None if node.is_leaf else node.predicate.text,
                "knowledge": None if node.is_leaf else node.knowledge,
                "synthetic": False if node.is_leaf else node.synthetic,
                "leaf": node.is_leaf,
                "chi": t.characteristic(node.id).text,
            })
        out_levels.append({"id": level_id, "title": title, "doc": doc, "tree_hash": t.hash, "nodes": nodes})
    unfold = {}
    for z, e in sorted(elements().items()):
        unfold[e.symbol] = [{"level": s.level, "title": s.title, "lines": s.lines, "node": s.node,
                             "chi": s.characteristic} for s in unfold_element(z)]
    return {
        "format": "delta-pyramid@0.1",
        "levels": out_levels,
        "composition": [
            {"from": "particles", "to": "nucleus", "rule": "ядро = Z протонов + N нейтронов"},
            {"from": "nucleus", "to": "atoms", "rule": "атом = ядро + Z электронов (слои по правилу Клечковского)"},
            {"from": "atoms", "to": "bonds", "rule": "связь = пара атомов (тип — дерево химической связи)"},
            {"from": "bonds", "to": "substances", "rule": "вещество = атомы и связи между ними"},
            {"from": "substances", "to": "matter", "rule": "агрегатное состояние вещества — узел дерева материи"},
        ],
        "constants": {"source": CODATA, "proton_u": str(PROTON["mass_u"]), "neutron_u": str(NEUTRON["mass_u"]),
                      "electron_u": str(ELECTRON["mass_u"]), "MeV_per_u": str(MEV_PER_U)},
        "unfold": unfold,
    }
