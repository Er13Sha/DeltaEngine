"""Визуализация пирамиды уровней материи: страница из деревьев ядра.

    python -m delta_app.pyramid_view [путь]     # по умолчанию docs/pyramid.html

Страница строится только из данных: деревья вилок уровней (`data/trees/*.json`),
функции-характеристики χ узлов, развёртки всех 118 элементов и 82 веществ
(`delta_science.pyramid`). Поэтому визуализация всегда совпадает с тем,
что проверяет ядро.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from delta_kernel import Tree
from delta_science.atoms import elements, structure
from delta_science.bonds import BOND_KIND, compound_structure, simple_substance
from delta_science.pyramid import (CLASS_NAMES, STATE_NODES, atom_features, matter_tree, tree, unfold_compound,
                                   unfold_element)
from delta_text.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("pyramid_template.html")
OUT = ROOT / "docs" / "pyramid.html"

BANDS = [
    {"id": "matter", "title": "Материя", "level": "matter",
     "doc": "Вершина: дерево материи проекта. Объект приходит сюда своим агрегатным состоянием."},
    {"id": "field", "title": "Поле → частица (модель)", "level": "field",
     "doc": "Трек E: целочисленная модель поля. При каких параметрах уплотнения поля устойчивы и одноимённые "
            "отталкиваются. Вилки открыл индуктор по перебору параметров. Это знание о модели (scope = model), "
            "а не о природе.",
     "link": "посылка H0.P2: уплотнённое поле — частица (гипотеза; проверена только на модели)"},
    {"id": "particles", "title": "Элементарные частицы", "level": "particles",
     "doc": "Модель школьного курса: протон, нейтрон, электрон. Константы CODATA 2018.",
     "link": "частицы модели сопоставляются с частицами школьного курса (соответствие не доказано)"},
    {"id": "nucleus", "title": "Атомное ядро", "level": None,
     "doc": "Уровень состава: ядро = Z протонов + N нейтронов, заряд +Z.",
     "link": "ядро = Z протонов + N нейтронов"},
    {"id": "atoms", "title": "Атом", "level": "atoms",
     "doc": "Класс элемента по строению атома. Признаки выведены из заряда ядра Z по правилу Клечковского, "
            "вилки открыл индуктор.",
     "link": "атом = ядро + Z электронов, слои по правилу Клечковского"},
    {"id": "bonds", "title": "Химическая связь", "level": "bonds",
     "doc": "Пара атомов: металлическая, ионная или ковалентная связь.",
     "link": "связь = пара атомов"},
    {"id": "substances", "title": "Вещество", "level": "substances",
     "doc": "Простое или сложное вещество, по главному типу связи.",
     "link": "вещество = атомы и связи между ними"},
]


def _level(level_id: str, t: Tree, kb: KnowledgeBase) -> dict[str, Any]:
    nodes = []
    for node in t.nodes.values():
        parent = t.parent[node.id]
        label = node.label or ""
        if node.is_leaf and not label:
            label = CLASS_NAMES.get(node.meta.get("class"), node.id)
        elif node.is_leaf and node.meta.get("class"):
            label = CLASS_NAMES.get(node.meta["class"], label)
        question = ""
        if not node.is_leaf:
            question = _say(kb, level_id, t, node.id) + "?"
        nodes.append({
            "id": node.id, "label": label or (node.id if node.is_leaf else ""), "parent": parent[0] if parent else None,
            "branch": parent[1] if parent else None, "leaf": node.is_leaf,
            "knowledge": None if node.is_leaf else node.knowledge,
            "synthetic": False if node.is_leaf else node.synthetic,
            "predicate": None if node.is_leaf else node.predicate.text,
            "chi": t.characteristic(node.id).text, "question": question,
        })
    return {"id": level_id, "nodes": nodes, "hash": t.hash}


def _say(kb: KnowledgeBase, level_id: str, t: Tree, fork_id: str) -> str:
    if level_id in kb.trees:
        return kb.say(level_id, fork_id, True)
    saved = kb.trees.get(level_id)
    kb.trees[level_id] = t  # дерево атомов не входит в словарь понятий: только словесная форма вилки
    try:
        return kb.say(level_id, fork_id, True)
    finally:
        if saved is None:
            kb.trees.pop(level_id)


def _element_object(z: int, title: str) -> dict[str, Any]:
    e = elements()[z]
    s = structure(z)
    steps = {st.level: st for st in unfold_element(z)}
    atoms_leaf = tree("atoms").classify(atom_features(z), certify=False).leaf
    bond = simple_substance(z)
    out = {
        "key": f"element:{e.symbol}", "kind": "element", "title": f"{title} ({e.symbol}, Z = {z})",
        "steps": {
            "particles": {"lines": steps["particles"].lines, "chi": "",
                          "nodes": ["Proton", "Electron"] + (["Neutron"] if s.neutrons else [])},
            "nucleus": {"lines": steps["nucleus"].lines, "chi": "", "nodes": []},
            "atoms": {"lines": steps["atoms"].lines, "chi": steps["atoms"].characteristic, "nodes": [atoms_leaf]},
            "bonds": {"lines": steps["bonds"].lines, "chi": steps["bonds"].characteristic,
                      "nodes": [steps["bonds"].node] if steps["bonds"].node else []},
            "substances": {"lines": steps["substances"].lines, "chi": steps["substances"].characteristic,
                           "nodes": [steps["substances"].node]},
        },
        "links": {
            "nucleus": f"{z} p + {s.neutrons} n",
            "atoms": f"+ {z} e; слои {', '.join(map(str, s.shells))}",
            "bonds": f"{e.symbol}–{e.symbol}" if bond is not None else "связей нет",
            "substances": f"простое вещество {e.symbol}",
        },
    }
    if "states" in steps:
        out["steps"]["matter"] = {"lines": steps["states"].lines, "chi": steps["states"].characteristic,
                                  "nodes": [steps["states"].node]}
    return out


def _compound_object(c: Any) -> dict[str, Any]:
    formula = dict(c.facts).get("формула", "")
    steps = {st.level: st for st in unfold_compound(c.smiles, _hill(c.smiles), c.phase)}
    _, bonds = compound_structure(c.smiles)
    leaves = sorted({leaf for leaf, kind in BOND_KIND.items() for b in bonds if b.kind == kind})
    out = {
        "key": c.key, "kind": "compound", "title": f"{c.name} ({formula})",
        "steps": {
            "particles": {"lines": steps["particles"].lines, "chi": "", "nodes": ["Proton", "Neutron", "Electron"]},
            "atoms": {"lines": steps["atoms"].lines, "chi": "", "nodes": []},
            "bonds": {"lines": steps["bonds"].lines, "chi": "", "nodes": leaves},
            "substances": {"lines": steps["substances"].lines, "chi": steps["substances"].characteristic,
                           "nodes": [steps["substances"].node]},
        },
        "links": {"bonds": formula, "substances": formula},
    }
    if "states" in steps:
        out["steps"]["matter"] = {"lines": steps["states"].lines, "chi": steps["states"].characteristic,
                                  "nodes": [steps["states"].node]}
    return out


def _hill(smiles: str) -> str:
    from delta_science.smiles import parse

    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(parse(smiles).formula().items()))


def build() -> dict[str, Any]:
    kb = KnowledgeBase()
    levels = {
        "matter": _level("matter", matter_tree(), kb),
        "field": _level("field", tree("field_model"), kb),
        "particles": _level("particles", tree("particles"), kb),
        "atoms": _level("atoms", tree("atoms"), kb),
        "bonds": _level("bonds", tree("bonds"), kb),
        "substances": _level("substances", tree("substances"), kb),
    }
    objects = []
    for z, e in sorted(elements().items()):
        concept = kb.concepts.get(f"element:{e.symbol}")
        objects.append(_element_object(z, concept.label.capitalize() if concept else e.name_en))
    for c in sorted((c for c in kb.concepts.values() if c.kind == "instance" and c.smiles),
                    key=lambda c: c.name):
        objects.append(_compound_object(c))
    agree = sum(
        tree("atoms").nodes[tree("atoms").classify(atom_features(z), certify=False).leaf].meta.get("class") == e.group_block
        for z, e in elements().items()
    )
    summary = json.loads((ROOT / "data" / "experiments" / "field" / "summary.json").read_text(encoding="utf-8"))
    fams = summary["families"]
    field_note = [
        f"Шип E1 (протокол {summary['protocol_hash'][7:19]}): семейство A — частиц нет; B — частицы есть, "
        f"но одноимённые притягиваются при касании; C — частицы есть, одноимённые отталкиваются, разноимённые "
        f"не взаимодействуют (прямого притяжения в правиле нет). Допущено: "
        + ", ".join(f for f, v in fams.items() if v["admitted"]) + ".",
        "Отрицательные контроли провалили проверку, как требуется: "
        + ("да" if summary["negative_controls_failed_as_required"] else "НЕТ") + ".",
        "Ограничения шипа: решётка 2D, динамика без инерции, параметры выбраны по пилотным прогонам; "
        "квантование заряда ещё не проверялось.",
    ]
    screening = json.loads((ROOT / "data" / "experiments" / "field" / "screening.json").read_text(encoding="utf-8"))
    transparent = next(r["ratio"] for r in screening["video_formula"]["rows"] if r["mu"] == 0.001)
    field_note += [
        "Закон взаимодействия — формула воздействия (видео Rek8C322Q5k): W = Q·N, Q = k·c1/(4πr²), N = h·S·ρ; "
        "при c1 = 4πG она точно даёт закон тяготения, при c1 = 1/ε₀ — закон Кулона.",
        f"Семейство D (экранирование изотропного потока): одно тело — сила "
        f"{screening['negative_control_single_body_force']:.0e}, пара притягивается по закону "
        f"r^{screening['distance_law']['exponent']:.4f}; формула видео выводится с c1 = 4π·I "
        f"(опыт/формула = {transparent:.3f} для прозрачных тел), для плотных тел насыщается.",
        "Одноимённые в потоке притягиваются, только если G·m² > k·q²; у протонов отталкивание сильнее в "
        f"{screening['like_charges']['real_world']['proton_pair_coulomb_over_gravity']:.2e} раз.",
    ]
    earth = json.loads((ROOT / "data" / "experiments" / "field" / "drag.json").read_text(encoding="utf-8"))["earth"]
    field_note.append(
        f"Проверка C6 отклонила D как механизм тяготения: при носителях со скоростью света Земля остановилась бы "
        f"на орбите за {earth['if_u_is_c']['stopping_time_days']:.0f} суток, а чтобы орбита выжила, нагрев "
        f"должен превышать наблюдаемый в {earth['excess']:.0e} раз; упругое переизлучение убирает само притяжение."
    )
    instant = json.loads((ROOT / "data" / "experiments" / "field" / "drag.json").read_text(encoding="utf-8"))
    field_note.append(
        f"Вариант заказчика «тяготение мгновенно» проходит C6, если носитель несёт импульс почти без энергии "
        f"(скорость ≥ {instant['instant_carriers']['superluminal_u_min_over_c']:.1e} c); тогда волны тяготения "
        f"(GW170817, двойной пульсар — скорость c) должны быть отдельным явлением среды."
    )
    light = json.loads((ROOT / "data" / "experiments" / "field" / "light.json").read_text(encoding="utf-8"))
    field_note.append(
        "Свет — частица в базовом поле: данные выбирают поле как среду для света (n = 1 + 2GM/(rc²)): отклонение у "
        f"Солнца {light['deflection_at_limb_arcsec']['medium']:.3f}″ и задержка сигнала совпадают с измерениями "
        f"({light['medium_sigma']:.1f} σ); частица, которую поле только притягивает, даёт половину отклонения и "
        "опережение вместо задержки."
    )
    unified = json.loads((ROOT / "data" / "experiments" / "field" / "unified.json").read_text(encoding="utf-8"))
    nuclear = unified["nuclear"]
    field_note.append(
        "Единая схема: магнетизм выводится из того же закона при конечной скорости передачи (F_маг/F_эл = v²/c²), "
        f"но несёт его заряд, а не масса (масса слабее в {1 / unified['magnetism']['wires']['mass_over_charge']:.0e} раз). "
        f"Ядро: энергия связи 20 ядер растёт как A^{nuclear['data_exponent_B_vs_A']:.2f}; формула без затухания "
        f"ошибается на {nuclear['fits'][-1]['rms_relative_error']:.0%}, с затуханием k_пр = e^(−r/r₀) при r₀ ≲ 1 фм — "
        f"на {nuclear['best']['rms_relative_error']:.0%}."
    )
    for band in BANDS:
        if band["id"] == "field":
            band["note"] = field_note
    return {
        "levels": levels,
        "levelsList": [{"id": k, "hash": v["hash"]} for k, v in levels.items()],
        "bands": BANDS,
        "objects": objects,
        "default": "element:Na",
        "constants": {"source": "CODATA 2018"},
        "atomsCheck": f"{agree} из {len(elements())}",
    }


def main(argv: list[str] | None = None) -> int:
    out = Path((argv or sys.argv[1:] or [str(OUT)])[0])
    data = json.dumps(build(), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", data)
    out.write_text(page, encoding="utf-8")
    print(f"{out}: {len(page) // 1024} КБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
