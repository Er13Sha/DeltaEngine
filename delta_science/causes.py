"""Причинные правила учебной физики и химии (`data/knowledge/causes.json`).

Правило: условие (предикат DSL над записью вещества) → следствие + механизм.
Условие проверяет ядро. Запись вещества строится из проверенных данных, и для каждого
признака хранится происхождение — из него собирается цепочка объяснения:

    натрий: конфигурация 1s2 2s2 2p6 3s1 (PubChem) → металл (класс PubChem)
    → простое вещество: металлическая связь (дерево связей) → свободные электроны
    → проводит ток (правило metal.conducts)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from delta_kernel import Language, Schema
from delta_kernel.dsl import Evaluation, Predicate
from delta_science.atoms import elements, format_configuration, structure
from delta_science.bonds import compound_structure, fmt, is_metal, main_bond, simple_substance

ROOT = Path(__file__).resolve().parents[1]
CAUSES = ROOT / "data" / "knowledge" / "causes.json"
BOND_NAMES = {
    "metallic": "металлическая связь",
    "ionic": "ионная связь",
    "covalent_polar": "ковалентная полярная связь",
    "covalent_nonpolar": "ковалентная неполярная связь",
    "none": "связей нет (отдельные атомы)",
}
STATE_NAMES = {"solid": "твёрдое", "liquid": "жидкое", "gas": "газообразное", "plasma": "плазма"}
# узлы деревьев знаний → признаки записи класса
CLASS_RECORDS = {
    "matter:Solid": {"state": "solid"},
    "matter:Liquid": {"state": "liquid"},
    "matter:gas.ionized": {"state": "gas"},
    "matter:NeutralGas": {"state": "gas"},
    "matter:Plasma": {"state": "plasma"},
    "bonds:MetallicBond": {"bond": "metallic"},
    "bonds:IonicBond": {"bond": "ionic"},
    "bonds:PolarCovalentBond": {"bond": "covalent_polar"},
    "bonds:NonpolarCovalentBond": {"bond": "covalent_nonpolar"},
}


@dataclass(frozen=True)
class Rule:
    id: str
    predicate: Predicate
    effect: str
    keywords: tuple[str, ...]
    mechanism: str
    terms: tuple[str, ...]
    source: str
    status: str  # unreviewed | accepted


@dataclass
class SubstanceRecord:
    record: dict[str, Any]
    provenance: dict[str, str]  # признак → откуда он взят
    caveats: list[str]


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    return json.loads(CAUSES.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def language() -> Language:
    return Language(Schema.from_dict(_data()["schema"]))


@lru_cache(maxsize=1)
def rules() -> tuple[Rule, ...]:
    lang = language()
    return tuple(
        Rule(r["id"], lang.compile(r["condition"]), r["effect"], tuple(r["keywords"]), r["mechanism"],
             tuple(r.get("terms", ())), r["source"], r.get("status", "unreviewed"))
        for r in _data()["rules"]
    )


def evaluate(rule: Rule, sub: SubstanceRecord) -> Evaluation:
    return rule.predicate.evaluate(language().schema.coerce_record(sub.record))


def _state(standard_state: str) -> str | None:
    base = standard_state.removeprefix("Expected to be a ").lower()
    return base if base in ("solid", "liquid", "gas") else None


def element_record(z: int) -> SubstanceRecord:
    e = elements()[z]
    s = structure(z)
    config = format_configuration(s.configuration)
    record: dict[str, Any] = {"kind": "element", "metal": is_metal(e.symbol), "block": s.block,
                              "outer_electrons": s.outer_electrons, "outer_full": s.outer_full}
    prov = {
        "metal": f"класс элемента по PubChem: «{e.group_block}»",
        "block": f"электронная конфигурация {config}: последним заполняется {s.block}-подуровень",
        "outer_electrons": f"электронная конфигурация {config} (PubChem): на внешнем слое "
                           f"{s.outer_electrons} электр.",
        "outer_full": f"электронная конфигурация {config}: внешний слой "
                      f"{'завершён' if s.outer_full else 'не завершён'}",
    }
    caveats = []
    state = _state(e.standard_state)
    if state:
        record["state"] = state
        prov["state"] = f"PubChem: при стандартных условиях «{e.standard_state}»"
        if e.standard_state.startswith("Expected"):
            caveats.append(f"агрегатное состояние «{e.standard_state}» в PubChem — ожидаемое, а не измеренное")
    bond = simple_substance(z)
    if bond is None:
        record["bond"] = "none"
        prov["bond"] = "внешний слой завершён: простое вещество состоит из отдельных атомов"
    elif bond.kind is not None:
        record["bond"] = bond.kind
        reason = "оба атома — металлы" if bond.kind == "metallic" else f"разность электроотрицательностей {fmt(bond.delta_en)}"
        prov["bond"] = f"простое вещество: {BOND_NAMES[bond.kind]} (дерево связей: {reason})"
    return SubstanceRecord(record, prov, caveats)


def compound_record(smiles: str, phase: str | None, phase_verified: bool) -> SubstanceRecord:
    _, bonds = compound_structure(smiles)
    record: dict[str, Any] = {"kind": "compound"}
    prov: dict[str, str] = {}
    caveats = []
    kind = main_bond(bonds)
    if kind:
        record["bond"] = kind
        prov["bond"] = f"структура SMILES {smiles} (PubChem): главная связь — {BOND_NAMES[kind]} (дерево связей)"
    if phase in ("solid", "liquid", "gas"):
        record["state"] = phase
        prov["state"] = "агрегатное состояние при обычных условиях"
        if not phase_verified:
            caveats.append("агрегатное состояние взято из локальных метаданных прототипа, а не из проверенного источника")
    return SubstanceRecord(record, prov, caveats)


def class_record(concept_key: str) -> SubstanceRecord | None:
    values = CLASS_RECORDS.get(concept_key)
    if values is None:
        return None
    prov = {k: f"по определению понятия ({STATE_NAMES.get(v, BOND_NAMES.get(v, v))})" for k, v in values.items()}
    return SubstanceRecord({"kind": "class", **values}, prov, [])


def attributes_of(rule: Rule) -> list[str]:
    """Признаки записи, на которые опирается условие правила (для цепочки объяснения)."""
    names = []

    def walk(node: Any) -> None:
        if node.kind in ("attr", "has") and node.value not in names:
            names.append(node.value)
        for a in node.args:
            walk(a)

    walk(rule.predicate.node)
    return names
