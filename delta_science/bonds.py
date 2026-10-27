"""Химическая связь и строение вещества.

Тип связи определяет дерево ядра `data/trees/bonds.json` по паре атомов:
оба ли металлы (дерево химических элементов), входят ли атомы в противоположно
заряженные ионы (структура SMILES PubChem), разность электроотрицательностей.

- `simple_substance(z)` — связь в простом веществе элемента (благородные газы —
  отдельные атомы, связей нет);
- `compound_structure(smiles)` — граф вещества: все связи (с неявными водородами)
  и ионные пары, сгруппированные по типу.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from delta_kernel import PROVEN, Tree
from delta_science.atoms import by_symbol, elements, structure
from delta_science.smiles import Molecule, parse

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = Fraction(17, 10)  # условная граница ионной связи (школьная химия)
BOND_KIND = {  # лист дерева → значение признака `bond` для причинных правил
    "MetallicBond": "metallic",
    "IonicBond": "ionic",
    "PolarCovalentBond": "covalent_polar",
    "NonpolarCovalentBond": "covalent_nonpolar",
}
ORDER_NAMES = {Fraction(1): "одинарная", Fraction(2): "двойная", Fraction(3): "тройная",
               Fraction(3, 2): "ароматическая"}


@lru_cache(maxsize=1)
def bond_tree() -> Tree:
    return Tree.from_dict(json.loads((ROOT / "data" / "trees" / "bonds.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def elements_tree() -> Tree:
    return Tree.from_dict(json.loads((ROOT / "data" / "trees" / "elements.json").read_text(encoding="utf-8")))


def is_metal(symbol: str) -> bool:
    """Металл ли элемент — по корневой вилке дерева химических элементов."""
    e = by_symbol(symbol)
    result = elements_tree().classify({"z": e.z, "group_block": e.group_block}, certify=False)
    return result.path[0].branch


@dataclass
class BondInfo:
    a: str  # символ или ион
    b: str
    kind: str | None  # metallic | ionic | covalent_polar | covalent_nonpolar | None (нет данных)
    leaf: str | None
    delta_en: Fraction | None
    orders: Counter  # порядок связи → число связей
    certificate: dict[str, Any] | None
    note: str = ""

    @property
    def count(self) -> int:
        return sum(self.orders.values())


def pair_record(a: str, b: str, ions: bool) -> dict[str, Any]:
    ea, eb = by_symbol(a), by_symbol(b)
    record: dict[str, Any] = {"both_metals": is_metal(a) and is_metal(b) and not ions, "ions": ions}
    if ea.electronegativity is not None and eb.electronegativity is not None:
        record["delta_en"] = abs(ea.electronegativity - eb.electronegativity)
    return record


def classify_pair(a: str, b: str, ions: bool = False) -> tuple[str | None, str | None, Fraction | None, dict | None, str]:
    record = pair_record(a, b, ions)
    result = bond_tree().classify(record)
    delta = record.get("delta_en")
    note = ""
    if delta is not None and not ions and abs(delta - THRESHOLD) <= Fraction(2, 10) and not record["both_metals"]:
        note = f"разность электроотрицательностей {fmt(delta)} близка к условной границе 1,7"
    elif delta is not None and not ions and 0 < delta < Fraction(4, 10):
        note = "слабо полярная (разность электроотрицательностей меньше 0,4)"
    if result.status != PROVEN or result.leaf is None:
        missing = ", ".join(result.missing) or "; ".join(result.errors)
        return None, None, delta, result.certificate, f"нет данных: {missing}"
    return BOND_KIND[result.leaf], result.leaf, delta, result.certificate, note


def simple_substance(z: int) -> BondInfo | None:
    """Связь в простом веществе; None — благородный газ (отдельные атомы)."""
    if structure(z).outer_full:
        return None
    sym = elements()[z].symbol
    kind, leaf, delta, cert, note = classify_pair(sym, sym)
    return BondInfo(sym, sym, kind, leaf, delta, Counter(), cert, note)


def compound_structure(smiles: str) -> tuple[Molecule, list[BondInfo]]:
    mol = parse(smiles)
    groups: dict[tuple[str, str, str | None], BondInfo] = {}

    def add(a: str, b: str, order: Fraction, ions: bool, label_a: str | None = None, label_b: str | None = None):
        kind, leaf, delta, cert, note = classify_pair(a, b, ions)
        if ions and (_polyatomic(label_a) or _polyatomic(label_b)):
            delta = None  # у сложных ионов разность ЭО «представителей» смысла не имеет
        la, lb = sorted([label_a or a, label_b or b])
        key = (la, lb, kind)
        if key not in groups:
            groups[key] = BondInfo(la, lb, kind, leaf, delta, Counter(), cert, note)
        groups[key].orders[order] += 1

    for i, j, order in mol.bonds:
        add(mol.atoms[i].element, mol.atoms[j].element, order, False)
    for atom in mol.atoms:
        for _ in range(atom.hydrogens):
            add(atom.element, "H", Fraction(1), False)
    # ионные пары: каждый катион с каждым видом аниона (связь между ионами, а не атомами)
    ions = []
    for comp in mol.components():
        charge = sum(mol.atoms[i].charge for i in comp)
        if charge:
            # представитель иона — атом с зарядом того же знака, что у иона (O⁻ в NO₃⁻, а не N⁺)
            charged = next(i for i in comp if mol.atoms[i].charge * charge > 0)
            ions.append((ion_label(mol, comp, charge), mol.atoms[charged].element, charge))
    for cat in sorted({ion for ion in ions if ion[2] > 0}):
        for an in sorted({ion for ion in ions if ion[2] < 0}):
            add(cat[1], an[1], Fraction(0), True, cat[0], an[0])
    return mol, list(groups.values())


def _polyatomic(label: str | None) -> bool:
    return label is not None and sum(ch.isupper() for ch in label) + any(ch in "₂₃₄₅₆₇₈₉" for ch in label) > 1


SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def ion_label(mol: Molecule, comp: list[int], charge: int) -> str:
    counts: Counter = Counter()
    for i in comp:
        counts[mol.atoms[i].element] += 1
        counts["H"] += mol.atoms[i].hydrogens
    order = sorted(e for e in counts if e not in ("O", "H")) + [e for e in ("O", "H") if counts[e]]
    formula = "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order).translate(SUBSCRIPT)
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    return formula + (f"{magnitude if magnitude > 1 else ''}{sign}").translate(SUPERSCRIPT)


def main_bond(bonds: list[BondInfo]) -> str | None:
    """Главный тип связи вещества: ионная, если есть; иначе полярная; иначе неполярная."""
    kinds = {b.kind for b in bonds}
    for kind in ("ionic", "metallic", "covalent_polar", "covalent_nonpolar"):
        if kind in kinds:
            return kind
    return None


def fmt(value: Fraction | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    scaled = round(value * 10 ** digits)
    text = f"{scaled / 10 ** digits:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")
