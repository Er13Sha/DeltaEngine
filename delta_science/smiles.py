"""Граф связей атомов по записи SMILES (подмножество, достаточное для структур PubChem).

Поддерживается: атомы органического подмножества (B C N O P S F Cl Br I) и
ароматические (b c n o p s), атомы в скобках с зарядом и числом водородов
([Na+], [NH4+], [O-2], [C@@H]), связи - = # :, ветви ( ), замыкания циклов
(цифры и %nn), точка — несвязанные части (ионы). Стереометки (@, /, \\) не влияют
на связи и отбрасываются.

Неявные атомы водорода атомов органического подмножества добавляются по правилу
SMILES: наименьшая обычная валентность, не меньшая суммы порядков связей.
Проверка разбора — сравнение состава с молекулярной формулой PubChem.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

ORGANIC = {"B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5), "S": (2, 4, 6),
           "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,)}
AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}
BOND_ORDER = {"-": Fraction(1), "=": Fraction(2), "#": Fraction(3), ":": Fraction(3, 2)}


class SmilesError(ValueError):
    pass


@dataclass
class Atom:
    element: str
    charge: int = 0
    hydrogens: int = 0  # явные (в скобках) или неявные
    aromatic: bool = False
    bracket: bool = False


@dataclass
class Molecule:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[tuple[int, int, Fraction]] = field(default_factory=list)  # (i, j, порядок)

    def formula(self) -> Counter:
        counts: Counter = Counter()
        for a in self.atoms:
            counts[a.element] += 1
            counts["H"] += a.hydrogens
        return +counts

    def components(self) -> list[list[int]]:
        """Связные части (молекулы и ионы)."""
        parent = list(range(len(self.atoms)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j, _ in self.bonds:
            parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i in range(len(self.atoms)):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())


_BRACKET = re.compile(r"\[(\d*)([A-Z][a-z]?|[bcnops])(@*)(H\d*)?([+-]\d*|[+-]+)?\]")


def parse(smiles: str) -> Molecule:
    mol = Molecule()
    i = 0
    prev: int | None = None
    stack: list[int | None] = []
    rings: dict[str, tuple[int, Fraction | None]] = {}
    pending: Fraction | None = None
    while i < len(smiles):
        c = smiles[i]
        if c in "/\\":
            i += 1
            continue
        if c in BOND_ORDER:
            pending = BOND_ORDER[c]
            i += 1
            continue
        if c == "(":
            stack.append(prev)
            i += 1
            continue
        if c == ")":
            if not stack:
                raise SmilesError(f"Лишняя «)» в позиции {i}")
            prev = stack.pop()
            i += 1
            continue
        if c == ".":
            prev, pending = None, None
            i += 1
            continue
        if c.isdigit() or c == "%":
            label = smiles[i + 1 : i + 3] if c == "%" else c
            i += 3 if c == "%" else 1
            if prev is None:
                raise SmilesError("Замыкание цикла без атома")
            if label in rings:
                other, order = rings.pop(label)
                _add_bond(mol, other, prev, pending or order)
            else:
                rings[label] = (prev, pending)
            pending = None
            continue
        if c == "[":
            m = _BRACKET.match(smiles, i)
            if not m:
                raise SmilesError(f"Не разобран атом в скобках в позиции {i}")
            _, sym, _, hpart, charge = m.groups()
            aromatic = sym in AROMATIC
            atom = Atom(AROMATIC.get(sym, sym), _charge(charge), _hcount(hpart), aromatic, True)
            i = m.end()
        else:
            two = smiles[i : i + 2]
            if two in ("Cl", "Br"):
                atom, i = Atom(two), i + 2
            elif c in ORGANIC:
                atom, i = Atom(c), i + 1
            elif c in AROMATIC:
                atom, i = Atom(AROMATIC[c], aromatic=True), i + 1
            else:
                raise SmilesError(f"Неизвестный символ {c!r} в позиции {i}")
        mol.atoms.append(atom)
        idx = len(mol.atoms) - 1
        if prev is not None:
            order = pending or (BOND_ORDER[":"] if atom.aromatic and mol.atoms[prev].aromatic else Fraction(1))
            _add_bond(mol, prev, idx, order)
        prev, pending = idx, None
    if rings:
        raise SmilesError(f"Незамкнутые циклы: {sorted(rings)}")
    if stack:
        raise SmilesError("Незакрытая «(»")
    _implicit_hydrogens(mol)
    return mol


def _add_bond(mol: Molecule, i: int, j: int, order: Fraction | None) -> None:
    if order is None:
        order = BOND_ORDER[":"] if mol.atoms[i].aromatic and mol.atoms[j].aromatic else Fraction(1)
    mol.bonds.append((i, j, order))


def _charge(text: str | None) -> int:
    if not text:
        return 0
    sign = 1 if text[0] == "+" else -1
    rest = text[1:]
    if rest.isdigit():
        return sign * int(rest)
    return sign * len(text)


def _hcount(text: str | None) -> int:
    if not text:
        return 0
    return int(text[1:]) if len(text) > 1 else 1


def _implicit_hydrogens(mol: Molecule) -> None:
    used = [Fraction(0)] * len(mol.atoms)
    for i, j, order in mol.bonds:
        used[i] += order
        used[j] += order
    for k, atom in enumerate(mol.atoms):
        if atom.bracket or atom.element not in ORGANIC:
            continue
        total = used[k]
        if atom.aromatic:
            total = Fraction(int(total) + (1 if total % 1 else 0)) if total % 1 else total + 1
        for valence in ORGANIC[atom.element]:
            if valence >= total:
                atom.hydrogens = int(valence - total)
                break


def parse_formula(formula: str) -> Counter:
    """Молекулярная формула PubChem (запись Хилла): «C2H6O» → {C: 2, H: 6, O: 1}."""
    counts: Counter = Counter()
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[sym] += int(num) if num else 1
    return counts
