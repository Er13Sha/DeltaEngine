"""Строение атома: из заряда ядра Z и проверенных данных PubChem.

Развёртка, а не хранение: электронная конфигурация выводится из Z по правилу
Клечковского (подуровни заполняются по возрастанию n + l, при равенстве — по n;
ёмкость подуровня 2·(2l + 1)). Результат сверяется с конфигурацией PubChem;
расхождения — известные «провалы» электрона (Cr, Cu, …) — фиксируются как аномалии,
а фактическое строение берётся из PubChem.

Число нейтронов — школьное правило: округлённая относительная атомная масса минус Z.
Оно даёт нейтроны самого распространённого изотопа не всегда, поэтому помечается.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "sources" / "pubchem" / "periodic-table.json"
L_LETTERS = "spdf"
CORES = {"He": 2, "Ne": 10, "Ar": 18, "Kr": 36, "Xe": 54, "Rn": 86}

Config = tuple[tuple[int, str, int], ...]  # ((n, буква подуровня, число электронов), …)


@dataclass(frozen=True)
class Element:
    z: int
    symbol: str
    name_en: str
    atomic_mass: Fraction
    configuration_text: str
    electronegativity: Fraction | None
    ionization_energy: Fraction | None  # эВ
    electron_affinity: Fraction | None  # эВ
    atomic_radius: Fraction | None  # пм
    oxidation_states: str
    standard_state: str
    melting_point: Fraction | None  # K
    boiling_point: Fraction | None  # K
    density: Fraction | None  # г/см³
    group_block: str


@dataclass(frozen=True)
class AtomStructure:
    z: int
    protons: int
    electrons: int
    neutrons: int  # школьное правило: round(Ar) − Z
    configuration: Config  # фактическая (PubChem)
    madelung: Config  # выведенная из Z
    anomaly: bool  # фактическая конфигурация отличается от выведенной
    shells: tuple[int, ...]  # электроны по слоям: Na → (2, 8, 1)
    outer_electrons: int  # электронов на внешнем слое
    outer_full: bool  # внешний слой завершён (s²p⁶ или 1s²)
    period: int
    group: int | None  # IUPAC 1–18; None у f-элементов
    block: str  # s | p | d | f
    notes: tuple[str, ...] = field(default=())


def _num(text: str) -> Fraction | None:
    text = text.strip()
    return Fraction(text) if text else None


@lru_cache(maxsize=1)
def elements() -> dict[int, Element]:
    data = json.loads(TABLE.read_text(encoding="utf-8"))
    cols = data["Table"]["Columns"]["Column"]
    out = {}
    for row in data["Table"]["Row"]:
        r = dict(zip(cols, row["Cell"]))
        z = int(r["AtomicNumber"])
        out[z] = Element(
            z=z, symbol=r["Symbol"], name_en=r["Name"], atomic_mass=Fraction(r["AtomicMass"]),
            configuration_text=r["ElectronConfiguration"],
            electronegativity=_num(r["Electronegativity"]), ionization_energy=_num(r["IonizationEnergy"]),
            electron_affinity=_num(r["ElectronAffinity"]), atomic_radius=_num(r["AtomicRadius"]),
            oxidation_states=r["OxidationStates"], standard_state=r["StandardState"],
            melting_point=_num(r["MeltingPoint"]), boiling_point=_num(r["BoilingPoint"]),
            density=_num(r["Density"]), group_block=r["GroupBlock"],
        )
    return out


def by_symbol(symbol: str) -> Element:
    for e in elements().values():
        if e.symbol == symbol:
            return e
    raise KeyError(symbol)


def madelung_order() -> list[tuple[int, str]]:
    subshells = [(n, l) for n in range(1, 8) for l in range(0, min(n, 4))]
    subshells.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return [(n, L_LETTERS[l]) for n, l in subshells]


def madelung(z: int) -> Config:
    """Конфигурация из Z по правилу Клечковского."""
    left = z
    out = []
    for n, letter in madelung_order():
        if left == 0:
            break
        take = min(left, 2 * (2 * L_LETTERS.index(letter) + 1))
        out.append((n, letter, take))
        left -= take
    return tuple(out)


_SUB = re.compile(r"(\d)([spdf])(\d+)")


def parse_configuration(text: str) -> Config:
    """«[Ne]3s1» → полная конфигурация; пометки «(calculated)» отбрасываются."""
    text = re.sub(r"\(.*?\)", "", text).strip()
    out: list[tuple[int, str, int]] = []
    core = re.match(r"\[(\w+)\]", text)
    if core:
        out += list(parse_configuration(by_symbol(core.group(1)).configuration_text))
        text = text[core.end():]
    out += [(int(n), letter, int(k)) for n, letter, k in _SUB.findall(text)]
    return _canonical(out)


def _canonical(config) -> Config:
    merged: dict[tuple[int, str], int] = {}
    for n, letter, k in config:
        merged[(n, letter)] = merged.get((n, letter), 0) + k
    return tuple(sorted(((n, letter, k) for (n, letter), k in merged.items() if k),
                        key=lambda t: (t[0], L_LETTERS.index(t[1]))))


def format_configuration(config: Config) -> str:
    order = {nl: i for i, nl in enumerate(madelung_order())}
    return " ".join(f"{n}{letter}{k}" for n, letter, k in sorted(config, key=lambda t: order[(t[0], t[1])]))


def structure(z: int) -> AtomStructure:
    e = elements()[z]
    actual = parse_configuration(e.configuration_text)
    derived = _canonical(madelung(z))
    shells_map: dict[int, int] = {}
    for n, _, k in actual:
        shells_map[n] = shells_map.get(n, 0) + k
    shells = tuple(shells_map[n] for n in sorted(shells_map))
    top = max(shells_map)
    outer = shells_map[top]
    outer_letters = {letter for n, letter, k in actual if n == top}
    outer_full = (top == 1 and outer == 2) or (outer == 8 and outer_letters == {"s", "p"})
    # период, группа и блок — по выведенной конфигурации (у Pd фактически нет 5s, но он в 5-м периоде)
    last_n, last_letter, _ = max(derived, key=lambda t: madelung_order().index((t[0], t[1])))
    period = max(n for n, _, _ in derived)
    s_outer = sum(k for n, letter, k in derived if n == period and letter == "s")
    if last_letter == "s":
        group = 18 if z == 2 else s_outer
    elif last_letter == "p":
        group = 10 + s_outer + sum(k for n, letter, k in derived if n == period and letter == "p")
    elif last_letter == "d":
        group = s_outer + sum(k for n, letter, k in derived if n == period - 1 and letter == "d")
    else:
        group = None
    notes = []
    if actual != derived:
        notes.append(f"по правилу Клечковского {format_configuration(derived)}, фактически "
                     f"{format_configuration(actual)} (провал электрона)")
    if "(" in e.configuration_text:
        notes.append("конфигурация в PubChem расчётная или предсказанная")
    return AtomStructure(
        z=z, protons=z, electrons=z, neutrons=round(e.atomic_mass) - z,
        configuration=actual, madelung=derived, anomaly=actual != derived,
        shells=shells, outer_electrons=outer, outer_full=outer_full,
        period=period, group=group, block=last_letter, notes=tuple(notes),
    )
