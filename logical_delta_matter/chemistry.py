from __future__ import annotations

import re
from typing import Any


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple chemical formula with parenthesized groups.

    The parser is intentionally strict and integer-only. Hydrates, charges and
    dot notation are outside this MVP.
    """

    stack: list[dict[str, int]] = [{}]
    index = 0
    while index < len(formula):
        char = formula[index]
        if char == "(":
            stack.append({})
            index += 1
            continue
        if char == ")":
            if len(stack) == 1:
                raise ValueError(f"Unmatched ')' in formula: {formula}")
            index += 1
            match = re.match(r"(\d*)", formula[index:])
            multiplier = int(match.group(1)) if match and match.group(1) else 1
            index += len(match.group(0)) if match else 0
            group = stack.pop()
            target = stack[-1]
            for symbol, count in group.items():
                target[symbol] = target.get(symbol, 0) + count * multiplier
            continue

        match = re.match(r"([A-Z][a-z]?)(\d*)", formula[index:])
        if not match:
            raise ValueError(f"Unsupported formula syntax at '{formula[index:]}' in {formula}")
        symbol = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        stack[-1][symbol] = stack[-1].get(symbol, 0) + count
        index += len(match.group(0))

    if len(stack) != 1:
        raise ValueError(f"Unclosed '(' in formula: {formula}")
    return stack[0]


def hill_formula(atoms: dict[str, int]) -> str:
    if not atoms:
        raise ValueError("Cannot build formula from empty atom map.")
    symbols = list(atoms)
    if "C" in atoms:
        ordered = ["C"]
        if "H" in atoms:
            ordered.append("H")
        ordered.extend(sorted(symbol for symbol in symbols if symbol not in {"C", "H"}))
    else:
        ordered = sorted(symbols)
    return "".join(f"{symbol}{atoms[symbol] if atoms[symbol] != 1 else ''}" for symbol in ordered)


def particle_requirement(atoms: dict[str, int], elements_by_symbol: dict[str, dict[str, Any]]) -> dict[str, int]:
    requirement = {"protons": 0, "neutrons": 0, "electrons": 0}
    for symbol, count in atoms.items():
        element = elements_by_symbol.get(symbol)
        if element is None:
            raise KeyError(f"Unknown element: {symbol}")
        z = int(element["atomic_number"])
        neutrons = int(element["model"]["neutrons"])
        requirement["protons"] += z * count
        requirement["neutrons"] += neutrons * count
        requirement["electrons"] += z * count
    return requirement


def capacity_for_requirement(requirement: dict[str, int], stock: dict[str, int]) -> int:
    limits: list[int] = []
    for key in ("protons", "neutrons", "electrons"):
        needed = int(requirement.get(key, 0))
        if needed > 0:
            limits.append(int(stock.get(key, 0)) // needed)
    return min(limits) if limits else 0


def shortage_for_requirement(requirement: dict[str, int], stock: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(requirement.get(key, 0)) - int(stock.get(key, 0)))
        for key in ("protons", "neutrons", "electrons")
    }
