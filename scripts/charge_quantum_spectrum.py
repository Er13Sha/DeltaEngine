"""Спектр частиц семейства C по числу порций заряда Q = n."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from adapters.field_lattice.model import (
    acceptance,
    clusters,
    kawasaki,
    kernel,
    periodic_distance,
)

OUT = Path("data/experiments/field/charge_quantum_spectrum_2026-10-27.json")
L = 96
J = 40
K = 1
R = 8
TEMP = (8, 1)
SWEEPS = 600
NS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SEEDS = (1, 2, 3)
ISOLATED = (4, 8, 16, 32)
SPACING = 12


def compact_cells(n: int) -> list[tuple[int, int]]:
    ranked: list[tuple[int, int, int]] = []
    span = int(math.ceil(math.sqrt(n))) + 2
    for dx in range(-span, span + 1):
        for dy in range(-span, span + 1):
            ranked.append((dx * dx + dy * dy, dx, dy))
    ranked.sort()
    cx = L // 2
    cy = L // 2
    return [((cx + dx) % L, (cy + dy) % L) for _, dx, dy in ranked[:n]]


def isolated_cells(n: int) -> list[tuple[int, int]]:
    side = int(math.ceil(math.sqrt(n)))
    start = (L - (side - 1) * SPACING) // 2
    cells = []
    for i in range(n):
        row, col = divmod(i, side)
        cells.append(((start + col * SPACING) % L, (start + row * SPACING) % L))
    return cells


def place(cells: list[tuple[int, int]]) -> np.ndarray:
    field = np.zeros((L, L), dtype=np.int64)
    for x, y in cells:
        if field[x, y] != 0:
            raise ValueError("overlap")
        field[x, y] = 1
    return field


def judge(start_cells: list[tuple[int, int]], after: list[dict]) -> dict[str, object]:
    n = len(start_cells)
    start_center = (
        sum(c[0] for c in start_cells) / n,
        sum(c[1] for c in start_cells) / n,
    )
    plus = [c for c in after if c["charge"] > 0]
    charges = sorted(int(c["charge"]) for c in plus)
    if not plus:
        kind = "evaporated"
        held = False
    elif (
        len(plus) == 1
        and plus[0]["charge"] == n
        and math.ceil(n / 2) <= plus[0]["cells"] <= 2 * n
        and periodic_distance(start_center, plus[0]["center"], L) <= 8
    ):
        kind = "held_n"
        held = True
    elif len(plus) > 1 and all(abs(c["charge"]) < n for c in plus):
        kind = "copies"
        held = False
    else:
        kind = "other"
        held = False
    return {
        "kind": kind,
        "held": bool(held),
        "charges": charges,
        "n_clusters": len(plus),
        "cells": [int(c["cells"]) for c in plus],
    }


def run_one(n: int, seed: int, isolated: bool) -> dict[str, object]:
    cells = isolated_cells(n) if isolated else compact_cells(n)
    field = place(cells)
    if int(field.sum()) != n:
        raise RuntimeError("charge not n")
    kern = kernel(R)
    thresh = acceptance(TEMP[0], TEMP[1], 400)
    kawasaki(field, J, K, kern, thresh, SWEEPS, seed)
    after = clusters(field)
    judged = judge(cells, after)
    print(
        f"{'iso' if isolated else 'cmp'} n={n} seed={seed} "
        f"{judged['kind']} Q={judged['charges']}"
    )
    return {
        "n": n,
        "seed": seed,
        "isolated": isolated,
        "sum_conserved": bool(int(field.sum()) == n),
        **judged,
    }


def main() -> None:
    compact = [run_one(n, seed, False) for n in NS for seed in SEEDS]
    isolated = [run_one(n, seed, True) for n in ISOLATED for seed in SEEDS]
    by_n: dict[int, list[dict[str, object]]] = {}
    for row in compact:
        by_n.setdefault(int(row["n"]), []).append(row)
    accepted = {n: all(row["held"] for row in rows) for n, rows in by_n.items()}
    elementary = bool(accepted.get(1, False))
    ladder = sum(1 for ok in accepted.values() if ok) >= 2
    copies_only = (not any(accepted[n] for n in accepted if n >= 2)) and all(
        row["kind"] != "held_n" for row in isolated
    )
    isolated_assembled = any(row["held"] for row in isolated)
    payload = {
        "scope": "model",
        "protocol": "docs/charge-quantum-spectrum-protocol-2026-10-27.md",
        "family": "C",
        "params": {"L": L, "J": J, "K": K, "R": R, "T": "8", "sweeps": SWEEPS},
        "compact": compact,
        "isolated": isolated,
        "accepted_n": {str(n): bool(ok) for n, ok in sorted(accepted.items())},
        "elementary_portion": elementary,
        "particle_ladder": ladder,
        "larger_n_only_copies": bool(copies_only),
        "isolated_assembled": bool(isolated_assembled),
        "all_sums_conserved": all(row["sum_conserved"] for row in compact + isolated),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"elementary={elementary} ladder={ladder} copies_only={copies_only} "
        f"iso_assembled={isolated_assembled} accepted={payload['accepted_n']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
