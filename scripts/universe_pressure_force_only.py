"""Две порции: только 1/r² и P, без столкновения в законе."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_force_only_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
IMPACTS = (0.0, 0.2, 0.5, 1.0, 2.0)
PRESSURES = (0.0, 1.0)
DT = 0.002
T_END = 40.0


def formula_gate() -> dict[str, object]:
    linked = coupling(C1)
    exact = linked == PiNumber(Fraction(1)) and linked.power == 0
    rows = []
    for radius in (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(10)):
        got = force(C1, radius, 1, 1)
        expected = PiNumber(Fraction(1) / radius**2)
        match = got == expected and got.power == 0
        exact = exact and match
        rows.append({"r": str(radius), "force": str(got), "match": bool(match)})
    return {"passed": bool(exact), "coupling": str(linked), "rows": rows}


@njit
def accelerations(pos, pressure):
    acc = np.zeros((2, 3))
    dx = pos[0, 0] - pos[1, 0]
    dy = pos[0, 1] - pos[1, 1]
    dz = pos[0, 2] - pos[1, 2]
    dist2 = dx * dx + dy * dy + dz * dz
    skipped = False
    if dist2 > 0.0:
        dist = math.sqrt(dist2)
        inv3 = 1.0 / (dist2 * dist)
        acc[0, 0] = inv3 * dx
        acc[0, 1] = inv3 * dy
        acc[0, 2] = inv3 * dz
        acc[1, 0] = -inv3 * dx
        acc[1, 1] = -inv3 * dy
        acc[1, 2] = -inv3 * dz
    else:
        skipped = True
    if pressure > 0.0:
        cx = 0.5 * (pos[0, 0] + pos[1, 0])
        cy = 0.5 * (pos[0, 1] + pos[1, 1])
        cz = 0.5 * (pos[0, 2] + pos[1, 2])
        for i in range(2):
            rx = pos[i, 0] - cx
            ry = pos[i, 1] - cy
            rz = pos[i, 2] - cz
            rr = math.sqrt(rx * rx + ry * ry + rz * rz)
            if rr > 0.0:
                s = -pressure / rr
                acc[i, 0] += s * rx
                acc[i, 1] += s * ry
                acc[i, 2] += s * rz
    dist = math.sqrt(dist2) if dist2 > 0.0 else 0.0
    return acc, dist, skipped


@njit
def march(pos, vel, pressure):
    min_r = 1.0e300
    skipped_steps = 0
    t = 0.0
    steps = 0
    while t < T_END:
        acc, dist, skipped = accelerations(pos, pressure)
        if dist < min_r:
            min_r = dist
        if skipped:
            skipped_steps += 1
        half = 0.5 * DT
        half_dt2 = 0.5 * DT * DT
        for i in range(2):
            for d in range(3):
                pos[i, d] += vel[i, d] * DT + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        acc, dist, skipped = accelerations(pos, pressure)
        if dist < min_r:
            min_r = dist
        if skipped:
            skipped_steps += 1
        for i in range(2):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        t += DT
        steps += 1
        if t > 5.0 and abs(pos[0, 0] - pos[1, 0]) > 20.0:
            break
    swapped = pos[0, 0] > pos[1, 0]
    rebound = pos[0, 0] < 0.0 and vel[0, 0] < 0.0 and not swapped
    return min_r, t, steps, skipped_steps, pos[0, 0], pos[1, 0], vel[0, 0], vel[1, 0], swapped, rebound


def one(b: float, pressure: float) -> dict[str, object]:
    pos = np.array([[-10.0, 0.5 * b, 0.0], [10.0, -0.5 * b, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    min_r, t, steps, skipped, x0, x1, vx0, vx1, swapped, rebound = march(
        pos, vel, pressure
    )
    print(
        f"b={b:g} P={pressure:g} minr={min_r:.4g} swapped={swapped} "
        f"rebound={rebound} skip={skipped}"
    )
    return {
        "b": b,
        "pressure": pressure,
        "min_separation": float(min_r),
        "t_done": float(t),
        "steps": int(steps),
        "undefined_pair_skips": int(skipped),
        "x0": float(x0),
        "x1": float(x1),
        "vx0": float(vx0),
        "vx1": float(vx1),
        "sides_swapped": bool(swapped),
        "turned_back": bool(rebound),
        "halted": False,
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-force-only-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(b, p) for p in PRESSURES for b in IMPACTS]
    payload.update(
        {
            "rows": rows,
            "all_finished": all(not row["halted"] for row in rows),
            "no_undefined_skips": all(row["undefined_pair_skips"] == 0 for row in rows),
            "grid_accepted": all(
                (not row["halted"]) and row["undefined_pair_skips"] == 0 for row in rows
            ),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"grid={payload['grid_accepted']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
