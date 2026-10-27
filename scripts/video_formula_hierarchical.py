"""Иерархические тройки по формуле видео. Протокол записан отдельно."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/video_formula_hierarchical_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
RADII = (3.0, 5.0, 10.0, 20.0)


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


def initial(radius: float, shift: float) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5
    speed_in = math.sqrt(0.5)
    speed_out = math.sqrt(2.0 / radius)
    pos = np.array(
        [[-half, 0.0], [half, 0.0], [radius + shift, 0.0]],
        dtype=np.float64,
    )
    vel = np.array(
        [[0.0, -speed_in], [0.0, speed_in], [0.0, speed_out]],
        dtype=np.float64,
    )
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
    return pos, vel


@njit
def run(pos, vel, steps, dt, radius, limit):
    n = 3
    mass = np.ones(n)
    acc = np.zeros((n, 2))
    max_inner = 0.0
    max_outer = 0.0
    min_pair = 1.0e300
    hit = False
    for _ in range(steps):
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist2 = dx * dx + dy * dy
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < limit:
                    hit = True
                inv3 = 1.0 / (dist2 * dist)
                s = mass[j] * inv3
                acc[i, 0] += s * dx
                acc[i, 1] += s * dy
                s = mass[i] * inv3
                acc[j, 0] -= s * dx
                acc[j, 1] -= s * dy
        if hit:
            break
        for i in range(n):
            pos[i, 0] += vel[i, 0] * dt + acc[i, 0] * (0.5 * dt * dt)
            pos[i, 1] += vel[i, 1] * dt + acc[i, 1] * (0.5 * dt * dt)
            vel[i, 0] += acc[i, 0] * (0.5 * dt)
            vel[i, 1] += acc[i, 1] * (0.5 * dt)
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist2 = dx * dx + dy * dy
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                s = mass[j] * inv3
                acc[i, 0] += s * dx
                acc[i, 1] += s * dy
                s = mass[i] * inv3
                acc[j, 0] -= s * dx
                acc[j, 1] -= s * dy
        for i in range(n):
            vel[i, 0] += acc[i, 0] * (0.5 * dt)
            vel[i, 1] += acc[i, 1] * (0.5 * dt)
        inner = math.hypot(pos[1, 0] - pos[0, 0], pos[1, 1] - pos[0, 1])
        cx = 0.5 * (pos[0, 0] + pos[1, 0])
        cy = 0.5 * (pos[0, 1] + pos[1, 1])
        outer = math.hypot(pos[2, 0] - cx, pos[2, 1] - cy)
        err_in = abs(inner - 1.0)
        err_out = abs(outer - radius)
        if err_in > max_inner:
            max_inner = err_in
        if err_out > max_outer:
            max_outer = err_out
    return max_inner, max_outer, min_pair, hit


def one(radius: float, shift: float) -> dict[str, object]:
    pos, vel = initial(radius, shift)
    period_in = 2.0 * math.pi / math.sqrt(2.0)
    period_out = 2.0 * math.pi * math.sqrt(radius**3 / 2.0)
    dt = period_in / 2000.0
    steps = int(math.ceil(20.0 * period_out / dt))
    max_inner, max_outer, min_pair, hit = run(pos, vel, steps, dt, radius, 0.2)
    accepted = (not hit) and max_inner <= 0.05 and max_outer <= 0.05 * radius
    return {
        "radius": radius,
        "shift": shift,
        "steps": steps,
        "max_inner_error": float(max_inner),
        "max_outer_error": float(max_outer),
        "min_pair": float(min_pair),
        "close_approach": bool(hit),
        "accepted": bool(accepted),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    rows = []
    if gate["passed"]:
        for radius in RADII:
            plain = one(radius, 0.0)
            moved = one(radius, 1e-3 * radius)
            stable = bool(plain["accepted"] and moved["accepted"])
            row = {"plain": plain, "shifted": moved, "stable": stable}
            rows.append(row)
            print(
                f"R={radius} inner={plain['max_inner_error']:.4g} "
                f"outer={plain['max_outer_error']:.4g} close={plain['close_approach']} "
                f"shift_ok={moved['accepted']} stable={stable}"
            )
    payload = {
        "scope": "model",
        "protocol": "docs/video-formula-hierarchical-protocol-2026-10-27.md",
        "formula_gate": gate,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
