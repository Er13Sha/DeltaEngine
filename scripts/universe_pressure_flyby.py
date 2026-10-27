"""Две порции: отталкивание без столкновения, пролёт при b > 0."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_flyby_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
IMPACTS = (0.0, 0.2, 0.5, 1.0, 2.0)
PRESSURES = (0.0, 1.0)


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
def march(pos, vel, pressure, t_end):
    min_r = 1.0e300
    t = 0.0
    base_dt = 0.002
    while t < t_end:
        dx = pos[0, 0] - pos[1, 0]
        dy = pos[0, 1] - pos[1, 1]
        dz = pos[0, 2] - pos[1, 2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < min_r:
            min_r = dist
        if dist <= 1.0e-4:
            break
        rel_vx = vel[0, 0] - vel[1, 0]
        rel_vy = vel[0, 1] - vel[1, 1]
        rel_vz = vel[0, 2] - vel[1, 2]
        rel_v = math.sqrt(rel_vx * rel_vx + rel_vy * rel_vy + rel_vz * rel_vz)
        dt = base_dt
        if dist < 1.0:
            if rel_v > 0.0:
                dt = min(dt, 0.02 * dist / rel_v)
            dt = min(dt, 0.2 * dist * math.sqrt(dist))
        if dt < 1.0e-12:
            dt = 1.0e-12
        if t + dt > t_end:
            dt = t_end - t
        cx = 0.5 * (pos[0, 0] + pos[1, 0])
        cy = 0.5 * (pos[0, 1] + pos[1, 1])
        cz = 0.5 * (pos[0, 2] + pos[1, 2])
        acc = np.zeros((2, 3))
        inv3 = 1.0 / (dist * dist * dist)
        acc[0, 0] += inv3 * dx
        acc[0, 1] += inv3 * dy
        acc[0, 2] += inv3 * dz
        acc[1, 0] -= inv3 * dx
        acc[1, 1] -= inv3 * dy
        acc[1, 2] -= inv3 * dz
        if pressure > 0.0:
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
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(2):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        dx = pos[0, 0] - pos[1, 0]
        dy = pos[0, 1] - pos[1, 1]
        dz = pos[0, 2] - pos[1, 2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < min_r:
            min_r = dist
        if dist <= 1.0e-4:
            break
        inv3 = 1.0 / (dist * dist * dist)
        acc[0, 0] = inv3 * dx
        acc[0, 1] = inv3 * dy
        acc[0, 2] = inv3 * dz
        acc[1, 0] = -inv3 * dx
        acc[1, 1] = -inv3 * dy
        acc[1, 2] = -inv3 * dz
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
        for i in range(2):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        t += dt
        sep_x = abs(pos[0, 0] - pos[1, 0])
        if t > 5.0 and sep_x > 20.0:
            break
    return min_r, t, pos[0, 0], pos[1, 0], vel[0, 0], vel[1, 0]


def one(b: float, pressure: float) -> dict[str, object]:
    pos = np.array([[-10.0, 0.5 * b, 0.0], [10.0, -0.5 * b, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    min_r, t, x0, x1, vx0, vx1 = march(pos, vel, pressure, 40.0)
    collision = min_r <= 1.0e-4
    if b == 0.0:
        kind_ok = (not collision) and x0 < 0.0 and vx0 < 0.0
        expected = "rebound"
    else:
        kind_ok = (not collision) and x0 > x1
        expected = "flyby"
    print(
        f"b={b:g} P={pressure:g} minr={min_r:.4g} x=({x0:.3f},{x1:.3f}) "
        f"ok={kind_ok} {expected}"
    )
    return {
        "b": b,
        "pressure": pressure,
        "min_separation": float(min_r),
        "t_done": float(t),
        "x0": float(x0),
        "x1": float(x1),
        "vx0": float(vx0),
        "vx1": float(vx1),
        "collision": bool(collision),
        "expected": expected,
        "accepted": bool(kind_ok),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-flyby-protocol-2026-10-27.md",
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
            "no_collisions": all(not row["collision"] for row in rows),
            "grid_accepted": all(row["accepted"] for row in rows),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"no_collisions={payload['no_collisions']} grid={payload['grid_accepted']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
