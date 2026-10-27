"""Две частицы под константой давления P: связь, слияние или разлёт."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_bond_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
NS = (2, 4)
PRESSURES = (1.0, 0.25, 0.0)
D_FACTORS = (2.0, 4.0, 8.0)
DT = 0.002
STEPS = int(round(50.0 / DT))


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


def seeds(n: int) -> np.ndarray:
    if n == 2:
        pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    else:
        pts = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
    pts -= pts.mean(axis=0)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def radial_repulsion(points: np.ndarray) -> float:
    n = points.shape[0]
    acc = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = points[i] - points[j]
            dist2 = float(delta @ delta)
            dist = math.sqrt(dist2)
            acc[i] += delta / (dist2 * dist)
    radial = 0.0
    for i in range(n):
        radial += float(acc[i] @ points[i]) / float(np.linalg.norm(points[i]))
    return radial / n


def balanced_radius(n: int, pressure: float) -> float:
    if pressure <= 0.0:
        return balanced_radius(n, 1.0)
    lo, hi = 1e-6, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pts = seeds(n) * mid
        if radial_repulsion(pts) > pressure:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit
def march(pos, vel, charge, pressure, steps, dt):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    min_pair = 1.0e300
    hit = False
    sum_d = 0.0
    min_d = 1.0e300
    max_d = 0.0
    done = 0
    half_n = n // 2
    for _ in range(steps):
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for i in range(n):
            cx += pos[i, 0]
            cy += pos[i, 1]
            cz += pos[i, 2]
        cx /= n
        cy /= n
        cz /= n
        for i in range(n):
            rx = pos[i, 0] - cx
            ry = pos[i, 1] - cy
            rz = pos[i, 2] - cz
            dist = math.sqrt(rx * rx + ry * ry + rz * rz)
            if dist == 0.0 or pressure == 0.0:
                acc[i, 0] = 0.0
                acc[i, 1] = 0.0
                acc[i, 2] = 0.0
            else:
                scale = -pressure / dist
                acc[i, 0] = scale * rx
                acc[i, 1] = scale * ry
                acc[i, 2] = scale * rz
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < 1.0e-4 or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = charge[i] * charge[j] / (dist2 * dist)
                acc[i, 0] += inv3 * dx
                acc[i, 1] += inv3 * dy
                acc[i, 2] += inv3 * dz
                acc[j, 0] -= inv3 * dx
                acc[j, 1] -= inv3 * dy
                acc[j, 2] -= inv3 * dz
        if hit:
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for i in range(n):
            cx += pos[i, 0]
            cy += pos[i, 1]
            cz += pos[i, 2]
        cx /= n
        cy /= n
        cz /= n
        for i in range(n):
            rx = pos[i, 0] - cx
            ry = pos[i, 1] - cy
            rz = pos[i, 2] - cz
            dist = math.sqrt(rx * rx + ry * ry + rz * rz)
            if dist == 0.0 or pressure == 0.0:
                acc[i, 0] = 0.0
                acc[i, 1] = 0.0
                acc[i, 2] = 0.0
            else:
                scale = -pressure / dist
                acc[i, 0] = scale * rx
                acc[i, 1] = scale * ry
                acc[i, 2] = scale * rz
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < 1.0e-4 or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = charge[i] * charge[j] / (dist2 * dist)
                acc[i, 0] += inv3 * dx
                acc[i, 1] += inv3 * dy
                acc[i, 2] += inv3 * dz
                acc[j, 0] -= inv3 * dx
                acc[j, 1] -= inv3 * dy
                acc[j, 2] -= inv3 * dz
        if hit:
            break
        for i in range(n):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        c0x = 0.0
        c0y = 0.0
        c0z = 0.0
        c1x = 0.0
        c1y = 0.0
        c1z = 0.0
        for i in range(half_n):
            c0x += pos[i, 0]
            c0y += pos[i, 1]
            c0z += pos[i, 2]
            c1x += pos[i + half_n, 0]
            c1y += pos[i + half_n, 1]
            c1z += pos[i + half_n, 2]
        c0x /= half_n
        c0y /= half_n
        c0z /= half_n
        c1x /= half_n
        c1y /= half_n
        c1z /= half_n
        dx = c1x - c0x
        dy = c1y - c0y
        dz = c1z - c0z
        sep = math.sqrt(dx * dx + dy * dy + dz * dz)
        if sep < min_d:
            min_d = sep
        if sep > max_d:
            max_d = sep
        sum_d += sep
        done += 1
    mean_d = sum_d / done if done else 0.0
    return min_pair, hit, mean_d, min_d, max_d, done, pos


def group_radius(pos: np.ndarray, sl: slice) -> float:
    chunk = pos[sl]
    center = chunk.mean(axis=0)
    return float(np.mean(np.linalg.norm(chunk - center, axis=1)))


def classify(pos: np.ndarray, n: int, radius: float, mean_d: float, min_d: float, max_d: float, hit: bool) -> str:
    if hit:
        return "close_approach"
    r0 = group_radius(pos, slice(0, n))
    r1 = group_radius(pos, slice(n, 2 * n))
    all_r = float(np.mean(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))
    width = max_d - min_d
    intact = r0 <= 2.0 * radius and r1 <= 2.0 * radius
    if all_r <= 3.0 * radius:
        return "merged"
    if mean_d > 20.0 * radius:
        return "scattered"
    if (
        intact
        and 1.5 * radius <= mean_d <= 12.0 * radius
        and width <= 0.20 * mean_d
    ):
        return "bond"
    return "other"


def one(n: int, pressure: float, factor: float, unlike: bool) -> dict[str, object]:
    radius = balanced_radius(n, pressure)
    left = seeds(n) * radius
    right = seeds(n) * radius
    gap = factor * radius
    left[:, 0] -= 0.5 * gap
    right[:, 0] += 0.5 * gap
    pos = np.vstack([left, right])
    vel = np.zeros_like(pos)
    charge = np.ones(2 * n)
    if unlike:
        charge[n:] = -1.0
    min_pair, hit, mean_d, min_d, max_d, done, pos = march(
        pos, vel, charge, pressure, STEPS, DT
    )
    kind = classify(pos, n, radius, mean_d, min_d, max_d, hit)
    print(
        f"{'unlike' if unlike else 'like'} n={n} P={pressure:g} D={factor:.0f}R "
        f"{kind} meanD={mean_d:.4g} r0={group_radius(pos, slice(0, n)):.4g}"
    )
    return {
        "n": n,
        "pressure": pressure,
        "d_over_r": factor,
        "unlike": unlike,
        "single_radius": radius,
        "mean_centers": float(mean_d),
        "min_centers": float(min_d),
        "max_centers": float(max_d),
        "group0_radius": group_radius(pos, slice(0, n)),
        "group1_radius": group_radius(pos, slice(n, 2 * n)),
        "all_radius": float(np.mean(np.linalg.norm(pos - pos.mean(axis=0), axis=1))),
        "min_pair": float(min_pair),
        "close_approach": bool(hit),
        "kind": kind,
        "steps_done": int(done),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-bond-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    r2 = balanced_radius(2, 1.0)
    if abs(r2 - 0.5) > 1e-9:
        payload["aborted"] = "n=2 radius is not 1/2"
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1)
    rows = []
    for unlike in (False, True):
        for n in NS:
            for pressure in PRESSURES:
                for factor in D_FACTORS:
                    rows.append(one(n, pressure, factor, unlike))
    bonds = [row for row in rows if row["kind"] == "bond" and row["pressure"] > 0]
    zero_same = [
        row
        for row in rows
        if row["pressure"] == 0
        and any(
            b["n"] == row["n"]
            and b["d_over_r"] == row["d_over_r"]
            and b["unlike"] == row["unlike"]
            and b["kind"] == "bond"
            for b in bonds
        )
        and row["kind"] == "bond"
    ]
    payload.update(
        {
            "rows": rows,
            "bonds": bonds,
            "substance_starts": bool(bonds) and not zero_same,
            "bond_count": len(bonds),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bonds={len(bonds)} substance_starts={payload['substance_starts']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
