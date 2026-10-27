"""Одна частица: число порций до 16000 при давлении P."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit, prange

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_large_n_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
NS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16000)
ZERO_N = (8, 64, 256)
DT = 0.002


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


def duration(n: int) -> float:
    if n <= 64:
        return 50.0
    if n <= 1024:
        return 5.0
    if n <= 4096:
        return 1.0
    return 0.4


def fibonacci_sphere(n: int) -> np.ndarray:
    pts = np.zeros((n, 3))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i + 1.0) / n
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i, 0] = math.cos(theta) * ring
        pts[i, 1] = y
        pts[i, 2] = math.sin(theta) * ring
    return pts


@njit(parallel=True)
def radial_repulsion(pos):
    n = pos.shape[0]
    total = 0.0
    for i in prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist2 = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(dist2)
            inv3 = 1.0 / (dist2 * dist)
            ax += inv3 * dx
            ay += inv3 * dy
            az += inv3 * dz
        rr = math.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2 + pos[i, 2] ** 2)
        total += (ax * pos[i, 0] + ay * pos[i, 1] + az * pos[i, 2]) / rr
    return total / n


def balanced_radius(n: int, pressure: float) -> float:
    unit = fibonacci_sphere(n)
    lo, hi = 1e-4, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if radial_repulsion(unit * mid) > pressure:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(parallel=True)
def accelerations(pos, acc, pressure):
    n = pos.shape[0]
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
    hit = False
    min_pair = 1.0e300
    for i in prange(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        rx = pos[i, 0] - cx
        ry = pos[i, 1] - cy
        rz = pos[i, 2] - cz
        dist = math.sqrt(rx * rx + ry * ry + rz * rz)
        if dist > 0.0 and pressure > 0.0:
            scale = -pressure / dist
            ax = scale * rx
            ay = scale * ry
            az = scale * rz
        local_min = 1.0e300
        local_hit = False
        for j in range(n):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist2 = dx * dx + dy * dy + dz * dz
            d = math.sqrt(dist2)
            if d < local_min:
                local_min = d
            if d < 1.0e-4 or dist2 == 0.0:
                local_hit = True
                continue
            inv3 = 1.0 / (dist2 * d)
            ax += inv3 * dx
            ay += inv3 * dy
            az += inv3 * dz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
        if local_hit:
            hit = True
        if local_min < min_pair:
            min_pair = local_min
    return min_pair, hit, cx, cy, cz


@njit
def mean_radius(pos, cx, cy, cz):
    n = pos.shape[0]
    s = 0.0
    for i in range(n):
        dx = pos[i, 0] - cx
        dy = pos[i, 1] - cy
        dz = pos[i, 2] - cz
        s += math.sqrt(dx * dx + dy * dy + dz * dz)
    return s / n


@njit
def march(pos, vel, acc, pressure, steps, dt):
    n = pos.shape[0]
    start_r = 0.0
    for i in range(n):
        start_r += math.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2 + pos[i, 2] ** 2)
    start_r /= n
    max_err = 0.0
    min_pair = 1.0e300
    sum_r = 0.0
    hit = False
    done = 0
    for _ in range(steps):
        pair, close, _, _, _ = accelerations(pos, acc, pressure)
        if pair < min_pair:
            min_pair = pair
        if close:
            hit = True
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            pos[i, 0] += vel[i, 0] * dt + acc[i, 0] * half_dt2
            pos[i, 1] += vel[i, 1] * dt + acc[i, 1] * half_dt2
            pos[i, 2] += vel[i, 2] * dt + acc[i, 2] * half_dt2
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        pair, close, cx, cy, cz = accelerations(pos, acc, pressure)
        if pair < min_pair:
            min_pair = pair
        if close:
            hit = True
            break
        for i in range(n):
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        rms = mean_radius(pos, cx, cy, cz)
        err = abs(rms - start_r)
        if err > max_err:
            max_err = err
        sum_r += rms
        done += 1
    mean_r = sum_r / done if done else 0.0
    return start_r, mean_r, max_err, min_pair, hit, done


def one(n: int, pressure: float) -> dict[str, object]:
    t_end = duration(n) if pressure > 0.0 else duration(min(n, 64))
    if pressure == 0.0 and n > 64:
        t_end = 5.0
    steps = int(round(t_end / DT))
    radius = balanced_radius(n, pressure if pressure > 0.0 else 1.0)
    pos = fibonacci_sphere(n) * radius
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    start_r, mean_r, max_err, min_pair, hit, done = march(
        pos, vel, acc, pressure, steps, DT
    )
    accepted = (not hit) and pressure > 0.0 and max_err <= 0.05 * start_r
    print(
        f"n={n} P={pressure:g} t={t_end:g} R0={start_r:.5f} mean={mean_r:.5f} "
        f"err={max_err:.4g} ok={accepted}"
    )
    return {
        "n": n,
        "pressure": pressure,
        "t": t_end,
        "steps": steps,
        "steps_done": int(done),
        "start_radius": float(start_r),
        "mean_radius": float(mean_r),
        "max_radius_error": float(max_err),
        "min_pair": float(min_pair),
        "close_approach": bool(hit),
        "accepted": bool(accepted),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-large-n-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(n, 1.0) for n in NS]
    zeros = [one(n, 0.0) for n in ZERO_N]
    n8 = next(row for row in rows if row["n"] == 8)
    sphere_ok = 0.5 <= n8["start_radius"] <= 3.0
    accepted = [row for row in rows if row["accepted"]]
    radii = [row["start_radius"] for row in accepted]
    ladder = len(radii) >= 2 and max(radii) > 1.05 * min(radii)
    payload.update(
        {
            "sphere_assembly_ok": bool(sphere_ok),
            "rows": rows,
            "zero_pressure": zeros,
            "accepted_n": [row["n"] for row in accepted],
            "particle_ladder": bool(ladder),
            "all_p1_accepted": all(row["accepted"] for row in rows),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"sphere_ok={sphere_ok} accepted={payload['accepted_n']} "
        f"ladder={ladder} all={payload['all_p1_accepted']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
