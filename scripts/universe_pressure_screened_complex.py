"""Порции закрывают комплекс друг другом; локальное 1/r² всегда."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_screened_complex_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
P_INF = 1.0
N_SKY = 72
DT = 0.002
T_END = 20.0
STEPS = int(round(T_END / DT))


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


def fibonacci_sky(count: int) -> np.ndarray:
    phi = math.pi * (3.0 - math.sqrt(5.0))
    pts = np.zeros((count, 3))
    for i in range(count):
        y = 1.0 - (2.0 * i + 1.0) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i, 0] = math.cos(theta) * ring
        pts[i, 1] = y
        pts[i, 2] = math.sin(theta) * ring
        nrm = math.sqrt(pts[i, 0] ** 2 + pts[i, 1] ** 2 + pts[i, 2] ** 2)
        pts[i] /= nrm
    return pts


def seeds(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 3))
    if n == 2:
        pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    elif n == 4:
        pts = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
    elif n == 9:
        cube = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        pts = np.vstack([cube, np.zeros((1, 3))])
    else:
        pts = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
    pts = pts.astype(np.float64)
    if n == 9:
        core = pts[:8]
        core -= core.mean(axis=0)
        norms = np.linalg.norm(core, axis=1, keepdims=True)
        pts[:8] = core / norms
        pts[8] = 0.0
        return pts
    pts -= pts.mean(axis=0)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


@njit
def accelerations(pos, sky, pressure, screened_sky):
    n = pos.shape[0]
    nsky = sky.shape[0]
    weight = 4.0 * math.pi / nsky
    acc = np.zeros((n, 3))
    local = np.zeros((n, 3))
    complex_f = np.zeros((n, 3))
    cover = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 == 0.0:
                continue
            dist = math.sqrt(dist2)
            inv3 = 1.0 / (dist2 * dist)
            local[i, 0] += inv3 * dx
            local[i, 1] += inv3 * dy
            local[i, 2] += inv3 * dz
        if screened_sky and pressure > 0.0:
            closed = 0.0
            for k in range(nsky):
                nx = sky[k, 0]
                ny = sky[k, 1]
                nz = sky[k, 2]
                occupied = False
                best = 1.0e300
                for j in range(n):
                    if i == j:
                        continue
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dz = pos[j, 2] - pos[i, 2]
                    dist2 = dx * dx + dy * dy + dz * dz
                    if dist2 == 0.0:
                        continue
                    dist = math.sqrt(dist2)
                    ux = dx / dist
                    uy = dy / dist
                    uz = dz / dist
                    omega = 1.0 / dist2
                    if omega > 2.0 * math.pi:
                        omega = 2.0 * math.pi
                    cos_a = 1.0 - omega / (2.0 * math.pi)
                    if cos_a < -1.0:
                        cos_a = -1.0
                    if ux * nx + uy * ny + uz * nz >= cos_a and dist < best:
                        occupied = True
                        best = dist
                if occupied:
                    closed += 1.0
                    complex_f[i, 0] += pressure * nx * weight
                    complex_f[i, 1] += pressure * ny * weight
                    complex_f[i, 2] += pressure * nz * weight
            cover[i] = closed / nsky
        acc[i, 0] = local[i, 0] + complex_f[i, 0]
        acc[i, 1] = local[i, 1] + complex_f[i, 1]
        acc[i, 2] = local[i, 2] + complex_f[i, 2]
    return acc, local, complex_f, cover


@njit
def mean_radius(pos):
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
    s = 0.0
    for i in range(n):
        dx = pos[i, 0] - cx
        dy = pos[i, 1] - cy
        dz = pos[i, 2] - cz
        s += math.sqrt(dx * dx + dy * dy + dz * dz)
    return s / n


@njit
def march(pos, vel, sky, pressure, screened_sky, steps, dt):
    radii = np.zeros(steps)
    for s in range(steps):
        acc, _, _, _ = accelerations(pos, sky, pressure, screened_sky)
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        n = pos.shape[0]
        for i in range(n):
            pos[i, 0] += vel[i, 0] * dt + acc[i, 0] * half_dt2
            pos[i, 1] += vel[i, 1] * dt + acc[i, 1] * half_dt2
            pos[i, 2] += vel[i, 2] * dt + acc[i, 2] * half_dt2
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        acc, _, _, _ = accelerations(pos, sky, pressure, screened_sky)
        for i in range(n):
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        radii[s] = mean_radius(pos)
    return radii


def snapshot(n: int, sky: np.ndarray) -> dict[str, object]:
    pos = seeds(n)
    acc, local, complex_f, cover = accelerations(pos, sky, P_INF, True)
    local_mag = np.linalg.norm(local, axis=1)
    complex_mag = np.linalg.norm(complex_f, axis=1)
    total_mag = np.linalg.norm(acc, axis=1)
    pair = None
    if n >= 2:
        delta = pos[0] - pos[1]
        dist = float(np.linalg.norm(delta))
        expected = 1.0 / dist**2
        got = abs(float(local[0] @ (delta / dist)))
        pair = {"r": dist, "local_radial": got, "expected_1_over_r2": expected, "match": abs(got - expected) < 1e-9}
    print(f"snap n={n} cover={cover} complex={complex_mag}")
    return {
        "n": n,
        "cover": cover.tolist(),
        "local_mag": local_mag.tolist(),
        "complex_mag": complex_mag.tolist(),
        "total_mag": total_mag.tolist(),
        "pair": pair,
    }


def run_mode(n: int, sky: np.ndarray, screened: bool) -> dict[str, object]:
    pos = seeds(n)
    vel = np.zeros_like(pos)
    radii = march(pos, vel, sky, P_INF, screened, STEPS, DT)
    half = radii[len(radii) // 2 :]
    label = "screened" if screened else "local_only"
    print(f"n={n} {label} R={float(half.mean()):.4g} span={float(half.max() - half.min()):.4g}")
    return {
        "n": n,
        "mode": label,
        "mean_second_half": float(half.mean()),
        "span_second_half": float(half.max() - half.min()),
        "last": float(radii[-1]),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    sky = fibonacci_sky(N_SKY)
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-screened-complex-protocol-2026-10-27.md",
        "formula_gate": gate,
        "n_sky": N_SKY,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    one = snapshot(1, sky)
    two = snapshot(2, sky)
    nine = snapshot(9, sky)
    pair_ok = bool(two["pair"] is not None and two["pair"]["match"])
    isolated = float(one["total_mag"][0]) < 1e-6
    center_cover = nine["cover"][8]
    corner_cover = float(np.mean(nine["cover"][:8]))
    center_complex = nine["complex_mag"][8]
    corner_complex = float(np.mean(nine["complex_mag"][:8]))
    interior = center_cover > corner_cover and center_complex < 0.5 * corner_complex
    runs = []
    for n in (2, 4, 9):
        runs.append(run_mode(n, sky, True))
        runs.append(run_mode(n, sky, False))
    two_s = next(r for r in runs if r["n"] == 2 and r["mode"] == "screened")
    two_l = next(r for r in runs if r["n"] == 2 and r["mode"] == "local_only")
    motion = two_s["mean_second_half"] < 0.95 * two_l["mean_second_half"]
    payload.update(
        {
            "isolated": one,
            "nine": nine,
            "runs": runs,
            "isolated_zero": bool(isolated),
            "pair_force_1r2": bool(pair_ok),
            "interior_screened": bool(interior),
            "screening_changes_motion": bool(motion),
            "grid_accepted": bool(isolated and pair_ok and interior and motion),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"isolated={isolated} pair={pair_ok} interior={interior} motion={motion} "
        f"grid={payload['grid_accepted']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
