"""Локальное 1/r² действует на комплекс: Q живой, без статического покоя."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_reactive_complex_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
NS = (2, 4, 8)
P_INF = 1.0
DT = 0.002
T_END = 50.0
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


def seeds(n: int) -> np.ndarray:
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
    else:
        pts = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
    pts = pts.astype(np.float64)
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
    total = 0.0
    for i in range(n):
        radius = float(np.linalg.norm(points[i]))
        total += float(acc[i] @ points[i]) / radius
    return total / n


def scale_radius(n: int, pressure: float) -> float:
    lo, hi = 1e-6, 20.0
    unit = seeds(n)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        extra = radial_repulsion(unit * mid) - pressure
        if extra > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit
def accelerations(pos, pressure):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
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
    f_out = 0.0
    mean_r = 0.0
    for i in range(n):
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
            if dist2 == 0.0:
                continue
            dist = math.sqrt(dist2)
            inv3 = 1.0 / (dist2 * dist)
            ax += inv3 * dx
            ay += inv3 * dy
            az += inv3 * dz
        rx = pos[i, 0] - cx
        ry = pos[i, 1] - cy
        rz = pos[i, 2] - cz
        rr = math.sqrt(rx * rx + ry * ry + rz * rz)
        mean_r += rr
        if rr > 0.0:
            f_out += (ax * rx + ay * ry + az * rz) / rr
            s = -pressure / rr
            ax += s * rx
            ay += s * ry
            az += s * rz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
    f_out /= n
    mean_r /= n
    return acc, f_out, mean_r


@njit
def march(pos, vel, p_inf, reactive, steps, dt):
    n = pos.shape[0]
    q = 0.0
    qv = 0.0
    radii = np.zeros(steps)
    qs = np.zeros(steps)
    speeds = np.zeros(steps)
    for s in range(steps):
        p = p_inf * math.exp(-q) if reactive else p_inf
        acc, f_out, mean_r = accelerations(pos, p)
        qacc = (f_out - p) if reactive else 0.0
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            pos[i, 0] += vel[i, 0] * dt + acc[i, 0] * half_dt2
            pos[i, 1] += vel[i, 1] * dt + acc[i, 1] * half_dt2
            pos[i, 2] += vel[i, 2] * dt + acc[i, 2] * half_dt2
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        q += qv * dt + qacc * half_dt2
        qv += qacc * half
        p = p_inf * math.exp(-q) if reactive else p_inf
        acc, f_out, mean_r = accelerations(pos, p)
        qacc = (f_out - p) if reactive else 0.0
        for i in range(n):
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        qv += qacc * half
        speed = 0.0
        for i in range(n):
            speed += math.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
        radii[s] = mean_r
        qs[s] = q
        speeds[s] = speed / n
    return radii, qs, speeds


def stats(arr: np.ndarray) -> dict[str, float]:
    half = arr[len(arr) // 2 :]
    return {
        "mean_second_half": float(half.mean()),
        "span_second_half": float(half.max() - half.min()),
        "max_abs": float(np.max(np.abs(arr))),
        "last": float(arr[-1]),
    }


def one(n: int, reactive: bool) -> dict[str, object]:
    radius = 0.8 * scale_radius(n, P_INF)
    pos = seeds(n) * radius
    vel = np.zeros_like(pos)
    radii, qs, speeds = march(pos, vel, P_INF, reactive, STEPS, DT)
    rstat = stats(radii)
    qstat = stats(qs)
    vstat = stats(speeds)
    label = "reactive" if reactive else "frozen"
    print(
        f"n={n} {label} R={rstat['mean_second_half']:.4g} "
        f"span={rstat['span_second_half']:.4g} Qmax={qstat['max_abs']:.4g}"
    )
    return {
        "n": n,
        "mode": label,
        "start_radius": float(radius),
        "radius": rstat,
        "q": qstat,
        "speed": vstat,
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-reactive-complex-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    live = [one(n, True) for n in NS]
    frozen = [one(n, False) for n in NS]
    reacted = all(row["q"]["max_abs"] > 0.01 for row in live)
    moving = all(
        row["radius"]["span_second_half"] > 0.02 * row["radius"]["mean_second_half"]
        for row in live
    )
    differed = False
    for a, b in zip(live, frozen):
        base = abs(b["radius"]["mean_second_half"])
        if base > 0.0:
            rel = abs(a["radius"]["mean_second_half"] - b["radius"]["mean_second_half"]) / base
            if rel > 0.02:
                differed = True
    payload.update(
        {
            "live": live,
            "frozen": frozen,
            "complex_reacted": bool(reacted),
            "no_statics": bool(moving),
            "frozen_differs": bool(differed),
            "grid_accepted": bool(reacted and moving and differed),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"reacted={reacted} moving={moving} differed={differed} "
        f"grid={payload['grid_accepted']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
