"""Много частиц (пар порций) под давлением P."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit, prange

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_pressure_many_particles_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
MS = (3, 4, 8, 16, 32, 64, 128)
R0 = 0.5
SPACING = 4.0 * R0
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


def duration(m: int) -> float:
    if m <= 16:
        return 5.0
    if m <= 64:
        return 1.0
    return 0.4


def centers_grid(m: int) -> np.ndarray:
    side = int(math.ceil(m ** (1.0 / 3.0)))
    pts = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                pts.append((x * SPACING, y * SPACING, z * SPACING))
                if len(pts) == m:
                    arr = np.array(pts, dtype=np.float64)
                    arr -= arr.mean(axis=0)
                    return arr
    raise RuntimeError("grid")


def initial(m: int) -> np.ndarray:
    centres = centers_grid(m)
    pos = np.zeros((2 * m, 3))
    for i, c in enumerate(centres):
        pos[2 * i] = c + np.array([-R0, 0.0, 0.0])
        pos[2 * i + 1] = c + np.array([R0, 0.0, 0.0])
    return pos


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
def cloud_rms(pos, m):
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for i in range(m):
        px = 0.5 * (pos[2 * i, 0] + pos[2 * i + 1, 0])
        py = 0.5 * (pos[2 * i, 1] + pos[2 * i + 1, 1])
        pz = 0.5 * (pos[2 * i, 2] + pos[2 * i + 1, 2])
        cx += px
        cy += py
        cz += pz
    cx /= m
    cy /= m
    cz /= m
    s = 0.0
    for i in range(m):
        px = 0.5 * (pos[2 * i, 0] + pos[2 * i + 1, 0]) - cx
        py = 0.5 * (pos[2 * i, 1] + pos[2 * i + 1, 1]) - cy
        pz = 0.5 * (pos[2 * i, 2] + pos[2 * i + 1, 2]) - cz
        s += px * px + py * py + pz * pz
    return math.sqrt(s / m)


@njit
def intact_count(pos, m, limit):
    count = 0
    for i in range(m):
        cx = 0.5 * (pos[2 * i, 0] + pos[2 * i + 1, 0])
        cy = 0.5 * (pos[2 * i, 1] + pos[2 * i + 1, 1])
        cz = 0.5 * (pos[2 * i, 2] + pos[2 * i + 1, 2])
        d0 = math.sqrt(
            (pos[2 * i, 0] - cx) ** 2 + (pos[2 * i, 1] - cy) ** 2 + (pos[2 * i, 2] - cz) ** 2
        )
        d1 = math.sqrt(
            (pos[2 * i + 1, 0] - cx) ** 2
            + (pos[2 * i + 1, 1] - cy) ** 2
            + (pos[2 * i + 1, 2] - cz) ** 2
        )
        if d0 <= limit and d1 <= limit:
            count += 1
    return count


@njit
def all_radius(pos, cx, cy, cz):
    n = pos.shape[0]
    s = 0.0
    for i in range(n):
        dx = pos[i, 0] - cx
        dy = pos[i, 1] - cy
        dz = pos[i, 2] - cz
        s += math.sqrt(dx * dx + dy * dy + dz * dz)
    return s / n


@njit
def march(pos, vel, acc, pressure, steps, dt, m):
    start_rms = cloud_rms(pos, m)
    max_rel = 0.0
    min_pair = 1.0e300
    hit = False
    done = 0
    last_rms = start_rms
    last_intact = m
    last_all = 0.0
    for _ in range(steps):
        pair, close, _, _, _ = accelerations(pos, acc, pressure)
        if pair < min_pair:
            min_pair = pair
        if close:
            hit = True
            break
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
        rms = cloud_rms(pos, m)
        rel = abs(rms - start_rms) / start_rms if start_rms > 0.0 else rms
        if rel > max_rel:
            max_rel = rel
        last_rms = rms
        last_intact = intact_count(pos, m, 2.0 * R0)
        last_all = all_radius(pos, cx, cy, cz)
        done += 1
    return start_rms, last_rms, max_rel, last_intact, last_all, min_pair, hit, done


def classify(start_rms, last_rms, max_rel, intact, all_r, hit) -> str:
    if hit:
        return "close_approach"
    if all_r <= 3.0 * R0:
        return "merged"
    if last_rms > 20.0 * start_rms:
        return "scattered"
    if intact >= 3 and max_rel <= 0.20:
        return "assembled"
    return "other"


def one(m: int, pressure: float) -> dict[str, object]:
    t_end = duration(m)
    if pressure == 0.0:
        t_end = 5.0
    steps = int(round(t_end / DT))
    pos = initial(m)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    start_rms, last_rms, max_rel, intact, all_r, min_pair, hit, done = march(
        pos, vel, acc, pressure, steps, DT, m
    )
    kind = classify(start_rms, last_rms, max_rel, intact, all_r, hit)
    print(
        f"M={m} P={pressure:g} {kind} intact={intact}/{m} "
        f"rms={last_rms:.4g} allR={all_r:.4g}"
    )
    return {
        "m": m,
        "pressure": pressure,
        "t": t_end,
        "kind": kind,
        "intact": int(intact),
        "start_rms": float(start_rms),
        "end_rms": float(last_rms),
        "max_relative_rms": float(max_rel),
        "all_radius": float(all_r),
        "min_pair": float(min_pair),
        "close_approach": bool(hit),
        "steps_done": int(done),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-many-particles-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(m, 1.0) for m in MS]
    zero = one(8, 0.0)
    assembled = [row for row in rows if row["kind"] == "assembled"]
    payload.update(
        {
            "rows": rows,
            "zero_M8": zero,
            "assembled_m": [row["m"] for row in assembled],
            "substance_on_grid": bool(assembled) and zero["kind"] != "assembled",
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"assembled={payload['assembled_m']} substance={payload['substance_on_grid']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
