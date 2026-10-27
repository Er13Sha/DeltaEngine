"""Сжатие вселенной как константа Ω² при отталкивании порций."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/universe_constant_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
OMEGAS = (1.0, 0.25, 0.0625, 0.0)
NS = (2, 3, 4, 6, 8)
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


def polygon(n: int, radius: float) -> np.ndarray:
    pos = np.zeros((n, 3))
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        pos[k, 0] = radius * math.cos(ang)
        pos[k, 1] = radius * math.sin(ang)
    pos -= pos.mean(axis=0)
    return pos


def radial_repulsion(pos: np.ndarray) -> float:
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = pos[i] - pos[j]
            dist2 = float(delta @ delta)
            dist = math.sqrt(dist2)
            acc[i] += delta / (dist2 * dist)
    radii = np.linalg.norm(pos, axis=1)
    radial = 0.0
    for i in range(n):
        radial += float(acc[i] @ pos[i]) / radii[i]
    return radial / n


def balanced_radius(n: int, omega2: float) -> float:
    if omega2 <= 0.0:
        return balanced_radius(n, 1.0)
    lo, hi = 1e-6, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        extra = radial_repulsion(polygon(n, mid)) - omega2 * mid
        if extra > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit
def march(pos, vel, omega2, steps, dt):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    start_r = 0.0
    for i in range(n):
        start_r += math.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2 + pos[i, 2] ** 2)
    start_r /= n
    max_err = 0.0
    min_pair = 1.0e300
    max_r = 0.0
    sum_r = 0.0
    hit = False
    done = 0
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
            acc[i, 0] = -omega2 * (pos[i, 0] - cx)
            acc[i, 1] = -omega2 * (pos[i, 1] - cy)
            acc[i, 2] = -omega2 * (pos[i, 2] - cz)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < 1.0e-4:
                    hit = True
                inv3 = 1.0 / (dist2 * dist)
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
            acc[i, 0] = -omega2 * (pos[i, 0] - cx)
            acc[i, 1] = -omega2 * (pos[i, 1] - cy)
            acc[i, 2] = -omega2 * (pos[i, 2] - cz)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += inv3 * dx
                acc[i, 1] += inv3 * dy
                acc[i, 2] += inv3 * dz
                acc[j, 0] -= inv3 * dx
                acc[j, 1] -= inv3 * dy
                acc[j, 2] -= inv3 * dz
        for i in range(n):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        rms = 0.0
        for i in range(n):
            rx = pos[i, 0] - cx
            ry = pos[i, 1] - cy
            rz = pos[i, 2] - cz
            rr = math.sqrt(rx * rx + ry * ry + rz * rz)
            rms += rr
            if rr > max_r:
                max_r = rr
        rms /= n
        err = abs(rms - start_r)
        if err > max_err:
            max_err = err
        sum_r += rms
        done += 1
    mean_r = sum_r / done if done else 0.0
    return start_r, mean_r, max_err, min_pair, max_r, hit, done


def one(n: int, omega2: float) -> dict[str, object]:
    radius = balanced_radius(n, omega2)
    pos = polygon(n, radius)
    vel = np.zeros_like(pos)
    start_r, mean_r, max_err, min_pair, max_r, hit, done = march(
        pos, vel, omega2, STEPS, DT
    )
    accepted = (not hit) and omega2 > 0.0 and max_err <= 0.05 * start_r
    print(
        f"n={n} Ω²={omega2:g} R0={start_r:.6f} mean={mean_r:.6f} "
        f"err={max_err:.4g} close={hit} ok={accepted}"
    )
    return {
        "n": n,
        "omega2": omega2,
        "start_radius": float(start_r),
        "mean_radius": float(mean_r),
        "max_radius_error": float(max_err),
        "min_pair": float(min_pair),
        "max_radius": float(max_r),
        "close_approach": bool(hit),
        "accepted": bool(accepted),
        "steps_done": int(done),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-constant-protocol-2026-10-27.md",
        "formula_gate": gate,
        "constant": "Omega2",
        "internal": "repulsion 1/r^2",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    analytic = (1.0 / 4.0) ** (1.0 / 3.0)
    r2 = balanced_radius(2, 1.0)
    radius_gate = abs(r2 - analytic) <= 1e-9
    print("n=2 radius gate", radius_gate, r2, analytic)
    if not radius_gate:
        payload["aborted"] = "n=2 radius does not match (2/Omega2)^(1/3)"
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1)
    rows = [one(n, omega2) for n in NS for omega2 in OMEGAS]
    by_n: dict[int, dict[float, dict[str, object]]] = {}
    for row in rows:
        by_n.setdefault(int(row["n"]), {})[float(row["omega2"])] = row
    size_set = True
    necessary = True
    accepted_at_one = []
    for n in NS:
        strong = by_n[n][1.0]
        weak = by_n[n][0.0625]
        empty = by_n[n][0.0]
        if strong["accepted"] and weak["accepted"]:
            ratio = weak["mean_radius"] / strong["mean_radius"]
            if ratio <= 2.0:
                size_set = False
        else:
            size_set = False
        grew = empty["mean_radius"] > 2.0 * strong["start_radius"]
        if strong["accepted"] and not grew:
            necessary = False
        if not strong["accepted"]:
            necessary = False
        if strong["accepted"]:
            accepted_at_one.append((n, strong["start_radius"]))
    ladder = False
    if len(accepted_at_one) >= 2:
        radii = [item[1] for item in accepted_at_one]
        ladder = max(radii) > 1.05 * min(radii)
    payload.update(
        {
            "n2_analytic_radius": analytic,
            "n2_start_radius": r2,
            "rows": rows,
            "constant_sets_size": bool(size_set),
            "constant_necessary": bool(necessary),
            "particle_ladder": bool(ladder),
            "accepted_n_at_omega2_1": [n for n, _ in accepted_at_one],
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"sets_size={size_set} necessary={necessary} ladder={ladder} "
        f"accepted={payload['accepted_n_at_omega2_1']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
