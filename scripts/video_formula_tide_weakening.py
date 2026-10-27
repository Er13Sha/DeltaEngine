"""Ослабление двух осевых масс: среднее расстояние пары и размах."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/video_formula_tide_weakening_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
SIDE_MASSES = (200.0, 150.0, 100.0, 75.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.0)
ENDS = np.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]])


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


def pair_state() -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((2, 3))
    pos[0, 0] = -0.5
    pos[1, 0] = 0.5
    omega = math.sqrt(2.0)
    vel = np.zeros_like(pos)
    vel[:, 1] = omega * pos[:, 0]
    return pos, vel


@njit
def source_acc(at, ext, mass_each):
    acc = np.zeros(3)
    for k in range(ext.shape[0]):
        dx = ext[k, 0] - at[0]
        dy = ext[k, 1] - at[1]
        dz = ext[k, 2] - at[2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        inv3 = mass_each / (dist2 * dist)
        acc[0] += inv3 * dx
        acc[1] += inv3 * dy
        acc[2] += inv3 * dz
    return acc


@njit
def march(pos, vel, ext, mass_each, steps, dt, sample_law):
    sum_sep = 0.0
    min_sep = 1.0e300
    max_sep = 0.0
    max_law = 0.0
    sample_every = steps // 20
    if sample_every < 1:
        sample_every = 1
    hit = False
    done = 0
    n = ext.shape[0]
    for step in range(steps):
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        if dist < min_sep:
            min_sep = dist
        if dist > max_sep:
            max_sep = dist
        if dist < 1.0e-4:
            hit = True
            break
        inv3 = 1.0 / (dist2 * dist)
        ax0 = inv3 * dx
        ay0 = inv3 * dy
        az0 = inv3 * dz
        ax1 = -ax0
        ay1 = -ay0
        az1 = -az0
        if sample_law and step % sample_every == 0:
            amag = math.sqrt(ax0 * ax0 + ay0 * ay0 + az0 * az0)
            law = abs(dist2 * amag - 1.0)
            if law > max_law:
                max_law = law
        for k in range(n):
            for i in range(2):
                rx = ext[k, 0] - pos[i, 0]
                ry = ext[k, 1] - pos[i, 1]
                rz = ext[k, 2] - pos[i, 2]
                r2 = rx * rx + ry * ry + rz * rz
                r = math.sqrt(r2)
                s = mass_each / (r2 * r)
                if i == 0:
                    ax0 += s * rx
                    ay0 += s * ry
                    az0 += s * rz
                else:
                    ax1 += s * rx
                    ay1 += s * ry
                    az1 += s * rz
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        pos[0, 0] += vel[0, 0] * dt + ax0 * half_dt2
        pos[0, 1] += vel[0, 1] * dt + ay0 * half_dt2
        pos[0, 2] += vel[0, 2] * dt + az0 * half_dt2
        pos[1, 0] += vel[1, 0] * dt + ax1 * half_dt2
        pos[1, 1] += vel[1, 1] * dt + ay1 * half_dt2
        pos[1, 2] += vel[1, 2] * dt + az1 * half_dt2
        vel[0, 0] += ax0 * half
        vel[0, 1] += ay0 * half
        vel[0, 2] += az0 * half
        vel[1, 0] += ax1 * half
        vel[1, 1] += ay1 * half
        vel[1, 2] += az1 * half
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        inv3 = 1.0 / (dist2 * dist)
        ax0 = inv3 * dx
        ay0 = inv3 * dy
        az0 = inv3 * dz
        ax1 = -ax0
        ay1 = -ay0
        az1 = -az0
        for k in range(n):
            for i in range(2):
                rx = ext[k, 0] - pos[i, 0]
                ry = ext[k, 1] - pos[i, 1]
                rz = ext[k, 2] - pos[i, 2]
                r2 = rx * rx + ry * ry + rz * rz
                r = math.sqrt(r2)
                s = mass_each / (r2 * r)
                if i == 0:
                    ax0 += s * rx
                    ay0 += s * ry
                    az0 += s * rz
                else:
                    ax1 += s * rx
                    ay1 += s * ry
                    az1 += s * rz
        vel[0, 0] += ax0 * half
        vel[0, 1] += ay0 * half
        vel[0, 2] += az0 * half
        vel[1, 0] += ax1 * half
        vel[1, 1] += ay1 * half
        vel[1, 2] += az1 * half
        if dist < min_sep:
            min_sep = dist
        if dist > max_sep:
            max_sep = dist
        sum_sep += dist
        done = step + 1
    mean_sep = sum_sep / done if done else 0.0
    return mean_sep, min_sep, max_sep, max_law, hit, done


def one(side_mass: float) -> dict[str, object]:
    pos, vel = pair_state()
    period = 2.0 * math.pi / math.sqrt(2.0)
    dt = period / 2000.0
    steps = 2000 * 50
    ext = ENDS.copy()
    a0 = source_acc(pos[0], ext, side_mass)
    a1 = source_acc(pos[1], ext, side_mass)
    cm = 0.5 * (a0 + a1)
    tidal = a1 - a0
    cm_acc = float(math.sqrt(cm @ cm))
    tidal_acc = float(math.sqrt(tidal @ tidal))
    mean_sep, min_sep, max_sep, max_law, hit, done = march(
        pos, vel, ext, side_mass, steps, dt, side_mass == 0.0
    )
    width = float(max_sep - min_sep)
    print(
        f"M={side_mass} S={mean_sep:.6f} min={min_sep:.4g} max={max_sep:.4g} "
        f"W={width:.4g} tidal={tidal_acc:.4g} close={hit}"
    )
    return {
        "side_mass": side_mass,
        "mean_separation": float(mean_sep),
        "min_separation": float(min_sep),
        "max_separation": float(max_sep),
        "width": width,
        "cm_acceleration": cm_acc,
        "tidal_acceleration": tidal_acc,
        "balanced": bool(cm_acc < 0.01),
        "max_law_error": float(max_law),
        "close_approach": bool(hit),
        "steps_done": int(done),
        "steps": steps,
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/video-formula-tide-weakening-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(mass) for mass in SIDE_MASSES]
    by_mass = {row["side_mass"]: row for row in rows}
    bare = by_mass[0.0]
    heavy = by_mass[200.0]
    light = by_mass[1.0]
    law_ok = bool(bare["max_law_error"] < 1.0e-9)
    balanced = all(row["balanced"] for row in rows)
    no_hit = all(not row["close_approach"] for row in rows)
    mean_shift = (heavy["mean_separation"] - bare["mean_separation"]) / bare["mean_separation"]
    pull_moves = bool(mean_shift > 0.01)
    span = abs(heavy["mean_separation"] - bare["mean_separation"])
    returns = bool(
        law_ok
        and balanced
        and no_hit
        and all(abs(row["mean_separation"] - bare["mean_separation"]) <= span + 1.0e-4 for row in rows)
        and abs(light["mean_separation"] - bare["mean_separation"]) / bare["mean_separation"] < 0.01
    )
    settled = bool(heavy["width"] / heavy["mean_separation"] <= 0.05) if heavy["mean_separation"] else False
    ordered = sorted(rows, key=lambda row: row["side_mass"], reverse=True)
    monotonic = all(
        ordered[i]["mean_separation"] + 1.0e-4 >= ordered[i + 1]["mean_separation"]
        for i in range(len(ordered) - 1)
    )
    payload.update(
        {
            "rows": rows,
            "bare_law_ok": law_ok,
            "all_balanced": bool(balanced),
            "pull_moves_mean": pull_moves,
            "mean_relative_shift": float(mean_shift),
            "weakening_returns_mean": returns,
            "settled_size_at_200": settled,
            "mean_monotonic": bool(monotonic),
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"law={law_ok} balanced={balanced} moves={pull_moves} "
        f"returns={returns} settled={settled} monotonic={monotonic}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
