"""Комплекс вселенной: заполненный шар против полости вокруг пары."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/video_formula_universe_complex_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
COUNT = 40000
R_OUT = 20.0
R_IN = 5.0
MASSES = (400.0, 100.0, 25.0, 0.0)


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


def volume_cloud(count: int, r_in: float, r_out: float) -> np.ndarray:
    pts = np.zeros((count, 3))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    inner3 = r_in ** 3
    span3 = r_out ** 3 - inner3
    for i in range(count):
        u = (i + 0.5) / count
        radius = (inner3 + span3 * u) ** (1.0 / 3.0)
        y = 1.0 - (2.0 * i + 1.0) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i, 0] = radius * math.cos(theta) * ring
        pts[i, 1] = radius * y
        pts[i, 2] = radius * math.sin(theta) * ring
    return pts


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
            rx = ext[k, 0] - pos[0, 0]
            ry = ext[k, 1] - pos[0, 1]
            rz = ext[k, 2] - pos[0, 2]
            r2 = rx * rx + ry * ry + rz * rz
            r = math.sqrt(r2)
            s = mass_each / (r2 * r)
            ax0 += s * rx
            ay0 += s * ry
            az0 += s * rz
            rx = ext[k, 0] - pos[1, 0]
            ry = ext[k, 1] - pos[1, 1]
            rz = ext[k, 2] - pos[1, 2]
            r2 = rx * rx + ry * ry + rz * rz
            r = math.sqrt(r2)
            s = mass_each / (r2 * r)
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
            rx = ext[k, 0] - pos[0, 0]
            ry = ext[k, 1] - pos[0, 1]
            rz = ext[k, 2] - pos[0, 2]
            r2 = rx * rx + ry * ry + rz * rz
            r = math.sqrt(r2)
            s = mass_each / (r2 * r)
            ax0 += s * rx
            ay0 += s * ry
            az0 += s * rz
            rx = ext[k, 0] - pos[1, 0]
            ry = ext[k, 1] - pos[1, 1]
            rz = ext[k, 2] - pos[1, 2]
            r2 = rx * rx + ry * ry + rz * rz
            r = math.sqrt(r2)
            s = mass_each / (r2 * r)
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


def one(ext: np.ndarray, total_mass: float, label: str) -> dict[str, object]:
    count = int(ext.shape[0])
    mass_each = total_mass / count if count else 0.0
    pos, vel = pair_state()
    a0 = source_acc(pos[0], ext, mass_each) if count else np.zeros(3)
    a1 = source_acc(pos[1], ext, mass_each) if count else np.zeros(3)
    cm = 0.5 * (a0 + a1)
    tidal = a1 - a0
    inward0 = float(-a0[0] * pos[0, 0] - a0[1] * pos[0, 1] - a0[2] * pos[0, 2])
    r0 = float(math.sqrt(pos[0] @ pos[0]))
    inward_proj = inward0 / r0 if r0 else 0.0
    period = 2.0 * math.pi / math.sqrt(2.0)
    dt = period / 2000.0
    steps = 2000 * 10
    mean_sep, min_sep, max_sep, max_law, hit, done = march(
        pos, vel, ext, mass_each, steps, dt, total_mass == 0.0
    )
    print(
        f"{label} S={mean_sep:.8f} tidal={math.sqrt(tidal @ tidal):.4g} "
        f"inward={inward_proj:.4g} close={hit}"
    )
    return {
        "label": label,
        "total_mass": total_mass,
        "mean_separation": float(mean_sep),
        "min_separation": float(min_sep),
        "max_separation": float(max_sep),
        "width": float(max_sep - min_sep),
        "cm_acceleration": float(math.sqrt(cm @ cm)),
        "tidal_acceleration": float(math.sqrt(tidal @ tidal)),
        "inward_projection": float(inward_proj),
        "presses_inward": bool(inward_proj > 0.01),
        "balanced": bool(math.sqrt(cm @ cm) < 0.01),
        "max_law_error": float(max_law),
        "close_approach": bool(hit),
        "steps_done": int(done),
    }


def family_verdict(rows: list[dict[str, object]]) -> dict[str, object]:
    bare = next(row for row in rows if row["total_mass"] == 0.0)
    heavy = next(row for row in rows if row["total_mass"] == 400.0)
    grew = bool(
        heavy["balanced"]
        and not heavy["close_approach"]
        and not bare["close_approach"]
        and bare["mean_separation"] > 1.01 * heavy["mean_separation"]
    )
    return {
        "size_grew_on_weakening": grew,
        "relative_change": float(
            (bare["mean_separation"] - heavy["mean_separation"]) / heavy["mean_separation"]
        ),
        "presses_inward_at_400": bool(heavy["presses_inward"]),
        "balanced_at_400": bool(heavy["balanced"]),
        "bare_law_ok": bool(bare["max_law_error"] < 1.0e-9),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/video-formula-universe-complex-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    filled_pts = volume_cloud(COUNT, 0.0, R_OUT)
    cavity_pts = volume_cloud(COUNT, R_IN, R_OUT)
    filled = [one(filled_pts, mass, f"filled M={mass}") for mass in MASSES]
    cavity = [one(cavity_pts, mass, f"cavity M={mass}") for mass in MASSES]
    filled_v = family_verdict(filled)
    cavity_v = family_verdict(cavity)
    payload.update(
        {
            "count": COUNT,
            "filled": filled,
            "cavity": cavity,
            "filled_verdict": filled_v,
            "cavity_verdict": cavity_v,
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"filled grew={filled_v['size_grew_on_weakening']} "
        f"rel={filled_v['relative_change']:.6g} inward={filled_v['presses_inward_at_400']}"
    )
    print(
        f"cavity grew={cavity_v['size_grew_on_weakening']} "
        f"rel={cavity_v['relative_change']:.6g} inward={cavity_v['presses_inward_at_400']}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
