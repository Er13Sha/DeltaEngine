"""Размер пары при ослаблении неподвижной сферической оболочки."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/video_formula_surroundings_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
SHELL_N = 40000
SHELL_R = 20.0
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


def fibonacci_shell(count: int, radius: float) -> np.ndarray:
    pts = np.zeros((count, 3))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - (2.0 * i + 1.0) / count
        ring = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i, 0] = radius * math.cos(theta) * ring
        pts[i, 1] = radius * y
        pts[i, 2] = radius * math.sin(theta) * ring
    return pts


def pair_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.zeros((2, 3))
    pos[0, 0] = -0.5
    pos[1, 0] = 0.5
    omega = math.sqrt(2.0)
    vel = np.zeros_like(pos)
    vel[:, 1] = omega * pos[:, 0]
    mass = np.ones(2)
    return pos, vel, mass


@njit
def source_acc(at, ext, mass_each):
    acc = np.zeros(3)
    n = ext.shape[0]
    for k in range(n):
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
def march(pos, vel, ext, mass_each, steps, dt):
    acc = np.zeros((2, 3))
    sum_sep = 0.0
    max_err = 0.0
    min_sep = 1.0e300
    max_cm = 0.0
    samples = 0
    max_law = 0.0
    sample_every = steps // 20
    if sample_every < 1:
        sample_every = 1
    hit = False
    done = 0
    for step in range(steps):
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        if dist < min_sep:
            min_sep = dist
        if dist < 1.0e-4:
            hit = True
            break
        inv3 = 1.0 / (dist2 * dist)
        acc[0, 0] = inv3 * dx
        acc[0, 1] = inv3 * dy
        acc[0, 2] = inv3 * dz
        acc[1, 0] = -inv3 * dx
        acc[1, 1] = -inv3 * dy
        acc[1, 2] = -inv3 * dz
        n = ext.shape[0]
        for k in range(n):
            for i in range(2):
                rx = ext[k, 0] - pos[i, 0]
                ry = ext[k, 1] - pos[i, 1]
                rz = ext[k, 2] - pos[i, 2]
                r2 = rx * rx + ry * ry + rz * rz
                r = math.sqrt(r2)
                s = mass_each / (r2 * r)
                acc[i, 0] += s * rx
                acc[i, 1] += s * ry
                acc[i, 2] += s * rz
        if step % sample_every == 0:
            amag = math.sqrt(acc[0, 0] ** 2 + acc[0, 1] ** 2 + acc[0, 2] ** 2)
            law = abs(dist2 * amag - 1.0)
            if law > max_law:
                max_law = law
            samples += 1
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(2):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        inv3 = 1.0 / (dist2 * dist)
        acc[0, 0] = inv3 * dx
        acc[0, 1] = inv3 * dy
        acc[0, 2] = inv3 * dz
        acc[1, 0] = -inv3 * dx
        acc[1, 1] = -inv3 * dy
        acc[1, 2] = -inv3 * dz
        for k in range(n):
            for i in range(2):
                rx = ext[k, 0] - pos[i, 0]
                ry = ext[k, 1] - pos[i, 1]
                rz = ext[k, 2] - pos[i, 2]
                r2 = rx * rx + ry * ry + rz * rz
                r = math.sqrt(r2)
                s = mass_each / (r2 * r)
                acc[i, 0] += s * rx
                acc[i, 1] += s * ry
                acc[i, 2] += s * rz
        for i in range(2):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        sep = dist
        sum_sep += sep
        err = abs(sep - 1.0)
        if err > max_err:
            max_err = err
        cx = 0.5 * (pos[0, 0] + pos[1, 0])
        cy = 0.5 * (pos[0, 1] + pos[1, 1])
        cz = 0.5 * (pos[0, 2] + pos[1, 2])
        cm = math.sqrt(cx * cx + cy * cy + cz * cz)
        if cm > max_cm:
            max_cm = cm
        done = step + 1
    mean_sep = sum_sep / done if done else 0.0
    return mean_sep, max_err, min_sep, max_cm, max_law, hit, done


def balance(ext: np.ndarray, mass_each: float) -> dict[str, float]:
    pos, _, _ = pair_state()
    a0 = source_acc(pos[0], ext, mass_each)
    a1 = source_acc(pos[1], ext, mass_each)
    cm = 0.5 * (a0 + a1)
    tidal = a1 - a0
    return {
        "cm_acceleration": float(math.sqrt(cm @ cm)),
        "tidal_acceleration": float(math.sqrt(tidal @ tidal)),
    }


def one(ext: np.ndarray, total_mass: float, label: str) -> dict[str, object]:
    count = int(ext.shape[0])
    mass_each = total_mass / count if count else 0.0
    pos, vel, _ = pair_state()
    period = 2.0 * math.pi / math.sqrt(2.0)
    dt = period / 2000.0
    steps = 2000 * 10
    forces = balance(ext, mass_each) if count else {
        "cm_acceleration": 0.0,
        "tidal_acceleration": 0.0,
    }
    mean_sep, max_err, min_sep, max_cm, max_law, hit, done = march(
        pos, vel, ext, mass_each, steps, dt
    )
    print(
        f"{label} S={mean_sep:.8f} err={max_err:.4g} "
        f"cm_a={forces['cm_acceleration']:.4g} tidal={forces['tidal_acceleration']:.4g}"
    )
    return {
        "label": label,
        "total_mass": total_mass,
        "count": count,
        "mass_each": mass_each,
        "mean_separation": float(mean_sep),
        "max_separation_error": float(max_err),
        "min_separation": float(min_sep),
        "max_cm_radius": float(max_cm),
        "max_law_error": float(max_law),
        "close_approach": bool(hit),
        "steps_done": int(done),
        "cm_acceleration": forces["cm_acceleration"],
        "tidal_acceleration": forces["tidal_acceleration"],
        "balanced": bool(forces["cm_acceleration"] < 0.01),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/video-formula-surroundings-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    shell = fibonacci_shell(SHELL_N, SHELL_R)
    rows = [one(shell, mass, f"shell M={mass}") for mass in MASSES]
    bare = next(row for row in rows if row["total_mass"] == 0.0)
    heavy = next(row for row in rows if row["total_mass"] == 400.0)
    grew = bool(
        heavy["balanced"]
        and bare["mean_separation"] > 1.01 * heavy["mean_separation"]
    )
    ends = np.array([[SHELL_R, 0.0, 0.0], [-SHELL_R, 0.0, 0.0]])
    control = one(ends, 400.0, "two masses")
    control_sees = bool(
        abs(control["mean_separation"] - bare["mean_separation"]) / bare["mean_separation"]
        > 0.01
    )
    payload.update(
        {
            "shell_count": SHELL_N,
            "shell_radius": SHELL_R,
            "rows": rows,
            "size_grew_on_weakening": grew,
            "relative_change": float(
                (bare["mean_separation"] - heavy["mean_separation"]) / heavy["mean_separation"]
            ),
            "bare_law_ok": bool(bare["max_law_error"] < 1.0e-9),
            "control": control,
            "control_sees_change": control_sees,
        }
    )
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"grew={grew} relative={payload['relative_change']:.6g} "
        f"control_sees={control_sees}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
