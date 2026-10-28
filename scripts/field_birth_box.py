"""Те же две сцены рождения внутри шара радиуса 4."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scripts.field_birth import (
    CORE,
    H_CAP,
    INNER,
    STEP_CAP,
    T_END,
    mean_radius,
    pair_notes,
    shell,
    word_of,
)
from models.law import seeds
from scripts.universe_pressure_complex_press import accelerations, formula_gate
from scripts.universe_pressure_orbits import SKY, add_missed, min_pair

OUT = Path("data/experiments/field/field_birth_box_2026-10-28.json")
WALL = 4.0


def reflect(pos: np.ndarray, vel: np.ndarray) -> int:
    hits = 0
    n = pos.shape[0]
    for i in range(n):
        radius = math.sqrt(float(pos[i] @ pos[i]))
        if radius <= WALL or radius == 0.0:
            continue
        hits += 1
        direction = pos[i] / radius
        overshoot = radius - WALL
        inward = WALL - overshoot
        if inward <= 0.0:
            pos[i] = direction * WALL
            outward = float(vel[i] @ direction)
            if outward > 0.0:
                vel[i] -= outward * direction
            continue
        pos[i] = direction * inward
        outward = float(vel[i] @ direction)
        if outward > 0.0:
            vel[i] -= 2.0 * outward * direction
    return hits


def march_box(
    pos: np.ndarray,
    vel: np.ndarray,
    samples: list[list[list[float]]] | None = None,
    sample_dt: float = 0.1,
) -> tuple[float, float, int, int]:
    closest = 1.0e300
    t = 0.0
    steps = 0
    hits = 0
    n = pos.shape[0]
    next_sample = sample_dt
    if samples is not None:
        samples.append(np.round(pos, 4).tolist())
    while t < T_END and steps < STEP_CAP:
        acc, _, _, _, _ = accelerations(pos, SKY, 0.0, True)
        acc = add_missed(pos, SKY, 0.0, acc)
        dist = float(min_pair(pos))
        if dist < closest:
            closest = dist
        rel = 0.0
        amax = 0.0
        room = 1.0e300
        for i in range(n):
            speed = math.sqrt(float(vel[i] @ vel[i]))
            if speed > rel:
                rel = speed
            am = math.sqrt(float(acc[i] @ acc[i]))
            if am > amax:
                amax = am
            radius = math.sqrt(float(pos[i] @ pos[i]))
            if radius > 0.0:
                outward = float(vel[i] @ pos[i]) / radius
                gap = WALL - radius
                if outward > 0.0 and gap > 0.0:
                    room = min(room, 0.2 * gap / outward)
        h = H_CAP
        if amax > 0.0 and dist > 0.0:
            h = min(h, 0.2 * math.sqrt(dist / amax))
        if rel > 0.0 and dist > 0.0:
            h = min(h, 0.02 * dist / rel)
        if room < 1.0e300:
            h = min(h, room)
        h = max(h, 1.0e-14)
        if t + h > T_END:
            h = T_END - t
        half = 0.5 * h
        pos += vel * h + acc * (half * h)
        vel += acc * half
        acc, _, _, _, _ = accelerations(pos, SKY, 0.0, True)
        acc = add_missed(pos, SKY, 0.0, acc)
        vel += acc * half
        hits += reflect(pos, vel)
        t += h
        steps += 1
        if samples is not None:
            while next_sample <= t + 1.0e-9 and next_sample <= T_END + 1.0e-9:
                samples.append(np.round(pos, 4).tolist())
                next_sample += sample_dt
    return closest, t, steps, hits


def scene(with_sea: bool) -> dict[str, object]:
    core = seeds(CORE)
    r0, _ = mean_radius(core)
    pos = np.vstack([core, shell(12, 2.0), shell(18, 2.5), shell(24, 3.0)]) if with_sea else core.copy()
    _, core_center = mean_radius(pos[:CORE])
    shell_start = None
    if with_sea:
        shell_start = float(np.linalg.norm(pos[CORE : CORE + INNER] - core_center, axis=1).mean())
    vel = np.zeros_like(pos)
    closest, t, steps, hits = march_box(pos, vel)
    exhausted = int(steps) >= STEP_CAP and float(t) < T_END - 1.0e-9
    r_end, core_center = mean_radius(pos[:CORE])
    ratio = r_end / r0
    shell_end = None
    rarefaction = None
    if with_sea:
        shell_end = float(np.linalg.norm(pos[CORE : CORE + INNER] - core_center, axis=1).mean())
        rarefaction = bool(shell_end > shell_start * 1.10)
    label = "среда" if with_sea else "пустая"
    text = word_of(ratio, float(closest), exhausted)
    print(
        f"{label} word={text} R={r_end:.4g} ratio={ratio:.4g} "
        f"shell={shell_end} rare={rarefaction} hits={hits} t={t:.4g} steps={steps}",
        flush=True,
    )
    return {
        "scene": label,
        "n": int(pos.shape[0]),
        "pressure": 0.0,
        "wall": WALL,
        "h_cap": H_CAP,
        "R0": r0,
        "R_end": r_end,
        "ratio": ratio,
        "shell_start": shell_start,
        "shell_end": shell_end,
        "rarefaction": rarefaction,
        "wall_hits": int(hits),
        "min_pair": float(closest),
        "t_done": float(t),
        "steps": int(steps),
        "steps_exhausted": bool(exhausted),
        "word": text,
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"], flush=True)
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/field-birth-box-protocol-2026-10-28.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    empty = scene(False)
    sea = scene(True)
    payload.update({"empty": empty, "sea": sea, "pair": pair_notes(empty, sea)})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pair={payload['pair']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
