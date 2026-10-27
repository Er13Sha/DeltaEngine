"""Пятно в своей среде: пустой куб и куб в трёх подвижных оболочках."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.universe_pressure_complex_press import fibonacci_sky, formula_gate
from scripts.universe_pressure_orbits import SKY, march

OUT = Path("data/experiments/field/field_birth_2026-10-28.json")
CORE = 8
INNER = 12
STEP_CAP = 400000
T_END = 10.0
H_CAP = 0.02


def shell(count: int, radius: float) -> np.ndarray:
    pts = fibonacci_sky(count).astype(np.float64)
    pts -= pts.mean(axis=0)
    mean_r = float(np.linalg.norm(pts, axis=1).mean())
    return pts * (radius / mean_r)


def mean_radius(pos: np.ndarray) -> tuple[float, np.ndarray]:
    center = pos.mean(axis=0)
    return float(np.linalg.norm(pos - center, axis=1).mean()), center


def word_of(ratio: float, min_pair: float, exhausted: bool) -> str:
    if exhausted:
        return "обрыв"
    if min_pair <= 1.0e-9:
        return "одна точка"
    if ratio < 0.5:
        return "схлопывание"
    if ratio > 2.0:
        return "расходится"
    return "держит"


def scene(with_sea: bool) -> dict[str, object]:
    core = seeds(CORE)
    r0, _ = mean_radius(core)
    if with_sea:
        pos = np.vstack([core, shell(12, 2.0), shell(18, 2.5), shell(24, 3.0)])
    else:
        pos = core.copy()
    _, core_center = mean_radius(pos[:CORE])
    shell_start = None
    if with_sea:
        shell_start = float(np.linalg.norm(pos[CORE : CORE + INNER] - core_center, axis=1).mean())
    vel = np.zeros_like(pos)
    _phi, _rmin, _rmax, closest, t, steps = march(
        pos, vel, SKY, 0.0, T_END, STEP_CAP, H_CAP, 0.0, pos.shape[0] + 1
    )
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
        f"shell={shell_end} rare={rarefaction} t={t:.4g} steps={steps} pair={float(closest):.3g}",
        flush=True,
    )
    return {
        "scene": label,
        "n": int(pos.shape[0]),
        "pressure": 0.0,
        "h_cap": H_CAP,
        "R0": r0,
        "R_end": r_end,
        "ratio": ratio,
        "shell_start": shell_start,
        "shell_end": shell_end,
        "rarefaction": rarefaction,
        "min_pair": float(closest),
        "t_done": float(t),
        "steps": int(steps),
        "steps_exhausted": bool(exhausted),
        "word": text,
    }


def pair_notes(empty: dict[str, object], sea: dict[str, object]) -> dict[str, object]:
    cut = bool(empty["steps_exhausted"] or sea["steps_exhausted"])
    rare = bool(sea["rarefaction"])
    sign = (
        (not cut)
        and empty["word"] == "расходится"
        and sea["word"] == "держит"
        and rare
    )
    notes: list[str] = []
    if cut:
        notes.append("обрыв")
    else:
        if empty["word"] != "расходится":
            notes.append("среда размер не объясняет")
        if empty["word"] == "расходится" and sea["word"] == "расходится":
            notes.append("схема не начинается")
        if sea["word"] == "держит" and not rare:
            notes.append("держит без разрежения")
        if sign:
            notes.append("знак схемы")
        if not notes:
            notes.append("знака нет")
    return {"sign": bool(sign), "notes": notes}


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"], flush=True)
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/field-birth-protocol-2026-10-28.md",
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
