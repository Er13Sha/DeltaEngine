"""Элемент при P=1: одна ячейка n=100."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.universe_pressure_complex_press import formula_gate
from scripts.universe_pressure_orbits import SKY, march

OUT = Path("data/experiments/field/universe_pressure_element_n100_2026-10-28.json")
N = 100
STEP_CAP = 20000


def one(h_cap: float) -> dict[str, object]:
    pos = seeds(N)
    vel = np.zeros_like(pos)
    phi, rmin, rmax, closest, t, done = march(
        pos, vel, SKY, 1.0, 200.0, STEP_CAP, h_cap, 8.0, 2
    )
    orbiting = 0
    inside = 0
    finite = []
    for i in range(N):
        if rmin[i] <= 0.0 or rmin[i] > 1.0e200:
            continue
        ratio = float(rmax[i] / rmin[i])
        finite.append(ratio)
        if ratio <= 8.0:
            inside += 1
        if abs(float(phi[i])) >= 2.0 * np.pi and ratio <= 8.0 and closest > 1.0e-9:
            orbiting += 1
    span = max(finite) if finite else None
    exhausted = int(done) >= STEP_CAP and orbiting < 2 and (span is None or span <= 8.0)
    accepted = orbiting >= 2
    print(
        f"n={N} h={h_cap:g} orbiting={orbiting} inside={inside} "
        f"phi={float(np.max(np.abs(phi))):.3f} span={span} t={t:.4g} "
        f"steps={done} cap={STEP_CAP} ok={accepted} cut={exhausted}",
        flush=True,
    )
    return {
        "n": N,
        "pressure": 1.0,
        "h_cap": h_cap,
        "step_cap": STEP_CAP,
        "orbiting": orbiting,
        "inside_ratio": inside,
        "max_abs_phi": float(np.max(np.abs(phi))),
        "max_radius_ratio": span,
        "min_pair": float(closest),
        "t_done": float(t),
        "steps": int(done),
        "steps_exhausted": bool(exhausted),
        "accepted": bool(accepted),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"], flush=True)
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-element-n100-protocol-2026-10-28.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    row = one(0.02)
    if row["accepted"]:
        row["confirm"] = one(0.002)
        row["accepted"] = bool(row["confirm"]["accepted"])
    payload.update({"row": row, "element_starts": bool(row["accepted"])})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"starts={bool(row['accepted'])}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
