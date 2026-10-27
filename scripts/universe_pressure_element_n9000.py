"""Элемент при P=1: число порций от 64 до 9000."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.universe_pressure_complex_press import formula_gate
from scripts.universe_pressure_orbits import SKY, march

OUT = Path("data/experiments/field/universe_pressure_element_n9000_2026-10-27.json")
CELLS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 9000)


def step_cap(n: int) -> int:
    if n <= 512:
        return 20000
    if n <= 1024:
        return 4000
    if n <= 2048:
        return 1000
    if n <= 4096:
        return 250
    if n <= 8192:
        return 50
    return 40


def one(n: int, h_cap: float, steps: int) -> dict[str, object]:
    pos = seeds(n)
    vel = np.zeros_like(pos)
    phi, rmin, rmax, closest, t, done = march(
        pos, vel, SKY, 1.0, 200.0, steps, h_cap, 8.0, 2
    )
    orbiting = 0
    finite = []
    for i in range(n):
        if rmin[i] <= 0.0 or rmin[i] > 1.0e200:
            continue
        ratio = float(rmax[i] / rmin[i])
        finite.append(ratio)
        if abs(float(phi[i])) >= 2.0 * np.pi and ratio <= 8.0 and closest > 1.0e-9:
            orbiting += 1
    span = max(finite) if finite else None
    exhausted = int(done) >= steps and orbiting < 2 and (span is None or span <= 8.0)
    accepted = orbiting >= 2
    print(
        f"n={n} h={h_cap:g} orbiting={orbiting} "
        f"phi={float(np.max(np.abs(phi))):.3f} span={span} t={t:.4g} "
        f"steps={done} cap={steps} ok={accepted} cut={exhausted}",
        flush=True,
    )
    return {
        "n": n,
        "pressure": 1.0,
        "h_cap": h_cap,
        "step_cap": steps,
        "orbiting": orbiting,
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
        "protocol": "docs/universe-pressure-element-n9000-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = []
    for n in CELLS:
        row = one(n, 0.02, step_cap(n))
        if row["accepted"] and n <= 256:
            row["confirm"] = one(n, 0.002, step_cap(n))
            row["accepted"] = bool(row["confirm"]["accepted"])
        rows.append(row)
    held = [row["n"] for row in rows if row["accepted"]]
    payload.update({"rows": rows, "held": held, "element_starts": bool(held)})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"held={held} starts={bool(held)}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
