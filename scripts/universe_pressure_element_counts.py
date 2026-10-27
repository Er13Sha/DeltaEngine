"""Другие n: оборот и размер в пределах 8 на той же сетке P."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.universe_pressure_complex_press import formula_gate
from scripts.universe_pressure_orbits import SKY, march

OUT = Path("data/experiments/field/universe_pressure_element_counts_2026-10-27.json")
COUNTS = (2, 3, 6, 8, 16, 32)
PRESSURES = (1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0)


def one(n: int, pressure: float) -> dict[str, object]:
    pos = seeds(n)
    vel = np.zeros_like(pos)
    phi, rmin, rmax, closest, t, steps = march(pos, vel, SKY, pressure)
    orbiting = 0
    ratios = []
    for i in range(n):
        if rmin[i] <= 0.0 or rmin[i] > 1.0e200:
            ratios.append(None)
            continue
        ratio = float(rmax[i] / rmin[i])
        ratios.append(ratio)
        if abs(float(phi[i])) >= 2.0 * np.pi and ratio <= 8.0 and closest > 1.0e-9:
            orbiting += 1
    accepted = orbiting >= 2
    finite = [item for item in ratios if item is not None]
    span = max(finite) if finite else None
    print(
        f"n={n} P={pressure:g} orbiting={orbiting} "
        f"phi={float(np.max(np.abs(phi))):.3f} span={span} ok={accepted}"
    )
    return {
        "n": n,
        "pressure": pressure,
        "orbiting": orbiting,
        "max_abs_phi": float(np.max(np.abs(phi))),
        "max_radius_ratio": span,
        "min_pair": float(closest),
        "t_done": float(t),
        "steps": int(steps),
        "accepted": bool(accepted),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-element-counts-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(n, pressure) for n in COUNTS for pressure in PRESSURES]
    held = [{"n": row["n"], "pressure": row["pressure"]} for row in rows if row["accepted"]]
    payload.update({"rows": rows, "held": held, "element_starts": bool(held)})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"held={held} starts={bool(held)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
