"""Тетраэдр n=4: какое P даёт оборот без растяжения размера больше чем в 8 раз."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.universe_pressure_complex_press import formula_gate
from scripts.universe_pressure_orbits import SKY, initial, march

OUT = Path("data/experiments/field/universe_pressure_element_scale_2026-10-27.json")
PRESSURES = (1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0)


def one(pressure: float) -> dict[str, object]:
    pos = initial(4)
    vel = np.zeros_like(pos)
    phi, rmin, rmax, closest, t, steps = march(pos, vel, SKY, pressure)
    orbiting = 0
    ratios = []
    for i in range(4):
        ratio = float(rmax[i] / rmin[i]) if rmin[i] > 0.0 else None
        ratios.append(ratio)
        bounded = ratio is not None and rmax[i] <= 8.0 * rmin[i]
        if abs(float(phi[i])) >= 2.0 * math_pi() and bounded and closest > 1.0e-9:
            orbiting += 1
    accepted = orbiting >= 2
    print(
        f"P={pressure:g} orbiting={orbiting} phi={float(np.max(np.abs(phi))):.3f} "
        f"minr={closest:.4g} ratios={ratios} ok={accepted}"
    )
    return {
        "n": 4,
        "pressure": pressure,
        "orbiting": orbiting,
        "max_abs_phi": float(np.max(np.abs(phi))),
        "radius_ratio": ratios,
        "min_pair": float(closest),
        "t_done": float(t),
        "steps": int(steps),
        "accepted": bool(accepted),
    }


def math_pi() -> float:
    return 2.0 * np.pi


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-element-scale-protocol-2026-10-27.md",
        "law": "models/law.py",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(pressure) for pressure in PRESSURES]
    held = [row["pressure"] for row in rows if row["accepted"]]
    payload.update({"rows": rows, "held_pressures": held, "element_starts": bool(held)})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"held={held} starts={payload['element_starts']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
