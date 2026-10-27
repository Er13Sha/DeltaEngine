"""Старт с радиуса, где открытое небо гасит отталкивание. Порог 8 прежний."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.universe_pressure_complex_press import accelerations, formula_gate
from scripts.universe_pressure_orbits import SKY, add_missed, march

OUT = Path("data/experiments/field/universe_pressure_element_radius_2026-10-27.json")
NS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9000)
LO = 0.25
HI = 400.0


def step_cap(n: int) -> int:
    if n <= 256:
        return 10000
    if n <= 512:
        return 4000
    if n <= 1024:
        return 1500
    return 200


def mean_outward(n: int, radius: float) -> float:
    pos = seeds(n) * radius
    acc, _, _, _, _ = accelerations(pos, SKY, 1.0, True)
    acc = add_missed(pos, SKY, 1.0, acc)
    rel = pos - pos.mean(axis=0)
    rad = np.linalg.norm(rel, axis=1)
    return float(np.mean(np.sum(acc * rel, axis=1) / np.maximum(rad, 1.0e-30)))


def balanced(n: int) -> tuple[float, float, bool]:
    lo, hi = LO, HI
    left = mean_outward(n, lo)
    right = mean_outward(n, hi)
    if left == 0.0:
        return lo, left, True
    if left * right > 0.0:
        edge = lo if abs(left) <= abs(right) else hi
        return edge, left if edge == lo else right, False
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        value = mean_outward(n, mid)
        if left > 0.0:
            if value > 0.0:
                lo, left = mid, value
            else:
                hi, right = mid, value
        elif value < 0.0:
            lo, left = mid, value
        else:
            hi, right = mid, value
    radius = 0.5 * (lo + hi)
    return radius, mean_outward(n, radius), True


def orbit(n: int, radius: float, h_cap: float, steps: int) -> dict[str, object]:
    pos = seeds(n) * radius
    vel = np.zeros_like(pos)
    phi, rmin, rmax, closest, t, done = march(pos, vel, SKY, 1.0, 200.0, steps, h_cap, 8.0, 2)
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
        f"  orbit h={h_cap:g} orbiting={orbiting} phi={float(np.max(np.abs(phi))):.3f} "
        f"span={span} t={t:.4g} steps={done} ok={accepted} cut={exhausted}",
        flush=True,
    )
    return {
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
        "protocol": "docs/universe-pressure-element-radius-protocol-2026-10-27.md",
        "formula_gate": gate,
        "rows": [],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows: list[dict[str, object]] = []
    for n in NS:
        radius, outward, rooted = balanced(n)
        row: dict[str, object] = {
            "n": n,
            "pressure": 1.0,
            "radius": radius,
            "mean_outward": outward,
            "rooted": rooted,
            "accepted": False,
        }
        print(f"n={n} R={radius:.6g} outward={outward:.4g} rooted={rooted}", flush=True)
        if rooted:
            run = orbit(n, radius, 0.02, step_cap(n))
            row["run"] = run
            row["accepted"] = bool(run["accepted"])
            if row["accepted"] and n <= 256:
                confirm = orbit(n, radius, 0.002, step_cap(n))
                row["confirm"] = confirm
                row["accepted"] = bool(confirm["accepted"])
        rows.append(row)
        payload["rows"] = rows
        held = [item["n"] for item in rows if item["accepted"]]
        payload["held"] = held
        payload["element_starts"] = bool(held)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"held={payload['held']} starts={payload['element_starts']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
