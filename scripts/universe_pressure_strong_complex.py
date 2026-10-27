"""P = 1000000: сближение, рост скорости, пролёт при смещении."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scripts.universe_pressure_complex_press import accelerations, fibonacci_sky, formula_gate

OUT = Path("data/experiments/field/universe_pressure_strong_complex_2026-10-27.json")
P = 1_000_000.0
SKY = fibonacci_sky(72)


def march(pos, vel, t_end=5.0):
    pos = pos.copy()
    vel = vel.copy()
    start_sep = float(np.linalg.norm(pos[0] - pos[1]))
    start_speed = float(np.linalg.norm(vel[0] - vel[1]))
    min_r = start_sep
    speed_at_min = start_speed
    order0 = pos[0, 0] < pos[1, 0]
    swapped = False
    t = 0.0
    steps = 0
    while t < t_end and steps < 2_000_000:
        acc, _, _, _, _ = accelerations(pos, SKY, P, True)
        dist = float(np.linalg.norm(pos[0] - pos[1]))
        rel = vel[0] - vel[1]
        rel_v = float(np.linalg.norm(rel))
        amax = max(float(np.linalg.norm(acc[0])), float(np.linalg.norm(acc[1])))
        h = 0.002
        if amax > 0.0 and dist > 0.0:
            h = min(h, 0.2 * math.sqrt(dist / amax))
        if rel_v > 0.0 and dist > 0.0:
            h = min(h, 0.02 * dist / rel_v)
        h = max(h, 1e-12)
        if t + h > t_end:
            h = t_end - t
        half = 0.5 * h
        pos += vel * h + acc * (half * h)
        vel += acc * half
        acc, _, _, _, _ = accelerations(pos, SKY, P, True)
        vel += acc * half
        dist = float(np.linalg.norm(pos[0] - pos[1]))
        rel_v = float(np.linalg.norm(vel[0] - vel[1]))
        if dist < min_r:
            min_r = dist
            speed_at_min = rel_v
        if (pos[0, 0] < pos[1, 0]) != order0:
            swapped = True
        t += h
        steps += 1
        if t > 0.001 and dist > 12.0 and dist > start_sep:
            break
    return {
        "min_separation": min_r,
        "start_separation": start_sep,
        "start_speed": start_speed,
        "speed_at_closest": speed_at_min,
        "sides_swapped": swapped,
        "t_done": t,
        "steps": steps,
        "x0": float(pos[0, 0]),
        "x1": float(pos[1, 0]),
    }


def one(name: str, pos, vel) -> dict[str, object]:
    row = march(np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64))
    approached = row["min_separation"] < 0.5 * row["start_separation"]
    faster = row["speed_at_closest"] > row["start_speed"]
    print(
        f"{name}: minr={row['min_separation']:.4g} "
        f"v {row['start_speed']:.4g}->{row['speed_at_closest']:.4g} "
        f"pass={row['sides_swapped']} steps={row['steps']}"
    )
    row.update({"name": name, "approached": approached, "gained_speed": faster})
    return row


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/universe-pressure-strong-complex-protocol-2026-10-27.md",
        "pressure": P,
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    line = one("line-rest", [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], [[0, 0, 0], [0, 0, 0]])
    offset = one(
        "offset-rest",
        [[-3.0, 0.15, 0.0], [3.0, -0.15, 0.0]],
        [[0, 0, 0], [0, 0, 0]],
    )
    moving = one(
        "offset-inertia",
        [[-3.0, 0.15, 0.0], [3.0, -0.15, 0.0]],
        [[0.4, 0, 0], [-0.4, 0, 0]],
    )
    accepted = (
        line["approached"]
        and line["gained_speed"]
        and offset["approached"]
        and offset["gained_speed"]
        and offset["sides_swapped"]
        and moving["approached"]
        and moving["gained_speed"]
        and moving["sides_swapped"]
    )
    payload.update({"rows": [line, offset, moving], "grid_accepted": bool(accepted)})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"grid={accepted}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
