"""Прямая гравитация двух масс. Протокол записан отдельно. Камеры и отдельного поля нет."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from delta_science.action import textbook_check

OUT = Path("data/experiments/field/direct_gravity_2026-10-27.json")
G = 1.0
MASS = 1.0
SEPARATION = 1.0
PERIODS = 10


def initial_state() -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * SEPARATION
    speed = math.sqrt(G * MASS / (2.0 * SEPARATION))
    position = np.array([[-half, 0.0], [half, 0.0]], dtype=np.float64)
    velocity = np.array([[0.0, -speed], [0.0, speed]], dtype=np.float64)
    return position, velocity


def acceleration(position: np.ndarray, gravity: float) -> np.ndarray:
    delta = position[1] - position[0]
    dist = float(np.linalg.norm(delta))
    scale = gravity * MASS / dist**3
    acc = np.zeros_like(position)
    acc[0] = scale * delta
    acc[1] = -scale * delta
    return acc


def integrate(gravity: float, steps: int, dt: float) -> dict[str, float]:
    position, velocity = initial_state()
    acc = acceleration(position, gravity)
    max_rel = 0.0
    for _ in range(steps):
        position = position + velocity * dt + acc * (0.5 * dt * dt)
        velocity = velocity + acc * (0.5 * dt)
        acc = acceleration(position, gravity)
        velocity = velocity + acc * (0.5 * dt)
        dist = float(np.linalg.norm(position[1] - position[0]))
        max_rel = max(max_rel, abs(dist - SEPARATION) / SEPARATION)
    dist_end = float(np.linalg.norm(position[1] - position[0]))
    return {"distance_end": dist_end, "max_relative_error": max_rel}


def center_of_square() -> dict[str, float]:
    corners = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    origin = np.zeros(2)
    total = np.zeros(2)
    magnitudes = []
    for corner in corners:
        delta = corner - origin
        dist = float(np.linalg.norm(delta))
        magnitudes.append(G * MASS / dist**2)
        total += G * MASS * delta / dist**3
    return {
        "each_magnitude": float(magnitudes[0]),
        "expected_magnitude": G * MASS / 2.0,
        "net_acceleration": float(np.linalg.norm(total)),
    }


def main() -> None:
    omega = math.sqrt(2.0 * G * MASS / SEPARATION**3)
    period = 2.0 * math.pi / omega
    steps = 4000 * PERIODS
    dt = period / 4000.0
    bound = integrate(G, steps, dt)
    free_steps = 4000
    free = integrate(0.0, free_steps, dt)
    free_expected = math.sqrt(1.0 + 2.0 * period * period)
    square = center_of_square()
    check = textbook_check()
    payload = {
        "scope": "model",
        "protocol": "docs/direct-gravity-protocol-2026-10-27.md",
        "chamber": False,
        "auxiliary_field": False,
        "G": G,
        "mass": MASS,
        "period": period,
        "dt": dt,
        "orbit": {
            **bound,
            "accepted": bound["max_relative_error"] < 1e-4,
        },
        "gravity_off": {
            **free,
            "expected_distance": free_expected,
            "relative_error": abs(free["distance_end"] - free_expected) / free_expected,
            "farther_than_2": free["distance_end"] > 2.0,
        },
        "square_center": {
            **square,
            "links_nonzero": square["each_magnitude"] > 0.0,
            "net_near_zero": square["net_acceleration"] < 1e-12,
        },
        "textbook_gravity_equals_newton": bool(check["gravity_equals_newton_exactly"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"orbit max_rel={bound['max_relative_error']:.3g} end={bound['distance_end']:.8g} "
        f"ok={payload['orbit']['accepted']}"
    )
    print(
        f"G=0 end={free['distance_end']:.8g} expected={free_expected:.8g} "
        f"err={payload['gravity_off']['relative_error']:.3g}"
    )
    print(
        f"square each={square['each_magnitude']:.8g} net={square['net_acceleration']:.3g}"
    )
    print("newton", payload["textbook_gravity_equals_newton"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
