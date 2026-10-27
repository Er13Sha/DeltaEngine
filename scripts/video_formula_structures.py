"""Формула видео без добавок, затем первые структуры. Протокол записан отдельно."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/video_formula_structures_2026-10-27.json")
C1 = 4 * PI * Fraction(1)
CLOSE = 1e-8


def formula_gate() -> dict[str, object]:
    linked = coupling(C1)
    rows = []
    exact = linked == PiNumber(Fraction(1)) and linked.power == 0
    for radius in (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(10)):
        got = force(C1, radius, 1, 1)
        expected = PiNumber(Fraction(1) / radius**2)
        match = got == expected and got.power == 0
        exact = exact and match
        rows.append({"r": str(radius), "force": str(got), "expected": str(expected), "match": bool(match)})
    return {"passed": bool(exact), "coupling": str(linked), "rows": rows}


def pair_acceleration(position: np.ndarray, mass: np.ndarray) -> np.ndarray:
    n = len(mass)
    acc = np.zeros_like(position)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = position[j] - position[i]
            dist = float(np.linalg.norm(delta))
            if dist < CLOSE:
                raise RuntimeError("close approach")
            magnitude = float(force(C1, dist, mass[j], mass[i])) / float(mass[i])
            acc[i] += magnitude * delta / dist
    return acc


def verlet(position: np.ndarray, velocity: np.ndarray, mass: np.ndarray, steps: int, dt: float, sample_every: int) -> dict[str, object]:
    acc = pair_acceleration(position, mass)
    samples = []
    min_dist = math.inf
    for step in range(steps):
        position = position + velocity * dt + acc * (0.5 * dt * dt)
        velocity = velocity + acc * (0.5 * dt)
        acc = pair_acceleration(position, mass)
        velocity = velocity + acc * (0.5 * dt)
        dist = pairwise_distances(position)
        min_dist = min(min_dist, float(dist.min()))
        if step % sample_every == 0:
            samples.append((position.copy(), acc.copy(), dist.copy()))
    samples.append((position.copy(), acc.copy(), pairwise_distances(position).copy()))
    return {"samples": samples, "min_distance": min_dist}


def pairwise_distances(position: np.ndarray) -> np.ndarray:
    n = len(position)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.linalg.norm(position[j] - position[i])))
    return np.array(out)


def binary() -> dict[str, object]:
    omega = math.sqrt(2.0)
    period = 2.0 * math.pi / omega
    steps_per = 20000
    steps = steps_per * 10
    dt = period / steps_per
    half = 0.5
    speed = math.sqrt(0.5)
    position = np.array([[-half, 0.0], [half, 0.0]])
    velocity = np.array([[0.0, -speed], [0.0, speed]])
    mass = np.array([1.0, 1.0])
    run = verlet(position, velocity, mass, steps, dt, steps_per)
    rel = []
    law = []
    for pos, acc, dist in run["samples"]:
        rel.append(abs(dist[0] - 1.0))
        scale = abs(float(np.linalg.norm(acc[0])) * dist[0] ** 2 - 1.0)
        law.append(scale)
    max_rel = max(rel)
    max_law = max(law)
    return {
        "max_relative_separation_error": max_rel,
        "max_step_law_error": max_law,
        "min_distance": run["min_distance"],
        "structure_accepted": bool(max_rel < 1e-6),
        "step_law_accepted": bool(max_law < 1e-12),
    }


def triangle(shift: float) -> dict[str, float]:
    omega = math.sqrt(3.0)
    period = 2.0 * math.pi / omega
    steps_per = 8000
    steps = steps_per * 5
    dt = period / steps_per
    radius = 1.0 / math.sqrt(3.0)
    angles = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    position = np.array([[radius * math.cos(a), radius * math.sin(a)] for a in angles])
    velocity = omega * np.column_stack((-position[:, 1], position[:, 0]))
    position[0, 0] += shift
    mass = np.ones(3)
    run = verlet(position, velocity, mass, steps, dt, steps_per)
    side_error = [float(np.max(np.abs(dist - 1.0))) for _, _, dist in run["samples"]]
    return {"max_side_error": max(side_error), "min_distance": run["min_distance"]}


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"], "coupling", gate["coupling"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/video-formula-structures-protocol-2026-10-27.md",
        "formula_gate": gate,
        "chamber": False,
        "k_transfer": 1,
    }
    if not gate["passed"]:
        payload["structures"] = "not_run"
    else:
        pair = binary()
        plain = triangle(0.0)
        bumped = triangle(1e-3)
        growth = bumped["max_side_error"] / max(plain["max_side_error"], 1e-30)
        payload["binary"] = pair
        payload["triangle"] = {
            "unperturbed_max_side_error": plain["max_side_error"],
            "relative_equilibrium": bool(plain["max_side_error"] < 1e-3),
            "perturbed_max_side_error": bumped["max_side_error"],
            "perturbation_growth": growth,
            "stable": bool(plain["max_side_error"] < 1e-3 and growth <= 10.0),
        }
        print(
            f"binary sep={pair['max_relative_separation_error']:.3g} "
            f"law={pair['max_step_law_error']:.3g} ok={pair['structure_accepted']}"
        )
        tri = payload["triangle"]
        print(
            f"triangle side={tri['unperturbed_max_side_error']:.3g} "
            f"growth={tri['perturbation_growth']:.3g} stable={tri['stable']}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
