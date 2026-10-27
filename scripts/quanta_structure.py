"""Связные фигуры квантов при P=1, n от 2 до 24, срок t=10."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.law import seeds
from scripts.universe_pressure_complex_press import formula_gate
from scripts.universe_pressure_orbits import SKY, march

OUT = Path("data/experiments/field/quanta_structure_2026-10-28.json")
STEP_CAP = 400000
T_END = 10.0
H_CAP = 0.02


def figure(n: int) -> str:
    return {
        2: "пара",
        3: "треугольник",
        4: "тетраэдр",
        6: "оси",
        8: "куб",
    }.get(n, "сфера")


def link_length(pos: np.ndarray) -> float:
    n = pos.shape[0]
    farthest = 0.0
    for i in range(n):
        best = 1.0e300
        for j in range(n):
            if i == j:
                continue
            delta = pos[i] - pos[j]
            dist = float(np.sqrt(delta @ delta))
            if dist < best:
                best = dist
        if best > farthest:
            farthest = best
    return farthest


def component_sizes(pos: np.ndarray, limit: float) -> list[int]:
    n = pos.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[i] - pos[j]
            if float(np.sqrt(delta @ delta)) <= limit:
                union(i, j)
    sizes: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def signature(sizes: list[int]) -> str:
    return "+".join(str(size) for size in sizes)


def scale_to_unit(pos: np.ndarray) -> tuple[np.ndarray, float]:
    center = pos.mean(axis=0)
    rel = pos - center
    radii = np.linalg.norm(rel, axis=1)
    mean_r = float(radii.mean())
    if mean_r <= 0.0:
        return rel, mean_r
    return rel / mean_r, mean_r


def one(n: int) -> dict[str, object]:
    pos = seeds(n)
    d_link = link_length(pos)
    limit = d_link * (1.0 + 1.0e-9)
    start_sizes = component_sizes(pos, limit)
    vel = np.zeros_like(pos)
    _phi, _rmin, _rmax, closest, t, steps = march(
        pos, vel, SKY, 1.0, T_END, STEP_CAP, H_CAP, 0.0, n + 1
    )
    exhausted = int(steps) >= STEP_CAP and float(t) < T_END - 1.0e-9
    min_pair = float(closest)
    if min_pair <= 1.0e-9:
        end_sizes = [n]
        mean_r = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
        word = "обрыв" if exhausted else "одна точка"
    else:
        scaled, mean_r = scale_to_unit(pos)
        end_sizes = component_sizes(scaled, limit)
        if exhausted:
            word = "обрыв"
        elif end_sizes == start_sizes:
            word = "та же"
        elif end_sizes == [1] * n:
            word = "россыпь"
        else:
            word = "другая"
    print(
        f"n={n} {figure(n)} {signature(start_sizes)} -> {signature(end_sizes)} "
        f"{word} R={mean_r:.4g} t={t:.4g} steps={steps} pair={min_pair:.3g}",
        flush=True,
    )
    return {
        "n": n,
        "figure": figure(n),
        "pressure": 1.0,
        "h_cap": H_CAP,
        "link_length": d_link,
        "start_signature": signature(start_sizes),
        "end_signature": signature(end_sizes),
        "word": word,
        "mean_radius": mean_r,
        "min_pair": min_pair,
        "t_done": float(t),
        "steps": int(steps),
        "steps_exhausted": bool(exhausted),
    }


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"], flush=True)
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/quanta-structure-protocol-2026-10-28.md",
        "name": "квант",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(n) for n in range(2, 25)]
    payload.update({"rows": rows})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
