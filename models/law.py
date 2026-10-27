"""Один шаг сцены: тот же закон, что в прогоне орбит."""

from __future__ import annotations

import math
import threading

import numpy as np

from scripts.universe_pressure_complex_press import accelerations
from scripts.universe_pressure_orbits import add_missed, fibonacci_sky

SKY = fibonacci_sky(72)


def seeds(n: int) -> np.ndarray:
    if n == 2:
        pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    elif n == 3:
        pts = np.array(
            [[math.cos(2.0 * math.pi * k / 3.0), math.sin(2.0 * math.pi * k / 3.0), 0.0] for k in range(3)]
        )
    elif n == 4:
        pts = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        )
    elif n == 6:
        pts = np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
    elif n == 8:
        pts = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    else:
        pts = fibonacci_sky(n)
    pts = pts.astype(np.float64)
    pts -= pts.mean(axis=0)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _forces(pos: np.ndarray, pressure: float) -> tuple[np.ndarray, float, float, float]:
    acc, _, _, cover, press = accelerations(pos, SKY, pressure, True)
    acc = add_missed(pos, SKY, pressure, acc)
    return acc, float(np.mean(cover)), float(np.mean(press)), _min_pair(pos)


def _min_pair(pos: np.ndarray) -> float:
    n = pos.shape[0]
    best = 1.0e300
    for i in range(n):
        delta = pos[i + 1 :] - pos[i]
        if delta.size == 0:
            continue
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        best = min(best, float(dist.min()))
    return best


def _mean_radius(pos: np.ndarray) -> float:
    center = pos.mean(axis=0)
    return float(np.linalg.norm(pos - center, axis=1).mean())


class Stand:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.n = 4
        self.pressure = 1_000_000.0
        self.paused = False
        self.pos = seeds(self.n)
        self.vel = np.zeros_like(self.pos)
        self.t = 0.0
        self.radius = 1.0
        self.min_pair = 2.0
        self.cover = 0.0
        self.press = 0.0
        self.steps = 0

    def reset(self, n: int | None = None, pressure: float | None = None) -> None:
        with self.lock:
            if n is not None:
                self.n = max(2, min(16000, int(n)))
            if pressure is not None:
                self.pressure = max(0.0, float(pressure))
            self.pos = seeds(self.n)
            self.vel = np.zeros_like(self.pos)
            self.t = 0.0
            self.radius = _mean_radius(self.pos)
            self.min_pair = _min_pair(self.pos)
            self.cover = 0.0
            self.press = 0.0
            self.steps = 0

    def step(self) -> float:
        with self.lock:
            acc, cover, press, dist = _forces(self.pos, self.pressure)
            speed = float(np.linalg.norm(self.vel, axis=1).max()) if self.n else 0.0
            amax = float(np.linalg.norm(acc, axis=1).max()) if self.n else 0.0
            h = 0.002
            if amax > 0.0 and dist > 0.0:
                h = min(h, 0.2 * math.sqrt(dist / amax))
            if speed > 0.0 and dist > 0.0:
                h = min(h, 0.02 * dist / speed)
            h = max(h, 1.0e-14)
            half = 0.5 * h
            self.pos += self.vel * h + acc * (half * h)
            self.vel += acc * half
            acc, cover, press, dist = _forces(self.pos, self.pressure)
            self.vel += acc * half
            self.t += h
            self.steps += 1
            self.cover = cover
            self.press = press
            self.radius = _mean_radius(self.pos)
            self.min_pair = dist
            return h

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "n": self.n,
                "pressure": self.pressure,
                "paused": self.paused,
                "t": self.t,
                "steps": self.steps,
                "radius": self.radius,
                "minPair": self.min_pair,
                "cover": self.cover,
                "press": self.press,
                "positions": self.pos.reshape(-1).tolist(),
            }
