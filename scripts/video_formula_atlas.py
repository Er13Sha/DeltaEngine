"""Атлас структур по формуле видео. Карточка записана отдельно."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

from delta_science.action import PI, PiNumber, coupling, force

OUT = Path("data/experiments/field/video_formula_atlas_2026-10-27.json")
C1 = 4 * PI * Fraction(1)


def formula_gate() -> dict[str, object]:
    linked = coupling(C1)
    exact = linked == PiNumber(Fraction(1)) and linked.power == 0
    rows = []
    for radius in (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(10)):
        got = force(C1, radius, 1, 1)
        expected = PiNumber(Fraction(1) / radius**2)
        match = got == expected and got.power == 0
        exact = exact and match
        rows.append({"r": str(radius), "force": str(got), "match": bool(match)})
    return {"passed": bool(exact), "coupling": str(linked), "rows": rows}


def accel_py(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i == j:
                continue
            delta = pos[j] - pos[i]
            dist2 = float(delta @ delta)
            dist = math.sqrt(dist2)
            acc[i] += mass[j] * delta / (dist2 * dist)
    return acc


def center(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> None:
    total = float(mass.sum())
    pos -= (mass[:, None] * pos).sum(axis=0) / total
    vel -= (mass[:, None] * vel).sum(axis=0) / total


@njit
def march_lengths(pos, vel, mass, dt, steps, pairs_i, pairs_j, targets, close_limit, err_stop):
    n = pos.shape[0]
    dim = pos.shape[1]
    acc = np.zeros((n, dim))
    n_len = pairs_i.shape[0]
    max_err = np.zeros(n_len)
    min_pair = 1.0e300
    hit = False
    early = False
    done = 0
    for step in range(steps):
        for i in range(n):
            for d in range(dim):
                acc[i, d] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < close_limit or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        if hit:
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            for d in range(dim):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        for i in range(n):
            for d in range(dim):
                acc[i, d] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        for i in range(n):
            for d in range(dim):
                vel[i, d] += acc[i, d] * half
        worst = 0.0
        for k in range(n_len):
            i = pairs_i[k]
            j = pairs_j[k]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            err = abs(dist - targets[k])
            if err > max_err[k]:
                max_err[k] = err
            if err > worst:
                worst = err
        done = step + 1
        if worst > err_stop:
            early = True
            break
    return max_err, min_pair, hit, early, done


@njit
def march_pair(pos, vel, mass, dt, steps, separation):
    acc = np.zeros((2, 3))
    max_rel = 0.0
    max_law = 0.0
    min_pair = 1.0e300
    sample_every = steps // 20
    if sample_every < 1:
        sample_every = 1
    for step in range(steps):
        for i in range(2):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        if dist < min_pair:
            min_pair = dist
        inv3 = 1.0 / (dist2 * dist)
        acc[0, 0] = mass[1] * inv3 * dx
        acc[0, 1] = mass[1] * inv3 * dy
        acc[0, 2] = mass[1] * inv3 * dz
        acc[1, 0] = -mass[0] * inv3 * dx
        acc[1, 1] = -mass[0] * inv3 * dy
        acc[1, 2] = -mass[0] * inv3 * dz
        if step % sample_every == 0:
            amag = math.sqrt(acc[0, 0] ** 2 + acc[0, 1] ** 2 + acc[0, 2] ** 2)
            law = abs(dist2 * amag - mass[1])
            if law > max_law:
                max_law = law
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(2):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        dist2 = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist2)
        inv3 = 1.0 / (dist2 * dist)
        acc[0, 0] = mass[1] * inv3 * dx
        acc[0, 1] = mass[1] * inv3 * dy
        acc[0, 2] = mass[1] * inv3 * dz
        acc[1, 0] = -mass[0] * inv3 * dx
        acc[1, 1] = -mass[0] * inv3 * dy
        acc[1, 2] = -mass[0] * inv3 * dz
        for i in range(2):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        rel = abs(dist - separation) / separation
        if rel > max_rel:
            max_rel = rel
    return max_rel, max_law, min_pair


@njit
def march_hier(pos, vel, mass, dt, steps, radius, close_limit):
    n = 3
    acc = np.zeros((n, 3))
    max_inner = 0.0
    max_outer = 0.0
    min_pair = 1.0e300
    hit = False
    for _ in range(steps):
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < close_limit or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        if hit:
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        for i in range(n):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        dx = pos[1, 0] - pos[0, 0]
        dy = pos[1, 1] - pos[0, 1]
        dz = pos[1, 2] - pos[0, 2]
        inner = math.sqrt(dx * dx + dy * dy + dz * dz)
        cx = 0.5 * (pos[0, 0] + pos[1, 0])
        cy = 0.5 * (pos[0, 1] + pos[1, 1])
        cz = 0.5 * (pos[0, 2] + pos[1, 2])
        ox = pos[2, 0] - cx
        oy = pos[2, 1] - cy
        oz = pos[2, 2] - cz
        outer = math.sqrt(ox * ox + oy * oy + oz * oz)
        err_in = abs(inner - 1.0)
        err_out = abs(outer - radius)
        if err_in > max_inner:
            max_inner = err_in
        if err_out > max_outer:
            max_outer = err_out
    return max_inner, max_outer, min_pair, hit


@njit
def march_bb(pos, vel, mass, dt, steps, distance, close_limit):
    n = 4
    acc = np.zeros((n, 3))
    max_a = 0.0
    max_b = 0.0
    max_d = 0.0
    min_pair = 1.0e300
    hit = False
    for _ in range(steps):
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < close_limit or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        if hit:
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        for i in range(n):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        ax = pos[1, 0] - pos[0, 0]
        ay = pos[1, 1] - pos[0, 1]
        az = pos[1, 2] - pos[0, 2]
        sep_a = math.sqrt(ax * ax + ay * ay + az * az)
        bx = pos[3, 0] - pos[2, 0]
        by = pos[3, 1] - pos[2, 1]
        bz = pos[3, 2] - pos[2, 2]
        sep_b = math.sqrt(bx * bx + by * by + bz * bz)
        cax = 0.5 * (pos[0, 0] + pos[1, 0])
        cay = 0.5 * (pos[0, 1] + pos[1, 1])
        caz = 0.5 * (pos[0, 2] + pos[1, 2])
        cbx = 0.5 * (pos[2, 0] + pos[3, 0])
        cby = 0.5 * (pos[2, 1] + pos[3, 1])
        cbz = 0.5 * (pos[2, 2] + pos[3, 2])
        dx = cbx - cax
        dy = cby - cay
        dz = cbz - caz
        sep_d = math.sqrt(dx * dx + dy * dy + dz * dz)
        err_a = abs(sep_a - 1.0)
        err_b = abs(sep_b - 1.0)
        err_d = abs(sep_d - distance)
        if err_a > max_a:
            max_a = err_a
        if err_b > max_b:
            max_b = err_b
        if err_d > max_d:
            max_d = err_d
    return max_a, max_b, max_d, min_pair, hit


@njit
def march_radii(pos, vel, mass, dt, steps, r0, init_dist, close_frac):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    max_rel = np.zeros(n)
    min_pair = 1.0e300
    hit = False
    for _ in range(steps):
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < close_frac * init_dist[i, j] or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        if hit:
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        for i in range(n):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
        for i in range(1, n):
            dx = pos[i, 0] - pos[0, 0]
            dy = pos[i, 1] - pos[0, 1]
            dz = pos[i, 2] - pos[0, 2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            rel = abs(dist / r0[i] - 1.0)
            if rel > max_rel[i]:
                max_rel[i] = rel
    return max_rel, min_pair, hit


@njit
def advance(pos, vel, mass, steps, dt, close_limit):
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    min_pair = 1.0e300
    max_r = 0.0
    hit = False
    for _ in range(steps):
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                if dist < min_pair:
                    min_pair = dist
                if dist < close_limit or dist2 == 0.0:
                    hit = True
                    continue
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        if hit:
            break
        half = 0.5 * dt
        half_dt2 = 0.5 * dt * dt
        for i in range(n):
            for d in range(3):
                pos[i, d] += vel[i, d] * dt + acc[i, d] * half_dt2
                vel[i, d] += acc[i, d] * half
        for i in range(n):
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                dist2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist)
                acc[i, 0] += mass[j] * inv3 * dx
                acc[i, 1] += mass[j] * inv3 * dy
                acc[i, 2] += mass[j] * inv3 * dz
                acc[j, 0] -= mass[i] * inv3 * dx
                acc[j, 1] -= mass[i] * inv3 * dy
                acc[j, 2] -= mass[i] * inv3 * dz
        for i in range(n):
            for d in range(3):
                vel[i, d] += acc[i, d] * half
            rr = math.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2 + pos[i, 2] ** 2)
            if rr > max_r:
                max_r = rr
    return min_pair, max_r, hit


def pack_lengths(pos, vel, mass, dt, steps, pairs, targets, close_limit, err_stop):
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    tgt = np.array(targets, dtype=np.float64)
    max_err, min_pair, hit, early, done = march_lengths(
        pos, vel, mass, dt, steps, ii, jj, tgt, close_limit, err_stop
    )
    return {
        "max_error": float(np.max(max_err) if max_err.size else 0.0),
        "errors": [float(x) for x in max_err],
        "min_pair": float(min_pair),
        "close_approach": bool(hit),
        "stopped_early": bool(early),
        "steps_done": int(done),
        "steps": int(steps),
    }


def shape_stable(plain: dict[str, object], moved: dict[str, object]) -> dict[str, object]:
    equilibrium = (
        plain["max_error"] < 1e-3
        and not plain["close_approach"]
        and not plain["stopped_early"]
    )
    held = (
        moved["max_error"] <= 0.01
        and not moved["close_approach"]
        and not moved["stopped_early"]
    )
    return {
        "plain": plain,
        "shifted": moved,
        "equilibrium": bool(equilibrium),
        "stable": bool(equilibrium and held),
    }


def gascheau(masses: tuple[float, float, float]) -> bool:
    total = masses[0] + masses[1] + masses[2]
    pairs = masses[0] * masses[1] + masses[0] * masses[2] + masses[1] * masses[2]
    return bool(total * total > 27.0 * pairs)


def equilateral_state(masses: tuple[float, float, float], shift: float):
    pos = np.zeros((3, 3))
    pos[1, 0] = 1.0
    pos[2, 0] = 0.5
    pos[2, 1] = math.sqrt(3.0) / 2.0
    mass = np.array(masses, dtype=np.float64)
    center(pos, np.zeros_like(pos), mass)
    omega = math.sqrt(float(mass.sum()))
    vel = np.zeros_like(pos)
    vel[:, 0] = -omega * pos[:, 1]
    vel[:, 1] = omega * pos[:, 0]
    pos[2, 0] += shift
    return pos, vel, mass, omega


def run_equilateral(masses, periods: int, label: str) -> dict[str, object]:
    base_pos, _, _, omega = equilateral_state(masses, 0.0)
    period = 2.0 * math.pi / omega
    dt = period / 4000.0
    steps = 4000 * periods
    pairs = ((0, 1), (1, 2), (0, 2))
    targets = (1.0, 1.0, 1.0)

    def once(shift: float):
        pos, vel, mass, _ = equilateral_state(masses, shift)
        return pack_lengths(pos, vel, mass, dt, steps, pairs, targets, 1e-4, 1.0)

    row = shape_stable(once(0.0), once(1e-3))
    row["label"] = label
    row["masses"] = list(masses)
    row["periods"] = periods
    row["classical_gascheau"] = gascheau(masses)
    row["control"] = label == "control"
    del base_pos
    print(
        f"equilateral {label} eq={row['equilibrium']} stable={row['stable']} "
        f"plain={row['plain']['max_error']:.4g} shift={row['shifted']['max_error']:.4g} "
        f"gascheau={row['classical_gascheau']}"
    )
    return row


def polygon_state(n: int, shift: float):
    pos = np.zeros((n, 3))
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        pos[k, 0] = math.cos(ang)
        pos[k, 1] = math.sin(ang)
    mass = np.ones(n)
    acc = accel_py(pos, mass)
    r2 = float(pos[0, 0] ** 2 + pos[0, 1] ** 2)
    omega2 = -float(acc[0, 0] * pos[0, 0] + acc[0, 1] * pos[0, 1]) / r2
    resid = acc + omega2 * pos
    residual = float(np.max(np.linalg.norm(resid, axis=1)))
    pairs = []
    targets = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[i] - pos[j]
            pairs.append((i, j))
            targets.append(float(math.sqrt(delta @ delta)))
    omega = math.sqrt(omega2)
    vel = np.zeros_like(pos)
    vel[:, 0] = -omega * pos[:, 1]
    vel[:, 1] = omega * pos[:, 0]
    pos[0, 0] += shift
    return pos, vel, mass, omega, residual, pairs, targets


def run_polygon(n: int) -> dict[str, object]:
    _, _, _, omega, residual, pairs, targets = polygon_state(n, 0.0)
    row: dict[str, object] = {
        "n": n,
        "residual": residual,
        "initial_gate": bool(residual < 1e-12),
        "control": n == 3,
    }
    if not row["initial_gate"]:
        row["equilibrium"] = False
        row["stable"] = False
        print(f"polygon N={n} residual {residual} gate failed")
        return row
    period = 2.0 * math.pi / omega
    dt = period / 8000.0
    steps = 8000 * 20

    def once(shift: float):
        pos, vel, mass, _, _, _, _ = polygon_state(n, shift)
        return pack_lengths(pos, vel, mass, dt, steps, pairs, targets, 1e-4, 1.0)

    judged = shape_stable(once(0.0), once(1e-3))
    row.update(judged)
    print(
        f"polygon N={n} eq={row['equilibrium']} stable={row['stable']} "
        f"plain={row['plain']['max_error']:.4g} shift={row['shifted']['max_error']:.4g}"
    )
    return row


def euler_state(shift: float):
    pos = np.zeros((3, 3))
    pos[0, 0] = -1.0
    pos[2, 0] = 1.0
    mass = np.ones(3)
    omega2 = 1.25
    acc = accel_py(pos, mass)
    resid = acc + omega2 * pos
    residual = float(np.max(np.linalg.norm(resid, axis=1)))
    omega = math.sqrt(omega2)
    vel = np.zeros_like(pos)
    vel[:, 0] = -omega * pos[:, 1]
    vel[:, 1] = omega * pos[:, 0]
    pos[2, 0] += shift
    return pos, vel, mass, omega, residual


def run_euler() -> dict[str, object]:
    _, _, _, omega, residual = euler_state(0.0)
    row: dict[str, object] = {"residual": residual, "initial_gate": bool(residual < 1e-12)}
    if not row["initial_gate"]:
        row["equilibrium"] = False
        row["stable"] = False
        print("euler gate failed", residual)
        return row
    period = 2.0 * math.pi / omega
    dt = period / 8000.0
    steps = 8000 * 20
    pairs = ((0, 1), (1, 2), (0, 2))
    targets = (1.0, 1.0, 2.0)

    def once(shift: float):
        pos, vel, mass, _, _ = euler_state(shift)
        return pack_lengths(pos, vel, mass, dt, steps, pairs, targets, 1e-4, 1.0)

    row.update(shape_stable(once(0.0), once(1e-3)))
    print(
        f"euler eq={row['equilibrium']} stable={row['stable']} "
        f"plain={row['plain']['max_error']:.4g} shift={row['shifted']['max_error']:.4g}"
    )
    return row


def binary_state(separation: float, m1: float, m2: float):
    pos = np.zeros((2, 3))
    total = m1 + m2
    pos[0, 0] = -m2 / total * separation
    pos[1, 0] = m1 / total * separation
    omega = math.sqrt(total / separation**3)
    vel = np.zeros_like(pos)
    vel[:, 1] = omega * pos[:, 0]
    mass = np.array([m1, m2], dtype=np.float64)
    return pos, vel, mass


def run_binary(separation: float, m1: float, m2: float) -> dict[str, object]:
    pos, vel, mass = binary_state(separation, m1, m2)
    omega = math.sqrt((m1 + m2) / separation**3)
    period = 2.0 * math.pi / omega
    dt = period / 20000.0
    steps = 20000 * 10
    max_rel, max_law, min_pair = march_pair(pos, vel, mass, dt, steps, separation)
    accepted = bool(max_rel < 1e-6 and min_pair > 1e-4)
    row = {
        "separation": separation,
        "masses": [m1, m2],
        "max_relative_error": float(max_rel),
        "max_law_error": float(max_law),
        "min_pair": float(min_pair),
        "accepted": accepted,
        "control": bool(separation == 1.0 and m1 == 1.0 and m2 == 1.0),
    }
    print(
        f"binary a={separation} m=({m1},{m2}) err={max_rel:.4g} "
        f"law={max_law:.4g} accepted={accepted}"
    )
    return row


def hierarchical_state(radius: float, q: float, shift: float, retrograde: bool):
    pos = np.zeros((3, 3))
    vel = np.zeros((3, 3))
    pos[0, 0] = -0.5
    pos[1, 0] = 0.5
    pos[2, 0] = radius + shift
    speed_in = math.sqrt(0.5)
    speed_out = math.sqrt((2.0 + q) / radius)
    if retrograde:
        speed_out = -speed_out
    vel[0, 1] = -speed_in
    vel[1, 1] = speed_in
    vel[2, 1] = speed_out
    mass = np.array([1.0, 1.0, q], dtype=np.float64)
    center(pos, vel, mass)
    binary_v = 0.5 * (vel[0] + vel[1])
    v_rel = vel[2] - binary_v
    speed2_r = float(v_rel @ v_rel) * radius
    return pos, vel, mass, speed2_r


def run_hier_once(radius, q, shift, retrograde, periods):
    pos, vel, mass, speed2_r = hierarchical_state(radius, q, shift, retrograde)
    period_in = 2.0 * math.pi / math.sqrt(2.0)
    period_out = 2.0 * math.pi * math.sqrt(radius**3 / (2.0 + q))
    dt = period_in / 2000.0
    steps = int(math.ceil(periods * period_out / dt))
    max_inner, max_outer, min_pair, hit = march_hier(pos, vel, mass, dt, steps, radius, 0.2)
    accepted = (not hit) and max_inner <= 0.05 and max_outer <= 0.05 * radius
    return {
        "radius": radius,
        "q": q,
        "shift": shift,
        "retrograde": bool(retrograde),
        "periods": periods,
        "speed2_times_r": speed2_r,
        "steps": steps,
        "max_inner_error": float(max_inner),
        "max_outer_error": float(max_outer),
        "min_pair": float(min_pair),
        "close_approach": bool(hit),
        "accepted": bool(accepted),
    }


def run_hier_pair(radius, q, retrograde, periods, control=False):
    plain = run_hier_once(radius, q, 0.0, retrograde, periods)
    gate = abs(plain["speed2_times_r"] - (2.0 + q)) <= 1e-12
    moved = run_hier_once(radius, q, 1e-3 * radius, retrograde, periods)
    stable = bool(gate and plain["accepted"] and moved["accepted"])
    row = {
        "plain": plain,
        "shifted": moved,
        "initial_gate": bool(gate),
        "stable": stable,
        "control": control,
        "periods": periods,
        "retrograde": bool(retrograde),
    }
    print(
        f"hier R={radius} q={q} retro={retrograde} P={periods} "
        f"inner={plain['max_inner_error']:.4g} outer={plain['max_outer_error']:.4g} "
        f"shift_ok={moved['accepted']} stable={stable}"
    )
    return row


def planet_state(r_out: float, shift: float):
    mass = np.array([1.0, 1e-3, 1e-3])
    pos = np.zeros((3, 3))
    vel = np.zeros((3, 3))
    pos[1, 0] = 1.0
    pos[2, 0] = r_out + shift
    vel[1, 1] = math.sqrt((1.0 + 1e-3) / 1.0)
    vel[2, 1] = math.sqrt((1.0 + 1e-3) / r_out)
    center(pos, vel, mass)
    return pos, vel, mass


def run_planets(r_out: float) -> dict[str, object]:
    m = 1e-3
    t_in = 2.0 * math.pi * math.sqrt(1.0 / (1.0 + m))
    t_out = 2.0 * math.pi * math.sqrt(r_out**3 / (1.0 + m))
    dt = min(t_in, t_out) / 2000.0
    periods = 200 if r_out <= 2.0 else 20
    steps = int(math.ceil(periods * t_out / dt))

    def once(shift: float):
        pos, vel, mass = planet_state(r_out, shift)
        r0 = np.zeros(3)
        init = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                delta = pos[i] - pos[j]
                init[i, j] = math.sqrt(float(delta @ delta))
            if i:
                r0[i] = init[i, 0]
        max_rel, min_pair, hit = march_radii(pos, vel, mass, dt, steps, r0, init, 0.5)
        worst = float(np.max(max_rel[1:]))
        return {
            "shift": shift,
            "max_relative_radius": worst,
            "per_planet": [float(x) for x in max_rel[1:]],
            "min_pair": float(min_pair),
            "close_approach": bool(hit),
            "accepted": bool((not hit) and worst <= 0.05),
            "steps": steps,
        }

    plain = once(0.0)
    moved = once(1e-3 * r_out)
    stable = bool(plain["accepted"] and moved["accepted"])
    print(
        f"planets r={r_out} P={periods} rel={plain['max_relative_radius']:.4g} "
        f"close={plain['close_approach']} shift_ok={moved['accepted']} stable={stable}"
    )
    return {"r_out": r_out, "periods": periods, "plain": plain, "shifted": moved, "stable": stable}


def binary_binary_state(distance: float, shift: float):
    pos = np.zeros((4, 3))
    vel = np.zeros((4, 3))
    speed_in = math.sqrt(0.5)
    v_cm = 0.5 * math.sqrt(4.0 / distance)
    pos[0, 0] = -distance / 2.0 - 0.5
    pos[1, 0] = -distance / 2.0 + 0.5
    pos[2, 0] = distance / 2.0 - 0.5 + shift
    pos[3, 0] = distance / 2.0 + 0.5 + shift
    vel[0, 1] = -speed_in - v_cm
    vel[1, 1] = speed_in - v_cm
    vel[2, 1] = -speed_in + v_cm
    vel[3, 1] = speed_in + v_cm
    mass = np.ones(4)
    center(pos, vel, mass)
    return pos, vel, mass


def run_binary_binary(distance: float) -> dict[str, object]:
    period_in = 2.0 * math.pi / math.sqrt(2.0)
    period_out = 2.0 * math.pi * math.sqrt(distance**3 / 4.0)
    dt = period_in / 2000.0
    steps = int(math.ceil(20.0 * period_out / dt))

    def once(shift: float):
        pos, vel, mass = binary_binary_state(distance, shift)
        max_a, max_b, max_d, min_pair, hit = march_bb(
            pos, vel, mass, dt, steps, distance, 0.2
        )
        accepted = (
            (not hit)
            and max_a <= 0.05
            and max_b <= 0.05
            and max_d <= 0.05 * distance
        )
        return {
            "shift": shift,
            "max_pair_a": float(max_a),
            "max_pair_b": float(max_b),
            "max_centers": float(max_d),
            "min_pair": float(min_pair),
            "close_approach": bool(hit),
            "accepted": bool(accepted),
            "steps": steps,
        }

    plain = once(0.0)
    moved = once(1e-3 * distance)
    stable = bool(plain["accepted"] and moved["accepted"])
    print(
        f"binary-binary D={distance} a={plain['max_pair_a']:.4g} "
        f"b={plain['max_pair_b']:.4g} centers={plain['max_centers']:.4g} "
        f"stable={stable}"
    )
    return {"distance": distance, "plain": plain, "shifted": moved, "stable": stable}


def moon_state(d: float, shift: float):
    star, planet, moon = 1.0, 1e-3, 1e-6
    pos = np.zeros((3, 3))
    vel = np.zeros((3, 3))
    mass = np.array([star, planet, moon])
    pair = planet + moon
    v_orb = math.sqrt((star + pair) / 1.0)
    v_rel = math.sqrt(pair / d)
    pos[1, 0] = 1.0 - (moon / pair) * d
    pos[2, 0] = 1.0 + (planet / pair) * d + shift
    vel[1, 1] = v_orb - (moon / pair) * v_rel
    vel[2, 1] = v_orb + (planet / pair) * v_rel
    center(pos, vel, mass)
    v_internal = vel[2] - vel[1]
    v_cm = (planet * vel[1] + moon * vel[2]) / pair
    v_around = v_cm - vel[0]
    internal = float(math.sqrt(v_internal @ v_internal))
    around = float(math.sqrt(v_around @ v_around))
    return pos, vel, mass, internal, around, v_rel, v_orb


def run_moon(d: float) -> dict[str, object]:
    star, planet, moon = 1.0, 1e-3, 1e-6
    t_planet = 2.0 * math.pi * math.sqrt(1.0 / (star + planet + moon))
    t_moon = 2.0 * math.pi * math.sqrt(d**3 / (planet + moon))
    dt = min(t_planet, t_moon) / 2000.0
    steps = int(math.ceil(50.0 * t_planet / dt))
    _, _, _, internal, around, v_rel, v_orb = moon_state(d, 0.0)
    gate = abs(internal - v_rel) <= 1e-12 and abs(around - v_orb) <= 1e-12

    def once(shift: float):
        pos, vel, mass, _, _, _, _ = moon_state(d, shift)
        summary = pack_lengths(
            pos,
            vel,
            mass,
            dt,
            steps,
            ((0, 1), (1, 2)),
            (1.0, d),
            1e-6,
            1.0,
        )
        star_err = summary["errors"][0]
        moon_err = summary["errors"][1]
        accepted = (
            (not summary["close_approach"])
            and star_err <= 0.05
            and moon_err <= 0.05 * d
        )
        summary["accepted"] = bool(accepted)
        return summary

    if not gate:
        print(f"moon d={d} speed gate failed {internal} {around}")
        return {"d": d, "initial_gate": False, "stable": False}
    plain = once(0.0)
    moved = once(1e-3 * d)
    stable = bool(plain["accepted"] and moved["accepted"])
    print(
        f"moon d={d} star={plain['errors'][0]:.4g} moon={plain['errors'][1]:.4g} "
        f"shift_ok={moved['accepted']} stable={stable} steps={steps}"
    )
    return {
        "d": d,
        "initial_gate": True,
        "steps": steps,
        "plain": plain,
        "shifted": moved,
        "stable": stable,
    }


def figure_eight(shift: float) -> dict[str, object]:
    x1 = 0.97000436
    y1 = -0.24308753
    vx3 = -0.93240737
    vy3 = -0.86473146
    pos = np.array(
        [[x1, y1, 0.0], [-x1, -y1, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    vel = np.array(
        [
            [-vx3 / 2.0, -vy3 / 2.0, 0.0],
            [-vx3 / 2.0, -vy3 / 2.0, 0.0],
            [vx3, vy3, 0.0],
        ],
        dtype=np.float64,
    )
    pos[0, 0] += shift
    start = pos.copy()
    mass = np.ones(3)
    period = 6.32591398
    steps_per = 20000
    dt = period / steps_per
    min_pair = 1.0e300
    max_r = 0.0
    hit = False
    err_at = {}
    for lap in range(1, 31):
        mn, mx, close = advance(pos, vel, mass, steps_per, dt, 1e-4)
        min_pair = min(min_pair, float(mn))
        max_r = max(max_r, float(mx))
        hit = hit or bool(close)
        if lap in (1, 10, 30):
            err = 0.0
            for i in range(3):
                delta = pos[i] - start[i]
                err = max(err, float(math.sqrt(delta @ delta)))
            err_at[lap] = err
        if hit or max_r > 5.0 or min_pair < 0.05:
            break
    periodic = bool(err_at.get(1, 1.0) < 1e-4 and not hit)
    bounded = bool(max_r <= 5.0 and min_pair >= 0.05 and not hit and lap == 30)
    return {
        "shift": shift,
        "return_error_period_1": err_at.get(1),
        "return_error_period_10": err_at.get(10),
        "return_error_period_30": err_at.get(30),
        "laps_done": lap,
        "max_radius": max_r,
        "min_pair": min_pair,
        "close_approach": bool(hit),
        "periodic": periodic,
        "bounded": bounded,
    }


def run_figure_eight() -> dict[str, object]:
    plain = figure_eight(0.0)
    small = figure_eight(1e-5)
    large = figure_eight(1e-3)
    row = {
        "plain": plain,
        "shift_1e-5": small,
        "shift_1e-3": large,
        "periodic": bool(plain["periodic"]),
        "stable_1e-5": bool(plain["periodic"] and small["bounded"]),
        "stable_1e-3": bool(plain["periodic"] and large["bounded"]),
    }
    print(
        f"eight periodic={row['periodic']} return={plain['return_error_period_1']} "
        f"stable_1e-5={row['stable_1e-5']} stable_1e-3={row['stable_1e-3']}"
    )
    return row


def main() -> None:
    gate = formula_gate()
    print("formula_gate", gate["passed"])
    payload: dict[str, object] = {
        "scope": "model",
        "protocol": "docs/video-formula-atlas-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    if not gate["passed"]:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("gate failed")
        return

    mu_star = (1.0 - math.sqrt(1.0 - 4.0 / 27.0)) / 2.0
    print("mu_star", mu_star)

    binaries = []
    for separation in (0.5, 1.0, 2.0, 10.0):
        for m2 in (1.0, 0.1, 10.0):
            binaries.append(run_binary(separation, 1.0, m2))
    unit = next(row for row in binaries if row["control"])
    if unit["max_law_error"] >= 1e-9 or not unit["accepted"]:
        payload["binary"] = binaries
        payload["aborted"] = "unit binary step check failed"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(1)

    control = run_hier_pair(5.0, 1.0, False, 20, control=True)
    if not control["plain"]["accepted"]:
        payload["binary"] = binaries
        payload["control_R5"] = control
        payload["aborted"] = "R=5 control not accepted"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(2)

    equilateral = [run_equilateral((1.0, 1.0, 1.0), 5, "control")]
    for mu in (0.5, 0.1, 0.05, mu_star, 0.02, 0.01, 0.001):
        equilateral.append(
            run_equilateral((1.0 - mu, mu, 1e-6), 200, f"mu={mu:.8g}")
        )
    for masses in (
        (1.0, 0.01, 0.01),
        (1.0, 0.02, 0.02),
        (1.0, 0.05, 0.05),
        (1.0, 0.1, 0.1),
    ):
        equilateral.append(run_equilateral(masses, 200, "finite"))

    euler = run_euler()
    polygons = [run_polygon(n) for n in (3, 4, 5, 6)]

    hierarchical = []
    for q in (0.01, 0.1, 10.0):
        for radius in (3.0, 5.0, 10.0, 20.0):
            hierarchical.append(run_hier_pair(radius, q, False, 20))
    retrograde = [
        run_hier_pair(radius, 1.0, True, 20) for radius in (3.0, 5.0, 10.0, 20.0)
    ]
    durability = [
        run_hier_pair(radius, 1.0, False, 200) for radius in (5.0, 10.0, 20.0)
    ]
    planets = [run_planets(r) for r in (1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)]
    doubles = [run_binary_binary(d) for d in (5.0, 10.0, 20.0)]
    moons = [run_moon(d) for d in (0.01, 0.05, 0.2)]
    eight = run_figure_eight()

    payload.update(
        {
            "mu_star": mu_star,
            "binary": binaries,
            "equilateral": equilateral,
            "euler": euler,
            "polygon": polygons,
            "control_R5": control,
            "hierarchical": hierarchical,
            "retrograde": retrograde,
            "durability_200": durability,
            "planets": planets,
            "binary_binary": doubles,
            "star_planet_moon": moons,
            "figure_eight": eight,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
