"""Обороты порций вокруг центра масс при P = 1000000."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numba import njit, prange

from scripts.universe_pressure_complex_press import accelerations, fibonacci_sky, formula_gate

OUT = Path("data/experiments/field/universe_pressure_orbits_2026-10-27.json")
SKY = fibonacci_sky(72)
P_STRONG = 1_000_000.0


def initial(n: int) -> np.ndarray:
    if n == 4:
        pts = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
            dtype=np.float64,
        )
    elif n == 8:
        pts = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float64,
        )
    else:
        pts = fibonacci_sky(n) * 1.0
    pts -= pts.mean(axis=0)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


@njit(parallel=True)
def add_missed(pos, sky, pressure, acc):
    n = pos.shape[0]
    nsky = sky.shape[0]
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 == 0.0:
                continue
            dist = math.sqrt(dist2)
            omega = 1.0 / dist2
            if omega > 2.0 * math.pi:
                omega = 2.0 * math.pi
            cos_a = 1.0 - omega / (2.0 * math.pi)
            if cos_a < -1.0:
                cos_a = -1.0
            ux = dx / dist
            uy = dy / dist
            uz = dz / dist
            hit = False
            for k in range(nsky):
                if ux * sky[k, 0] + uy * sky[k, 1] + uz * sky[k, 2] >= cos_a:
                    hit = True
                    break
            if hit:
                continue
            mag = pressure * omega * (1.0 - omega / (4.0 * math.pi))
            acc[i, 0] += ux * mag
            acc[i, 1] += uy * mag
            acc[i, 2] += uz * mag
    return acc


@njit
def min_pair(pos):
    n = pos.shape[0]
    best = 1.0e300
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < best:
                best = dist
    return best


@njit
def center(pos):
    n = pos.shape[0]
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for i in range(n):
        cx += pos[i, 0]
        cy += pos[i, 1]
        cz += pos[i, 2]
    return cx / n, cy / n, cz / n


@njit
def march(pos, vel, sky, pressure, t_end=1.0, max_steps=400000, h_cap=0.002, stop_ratio=0.0, need_turns=1):
    n = pos.shape[0]
    phi = np.zeros(n)
    rmin = np.full(n, 1.0e300)
    rmax = np.zeros(n)
    prevx = np.zeros(n)
    prevy = np.zeros(n)
    prevz = np.zeros(n)
    axis = np.zeros(3)
    have_axis = False
    closest = 1.0e300
    t = 0.0
    steps = 0
    turned = False
    while t < t_end and steps < max_steps:
        acc, _, _, _, _ = accelerations(pos, sky, pressure, True)
        acc = add_missed(pos, sky, pressure, acc)
        dist = min_pair(pos)
        if dist < closest:
            closest = dist
        rel = 0.0
        amax = 0.0
        for i in range(n):
            speed = math.sqrt(vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
            if speed > rel:
                rel = speed
            am = math.sqrt(acc[i, 0] ** 2 + acc[i, 1] ** 2 + acc[i, 2] ** 2)
            if am > amax:
                amax = am
        h = h_cap
        if amax > 0.0 and dist > 0.0:
            h = min(h, 0.2 * math.sqrt(dist / amax))
        if rel > 0.0 and dist > 0.0:
            h = min(h, 0.02 * dist / rel)
        h = max(h, 1.0e-14)
        if t + h > t_end:
            h = t_end - t
        half = 0.5 * h
        for i in range(n):
            pos[i, 0] += vel[i, 0] * h + acc[i, 0] * half * h
            pos[i, 1] += vel[i, 1] * h + acc[i, 1] * half * h
            pos[i, 2] += vel[i, 2] * h + acc[i, 2] * half * h
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        acc, _, _, _, _ = accelerations(pos, sky, pressure, True)
        acc = add_missed(pos, sky, pressure, acc)
        for i in range(n):
            vel[i, 0] += acc[i, 0] * half
            vel[i, 1] += acc[i, 1] * half
            vel[i, 2] += acc[i, 2] * half
        cx, cy, cz = center(pos)
        lx = 0.0
        ly = 0.0
        lz = 0.0
        for i in range(n):
            rx = pos[i, 0] - cx
            ry = pos[i, 1] - cy
            rz = pos[i, 2] - cz
            lx += ry * vel[i, 2] - rz * vel[i, 1]
            ly += rz * vel[i, 0] - rx * vel[i, 2]
            lz += rx * vel[i, 1] - ry * vel[i, 0]
        lnorm = math.sqrt(lx * lx + ly * ly + lz * lz)
        if (not have_axis) and lnorm > 1.0e-6:
            axis[0] = lx / lnorm
            axis[1] = ly / lnorm
            axis[2] = lz / lnorm
            have_axis = True
            for i in range(n):
                rx = pos[i, 0] - cx
                ry = pos[i, 1] - cy
                rz = pos[i, 2] - cz
                radius = math.sqrt(rx * rx + ry * ry + rz * rz)
                if radius < rmin[i]:
                    rmin[i] = radius
                if radius > rmax[i]:
                    rmax[i] = radius
                along = rx * axis[0] + ry * axis[1] + rz * axis[2]
                prevx[i] = rx - along * axis[0]
                prevy[i] = ry - along * axis[1]
                prevz[i] = rz - along * axis[2]
        elif have_axis:
            alive = 0
            for i in range(n):
                rx = pos[i, 0] - cx
                ry = pos[i, 1] - cy
                rz = pos[i, 2] - cz
                radius = math.sqrt(rx * rx + ry * ry + rz * rz)
                if radius < rmin[i]:
                    rmin[i] = radius
                if radius > rmax[i]:
                    rmax[i] = radius
                if rmin[i] > 0.0 and rmax[i] <= stop_ratio * rmin[i]:
                    alive += 1
                along = rx * axis[0] + ry * axis[1] + rz * axis[2]
                px = rx - along * axis[0]
                py = ry - along * axis[1]
                pz = rz - along * axis[2]
                cxv = prevy[i] * pz - prevz[i] * py
                cyv = prevz[i] * px - prevx[i] * pz
                czv = prevx[i] * py - prevy[i] * px
                sin_a = cxv * axis[0] + cyv * axis[1] + czv * axis[2]
                cos_a = prevx[i] * px + prevy[i] * py + prevz[i] * pz
                phi[i] += math.atan2(sin_a, cos_a)
                prevx[i] = px
                prevy[i] = py
                prevz[i] = pz
            n_turned = 0
            for i in range(n):
                if abs(phi[i]) >= 2.0 * math.pi:
                    n_turned += 1
            if n_turned >= need_turns or (stop_ratio > 0.0 and alive < 2):
                turned = True
        t += h
        steps += 1
        if turned:
            break
    return phi, rmin, rmax, closest, t, steps


def one(n: int, pressure: float) -> dict[str, object]:
    pos = initial(n)
    vel = np.zeros_like(pos)
    phi, rmin, rmax, closest, t, steps = march(pos, vel, SKY, pressure)
    orbiting = 0
    for i in range(n):
        bounded = rmin[i] > 0.0 and rmax[i] <= 8.0 * rmin[i]
        if abs(float(phi[i])) >= 2.0 * math.pi and bounded and closest > 1.0e-9:
            orbiting += 1
    accepted = orbiting >= 2
    print(
        f"n={n} P={pressure:g} orbiting={orbiting} minr={closest:.4g} "
        f"phi={float(np.max(np.abs(phi))):.3f} t={t:.4g} steps={steps} ok={accepted}"
    )
    return {
        "n": n,
        "pressure": pressure,
        "orbiting": orbiting,
        "max_abs_phi": float(np.max(np.abs(phi))),
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
        "protocol": "docs/universe-pressure-orbits-protocol-2026-10-27.md",
        "formula_gate": gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not gate["passed"]:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    rows = [one(n, P_STRONG) for n in (4, 8, 16)]
    control = one(4, 1.0)
    accepted = any(row["accepted"] for row in rows) and not control["accepted"]
    payload.update({"rows": rows, "control_n4_p1": control, "grid_accepted": bool(accepted)})
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"grid={accepted}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
