"""E-COMP-v1: контроль сферического окружения и точная N-body динамика.

Эффективные узлы, парное отталкивание 1/r², без смягчения, отсечения и стенок.
Протокол: docs/external-compression-protocol-v1.md.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/field/external_compression_v1"
PROTOCOL = ROOT / "data/experiments/field/external_compression_protocol_v1.json"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def provenance():
    registration = json.loads(PROTOCOL.with_name("external_compression_protocol_v1.registration.json").read_text(encoding="utf-8"))
    for item in registration["files"]:
        if hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Зарегистрированный протокол изменился: " + item["path"])
    return {"scope": "model", "protocol_id": "E-COMP-v1", "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__}}


def shell_force(radius, order, shell_radius=1., source=1., probe=1.):
    """Полная осевая проекция: азимут проинтегрирован, dN = Ns du/2."""
    u, weights = np.polynomial.legendre.leggauss(order)
    d2 = shell_radius**2 + radius**2 - 2*shell_radius*radius*u
    return float(source*probe/2*np.sum(weights*(radius-shell_radius*u)/d2**1.5))


def p0():
    rows = []
    for order in (16, 32, 64, 128):
        for r in (0., .25, .5, .75, 1.25, 2.):
            force = shell_force(r, order)
            expected = 0. if r < 1 else 1/r**2
            error = abs(force-expected)
            rows.append({"r": r, "order": order, "force": force, "expected": expected,
                         "absolute_error": error, "relative_error": error/expected if expected else None})
    fine = [x for x in rows if x["order"] == 128]
    passed = all(x["absolute_error"] < 1e-8 if x["r"] < 1 else x["relative_error"] < 1e-8 for x in fine)
    result = {**provenance(), "stage": "P0", "passed": passed, "rows": rows}
    write_json(OUT/"p0.json", result)
    print("P0:", "PASS" if passed else "FAIL", "max interior residual:", max(x["absolute_error"] for x in fine if x["r"] < 1), flush=True)
    return result


def initial_state(radius, seed, omega, no_environment=False):
    central = np.array(list(itertools.product((-1., 1.), repeat=3)))*(.5/math.sqrt(3))
    lattice = np.array(list(itertools.product(range(-8, 9), repeat=3)), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    lattice += rng.uniform(-.1, .1, size=lattice.shape)
    radial = np.linalg.norm(lattice, axis=1)
    external = lattice[(radial >= 1.5) & (radial <= radius)]
    if no_environment:
        external = external[:0]
    positions = np.concatenate((central, external))
    velocities = np.zeros_like(positions)
    velocities[:8, 0] = -omega*central[:, 1]
    velocities[:8, 1] = omega*central[:, 0]
    return positions, velocities


@njit(cache=True, parallel=True)
def forces(positions, active):
    """Все N_i=mu_i=kappa=1. По строкам, без общих записей и отсечения."""
    n = len(positions)
    result = np.empty((active, 3))
    distances = np.empty(active)
    for i in prange(active):
        fx, fy, fz, minimum = 0., 0., 0., math.inf
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        for j in range(n):
            if i == j:
                continue
            dx, dy, dz = xi-positions[j, 0], yi-positions[j, 1], zi-positions[j, 2]
            d2 = dx*dx+dy*dy+dz*dz
            minimum = min(minimum, d2)
            if d2 == 0:
                fx, fy, fz = math.nan, math.nan, math.nan
                continue
            inverse = 1/(d2*math.sqrt(d2))
            fx += dx*inverse
            fy += dy*inverse
            fz += dz*inverse
        result[i, 0], result[i, 1], result[i, 2] = fx, fy, fz
        distances[i] = math.sqrt(minimum)
    return result, float(np.min(distances))


@njit(cache=True)
def potential(positions):
    energy = 0.
    for i in range(len(positions)):
        for j in range(i):
            dx = positions[i, 0]-positions[j, 0]
            dy = positions[i, 1]-positions[j, 1]
            dz = positions[i, 2]-positions[j, 2]
            energy += 1/math.sqrt(dx*dx+dy*dy+dz*dz)
    return energy


@njit(cache=True)
def external_virial(positions):
    center = np.sum(positions[:8], axis=0)/8
    virial = 0.
    for i in range(8):
        for j in range(8, len(positions)):
            dx, dy, dz = positions[i, 0]-positions[j, 0], positions[i, 1]-positions[j, 1], positions[i, 2]-positions[j, 2]
            d2 = dx*dx+dy*dy+dz*dz
            virial += ((positions[i, 0]-center[0])*dx + (positions[i, 1]-center[1])*dy + (positions[i, 2]-center[2])*dz)/(d2*math.sqrt(d2))
    return virial


@njit(cache=True)
def energy_momentum(positions, velocities):
    return potential(positions)+np.sum(velocities*velocities)/2, np.sum(velocities, axis=0)


def sample(positions, velocities, t, minimum):
    displacement = positions[:8]-positions[:8].mean(axis=0)
    radii = np.linalg.norm(displacement, axis=1)
    if len(positions) > 8:
        er = np.linalg.norm(positions[8:]-positions[8:].mean(axis=0), axis=1)
        r95 = float(np.sort(er)[math.ceil(.95*len(er))-1])
        density = .95*len(er)/(4*math.pi*r95**3/3)
        histogram = np.histogram(np.linalg.norm(positions[8:]-positions[:8].mean(axis=0), axis=1), bins=[0, 1.5, 2, 3, 4, 6, 8, 12, 16, 32, 64, math.inf])[0].tolist()
    else:
        r95, density, histogram = None, None, [0]*11
    energy, momentum = energy_momentum(positions, velocities)
    angular = np.sum(np.cross(positions, velocities), axis=0)
    return {"t": t, "rms_radius": float(np.sqrt(np.mean(radii*radii))), "max_radius": float(radii.max()),
            "C_ext": float(external_virial(positions)), "external_r95": r95, "external_density": density,
            "external_radial_counts": histogram, "energy": float(energy), "momentum": momentum.tolist(),
            "angular_momentum": angular.tolist(), "minimum_pair_distance": minimum}


@njit(cache=True)
def kick_drift(positions, velocities, acceleration, dt, active):
    for i in range(active):
        for axis in range(3):
            velocities[i, axis] += .5*dt*acceleration[i, axis]
            positions[i, axis] += dt*velocities[i, axis]


@njit(cache=True)
def kick(velocities, acceleration, dt, active):
    for i in range(active):
        for axis in range(3):
            velocities[i, axis] += .5*dt*acceleration[i, axis]


def run(radius=4, seed=1, omega=0., dt=.002, mode="mobile", tmax=10.):
    positions, velocities = initial_state(radius, seed, omega, mode == "isolated")
    initial_positions, initial_velocities = positions.copy(), velocities.copy()
    active = 8 if mode == "frozen" else len(positions)
    acceleration, minimum = forces(positions, active)
    trace = [sample(positions, velocities, 0., minimum)]
    every = round(.02/dt)
    status = "complete"
    started = time.perf_counter()
    for step in range(1, round(tmax/dt)+1):
        kick_drift(positions, velocities, acceleration, dt, active)
        acceleration, minimum = forces(positions, active)
        if minimum < .05 or not np.isfinite(acceleration).all():
            status = "unresolved_close_pair"
            break
        kick(velocities, acceleration, dt, active)
        if step % every == 0:
            trace.append(sample(positions, velocities, step*dt, minimum))
    energy0 = trace[0]["energy"]
    momentum0 = np.array(trace[0]["momentum"])
    energy_error = max(abs(x["energy"]/energy0-1) for x in trace)
    momentum_error = max(np.linalg.norm(np.array(x["momentum"])-momentum0)/len(positions) for x in trace)
    window = [x for x in trace if x["t"] >= 5]
    bounded = all(x["max_radius"] <= 1. for x in trace)
    radius_ratio = max(x["rms_radius"] for x in window)/min(x["rms_radius"] for x in window) if window else None
    density_error = max(abs(x["external_density"]/trace[0]["external_density"]-1) for x in trace) if len(positions) > 8 else None
    quality = energy_error < .001 and (mode == "frozen" or momentum_error < 1e-8) and status == "complete"
    candidate = bool(quality and bounded and radius_ratio is not None and radius_ratio <= 1.1)
    result = {**provenance(), "parameters": {"radius": radius, "seed": seed, "omega": omega, "dt": dt, "mode": mode, "tmax": tmax},
              "nodes": len(positions), "active_nodes": active, "status": status, "elapsed_seconds": time.perf_counter()-started,
              "quality": quality, "energy_relative_drift": energy_error, "normalized_momentum_drift": float(momentum_error),
              "bounded": bounded, "radius_window_max_over_min": radius_ratio, "candidate": candidate,
              "external_density_relative_drift": density_error, "stationary_environment": density_error is None or density_error <= .1,
              "radial_bin_edges": [0, 1.5, 2, 3, 4, 6, 8, 12, 16, 32, 64, "infinity"], "trace": trace}
    return result, {"initial_positions": initial_positions, "initial_velocities": initial_velocities,
                    "final_positions": positions, "final_velocities": velocities}


def run_key(radius, seed, omega, dt, mode):
    return f"{mode}-R{radius}-seed{seed}-omega{omega:g}-dt{dt:g}"


def save_run(radius, seed, omega, dt, mode, folder="runs"):
    key = run_key(radius, seed, omega, dt, mode)
    path = OUT/folder/(key+".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result, arrays = run(radius, seed, omega, dt, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    write_json(path, result)
    print(key, "N=", result["nodes"], "R_end=", round(result["trace"][-1]["rms_radius"], 5),
          "quality=", result["quality"], "candidate=", result["candidate"],
          "seconds=", round(result["elapsed_seconds"], 1), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("p0", "pilot", "campaign", "controls"), default="p0")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "p0":
        if not p0()["passed"]:
            raise SystemExit(1)
        return
    gate = json.loads((OUT/"p0.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise SystemExit("P0 не пройден")
    if args.stage == "pilot":
        save_run(4, 1, 0., .002, "mobile", folder="pilot")
    elif args.stage == "campaign":
        for radius, seed, omega in itertools.product((4, 6, 8), (1, 2, 3), (0., .5, 1.)):
            save_run(radius, seed, omega, .002, "mobile")
    elif args.stage == "controls":
        for omega in (0., .5, 1.):
            save_run(4, 1, omega, .002, "isolated", folder="controls")
        for radius, seed, omega in itertools.product((4, 6, 8), (1, 2, 3), (0., .5, 1.)):
            save_run(radius, seed, omega, .002, "frozen", folder="controls")
        for radius, omega in itertools.product((4, 8), (0., .5, 1.)):
            save_run(radius, 1, omega, .001, "mobile", folder="controls")


if __name__ == "__main__":
    main()
