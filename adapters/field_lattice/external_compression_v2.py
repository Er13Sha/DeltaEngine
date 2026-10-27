"""E-COMP-v2: реакционный пояс и сферическая замена дальнего комплекса.

Парное отталкивание 1/r², внутри сферы замены F_∞ = 0, радиальная стенка
только у узлов резервуара. Протокол: docs/external-compression-protocol-v2.md.
P2 этим модулем не запускается.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from fractions import Fraction
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

from adapters.field_lattice.external_compression import shell_force

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/field/external_compression_v2"
PROTOCOL = ROOT / "data/experiments/field/external_compression_protocol_v2.json"
REGISTRATION = ROOT / "data/experiments/field/external_compression_protocol_v2.registration.json"

FAMILIES = ("isolated", "frozen", "mobile_open", "mobile_reservoir")
R0 = 0.5
R_CAVITY = 1.5
R_IN = 3.0
R_OUT = 4.0
R_INF = 20.0
N_INF_CAMPAIGN = 33144
N_INF_PROBE = 1
UNRESOLVED = 0.05
BELT_VOLUME = 4.0 * math.pi / 3.0 * (R_OUT ** 3 - R_IN ** 3)
ENERGY_MAX = 1e-3
MOMENTUM_MAX = 1e-8
DT_REL_MAX = 0.01
RMS_RATIO_MAX = 1.1
HOLD_RADIUS = 2.0 * R0
SATURATION = 0.1
DENSITY_FLAG = 0.1
WINDOW = (5.0, 10.0)
SEEDS = (1, 2, 3)
OMEGAS = (0.0, 0.5, 1.0)
DT = 0.002
TMAX = 10.0


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float")
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def protocol_registration_ok():
    """Хэши текста и карточки, не прогонов."""
    items = json.loads(REGISTRATION.read_text(encoding="utf-8"))["files"]
    for item in items:
        digest = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            return False
    return True


def provenance():
    if not protocol_registration_ok():
        raise ValueError("Зарегистрированный протокол v2 изменился")
    return {
        "scope": "model",
        "protocol_id": "E-COMP-v2",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__},
    }


def count_cubic_points(r_min, r_max, index_span, *, min_exclusive=False):
    n = 0
    for i in range(-index_span, index_span + 1):
        for j in range(-index_span, index_span + 1):
            for k in range(-index_span, index_span + 1):
                radius = math.sqrt(i * i + j * j + k * k)
                if min_exclusive:
                    if r_min < radius <= r_max:
                        n += 1
                elif r_min <= radius <= r_max:
                    n += 1
    return n


def campaign_n_infinity():
    return count_cubic_points(4.0, 20.0, 20, min_exclusive=True)


def belt_nodes_no_jitter():
    return count_cubic_points(R_IN, R_OUT, 8)


def pair_force_exact(separation: Fraction) -> Fraction:
    return Fraction(1, 1) / (separation * separation)


def initial_state(seed, omega, family):
    if family not in FAMILIES:
        raise ValueError(family)
    central = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * (R0 / math.sqrt(3))
    if family == "isolated":
        positions = central.copy()
        reservoir = np.zeros(len(positions), dtype=bool)
        n_inf = 0.0
    else:
        lattice = np.array(list(itertools.product(range(-8, 9), repeat=3)), dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(seed))
        lattice += rng.uniform(-0.1, 0.1, size=lattice.shape)
        radial = np.linalg.norm(lattice, axis=1)
        external = lattice[(radial >= R_IN) & (radial <= R_OUT)]
        positions = np.concatenate((central, external))
        reservoir = np.zeros(len(positions), dtype=bool)
        reservoir[8:] = True
        n_inf = float(N_INF_CAMPAIGN)
    velocities = np.zeros_like(positions)
    velocities[:8, 0] = -omega * central[:, 1]
    velocities[:8, 1] = omega * central[:, 0]
    return positions, velocities, reservoir, n_inf


@njit(cache=True, parallel=True)
def pair_forces(positions, active):
    n = len(positions)
    result = np.empty((active, 3))
    distances = np.empty(active)
    for i in prange(active):
        fx, fy, fz, minimum = 0.0, 0.0, 0.0, math.inf
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        for j in range(n):
            if i == j:
                continue
            dx, dy, dz = xi - positions[j, 0], yi - positions[j, 1], zi - positions[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            minimum = min(minimum, d2)
            if d2 == 0:
                fx, fy, fz = math.nan, math.nan, math.nan
                continue
            inverse = 1 / (d2 * math.sqrt(d2))
            fx += dx * inverse
            fy += dy * inverse
            fz += dz * inverse
        result[i, 0], result[i, 1], result[i, 2] = fx, fy, fz
        distances[i] = math.sqrt(minimum)
    return result, float(np.min(distances))


def far_field_forces(positions, n_inf, r_inf, active):
    extra = np.zeros((active, 3))
    if n_inf == 0.0:
        return extra
    radii = np.linalg.norm(positions[:active], axis=1)
    outside = radii >= r_inf
    scale = np.zeros(active)
    safe = outside & (radii > 0)
    scale[safe] = n_inf / radii[safe] ** 3
    extra = positions[:active] * scale[:, None]
    return extra


def forces(positions, active, n_inf=0.0, r_inf=R_INF):
    pair, minimum = pair_forces(positions, active)
    return pair + far_field_forces(positions, n_inf, r_inf, active), minimum


@njit(cache=True)
def pair_potential(positions):
    energy = 0.0
    for i in range(len(positions)):
        for j in range(i):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            energy += 1 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return energy


@njit(cache=True)
def external_virial(positions):
    """C_ext: вклад слоя, не дальнего комплекса и не стенки."""
    if len(positions) <= 8:
        return 0.0
    center = np.sum(positions[:8], axis=0) / 8
    virial = 0.0
    for i in range(8):
        for j in range(8, len(positions)):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            d2 = dx * dx + dy * dy + dz * dz
            virial += (
                (positions[i, 0] - center[0]) * dx
                + (positions[i, 1] - center[1]) * dy
                + (positions[i, 2] - center[2]) * dz
            ) / (d2 * math.sqrt(d2))
    return virial


def energy(positions, velocities, n_inf=0.0, r_inf=R_INF):
    value = pair_potential(positions) + 0.5 * np.sum(velocities * velocities)
    if n_inf != 0.0:
        radii = np.linalg.norm(positions, axis=1)
        outside = radii >= r_inf
        if np.any(outside):
            value += n_inf * np.sum(1.0 / radii[outside] - 1.0 / r_inf)
    return float(value)


def apply_reservoir_constraint(positions, velocities, reservoir, r_in=R_IN, r_out=R_OUT):
    """Проекция только резервуара: не сила на восьмёрку и не −k x."""
    hit = False
    for i in np.flatnonzero(reservoir):
        radius = float(np.linalg.norm(positions[i]))
        if radius == 0.0:
            continue
        if radius < r_in or radius > r_out:
            hit = True
            target = r_in if radius < r_in else r_out
            hat = positions[i] / radius
            positions[i] = hat * target
            velocities[i] -= np.dot(velocities[i], hat) * hat
    return hit


@njit(cache=True)
def kick_drift(positions, velocities, acceleration, dt, active):
    for i in range(active):
        for axis in range(3):
            velocities[i, axis] += 0.5 * dt * acceleration[i, axis]
            positions[i, axis] += dt * velocities[i, axis]


@njit(cache=True)
def kick(velocities, acceleration, dt, active):
    for i in range(active):
        for axis in range(3):
            velocities[i, axis] += 0.5 * dt * acceleration[i, axis]


def thick_shell_force(radius, r_in=R_IN, r_out=R_OUT, order=128, n_radial=48, total=1.0):
    """Осевая сила на пробу в начале отсчёта от сферически симметричной толстой оболочки."""
    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    radii = 0.5 * (r_out - r_in) * nodes + 0.5 * (r_out + r_in)
    radial_weight = 0.5 * (r_out - r_in) * weights
    volume = r_out ** 3 - r_in ** 3
    total_force = 0.0
    for shell_radius, weight in zip(radii, radial_weight):
        source = total * 3.0 * shell_radius ** 2 * weight / volume
        total_force += shell_force(radius, order, shell_radius=float(shell_radius), source=float(source))
    return float(total_force)


def sample(positions, velocities, t, minimum, n_inf=0.0, r_inf=R_INF, constraint_hit=False, constraint_hits_in_interval=0):
    n_central = min(8, len(positions))
    center = positions[:n_central].mean(axis=0)
    radii = np.linalg.norm(positions[:n_central] - center, axis=1)
    belt = positions[n_central:]
    if len(belt):
        belt_radii = np.linalg.norm(belt, axis=1)
        inner_frac = float(np.mean(np.abs(belt_radii - R_IN) < 1e-12))
        outer_frac = float(np.mean(np.abs(belt_radii - R_OUT) < 1e-12))
        in_band = int(np.sum((belt_radii >= R_IN) & (belt_radii <= R_OUT)))
        band_density = in_band / BELT_VOLUME
        relative = np.linalg.norm(belt - belt.mean(axis=0), axis=1)
        r95 = float(np.sort(relative)[math.ceil(0.95 * len(relative)) - 1])
        r95_density = 0.95 * len(belt) / (4 * math.pi * r95 ** 3 / 3)
        hats = belt / np.maximum(belt_radii, 1e-300)[:, None]
        mean_hat = hats.mean(axis=0)
        angular_mean = float(np.linalg.norm(mean_hat))
        angular_rms = float(np.sqrt(np.mean(np.sum((hats - mean_hat) ** 2, axis=1))))
    else:
        inner_frac, outer_frac = 0.0, 0.0
        in_band, band_density, r95, r95_density = 0, None, None, None
        angular_mean, angular_rms = None, None
    angular = np.sum(np.cross(positions, velocities), axis=0)
    return {
        "t": t,
        "rms_radius": float(np.sqrt(np.mean(radii * radii))),
        "max_radius": float(radii.max()),
        "C_ext": float(external_virial(positions)),
        "energy": energy(positions, velocities, n_inf, r_inf),
        "momentum": np.sum(velocities, axis=0).tolist(),
        "angular_momentum": angular.tolist(),
        "minimum_pair_distance": minimum,
        "belt_inner_boundary_fraction": inner_frac,
        "belt_outer_boundary_fraction": outer_frac,
        "belt_nodes_in_band": in_band,
        "belt_band_density": band_density,
        "belt_r95": r95,
        "belt_r95_density": r95_density,
        "belt_angular_mean_rhat": angular_mean,
        "belt_angular_rms": angular_rms,
        "constraint_hit": bool(constraint_hit),
        "constraint_hits_in_interval": int(constraint_hits_in_interval),
    }


def integrate(
    positions,
    velocities,
    *,
    family,
    reservoir,
    n_inf,
    r_inf=R_INF,
    dt=0.002,
    tmax,
    r_in=R_IN,
    r_out=R_OUT,
):
    """Короткий интегратор. tmax задаётся явно; сетка P1 не является значением по умолчанию."""
    if tmax is None:
        raise ValueError("tmax must be explicit; the registered P1 grid is not a default")
    positions = np.array(positions, dtype=np.float64, copy=True)
    velocities = np.array(velocities, dtype=np.float64, copy=True)
    initial_positions = positions.copy()
    initial_velocities = velocities.copy()
    reservoir = np.asarray(reservoir, dtype=bool)
    if family == "frozen":
        active = min(8, len(positions))
    else:
        active = len(positions)
    acceleration, minimum = forces(positions, active, n_inf, r_inf)
    trace = [sample(positions, velocities, 0.0, minimum, n_inf, r_inf)]
    steps = round(tmax / dt)
    hits = 0
    interval_hits = 0
    status = "complete"
    every = max(1, round(0.02 / dt))
    started = time.perf_counter()
    for step in range(1, steps + 1):
        kick_drift(positions, velocities, acceleration, dt, active)
        acceleration, minimum = forces(positions, active, n_inf, r_inf)
        if minimum < UNRESOLVED or not np.isfinite(acceleration).all():
            status = "unresolved_close_pair"
            break
        kick(velocities, acceleration, dt, active)
        if family == "mobile_reservoir":
            if apply_reservoir_constraint(positions, velocities, reservoir, r_in, r_out):
                hits += 1
                interval_hits += 1
                acceleration, minimum = forces(positions, active, n_inf, r_inf)
        if step % every == 0:
            trace.append(
                sample(
                    positions,
                    velocities,
                    step * dt,
                    minimum,
                    n_inf,
                    r_inf,
                    constraint_hit=interval_hits > 0,
                    constraint_hits_in_interval=interval_hits,
                )
            )
            interval_hits = 0
    return {
        "family": family,
        "status": status,
        "nodes": len(positions),
        "active_nodes": active,
        "n_inf": n_inf,
        "constraint_hit_fraction": hits / steps if steps else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "trace": trace,
        "positions": positions,
        "velocities": velocities,
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
    }


def run(*, family, seed=1, omega=0.0, dt=0.002, tmax, r_inf=R_INF):
    positions, velocities, reservoir, n_inf = initial_state(seed, omega, family)
    if family == "isolated":
        n_inf = 0.0
    return integrate(
        positions, velocities, family=family, reservoir=reservoir, n_inf=n_inf, r_inf=r_inf, dt=dt, tmax=tmax
    )


def unconstrained_drift(trace, energy0, mass):
    max_energy = 0.0
    max_momentum = 0.0
    ref = trace[0]
    for row in trace[1:]:
        if row.get("constraint_hit"):
            ref = None
            continue
        if ref is None:
            ref = row
            continue
        if energy0:
            max_energy = max(max_energy, abs(row["energy"] - ref["energy"]) / abs(energy0))
        delta = np.linalg.norm(np.array(row["momentum"]) - np.array(ref["momentum"]))
        max_momentum = max(max_momentum, delta / mass)
    return max_energy, max_momentum


def assess(raw, *, family, seed, omega, dt, tmax):
    trace = raw["trace"]
    energy0 = trace[0]["energy"]
    mass = float(raw["active_nodes"])
    energy_error, momentum_error = unconstrained_drift(trace, energy0, mass)
    window = [row for row in trace if WINDOW[0] <= row["t"] <= WINDOW[1]]
    bounded = all(row["max_radius"] <= HOLD_RADIUS for row in trace)
    ratio = None
    if window and min(row["rms_radius"] for row in window) > 0:
        ratio = max(row["rms_radius"] for row in window) / min(row["rms_radius"] for row in window)
    hit_fraction = raw["constraint_hit_fraction"]
    saturated = family == "mobile_reservoir" and hit_fraction > SATURATION
    if saturated and raw["status"] == "complete":
        status = "constraint_saturated"
    else:
        status = raw["status"]
    energy_ok = energy_error < ENERGY_MAX
    momentum_ok = family == "frozen" or momentum_error < MOMENTUM_MAX
    numerics_ok = status in {"complete", "constraint_saturated"} and energy_ok and momentum_ok
    candidate = bool(
        status == "complete"
        and numerics_ok
        and not saturated
        and bounded
        and ratio is not None
        and ratio <= RMS_RATIO_MAX
    )
    first_density = trace[0]["belt_band_density"]
    if first_density:
        density_error = max(abs(row["belt_band_density"] / first_density - 1) for row in trace if row["belt_band_density"])
        r95_0 = trace[0]["belt_r95_density"]
        r95_error = max(abs(row["belt_r95_density"] / r95_0 - 1) for row in trace if row["belt_r95_density"]) if r95_0 else None
    else:
        density_error, r95_error = None, None
    return {
        **provenance(),
        "parameters": {
            "family": family,
            "seed": seed,
            "omega": omega,
            "dt": dt,
            "tmax": tmax,
            "R_in": R_IN,
            "R_out": R_OUT,
            "R_infinity": R_INF,
            "N_infinity": raw["n_inf"],
            "cavity_radius": R_CAVITY,
            "R0": R0,
        },
        "nodes": raw["nodes"],
        "active_nodes": raw["active_nodes"],
        "n_inf": raw["n_inf"],
        "status": status,
        "elapsed_seconds": raw["elapsed_seconds"],
        "constraint_hit_fraction": hit_fraction,
        "constraint_saturated": saturated,
        "quality": numerics_ok and status == "complete",
        "numerics_ok": numerics_ok,
        "energy_relative_drift": energy_error,
        "normalized_momentum_drift": float(momentum_error),
        "bounded": bounded,
        "radius_window_max_over_min": ratio,
        "candidate": candidate,
        "belt_band_density_relative_drift": density_error,
        "belt_r95_density_relative_drift": r95_error,
        "trace": trace,
    }


def run_key(family, seed, omega, dt):
    return f"{family}-seed{seed}-omega{omega:g}-dt{dt:g}"


def save_run(*, family, seed=1, omega=0.0, dt=DT, tmax=TMAX, folder="runs"):
    key = run_key(family, seed, omega, dt)
    path = OUT / folder / (key + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raw = run(family=family, seed=seed, omega=omega, dt=dt, tmax=tmax)
    result = assess(raw, family=family, seed=seed, omega=omega, dt=dt, tmax=tmax)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path.with_suffix(".npz"),
        initial_positions=raw["initial_positions"],
        initial_velocities=raw["initial_velocities"],
        final_positions=raw["positions"],
        final_velocities=raw["velocities"],
    )
    write_json(path, result)
    print(
        key,
        "N=",
        result["nodes"],
        "R_end=",
        round(result["trace"][-1]["rms_radius"], 5),
        "status=",
        result["status"],
        "quality=",
        result["quality"],
        "candidate=",
        result["candidate"],
        "seconds=",
        round(result["elapsed_seconds"], 1),
        flush=True,
    )
    return result


def t0_checks():
    rows = []
    ok = True
    for radius in (0.0, 0.25, 0.5, 0.75, 1.25, 2.0):
        force = shell_force(radius, 128)
        expected = 0.0 if radius < 1 else 1 / radius ** 2
        error = abs(force - expected)
        relative = None if expected == 0 else error / expected
        passed = error < 1e-8 if radius < 1 else relative < 1e-8
        ok = ok and passed
        rows.append({"id": "T0a", "r": radius, "force": force, "expected": expected, "passed": passed})
    unit = 1.0 / R_IN ** 2
    for frac in (0.0, 0.25, 0.5, 0.75):
        radius = frac * R_IN
        force = thick_shell_force(radius)
        passed = abs(force) < 1e-8 * unit
        ok = ok and passed
        rows.append({"id": "T0b", "r": radius, "force": force, "passed": passed})
    displaced = thick_shell_force(0.2)
    passed_c = abs(displaced) < 1e-8 * unit
    ok = ok and passed_c
    rows.append({"id": "T0c", "r": 0.2, "force": displaced, "passed": passed_c})
    probe = np.zeros((1, 3))
    for radius in (25.0, 40.0):
        probe[0, 0] = radius
        force, _ = forces(probe, 1, n_inf=float(N_INF_PROBE), r_inf=R_INF)
        expected = N_INF_PROBE / radius ** 2
        relative = abs(force[0, 0] - expected) / expected
        log_identity = math.log(abs(force[0, 0])) + 2.0 * math.log(radius) - math.log(N_INF_PROBE)
        passed = relative < 1e-8 and abs(log_identity) < 1e-10
        ok = ok and passed
        rows.append({"id": "T0d", "r": radius, "force": float(force[0, 0]), "expected": expected, "passed": passed})
    return ok, rows


def save_t0():
    passed, rows = t0_checks()
    result = {**provenance(), "stage": "T0", "passed": passed, "rows": rows}
    write_json(OUT / "t0.json", result)
    failed = [row for row in rows if not row["passed"]]
    print("T0:", "PASS" if passed else "FAIL", "rows:", len(rows), "failed:", len(failed), flush=True)
    return result


def run_controls():
    for omega in OMEGAS:
        save_run(family="isolated", seed=1, omega=omega, folder="controls")
    for seed, omega in itertools.product(SEEDS, OMEGAS):
        save_run(family="frozen", seed=seed, omega=omega, folder="controls")
    for omega in OMEGAS:
        save_run(family="mobile_open", seed=1, omega=omega, folder="controls")
    for omega in OMEGAS:
        save_run(family="mobile_reservoir", seed=1, omega=omega, dt=0.001, folder="controls")


def run_campaign():
    save_run(family="mobile_reservoir", seed=1, omega=0.0, folder="pilot")
    for seed, omega in itertools.product(SEEDS, OMEGAS):
        save_run(family="mobile_reservoir", seed=seed, omega=omega, folder="runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("t0", "pilot", "controls", "campaign", "all"), default="t0")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("t0", "all"):
        if not save_t0()["passed"]:
            raise SystemExit(1)
        if args.stage == "t0":
            return
    gate = json.loads((OUT / "t0.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise SystemExit("T0 не пройден")
    if args.stage == "pilot":
        save_run(family="mobile_reservoir", seed=1, omega=0.0, folder="pilot")
        return
    if args.stage in ("controls", "all"):
        run_controls()
        if args.stage == "controls":
            return
    if args.stage in ("campaign", "all"):
        run_campaign()


if __name__ == "__main__":
    main()
