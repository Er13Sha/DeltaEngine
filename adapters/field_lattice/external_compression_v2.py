"""E-COMP v2: полость в равновесном базовом поле.

Протокол: docs/external-compression-protocol-v2.md.
Коэффициент полости выводится из парной силы, а не подбирается.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/field/external_compression_v2"
PROTOCOL = ROOT / "data/experiments/field/external_compression_protocol_v2.json"
REGISTRATION = PROTOCOL.with_name("external_compression_protocol_v2.registration.json")

B_CUBE = 0.25 + 1 / (4 * math.sqrt(2)) + 1 / (12 * math.sqrt(3))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def provenance():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for item in registration["files"]:
        digest = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Зарегистрированный протокол изменился: " + item["path"])
    return {
        "scope": "model",
        "protocol_id": "E-COMP-v2",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def spring(rho):
    return (4 * math.pi / 3) * rho


def cube_scale(rho):
    if rho <= 0:
        raise ValueError("У равновесия при нулевой плотности нет конечного корня")
    return (B_CUBE / spring(rho)) ** (1 / 3)


def cube_radius(rho):
    return cube_scale(rho) * math.sqrt(3)


def cavity_radius(rho):
    return (6 / (math.pi * rho)) ** (1 / 3)


def cube_positions(scale):
    return np.array(list(itertools.product((-scale, scale), repeat=3)), dtype=np.float64)


def pair_forces_numpy(positions):
    delta = positions[:, None, :] - positions[None, :, :]
    distance_squared = np.sum(delta * delta, axis=2)
    np.fill_diagonal(distance_squared, np.inf)
    inverse = 1 / (distance_squared * np.sqrt(distance_squared))
    return np.sum(delta * inverse[:, :, None], axis=1), float(np.sqrt(distance_squared.min()))


def cavity_forces_group(positions, rho):
    pair, minimum = pair_forces_numpy(positions)
    center = positions.mean(axis=0)
    return pair - spring(rho) * (positions - center), minimum


def rms_radius(positions):
    center = positions.mean(axis=0)
    offset = positions - center
    return float(np.sqrt(np.mean(np.sum(offset * offset, axis=1))))


def group_energy(positions, velocities, rho):
    total = 0.0
    for i in range(len(positions)):
        for j in range(i):
            total += 1 / np.linalg.norm(positions[i] - positions[j])
    center = positions.mean(axis=0)
    total += 0.5 * spring(rho) * np.sum((positions - center) ** 2)
    total += 0.5 * np.sum(velocities * velocities)
    return float(total)


def ball_quadrature(radius, rho, shells=400, order=64, ball_radius=2.0):
    from adapters.field_lattice.external_compression import shell_force

    samples = np.linspace(0, ball_radius, shells)
    step = samples[1] - samples[0]
    total = 0.0
    for sample in samples[1:]:
        total += shell_force(radius, order, shell_radius=float(sample), source=rho * 4 * math.pi * sample * sample * step)
    return total


def integrate_group(scale, rho, dt, tmax, stretch=1.0):
    positions = cube_positions(scale) * stretch
    velocities = np.zeros_like(positions)
    acceleration, minimum = cavity_forces_group(positions, rho)
    every = round(0.02 / dt)
    trace = []
    steps = round(tmax / dt)
    status = "complete"
    energy0 = None
    for step in range(steps + 1):
        if step % every == 0:
            energy = group_energy(positions, velocities, rho)
            energy0 = energy if energy0 is None else energy0
            trace.append({"t": step * dt, "rms_radius": rms_radius(positions), "energy": energy, "minimum_pair_distance": minimum})
        if step == steps:
            break
        velocities += 0.5 * dt * acceleration
        positions += dt * velocities
        acceleration, minimum = cavity_forces_group(positions, rho)
        velocities += 0.5 * dt * acceleration
        if minimum < 0.05 or not np.isfinite(acceleration).all():
            status = "unresolved_close_pair"
            break
    drift = max(abs(row["energy"] / energy0 - 1) for row in trace)
    return {"status": status, "energy_relative_drift": drift, "trace": trace, "rho": rho, "stretch": stretch, "dt": dt, "scale": scale}


def stage_a():
    protocol = load_protocol()["stage_A"]
    runs = []
    for rho in (1.0, 0.5):
        scale = cube_scale(rho)
        runs.append(integrate_group(scale, rho, protocol["dt"], protocol["tmax_bound"], 1.0))
        runs.append(integrate_group(scale, rho, protocol["dt"], protocol["tmax_bound"], protocol["stretch"]))
    runs.append(integrate_group(cube_scale(1.0), 0.0, protocol["dt"], protocol["tmax_free"], 1.0))
    runs.append(integrate_group(cube_scale(1.0), 1.0, protocol["dt"] / 2, protocol["tmax_bound"], 1.0))
    result = {**provenance(), "stage": "A", "predicted_radius": {str(rho): cube_radius(rho) for rho in (1.0, 0.5)}, "runs": runs}
    result["decision"] = evaluate_stage_a(result, load_protocol())
    write_json(OUT / "stage_a.json", result)
    print("stage A", result["decision"]["passed"], flush=True)
    return result


def evaluate_stage_a(result, protocol):
    rules = protocol["stage_A"]
    runs = result["runs"]
    stationary = {row["rho"]: row for row in runs if row["stretch"] == 1 and row["dt"] == rules["dt"] and row["rho"] > 0}
    perturbed = {row["rho"]: row for row in runs if abs(row["stretch"] - rules["stretch"]) < 1e-12}
    free = next(row for row in runs if row["rho"] == 0)
    fine = next(row for row in runs if row["rho"] == 1 and row["dt"] == rules["dt"] / 2)
    predicted = {float(key): value for key, value in result["predicted_radius"].items()}
    checks = []

    def add(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for rho, run in stationary.items():
        deviation = max(abs(row["rms_radius"] / predicted[rho] - 1) for row in run["trace"])
        add(f"stationary-{rho:g}", run["status"] == "complete" and run["energy_relative_drift"] < rules["energy_relative_drift_max"] and deviation < rules["stationary_relative_tolerance"], deviation)
    for rho, run in perturbed.items():
        radii = [row["rms_radius"] / predicted[rho] for row in run["trace"]]
        late = [row["rms_radius"] for row in run["trace"] if row["t"] >= rules["late_time"]]
        band = rules["perturbed_rms_band"]
        mean_error = abs(sum(late) / len(late) / predicted[rho] - 1)
        add(
            f"perturbed-{rho:g}",
            run["status"] == "complete" and run["energy_relative_drift"] < rules["energy_relative_drift_max"] and min(radii) > band[0] and max(radii) < band[1] and mean_error < rules["perturbed_late_mean_tolerance"],
            {"mean_error": mean_error, "min": min(radii), "max": max(radii)},
        )
    ratio = (sum(row["rms_radius"] for row in stationary[0.5]["trace"] if row["t"] >= rules["late_time"]) / sum(1 for row in stationary[0.5]["trace"] if row["t"] >= rules["late_time"])) / (
        sum(row["rms_radius"] for row in stationary[1.0]["trace"] if row["t"] >= rules["late_time"]) / sum(1 for row in stationary[1.0]["trace"] if row["t"] >= rules["late_time"])
    )
    add("scale", abs(ratio / rules["scale_ratio"] - 1) < rules["scale_relative_tolerance"], ratio)
    coarse = {round(row["t"], 6): row["rms_radius"] for row in stationary[1.0]["trace"]}
    halved = {round(row["t"], 6): row["rms_radius"] for row in fine["trace"]}
    times = sorted(set(coarse) & set(halved))
    step_error = max(abs(coarse[time] - halved[time]) / max(abs(coarse[time]), abs(halved[time]), 1e-12) for time in times)
    add("half-step", step_error < rules["half_step_relative_tolerance"], step_error)
    expansion = free["trace"][-1]["rms_radius"] / free["trace"][0]["rms_radius"]
    add("free", free["status"] == "complete" and expansion >= rules["free_expansion_min"], expansion)
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def lattice(spacing, extent):
    count = int(math.floor(extent / spacing))
    points = []
    for index in itertools.product(range(-count, count + 1), repeat=3):
        point = spacing * np.array(index, dtype=np.float64)
        if point.dot(point) <= extent * extent + 1e-12:
            points.append(point)
    return np.vstack(points)


def fibonacci(count, radius):
    points = np.empty((count, 3))
    golden = math.pi * (1 + 5 ** 0.5)
    for index in range(count):
        height = 1 - 2 * (index + 0.5) / count
        ring = math.sqrt(max(0.0, 1 - height * height))
        angle = golden * index
        points[index] = (radius * ring * math.cos(angle), radius * height, radius * ring * math.sin(angle))
    return points


def medium_positions(spacing, extent):
    rho = spacing ** -3
    hole = cavity_radius(rho)
    points = lattice(spacing, extent)
    distance = np.linalg.norm(points, axis=1)
    inner = points[distance < hole - 1e-9]
    outer = points[distance >= hole - 1e-9]
    moved = fibonacci(len(inner), hole + 2 * spacing) if len(inner) else np.zeros((0, 3))
    environment = np.vstack((outer, moved)) if len(moved) else outer
    group = cube_positions(cube_scale(rho))
    return np.vstack((group, environment)), hole, rho


@njit(cache=True, parallel=True)
def medium_forces(positions, spring_constant):
    count = len(positions)
    result = np.empty((count, 3))
    distances = np.empty(count)
    for i in prange(count):
        fx, fy, fz, nearest = 0.0, 0.0, 0.0, math.inf
        xi, yi, zi = positions[i, 0], positions[i, 1], positions[i, 2]
        for j in range(count):
            if i == j:
                continue
            dx, dy, dz = xi - positions[j, 0], yi - positions[j, 1], zi - positions[j, 2]
            squared = dx * dx + dy * dy + dz * dz
            nearest = min(nearest, squared)
            inverse = 1 / (squared * math.sqrt(squared))
            fx += dx * inverse
            fy += dy * inverse
            fz += dz * inverse
        result[i, 0] = fx - spring_constant * xi
        result[i, 1] = fy - spring_constant * yi
        result[i, 2] = fz - spring_constant * zi
        distances[i] = math.sqrt(nearest)
    return result, float(np.min(distances))


@njit(cache=True)
def medium_potential(positions, spring_constant):
    energy = 0.0
    count = len(positions)
    for i in range(count):
        energy += 0.5 * spring_constant * (positions[i, 0] ** 2 + positions[i, 1] ** 2 + positions[i, 2] ** 2)
        for j in range(i):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            energy += 1 / math.sqrt(dx * dx + dy * dy + dz * dz)
    return energy


def lattice_residual(spacing, extent):
    positions = lattice(spacing, extent)
    rho = spacing ** -3
    forces, _ = medium_forces(positions, spring(rho))
    distance = np.linalg.norm(positions, axis=1)
    inner = forces[distance < extent / 2]
    return {"spacing": spacing, "nodes": len(positions), "max_force": float(np.linalg.norm(inner, axis=1).max()), "threshold": 0.1 * spring(rho) * spacing}


def sample_medium(positions, velocities, time, minimum, rho):
    group = positions[:8]
    radius = rms_radius(group)
    predicted = cube_radius(rho)
    intruders = int(np.sum(np.linalg.norm(positions[8:], axis=1) < predicted))
    energy = float(medium_potential(positions, spring(rho)) + 0.5 * np.sum(velocities * velocities))
    return {"t": time, "rms_radius": radius, "intruders": intruders, "energy": energy, "minimum_pair_distance": minimum}


def integrate_medium(positions, rho, dt, tmax):
    velocities = np.zeros_like(positions)
    acceleration, minimum = medium_forces(positions, spring(rho))
    every = round(0.02 / dt)
    trace = [sample_medium(positions, velocities, 0.0, minimum, rho)]
    status = "complete"
    for step in range(1, round(tmax / dt) + 1):
        velocities += 0.5 * dt * acceleration
        positions += dt * velocities
        acceleration, minimum = medium_forces(positions, spring(rho))
        velocities += 0.5 * dt * acceleration
        if minimum < 0.05 or not np.isfinite(acceleration).all():
            status = "unresolved_close_pair"
            trace.append(sample_medium(positions, velocities, step * dt, minimum, rho))
            break
        if step % every == 0:
            trace.append(sample_medium(positions, velocities, step * dt, minimum, rho))
    drift = max(abs(row["energy"] / trace[0]["energy"] - 1) for row in trace)
    return {"status": status, "energy_relative_drift": drift, "trace": trace, "nodes": len(positions)}


def stage_b():
    protocol = load_protocol()["stage_B"]
    controls = []
    runs = []
    for spacing in protocol["spacings"]:
        control = lattice_residual(spacing, protocol["extent"])
        control["passed"] = control["max_force"] < control["threshold"]
        controls.append(control)
        print("B0", control, flush=True)
        if not control["passed"]:
            continue
        rho = spacing ** -3
        positions, hole, _ = medium_positions(spacing, protocol["extent"])
        outcome = integrate_medium(positions, rho, protocol["dt"], protocol["tmax"])
        outcome.update({"spacing": spacing, "rho": rho, "hole_radius": hole, "dt": protocol["dt"], "predicted_radius": cube_radius(rho)})
        runs.append(outcome)
        print("stage B", spacing, outcome["status"], "R=", round(outcome["trace"][-1]["rms_radius"], 4), flush=True)
        if spacing == 1:
            fine_positions, _, _ = medium_positions(spacing, protocol["extent"])
            fine = integrate_medium(fine_positions, rho, protocol["dt"] / 2, protocol["tmax"])
            fine.update({"spacing": spacing, "rho": rho, "dt": protocol["dt"] / 2, "predicted_radius": cube_radius(rho)})
            runs.append(fine)
    result = {**provenance(), "stage": "B", "controls": controls, "runs": runs}
    result["decision"] = evaluate_stage_b(result, load_protocol())
    write_json(OUT / "stage_b.json", result)
    print("stage B decision", result["decision"]["passed"], result["decision"]["status"], flush=True)
    return result


def evaluate_stage_b(result, protocol):
    rules = protocol["stage_B"]
    if not any(control["passed"] for control in result["controls"]):
        return {"passed": False, "status": "numerically_unresolved", "checks": []}
    usable = [run for run in result["runs"] if run["dt"] == rules["dt"]]
    checks = []

    def add(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    late_mean = {}
    for run in usable:
        predicted = run["predicted_radius"]
        late = [row for row in run["trace"] if row["t"] >= rules["late_time"]]
        mean = sum(row["rms_radius"] for row in late) / len(late)
        intruders = sum(row["intruders"] for row in late) / len(late)
        peak = max(row["rms_radius"] for row in run["trace"])
        late_mean[run["spacing"]] = mean
        quality = run["status"] == "complete" and run["energy_relative_drift"] < rules["energy_relative_drift_max"]
        add(f"size-{run['spacing']}", quality and abs(mean / predicted - 1) < rules["mean_radius_tolerance"] and peak < rules["max_radius_factor"] * predicted, {"mean": mean, "predicted": predicted, "peak": peak})
        add(f"cavity-{run['spacing']}", quality and intruders <= rules["intruder_mean_max"], intruders)
    if 1 in late_mean and 0.5 in late_mean:
        ratio = late_mean[1] / late_mean[0.5]
        add("scale", abs(ratio / rules["spacing_ratio"] - 1) < rules["spacing_ratio_tolerance"], ratio)
    coarse = next((run for run in usable if run["spacing"] == 1), None)
    fine = next((run for run in result["runs"] if run["spacing"] == 1 and run["dt"] == rules["dt"] / 2), None)
    if coarse and fine and coarse["status"] == "complete" and fine["status"] == "complete":
        left = {round(row["t"], 6): row["rms_radius"] for row in coarse["trace"]}
        right = {round(row["t"], 6): row["rms_radius"] for row in fine["trace"]}
        times = sorted(set(left) & set(right))
        error = max(abs(left[time] - right[time]) / max(abs(left[time]), 1e-12) for time in times)
        add("half-step", error < rules["half_step_relative_tolerance"], error)
    return {"passed": all(item["passed"] for item in checks) and len(checks) > 0, "status": "evaluated", "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    provenance()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "a":
        stage_a()
    else:
        stage_b()


if __name__ == "__main__":
    main()
