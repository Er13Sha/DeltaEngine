"""Бюджет сил песочницы SimFirstStableVersion. Протокол записан отдельно.

Формулы скопированы из galaxy_cuda_vispy.py. Сам файл песочницы не импортируется:
импорт numba.cuda в этом окружении роняет процесс.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE = Path(r"C:\IT\ML\SimFirstStableVersion\galaxy_cuda_vispy.py")
OUT = Path("data/experiments/field/galaxy_sandbox_force_budget_2026-10-27.json")

N = 128
SEEDS = (7, 8, 9)
DT = np.float64(0.012)
STEPS = 200
SAMPLE_STEPS = (0, 10, 50, 200)
BLOW = 1.0e5
DISK_RADIUS = 35.0
CHAMBER_RADIUS = 42.0


@dataclass
class Config:
    n_particles: int = N
    disk_radius: float = DISK_RADIUS
    disk_height: float = 0.35
    bulge_fraction: float = 0.16
    bulge_radius: float = 4.5
    arms: int = 4
    arm_twist: float = 5.4
    arm_spread: float = 0.23
    mass_min: float = 1.0
    mass_max: float = 10000.0
    mass_distribution: str = "log"
    giant_count: int = 200
    giant_mass_min: float = 100000.0
    giant_mass_max: float = 1000000.0
    external_universe_count: int = 512
    external_universe_radius: float = 3000.0
    external_universe_mass_min: float = 1000000.0
    external_universe_mass_max: float = 10000000.0
    body_radius_scale: float = 0.002
    body_radius_min: float = 0.0005
    solid_angle_pressure: float = 1e-5
    contact_stiffness: float = 10.0
    contact_damping: float = 1.0
    contact_projection: float = 0.8
    g_const: float = 1.0
    pressure_lambda: float = 0.045
    confinement_scale: float = 1.0
    softening: float = 0.08
    chamber_enabled: bool = True
    chamber_radius: float = CHAMBER_RADIUS
    chamber_stiffness: float = 10.0
    chamber_damping: float = 1.0
    chamber_restitution: float = 0.0
    dt: float = float(DT)
    seed: int = 7


def effective_giant_count(n: int, requested: int) -> int:
    requested = max(0, int(requested))
    if requested == 0 or n < 5:
        return 0
    diversity_cap = max(1, n // 16)
    normal_reserve_cap = max(0, n - 4)
    return min(requested, diversity_cap, normal_reserve_cap)


def body_radius(mass: np.ndarray, config: Config) -> np.ndarray:
    radius = config.body_radius_scale * np.cbrt(np.maximum(mass, 0.0))
    return np.maximum(radius, config.body_radius_min)


def generate_spiral(config: Config) -> dict[str, np.ndarray]:
    """Та же последовательность вызовов генератора, что в песочнице, затем float64."""
    rng = np.random.default_rng(config.seed)
    n = int(config.n_particles)
    n_bulge = max(1, int(n * config.bulge_fraction))
    n_disk = n - n_bulge
    disk_u = rng.random(n_disk, dtype=np.float32)
    r_disk = np.float32(config.disk_radius) * np.sqrt(disk_u)
    arm_id = rng.integers(0, max(1, config.arms), size=n_disk)
    base_theta = (2.0 * np.pi * arm_id / max(1, config.arms)).astype(np.float32)
    twist = np.float32(config.arm_twist) * (r_disk / np.float32(max(config.disk_radius, 1.0)))
    theta_disk = base_theta + twist + rng.normal(0.0, config.arm_spread, n_disk).astype(np.float32)
    x_disk = r_disk * np.cos(theta_disk)
    y_disk = r_disk * np.sin(theta_disk)
    z_disk = rng.normal(0.0, config.disk_height, n_disk).astype(np.float32)
    bulge_r = rng.gamma(shape=2.0, scale=config.bulge_radius / 3.0, size=n_bulge)
    bulge_r = np.minimum(bulge_r, config.bulge_radius * 2.5)
    cos_phi = rng.uniform(-1.0, 1.0, n_bulge)
    sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi * cos_phi))
    theta_bulge = rng.uniform(0.0, 2.0 * np.pi, n_bulge)
    x_bulge = bulge_r * sin_phi * np.cos(theta_bulge)
    y_bulge = bulge_r * sin_phi * np.sin(theta_bulge)
    z_bulge = bulge_r * cos_phi
    x = np.concatenate([x_bulge, x_disk]).astype(np.float64)
    y = np.concatenate([y_bulge, y_disk]).astype(np.float64)
    z = np.concatenate([z_bulge, z_disk]).astype(np.float64)
    log_min = math.log(config.mass_min)
    log_max = math.log(config.mass_max)
    mass = np.exp(rng.uniform(log_min, log_max, n)).astype(np.float64)
    giant_count = effective_giant_count(n, config.giant_count)
    normal_mask = np.ones(n, dtype=bool)
    if giant_count:
        giant_indices = rng.choice(n, size=giant_count, replace=False)
        normal_mask[giant_indices] = False
        mass[giant_indices] = rng.uniform(config.giant_mass_min, config.giant_mass_max, giant_count)
        mass[giant_indices[0]] = config.giant_mass_min
        if giant_count >= 2:
            mass[giant_indices[1]] = config.giant_mass_max
    normal_indices = np.flatnonzero(normal_mask)
    if normal_indices.size >= 1:
        mass[normal_indices[0]] = config.mass_min
    if normal_indices.size >= 2:
        mass[normal_indices[1]] = config.mass_max
    radius = body_radius(mass, config)
    x = x - np.average(x, weights=mass)
    y = y - np.average(y, weights=mass)
    z = z - np.average(z, weights=mass)
    zeros = np.zeros(n, dtype=np.float64)
    return {
        "x": x,
        "y": y,
        "z": z,
        "vx": zeros.copy(),
        "vy": zeros.copy(),
        "vz": zeros.copy(),
        "mass": mass.astype(np.float64),
        "radius": radius.astype(np.float64),
    }


def generate_external(config: Config) -> dict[str, np.ndarray]:
    count = max(0, int(config.external_universe_count))
    if count == 0:
        zeros = np.zeros(1)
        return {"x": zeros, "y": zeros.copy(), "z": zeros.copy(), "mass": zeros.copy(), "radius": zeros.copy(), "n": 0}
    rng = np.random.default_rng(config.seed + 10007)
    pair_count = count // 2
    extra_count = count - 2 * pair_count
    directions = rng.normal(0.0, 1.0, size=(pair_count + extra_count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.uniform(0.85, 1.15, pair_count + extra_count).astype(np.float32) * np.float32(config.external_universe_radius)
    masses = rng.uniform(config.external_universe_mass_min, config.external_universe_mass_max, pair_count + extra_count).astype(np.float64)
    x = np.empty(count)
    y = np.empty(count)
    z = np.empty(count)
    mass = np.empty(count)
    write = 0
    for i in range(pair_count):
        px = directions[i, 0] * radii[i]
        py = directions[i, 1] * radii[i]
        pz = directions[i, 2] * radii[i]
        x[write] = px
        y[write] = py
        z[write] = pz
        mass[write] = masses[i]
        write += 1
        x[write] = -px
        y[write] = -py
        z[write] = -pz
        mass[write] = masses[i]
        write += 1
    if extra_count:
        i = pair_count
        x[write] = directions[i, 0] * radii[i]
        y[write] = directions[i, 1] * radii[i]
        z[write] = directions[i, 2] * radii[i]
        mass[write] = masses[i]
    return {"x": x, "y": y, "z": z, "mass": mass, "radius": body_radius(mass, config), "n": count}


def _accumulate(dx: np.ndarray, dy: np.ndarray, dz: np.ndarray, mass_j: np.ndarray, radius_j: np.ndarray, g: float, solid: float, softening2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sep2 = dx * dx + dy * dy + dz * dz
    dist2 = sep2 + softening2
    dist = np.sqrt(dist2)
    inv = 1.0 / dist
    inv3 = inv * inv * inv
    scale = g * mass_j * inv3
    ax = np.sum(scale * dx, axis=1)
    ay = np.sum(scale * dy, axis=1)
    az = np.sum(scale * dz, axis=1)
    if solid != 0.0:
        omega = (2.0 * math.pi) * (1.0 - dist / np.sqrt(dist2 + radius_j * radius_j))
        angle = solid * omega * inv
        sx = np.sum(angle * dx, axis=1)
        sy = np.sum(angle * dy, axis=1)
        sz = np.sum(angle * dz, axis=1)
    else:
        sx = np.zeros(dx.shape[0])
        sy = np.zeros(dx.shape[0])
        sz = np.zeros(dx.shape[0])
    return ax, ay, az, sx, sy, sz


def accelerations(state: dict[str, np.ndarray], external: dict[str, np.ndarray], config: Config, *, g: float, solid: float, contact: bool, use_lambda: bool, use_external: bool, use_chamber: bool) -> dict[str, np.ndarray]:
    x = state["x"]
    y = state["y"]
    z = state["z"]
    n = x.shape[0]
    softening2 = config.softening * config.softening
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dz = z[None, :] - z[:, None]
    np.fill_diagonal(dx, 0.0)
    np.fill_diagonal(dy, 0.0)
    np.fill_diagonal(dz, 0.0)
    mass_j = np.broadcast_to(state["mass"][None, :], (n, n)).copy()
    radius_j = np.broadcast_to(state["radius"][None, :], (n, n)).copy()
    np.fill_diagonal(mass_j, 0.0)
    pair_x, pair_y, pair_z, ang_x, ang_y, ang_z = _accumulate(dx, dy, dz, mass_j, radius_j, g, solid, softening2)
    contact_x = np.zeros(n)
    contact_y = np.zeros(n)
    contact_z = np.zeros(n)
    if contact and config.contact_stiffness != 0.0:
        sep2 = dx * dx + dy * dy + dz * dz
        physical = np.sqrt(sep2 + 1e-12)
        reach = state["radius"][:, None] + state["radius"][None, :]
        hit = (physical < reach) & ~np.eye(n, dtype=bool)
        inv_p = np.zeros_like(physical)
        np.divide(1.0, physical, out=inv_p, where=hit)
        nx = dx * inv_p
        ny = dy * inv_p
        nz = dz * inv_p
        overlap = reach - physical
        rvx = state["vx"][None, :] - state["vx"][:, None]
        rvy = state["vy"][None, :] - state["vy"][:, None]
        rvz = state["vz"][None, :] - state["vz"][:, None]
        normal_speed = rvx * nx + rvy * ny + rvz * nz
        force = config.contact_stiffness * overlap
        closing = hit & (normal_speed < 0.0)
        force = np.where(closing, force + config.contact_damping * (-normal_speed), force)
        force = np.where(hit, force, 0.0)
        contact_scale = force / np.sqrt(state["mass"][:, None] ** 2 + 1e-20)
        contact_x = np.sum(-contact_scale * nx, axis=1)
        contact_y = np.sum(-contact_scale * ny, axis=1)
        contact_z = np.sum(-contact_scale * nz, axis=1)
    ext_x = np.zeros(n)
    ext_y = np.zeros(n)
    ext_z = np.zeros(n)
    ext_ang_x = np.zeros(n)
    ext_ang_y = np.zeros(n)
    ext_ang_z = np.zeros(n)
    if use_external and external["n"] > 0:
        ex = external["x"][None, :] - x[:, None]
        ey = external["y"][None, :] - y[:, None]
        ez = external["z"][None, :] - z[:, None]
        em = np.broadcast_to(external["mass"][None, :], ex.shape)
        er = np.broadcast_to(external["radius"][None, :], ex.shape)
        ext_x, ext_y, ext_z, ext_ang_x, ext_ang_y, ext_ang_z = _accumulate(ex, ey, ez, em, er, g, solid, softening2)
    lam_x = np.zeros(n)
    lam_y = np.zeros(n)
    lam_z = np.zeros(n)
    if use_lambda and config.pressure_lambda != 0.0:
        radius = np.sqrt(x * x + y * y + z * z + softening2)
        psi = 1.0 / (radius + config.confinement_scale)
        lam_x = -config.pressure_lambda * x * psi
        lam_y = -config.pressure_lambda * y * psi
        lam_z = -config.pressure_lambda * z * psi
    ch_x = np.zeros(n)
    ch_y = np.zeros(n)
    ch_z = np.zeros(n)
    if use_chamber and config.chamber_enabled:
        r = np.sqrt(x * x + y * y + z * z + 1e-12)
        inner = config.chamber_radius - state["radius"]
        outside = (inner > 0.0) & (r > inner)
        inv = np.zeros_like(r)
        np.divide(1.0, r, out=inv, where=outside)
        nx = x * inv
        ny = y * inv
        nz = z * inv
        overlap = r - inner
        radial_speed = state["vx"] * nx + state["vy"] * ny + state["vz"] * nz
        force = config.chamber_stiffness * overlap
        outward = outside & (radial_speed > 0.0)
        force = np.where(outward, force + config.chamber_damping * radial_speed, force)
        force = np.where(outside, force, 0.0)
        ch_x = -force * nx
        ch_y = -force * ny
        ch_z = -force * nz
    parts = {
        "pair": (pair_x, pair_y, pair_z),
        "solid_angle": (ang_x + ext_ang_x, ang_y + ext_ang_y, ang_z + ext_ang_z),
        "contact": (contact_x, contact_y, contact_z),
        "external_newton": (ext_x, ext_y, ext_z),
        "lambda": (lam_x, lam_y, lam_z),
        "chamber": (ch_x, ch_y, ch_z),
    }
    total = [np.zeros(n), np.zeros(n), np.zeros(n)]
    for vectors in parts.values():
        total[0] = total[0] + vectors[0]
        total[1] = total[1] + vectors[1]
        total[2] = total[2] + vectors[2]
    parts["total"] = (total[0], total[1], total[2])
    return parts


def radial_means(state: dict[str, np.ndarray], parts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, float]:
    r = np.sqrt(state["x"] ** 2 + state["y"] ** 2 + state["z"] ** 2)
    inv = np.zeros_like(r)
    np.divide(1.0, r, out=inv, where=r > 0.0)
    rx = state["x"] * inv
    ry = state["y"] * inv
    rz = state["z"] * inv
    out: dict[str, float] = {}
    for name, (ax, ay, az) in parts.items():
        out[name] = float(np.mean(ax * rx + ay * ry + az * rz))
    return out


def rms_radius(state: dict[str, np.ndarray]) -> float:
    r2 = state["x"] ** 2 + state["y"] ** 2 + state["z"] ** 2
    return float(np.sqrt(np.average(r2, weights=state["mass"])))


def max_radius(state: dict[str, np.ndarray]) -> float:
    return float(np.max(np.sqrt(state["x"] ** 2 + state["y"] ** 2 + state["z"] ** 2)))


def contact_shift(state: dict[str, np.ndarray], config: Config) -> None:
    if config.contact_projection <= 0.0:
        return
    x = state["x"]
    y = state["y"]
    z = state["z"]
    n = x.shape[0]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dz = z[None, :] - z[:, None]
    sep2 = dx * dx + dy * dy + dz * dz
    eye = np.eye(n, dtype=bool)
    coincident = sep2 < 1e-18
    upper = coincident & np.triu(np.ones((n, n), dtype=bool), k=1)
    dx = dx.copy()
    dx[upper] = 1e-6
    dx[upper.T] = -1e-6
    sep2 = dx * dx + dy * dy + dz * dz
    dist = np.sqrt(sep2)
    reach = state["radius"][:, None] + state["radius"][None, :]
    hit = (dist < reach) & ~eye
    inv = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv, where=hit)
    overlap = reach - dist
    share = state["mass"][None, :] / (state["mass"][:, None] + state["mass"][None, :] + 1e-20)
    scale = np.where(hit, config.contact_projection * overlap * share * inv, 0.0)
    state["x"] = x + np.sum(-dx * scale, axis=1)
    state["y"] = y + np.sum(-dy * scale, axis=1)
    state["z"] = z + np.sum(-dz * scale, axis=1)


def chamber_project(state: dict[str, np.ndarray], config: Config) -> None:
    if not config.chamber_enabled:
        return
    r = np.sqrt(state["x"] ** 2 + state["y"] ** 2 + state["z"] ** 2 + 1e-12)
    inner = config.chamber_radius - state["radius"]
    outside = (inner > 0.0) & (r > inner)
    if not np.any(outside):
        return
    inv = 1.0 / r
    nx = state["x"] * inv
    ny = state["y"] * inv
    nz = state["z"] * inv
    state["x"] = np.where(outside, nx * inner, state["x"])
    state["y"] = np.where(outside, ny * inner, state["y"])
    state["z"] = np.where(outside, nz * inner, state["z"])
    radial_speed = state["vx"] * nx + state["vy"] * ny + state["vz"] * nz
    leaving = outside & (radial_speed > 0.0)
    scale = (1.0 + config.chamber_restitution) * radial_speed
    state["vx"] = np.where(leaving, state["vx"] - scale * nx, state["vx"])
    state["vy"] = np.where(leaving, state["vy"] - scale * ny, state["vy"])
    state["vz"] = np.where(leaving, state["vz"] - scale * nz, state["vz"])


def family_flags(name: str) -> dict[str, bool | float]:
    if name == "full":
        return {"g": 1.0, "solid": 1e-5, "contact": True, "use_lambda": True, "use_external": True, "use_chamber": True}
    if name == "no_holder":
        return {"g": 1.0, "solid": 1e-5, "contact": True, "use_lambda": False, "use_external": True, "use_chamber": False}
    if name == "pairs_only":
        return {"g": 1.0, "solid": 0.0, "contact": False, "use_lambda": False, "use_external": False, "use_chamber": False}
    if name == "lambda_only":
        return {"g": 0.0, "solid": 0.0, "contact": False, "use_lambda": True, "use_external": False, "use_chamber": True}
    raise KeyError(name)


def integrate(state: dict[str, np.ndarray], external: dict[str, np.ndarray], config: Config, name: str) -> dict[str, object]:
    flags = family_flags(name)
    samples = []
    blow = False
    accel = accelerations(state, external, config, **flags)
    for step in range(STEPS + 1):
        if step in SAMPLE_STEPS:
            samples.append({"step": step, "t": step * float(DT), "rms": rms_radius(state), "max_r": max_radius(state)})
            if samples[-1]["max_r"] > BLOW:
                blow = True
                break
        if step == STEPS:
            break
        dt = float(DT)
        ax, ay, az = accel["total"]
        state["x"] = state["x"] + state["vx"] * dt + ax * (0.5 * dt * dt)
        state["y"] = state["y"] + state["vy"] * dt + ay * (0.5 * dt * dt)
        state["z"] = state["z"] + state["vz"] * dt + az * (0.5 * dt * dt)
        state["vx"] = state["vx"] + ax * (0.5 * dt)
        state["vy"] = state["vy"] + ay * (0.5 * dt)
        state["vz"] = state["vz"] + az * (0.5 * dt)
        if flags["contact"]:
            contact_shift(state, config)
        if flags["use_chamber"]:
            chamber_project(state, config)
        accel = accelerations(state, external, config, **flags)
        ax, ay, az = accel["total"]
        state["vx"] = state["vx"] + ax * (0.5 * dt)
        state["vy"] = state["vy"] + ay * (0.5 * dt)
        state["vz"] = state["vz"] + az * (0.5 * dt)
    return {"family": name, "numerical_blowup": blow, "samples": samples}


def copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in state.items()}


def main() -> None:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    rows = []
    for seed in SEEDS:
        config = Config(seed=seed)
        state0 = generate_spiral(config)
        external = generate_external(config)
        parts = accelerations(
            state0,
            external,
            config,
            g=config.g_const,
            solid=config.solid_angle_pressure,
            contact=True,
            use_lambda=True,
            use_external=True,
            use_chamber=True,
        )
        means = radial_means(state0, parts)
        dominant = max((abs(means[name]), name) for name in ("pair", "solid_angle", "contact", "external_newton", "lambda", "chamber"))
        budget = {
            "seed": seed,
            "rms0": rms_radius(state0),
            "max_r0": max_radius(state0),
            "giants": int(np.count_nonzero(state0["mass"] > config.mass_max)),
            "radial_mean": means,
            "dominant": dominant[1],
            "dominant_abs": dominant[0],
        }
        families = {}
        for name in ("full", "no_holder", "pairs_only", "lambda_only"):
            families[name] = integrate(copy_state(state0), external, config, name)
        rows.append({"budget": budget, "families": families})
        print(
            f"seed={seed} rms0={budget['rms0']:.4f} dominant={budget['dominant']} "
            f"pair={means['pair']:.6g} lambda={means['lambda']:.6g} chamber={means['chamber']:.6g} "
            f"external={means['external_newton']:.6g}"
        )
        for name, run in families.items():
            last = run["samples"][-1]
            print(f"  {name}: step={last['step']} rms={last['rms']:.6g} max_r={last['max_r']:.6g} blow={run['numerical_blowup']}")
    payload = {
        "scope": "model",
        "protocol": "docs/galaxy-sandbox-force-budget-protocol-2026-10-27.md",
        "source": str(SOURCE),
        "source_sha256": source_hash,
        "arithmetic": "float64",
        "note": "CPU transcription of the sandbox formulas. Not a CUDA replay and not an E-COMP run.",
        "seeds": list(SEEDS),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
