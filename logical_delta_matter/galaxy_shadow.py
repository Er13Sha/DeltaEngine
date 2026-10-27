from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence


CUDA_SHADOW_KERNEL_SOURCE = r"""
extern "C" __global__
void shadow_forces(
    const double* positions,
    const double* radii,
    const int n,
    const double cosmic_pressure,
    const double shadow_coeff,
    const double softening,
    double* forces
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double xi = positions[3 * i + 0];
    double yi = positions[3 * i + 1];
    double zi = positions[3 * i + 2];
    double ri = radii[i];
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double dx = positions[3 * j + 0] - xi;
        double dy = positions[3 * j + 1] - yi;
        double dz = positions[3 * j + 2] - zi;
        double dist2 = dx * dx + dy * dy + dz * dz + softening;
        double inv_dist = rsqrt(dist2);
        double rj = radii[j];
        double magnitude = cosmic_pressure * shadow_coeff * ri * ri * rj * rj / dist2;
        fx += magnitude * dx * inv_dist;
        fy += magnitude * dy * inv_dist;
        fz += magnitude * dz * inv_dist;
    }
    forces[3 * i + 0] = fx;
    forces[3 * i + 1] = fy;
    forces[3 * i + 2] = fz;
}
"""


@dataclass(frozen=True, slots=True)
class ShadowConfig:
    cosmic_pressure: float = 1.0
    shadow_coeff: float = 1.0
    local_g: float = 1.0
    pressure_lambda: float = 0.0
    confinement_scale: float = 1.0
    softening: float = 1e-9


def direct_shadow_forces_cpu(
    positions: Sequence[tuple[float, float, float]],
    radii: Sequence[float],
    config: ShadowConfig,
) -> list[tuple[float, float, float]]:
    """Direct O(N^2) reference implementation of pressure-shadow interaction.

    This physics layer is intentionally separate from the trusted logical core:
    it may use floating point for simulation, while formal classification code
    remains exact and boolean.
    """

    if len(positions) != len(radii):
        raise ValueError("positions and radii must have the same length.")
    forces = [[0.0, 0.0, 0.0] for _ in positions]
    for i in range(len(positions)):
        xi, yi, zi = positions[i]
        for j in range(i + 1, len(positions)):
            xj, yj, zj = positions[j]
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            dist2 = dx * dx + dy * dy + dz * dz + config.softening
            inv_dist = 1.0 / sqrt(dist2)
            magnitude = config.cosmic_pressure * config.shadow_coeff * radii[i] ** 2 * radii[j] ** 2 / dist2
            fx = magnitude * dx * inv_dist
            fy = magnitude * dy * inv_dist
            fz = magnitude * dz * inv_dist
            forces[i][0] += fx
            forces[i][1] += fy
            forces[i][2] += fz
            forces[j][0] -= fx
            forces[j][1] -= fy
            forces[j][2] -= fz
    return [tuple(row) for row in forces]


def direct_shadow_acceleration_soa_cpu(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    mass: Sequence[float],
    config: ShadowConfig,
) -> tuple[list[float], list[float], list[float]]:
    """SOA CPU reference matching the CUDA backend contract."""

    n = len(x)
    if not (len(y) == len(z) == len(mass) == n):
        raise ValueError("x, y, z and mass arrays must have the same length.")
    ax = [0.0 for _ in range(n)]
    ay = [0.0 for _ in range(n)]
    az = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]
            dist2 = dx * dx + dy * dy + dz * dz + config.softening
            inv_dist = 1.0 / sqrt(dist2)
            inv_dist3 = inv_dist * inv_dist * inv_dist
            left = config.local_g * mass[j] * inv_dist3
            right = config.local_g * mass[i] * inv_dist3
            ax[i] += left * dx
            ay[i] += left * dy
            az[i] += left * dz
            ax[j] -= right * dx
            ay[j] -= right * dy
            az[j] -= right * dz

    if config.pressure_lambda:
        for i in range(n):
            radius = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i] + config.softening)
            psi = 1.0 / (radius + config.confinement_scale)
            ax[i] += -config.pressure_lambda * x[i] * psi
            ay[i] += -config.pressure_lambda * y[i] * psi
            az[i] += -config.pressure_lambda * z[i] * psi
    return ax, ay, az


def velocity_verlet_step_soa_cpu(
    x: list[float],
    y: list[float],
    z: list[float],
    vx: list[float],
    vy: list[float],
    vz: list[float],
    mass: Sequence[float],
    dt: float,
    config: ShadowConfig,
) -> tuple[list[float], list[float], list[float]]:
    """One Velocity Verlet step for small CPU reference runs."""

    ax, ay, az = direct_shadow_acceleration_soa_cpu(x, y, z, mass, config)
    half_dt = 0.5 * dt
    half_dt2 = 0.5 * dt * dt
    for i in range(len(x)):
        x[i] += vx[i] * dt + ax[i] * half_dt2
        y[i] += vy[i] * dt + ay[i] * half_dt2
        z[i] += vz[i] * dt + az[i] * half_dt2
        vx[i] += ax[i] * half_dt
        vy[i] += ay[i] * half_dt
        vz[i] += az[i] * half_dt

    new_ax, new_ay, new_az = direct_shadow_acceleration_soa_cpu(x, y, z, mass, config)
    for i in range(len(x)):
        vx[i] += new_ax[i] * half_dt
        vy[i] += new_ay[i] * half_dt
        vz[i] += new_az[i] * half_dt
    return new_ax, new_ay, new_az


def build_numba_shadow_kernel():
    """Return a Numba CUDA kernel when numba is installed, otherwise None."""

    try:
        from numba import cuda
    except ModuleNotFoundError:
        return None

    @cuda.jit
    def shadow_forces_numba(positions, radii, cosmic_pressure, shadow_coeff, softening, forces):
        i = cuda.grid(1)
        n = positions.shape[0]
        if i >= n:
            return
        fx = 0.0
        fy = 0.0
        fz = 0.0
        xi = positions[i, 0]
        yi = positions[i, 1]
        zi = positions[i, 2]
        ri = radii[i]
        for j in range(n):
            if i == j:
                continue
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dz = positions[j, 2] - zi
            dist2 = dx * dx + dy * dy + dz * dz + softening
            inv_dist = 1.0 / cuda.sqrt(dist2)
            magnitude = cosmic_pressure * shadow_coeff * ri * ri * radii[j] * radii[j] / dist2
            fx += magnitude * dx * inv_dist
            fy += magnitude * dy * inv_dist
            fz += magnitude * dz * inv_dist
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz

    return shadow_forces_numba


def cupy_raw_kernel_source() -> str:
    return CUDA_SHADOW_KERNEL_SOURCE


def build_cupy_raw_kernel():
    """Return cupy.RawKernel when CuPy is installed, otherwise None."""

    try:
        import cupy
    except ModuleNotFoundError:
        return None
    return cupy.RawKernel(CUDA_SHADOW_KERNEL_SOURCE, "shadow_forces")
