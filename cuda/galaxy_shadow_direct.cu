// Direct N-Body pressure-shadow kernel.
//
// Research layer, not trusted formal logic. The trusted path stays in exact
// arithmetic; this file is the high-performance simulation backend.
//
// Design constraints:
// - SOA layout: x[], y[], z[], m[] for coalesced VRAM reads.
// - Direct O(N^2): no Barnes-Hut, no Particle-Mesh grouping.
// - Tiled shared memory: each block loads a particle tile into L1/shared memory.
// - Symplectic integration: Velocity Verlet helpers below.

#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

extern "C" __global__ void compute_forces_shadow_soa(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ mass,
    const int n,
    const double local_g,
    const double pressure_lambda,
    const double confinement_scale,
    const double softening,
    double* __restrict__ ax,
    double* __restrict__ ay,
    double* __restrict__ az
) {
    __shared__ double sx[TILE_SIZE];
    __shared__ double sy[TILE_SIZE];
    __shared__ double sz[TILE_SIZE];
    __shared__ double sm[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double xi = x[i];
    const double yi = y[i];
    const double zi = z[i];
    double aix = 0.0;
    double aiy = 0.0;
    double aiz = 0.0;

    for (int tile = 0; tile < n; tile += TILE_SIZE) {
        const int j = tile + threadIdx.x;
        if (threadIdx.x < TILE_SIZE && j < n) {
            sx[threadIdx.x] = x[j];
            sy[threadIdx.x] = y[j];
            sz[threadIdx.x] = z[j];
            sm[threadIdx.x] = mass[j];
        }
        __syncthreads();

        const int tile_count = min(TILE_SIZE, n - tile);
        for (int k = 0; k < tile_count; ++k) {
            const int j_global = tile + k;
            if (j_global == i) {
                continue;
            }
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz + softening;
            const double inv_r = rsqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double local = local_g * sm[k] * inv_r3;
            aix += local * dx;
            aiy += local * dy;
            aiz += local * dz;
        }
        __syncthreads();
    }

    // Isotropic external pressure confinement. psi is calibrated so the
    // peripheral term decays approximately as 1/r for large r.
    const double r2_center = xi * xi + yi * yi + zi * zi + softening;
    const double r_center = sqrt(r2_center);
    const double psi = 1.0 / (r_center + confinement_scale);
    aix += -pressure_lambda * xi * psi;
    aiy += -pressure_lambda * yi * psi;
    aiz += -pressure_lambda * zi * psi;

    ax[i] = aix;
    ay[i] = aiy;
    az[i] = aiz;
}

extern "C" __global__ void velocity_verlet_drift_soa(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    const double* __restrict__ ax,
    const double* __restrict__ ay,
    const double* __restrict__ az,
    const int n,
    const double dt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const double half_dt2 = 0.5 * dt * dt;
    x[i] += vx[i] * dt + ax[i] * half_dt2;
    y[i] += vy[i] * dt + ay[i] * half_dt2;
    z[i] += vz[i] * dt + az[i] * half_dt2;
    vx[i] += 0.5 * ax[i] * dt;
    vy[i] += 0.5 * ay[i] * dt;
    vz[i] += 0.5 * az[i] * dt;
}

extern "C" __global__ void velocity_verlet_kick_soa(
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    const double* __restrict__ ax,
    const double* __restrict__ ay,
    const double* __restrict__ az,
    const int n,
    const double dt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    vx[i] += 0.5 * ax[i] * dt;
    vy[i] += 0.5 * ay[i] * dt;
    vz[i] += 0.5 * az[i] * dt;
}
