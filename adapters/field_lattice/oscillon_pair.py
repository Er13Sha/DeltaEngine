"""E3-волна, этап 2: сила между двумя частицами поля (протокол PAIR_PROTOCOL записан до расчёта).

Поле то же, что в oscillon.py: φ_tt = ∇²φ − V'(φ), V = (φ² − 1)²/8, c = F = 1, свободных
параметров нет. Две частицы (осциллоны) ставятся на ось z на расстоянии d, в фазе или в
противофазе, и отпускаются из покоя. Расчёт в 3D с осевой симметрией: координаты (ρ, z),
φ_tt = φ_ρρ + φ_ρ/ρ + φ_zz − V'(φ), схема «чехарда», поглощающий слой по краям.

Измеряется расстояние D(t) между центрами энергии половин z > 0 и z < 0, по нему —
относительное ускорение a, сила на каждую частицу f = E·a/2 (масса = энергия при c = 1).
Безразмерная сила взаимодействия α_модель = g/S, где g — коэффициент закона
f(d) = g·(1 + κd)·e^(−κd)/d² (сила между источниками поля с частотой ниже F), а S = E/ω —
действие частицы за период. Нелинейность λ сокращается: α_модель — число самого поля.
Предсказание до расчёта: радиус силы задаётся частотой частицы, κ = √(F² − ω²).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from numba import njit, prange

from adapters.field_lattice.oscillon import OSC_PROTOCOL, _evolve

PAIR_PROTOCOL = {
    "format": "delta-protocol@0.1",
    "track": "E",
    "stage": "E3-волна, этап 2: сила между двумя частицами поля",
    "field": OSC_PROTOCOL["equation"],
    "postulates": "P1–P4 из oscillon_protocol.json, новых нет",
    "free_dimensionless_parameters": 0,
    "particle": {"source": "радиальный расчёт R₀ = 4, установившийся к t = 2000", "phase0": "максимум φ(0, t)",
                 "antiphase": "тот же осциллон через полпериода"},
    "geometry": "3D, осевая симметрия (ρ, z)",
    "numerics": {"h": 0.1, "dt": 0.05, "rho_max": 50.0, "z_max": 70.0, "absorb_width": 12.0, "t_max": 150.0},
    "separations": [8, 10, 12, 14, 16],
    "control": {"separation": 32, "expectation": "|a| много меньше, чем при d = 16"},
    "phases": ["in", "anti"],
    "measurement": "D(t) по центрам энергии половин; a — из подгонки D = D₀ + v₀t + a·t²/2 на [0, t_max]; f = E·a/2",
    "law_fit": "f(d) = g·(1 + κd)·e^(−κd)/d² по разделённым d, отдельно для фаз",
    "prediction_before_run": "κ = √(1 − ω²) при ω = частота частицы; в фазе — притяжение, в противофазе — отталкивание",
    "alpha_model": "g/S, S = E/ω",
    "comparison_after_run": {
        "primary": "ядерная сила: α_N = 0,355 (дейтрон, track-e-frequency.md) — сила короткодействующая",
        "secondary": "электричество: α = 1/137 — приводится для полноты, сила модели не дальнодействующая",
    },
}


def protocol_hash() -> str:
    return "sha256:" + hashlib.sha256(json.dumps(PAIR_PROTOCOL, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare_particle() -> dict:
    """Установившийся осциллон из радиального расчёта: профили δ = φ + 1 и π в фазе и противофазе."""
    num = OSC_PROTOCOL["numerics"]
    dr, dt = num["dr"], num["dt"]
    n = int(num["r_max"] / dr)
    r = (np.arange(n) + 0.5) * dr
    phi = -1 + 2 * np.exp(-(r / 4.0) ** 2)
    pi = np.zeros(n)
    _evolve(phi, pi, dr, dt, int(2000 / dt), num["absorb_from"], int(num["core_radius"] / dr), 1000)
    # дойти до максимума φ(0) (фаза 0), затем записать профиль через полпериода
    trace = []
    for _ in range(800):
        _evolve(phi, pi, dr, dt, 1, num["absorb_from"], 2, 1)
        trace.append((phi[0], phi.copy(), pi.copy()))
    centers = np.array([t[0] for t in trace])
    peaks = [i for i in range(1, len(centers) - 1) if centers[i - 1] < centers[i] >= centers[i + 1]]
    i0, i1 = peaks[0], peaks[1]
    period = (i1 - i0) * dt
    half = i0 + int(round((i1 - i0) / 2))
    core = r < 15
    e_density = 0.5 * trace[i0][2] ** 2 + 0.5 * np.gradient(trace[i0][1], dr) ** 2 + (trace[i0][1] ** 2 - 1) ** 2 / 8
    energy = float(np.sum(e_density[core] * 4 * math.pi * r[core] ** 2) * dr)
    omega = 2 * math.pi / period
    return {"r": r, "in": (trace[i0][1] + 1, trace[i0][2]), "anti": (trace[half][1] + 1, trace[half][2]),
            "omega": omega, "energy": energy, "action": energy / omega}


@njit(cache=True, parallel=True)
def _step_axisym(phi, pi, h, dt, damp, steps, n_rho, n_z, z0_index, out):
    """Чехарда в (ρ, z); out[k] = (центр энергии z > 0, центр энергии z < 0)."""
    lap = np.empty_like(phi)
    for step in range(steps):
        for i in prange(n_rho):
            rho = (i + 0.5) * h
            for j in range(n_z):
                c = phi[i, j]
                rl = phi[i - 1, j] if i > 0 else phi[0, j]
                rr = phi[i + 1, j] if i < n_rho - 1 else -1.0
                zl = phi[i, j - 1] if j > 0 else -1.0
                zr = phi[i, j + 1] if j < n_z - 1 else -1.0
                lap[i, j] = (rr - 2 * c + rl + zr - 2 * c + zl) / (h * h) + (rr - rl) / (2 * h * rho)
        for i in prange(n_rho):
            for j in range(n_z):
                c = phi[i, j]
                pi[i, j] += dt * (lap[i, j] - 0.5 * c * (c * c - 1) - damp[i, j] * pi[i, j])
                phi[i, j] += dt * pi[i, j]
        # центры энергии половин
        wu = 0.0
        zu = 0.0
        wd = 0.0
        zd = 0.0
        for i in range(n_rho - 1):
            rho = (i + 0.5) * h
            for j in range(n_z - 1):
                c = phi[i, j]
                gr = (phi[i + 1, j] - c) / h
                gz = (phi[i, j + 1] - c) / h
                e = (0.5 * pi[i, j] ** 2 + 0.5 * (gr * gr + gz * gz) + (c * c - 1) ** 2 / 8) * rho
                z = (j - z0_index) * h
                if z > 0:
                    wu += e
                    zu += e * z
                else:
                    wd += e
                    zd += e * z
        out[step, 0] = zu / wu
        out[step, 1] = zd / wd


def pair_run(particle: dict, d: float, phase: str) -> dict:
    num = PAIR_PROTOCOL["numerics"]
    h, dt = num["h"], num["dt"]
    n_rho = int(num["rho_max"] / h)
    n_z = int(2 * num["z_max"] / h)
    rho = (np.arange(n_rho) + 0.5) * h
    z = (np.arange(n_z) - n_z // 2) * h
    R, Z = np.meshgrid(rho, z, indexing="ij")
    r_prof = particle["r"]

    def place(profile, zc):
        dist = np.sqrt(R ** 2 + (Z - zc) ** 2)
        return np.interp(dist, r_prof, profile[0], right=0.0), np.interp(dist, r_prof, profile[1], right=0.0)

    d1, p1 = place(particle["in"], -d / 2)
    d2, p2 = place(particle["in" if phase == "in" else "anti"], d / 2)
    phi = -1 + d1 + d2
    pi = p1 + p2
    w = num["absorb_width"]
    edge = np.maximum(np.maximum(R - (num["rho_max"] - w), 0), np.maximum(np.abs(Z) - (num["z_max"] - w), 0)) / w
    damp = 2.0 * edge ** 2
    steps = int(num["t_max"] / dt)
    out = np.empty((steps, 2))
    _step_axisym(phi, pi, h, dt, damp, steps, n_rho, n_z, n_z // 2, out)
    t = (np.arange(steps) + 1) * dt
    sep = out[:, 0] - out[:, 1]
    coef = np.polyfit(t, sep, 2)  # D = c2·t² + c1·t + c0
    a = 2 * coef[0]
    return {"d": d, "phase": phase, "D_start": float(sep[0]), "D_end": float(sep[-1]), "a": float(a),
            "force": float(particle["energy"] * a / 2), "verdict": "attraction" if a < 0 else "repulsion"}


def fit_law(rows: list[dict]) -> dict:
    """f(d) = g·(1 + κd)·e^(−κd)/d²: перебор κ, g — МНК при каждом κ."""
    d = np.array([r["d"] for r in rows], dtype=float)
    f = np.abs(np.array([r["force"] for r in rows]))
    best = None
    for kappa in np.linspace(0.05, 1.5, 2901):
        shape = (1 + kappa * d) * np.exp(-kappa * d) / d ** 2
        g = float(np.dot(np.log(f) - np.log(shape), np.ones_like(d)) / d.size)
        resid = float(np.sqrt(np.mean((np.log(f) - np.log(shape) - g) ** 2)))
        if best is None or resid < best[2]:
            best = (float(kappa), math.exp(g), resid)
    return {"kappa": best[0], "g": best[1], "log_rms": best[2]}


def experiment() -> dict:
    particle = prepare_particle()
    rows = [pair_run(particle, d, ph) for ph in PAIR_PROTOCOL["phases"] for d in PAIR_PROTOCOL["separations"]]
    control = [pair_run(particle, PAIR_PROTOCOL["control"]["separation"], ph) for ph in PAIR_PROTOCOL["phases"]]
    laws = {ph: fit_law([r for r in rows if r["phase"] == ph]) for ph in PAIR_PROTOCOL["phases"]}
    kappa_pred = math.sqrt(max(0.0, 1 - particle["omega"] ** 2))
    return {
        "protocol_hash": protocol_hash(), "scope": "model",
        "particle": {"omega_over_F": particle["omega"], "energy": particle["energy"], "action": particle["action"]},
        "runs": rows, "control": control, "laws": laws, "kappa_predicted": kappa_pred,
        "alpha_model": {ph: laws[ph]["g"] / particle["action"] for ph in laws},
    }
