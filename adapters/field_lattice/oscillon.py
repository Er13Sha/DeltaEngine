"""Путь 1 к частотам силы: собственная частота частицы — уплотнения поля (этап E3-волна).

Модель решётки E1 (динамика Кавасаки) передемпфирована: у неё нет инерции, и колебаний в ней
не бывает, поэтому частоту частицы она дать не может. Здесь поле получает инерцию, и всё
задаётся постулатами, записанными до расчёта (OSC_PROTOCOL, хэш в oscillon.json):

P1  пространство заполнено полем φ (H0.P1);
P2  поле обладает инерцией: возмущения бегут с конечной скоростью c (уточнение заказчика о свете);
P3  у поля два равновесных состояния φ = ±1 — «A / не-A», дельта поля;
P4  частица — долгоживущее локализованное колебание поля (уплотнение, H0.P2).

Простейшее поле с P1–P3: φ_tt = ∇²φ − V'(φ), V = (φ² − 1)²/8. В единицах, где c = 1 и частота
малых колебаний поля около равновесия F = 1 (общая частота поля), в уравнении **не остаётся
ни одного безразмерного параметра**: коэффициент нелинейности выносится из φ и входит только
общим множителем в энергию. Поэтому отношение частоты частицы к общей частоте поля ω/F —
число, которое задаёт само поле. Справочные данные в расчёт не входят.

Расчёт: сферически-симметричное уплотнение в 3D, φ_tt = φ_rr + (2/r)φ_r − V'(φ), схема
«чехарда» (leapfrog), поглощающий слой на краю, чтобы излучение уходило. Начальное
уплотнение — пузырь противоположного состояния радиуса R₀; перебираются R₀.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from numba import njit

OSC_PROTOCOL = {
    "format": "delta-protocol@0.1",
    "track": "E",
    "stage": "E3-волна: собственная частота уплотнения поля",
    "postulates": {
        "P1": "пространство заполнено полем φ",
        "P2": "поле обладает инерцией, возмущения распространяются с конечной скоростью c",
        "P3": "у поля два равновесных состояния φ = ±1 (дельта поля)",
        "P4": "частица — долгоживущее локализованное колебание поля",
    },
    "equation": "φ_tt = ∇²φ − V'(φ), V = (φ² − 1)²/8; единицы c = 1, F = √V''(±1) = 1",
    "free_dimensionless_parameters": 0,
    "geometry": "3D, сферическая симметрия",
    "numerics": {"dr": 0.05, "dt": 0.02, "r_max": 240.0, "absorb_from": 180.0, "t_max": 20000.0,
                 "core_radius": 15.0},
    "initial": "φ(r, 0) = −1 + 2·exp(−r²/R₀²), φ_t = 0",
    "scan_R0": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "particle_criterion": "энергия в ядре r < 15 держится выше 10 % начальной дольше 100 периодов 2π/F",
    "outputs": ["ω/F на поздней стадии (по нулям колебания φ(0, t) + 1)", "время жизни в периодах",
                "энергия E и действие за период S = E/ω (в единицах 1/λ, сокращаются в отношениях)"],
    "comparison": "после расчёта — со структурой частиц природы (см. отчёт); подгонки нет: параметров нет",
}


def protocol_hash() -> str:
    return "sha256:" + hashlib.sha256(json.dumps(OSC_PROTOCOL, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@njit(cache=True)
def _evolve(phi, pi, dr, dt, steps, absorb_from, core_n, sample_every):
    """Чехарда для φ_tt = φ_rr + (2/r)φ_r − φ(φ² − 1)/2 с затуханием в поглощающем слое."""
    n = phi.shape[0]
    r = (np.arange(n) + 0.5) * dr
    r_end = r[-1]
    damp = np.zeros(n)
    for i in range(n):
        if r[i] > absorb_from:
            x = (r[i] - absorb_from) / (r_end - absorb_from)
            damp[i] = 2.0 * x * x
    n_samples = steps // sample_every + 1
    center = np.empty(n_samples)
    core_energy = np.empty(n_samples)
    lap = np.empty(n)
    k = 0
    for step in range(steps + 1):
        # лапласиан с отражением в центре (φ_r(0) = 0) и φ = −1 на внешней границе
        for i in range(n):
            left = phi[i - 1] if i > 0 else phi[0]
            right = phi[i + 1] if i < n - 1 else -1.0
            lap[i] = (right - 2 * phi[i] + left) / (dr * dr) + (right - left) / (dr * r[i])
        if step % sample_every == 0:
            e = 0.0
            for i in range(core_n):
                grad = (phi[i + 1] - phi[i]) / dr
                v = (phi[i] * phi[i] - 1) ** 2 / 8
                e += (0.5 * pi[i] * pi[i] + 0.5 * grad * grad + v) * 4 * math.pi * r[i] * r[i] * dr
            center[k] = phi[0]
            core_energy[k] = e
            k += 1
        if step == steps:
            break
        for i in range(n):
            pi[i] += dt * (lap[i] - 0.5 * phi[i] * (phi[i] * phi[i] - 1) - damp[i] * pi[i])
        for i in range(n):
            phi[i] += dt * pi[i]
    return center, core_energy


def run(r0: float) -> dict:
    num = OSC_PROTOCOL["numerics"]
    dr, dt = num["dr"], num["dt"]
    n = int(num["r_max"] / dr)
    r = (np.arange(n) + 0.5) * dr
    phi = -1 + 2 * np.exp(-(r / r0) ** 2)
    pi = np.zeros(n)
    steps = int(num["t_max"] / dt)
    every = 10
    center, core = _evolve(phi, pi, dr, dt, steps, num["absorb_from"], int(num["core_radius"] / dr), every)
    t = np.arange(center.size) * dt * every
    e0 = core[0]
    alive = core > 0.1 * e0
    death = float(t[np.argmin(alive)]) if not alive.all() else float(t[-1])
    lifetime_periods = death / (2 * math.pi)
    # частота на поздней стадии: последние 20 % жизни, нули колебания φ(0) около его среднего
    end = int(np.searchsorted(t, death))
    start = max(0, int(end * 0.6))
    seg, ts = center[start:end], t[start:end]
    omega = float("nan")
    energy = float("nan")
    if seg.size > 100:
        x = seg - seg.mean()
        idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
        if idx.size >= 3:
            crossings = ts[idx] - x[idx] * (ts[idx + 1] - ts[idx]) / (x[idx + 1] - x[idx])
            omega = 2 * math.pi / float(np.mean(np.diff(crossings)))
        energy = float(np.mean(core[start:end]))
    return {"R0": r0, "initial_core_energy": float(e0), "lifetime_t": death,
            "lifetime_periods": lifetime_periods, "particle": lifetime_periods > 100,
            "omega_over_F": omega, "energy": energy, "action_per_period": energy / omega if omega == omega else None}


def experiment() -> dict:
    rows = [run(r0) for r0 in OSC_PROTOCOL["scan_R0"]]
    particles = [r for r in rows if r["particle"]]
    return {"protocol": OSC_PROTOCOL, "protocol_hash": protocol_hash(), "scope": "model", "runs": rows,
            "particles": len(particles),
            "omega_over_F_range": [min(r["omega_over_F"] for r in particles), max(r["omega_over_F"] for r in particles)]
            if particles else None}
