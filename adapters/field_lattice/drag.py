"""Семейство D, проверка C6: торможение и нагрев тела, движущегося в изотропном потоке.

Поток из семейства D (screening.py) состоит из носителей массы m со скоростью u и
плотностью n; с каждого направления на единицу площади приходит импульс
I = n·m·u²/(4π) на стерадиан. Тело с поглощающим сечением N (у прозрачного N = μ·V)
притягивает соседа силой F = I·N_A·N_B/d², то есть G = I/λ², где λ = M/N — масса на
единицу поглощающего сечения.

Тело, движущееся со скоростью v, встречает больше носителей спереди, чем сзади:
    торможение   F = −(n·m·u·N)·u·K(v/u),  K(s) = ½∫ x·√(1 + s² − 2sx) dx → −s/3
    нагрев       P = ½·n·m·u³·N            (вся энергия поглощённых носителей)
Отсюда a = (4π/3)·G·λ·v/u и P/M = 2π·G·λ·u: обе величины выражаются через G и λ.

Выход без нагрева — переизлучать носителей. Тогда поглощённый импульс уходит обратно
потоком от тела, и он расталкивает соседа силой β·I·N_A·N_B/d² (β — доля скорости
переизлучённых носителей): тяготение уменьшается в (1 − β) раз, при β = 1 исчезает.

Расчёт в плавающей точке, носители нерелятивистские; для u → c поправки — множители
порядка единицы, на порядки величин в итоге они не влияют.
"""
from __future__ import annotations

import math

import numpy as np

from adapters.field_lattice.screening import G_NEWTON, absorbed, shadow_force

C_LIGHT = 299_792_458.0  # м/с, точно
SIGMA_SB = 5.670374419e-8  # Вт/(м²·К⁴), CODATA 2018
YEAR = 3.15576e7  # юлианский год, с
EARTH = {
    "mass_kg": 5.9722e24,  # IERS / IAU
    "radius_m": 6.3710e6,  # средний радиус
    "age_yr": 4.54e9,  # ±1 %
    "orbital_speed_m_s": 2.978e4,
    "heat_flow_W": 47e12,  # тепловой поток из недр, Davies & Davies 2010: 47 ± 2 ТВт
}


def drag_integral(s: float, n: int = 64) -> float:
    """K(s) = ½∫₋₁¹ x·√(1 + s² − 2sx) dx: проекция на скорость тела потока импульса,
    поглощённого телом, в единицах n·m·u²·N (носители изотропны в покое, s = v/u < 1)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return float(0.5 * np.sum(w * x * np.sqrt(1 + s * s - 2 * s * x)))


def absorbing_section(r: float, mu: float, n: int = 64) -> float:
    """Поглощающее сечение шара N = ∫(1 − e^(−μL)) dS; у прозрачного → μ·V."""
    x, w = np.polynomial.legendre.leggauss(n)
    rho = (x + 1) / 2 * r
    return float(np.sum(w * r / 2 * 2 * math.pi * rho * absorbed(mu, 2 * np.sqrt(r * r - rho * rho))))


def reemission_balance(d: float, r: float, mu: float, beta: float) -> dict:
    """Сила тени минус отталкивание переизлучённым потоком (β = 1 — упругое переизлучение)."""
    grid = dict(n_theta=48, n_phi=36, n_rho=24, n_psi=24)
    shadow = shadow_force(d, r, r, mu, mu, **grid)
    n_a = absorbing_section(r, mu)
    push = beta * n_a * n_a / d ** 2  # I = 1
    return {"mu": mu, "beta": beta, "shadow": shadow, "shadow_over_formula": shadow / (n_a * n_a / d ** 2),
            "reemission_push": push, "net_over_shadow": (shadow - push) / shadow}


def earth_bounds() -> dict:
    """Самая мягкая граница: Земля хотя бы не непрозрачна (μ·2R ≤ 1), её орбита не затухла
    за возраст Земли (Δv ≤ v, допуск в 100 %). Получаем нижние границы u и нагрева."""
    m, r = EARTH["mass_kg"], EARTH["radius_m"]
    density = m / (4 / 3 * math.pi * r ** 3)
    lam = 2 * r * density  # λ = ρ/μ при μ = 1/(2R): меньше быть не может
    age = EARTH["age_yr"] * YEAR
    rate = 4 * math.pi / 3 * G_NEWTON * lam  # a = rate·v/u
    u_min = rate * age  # орбита живёт возраст Земли: rate·age/u ≤ 1
    heat_per_kg = 2 * math.pi * G_NEWTON * lam * u_min  # = (8π²/3)·G²·λ²·t, от u не зависит
    total = heat_per_kg * m
    temperature = (total / (SIGMA_SB * 4 * math.pi * r * r)) ** 0.25
    return {
        "assumptions": "μ·2R ≤ 1 (Земля не непрозрачна для потока), орбита не затухла за 4,54 млрд лет "
                       "(допуск Δv ≤ v); носители поглощаются",
        "earth_mean_density_kg_m3": density,
        "lambda_min_kg_m2": lam,
        "if_u_is_c": {"stopping_time_s": C_LIGHT / rate, "stopping_time_days": C_LIGHT / rate / 86400,
                      "heat_per_kg_W": 2 * math.pi * G_NEWTON * lam * C_LIGHT,
                      "mass_doubling_days": C_LIGHT ** 2 / (2 * math.pi * G_NEWTON * lam * C_LIGHT) / 86400},
        "u_min_m_s": u_min,
        "u_min_over_c": u_min / C_LIGHT,
        "heat_per_kg_min_W": heat_per_kg,
        "earth_heat_min_W": total,
        # если энергия не греет, а идёт в массу: dM/dt = P/c²
        "mass_doubling_time_min_s": C_LIGHT ** 2 / heat_per_kg,
        "earth_heat_observed_W": EARTH["heat_flow_W"],
        "excess": total / EARTH["heat_flow_W"],
        "equilibrium_surface_temperature_K": temperature,
    }


def experiment() -> dict:
    s_values = [1e-4, 1e-3, 1e-2, 0.1]
    drag = [{"s": s, "K": drag_integral(s), "K_over_minus_s_3": drag_integral(s) / (-s / 3)} for s in s_values]
    balance = [reemission_balance(48, 1, mu, beta) for mu in (0.01, 1.0) for beta in (0.0, 0.5, 1.0)]
    return {
        "family": "D", "check": "C6", "scope": "model",
        "method": "квадратура Гаусса—Лежандра; носители нерелятивистские; оценки в плавающей точке",
        "negative_control_at_rest": drag_integral(0.0),
        "drag": drag,
        "reemission": balance,
        "earth": earth_bounds(),
    }
