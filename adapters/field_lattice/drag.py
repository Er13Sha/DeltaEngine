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


def big_and_small(d: float = 24, r_big: float = 3, r_small: float = 1, mu: float = 0.01) -> dict:
    """Большое тело экранирует малое: сила на малое, сила на большое и формула I·N_A·N_B/d²."""
    grid = dict(n_theta=48, n_phi=36, n_rho=24, n_psi=24)
    on_small = shadow_force(d, r_big, r_small, mu, mu, **grid)
    on_big = shadow_force(d, r_small, r_big, mu, mu, **grid)
    formula = absorbing_section(r_big, mu) * absorbing_section(r_small, mu) / d ** 2
    return {"r_big": r_big, "r_small": r_small, "on_small": on_small, "on_big": on_big,
            "third_law_ratio": on_small / on_big, "on_small_over_formula": on_small / formula}


def universal_over_local() -> list[dict]:
    """Условие заказчика: поток Вселенной с одной стороны сильнее местного тяготения.

    Поток с полусферы давит на тело силой π·I·N = π·G·λ·m, местное тяготение у поверхности
    Земли — G·M·m/R². Отношение X = π·λ·R²/M. Чем оно больше, тем больше λ, а торможение
    и нагрев растут с λ (нагрев — как X²).
    """
    m, r = EARTH["mass_kg"], EARTH["radius_m"]
    age = EARTH["age_yr"] * YEAR
    lam_min = earth_bounds()["lambda_min_kg_m2"]
    x_min = math.pi * lam_min * r * r / m
    rows = []
    for x in (x_min, 10.0, 1e3, 1e6):
        lam = x * m / (math.pi * r * r)
        rate = 4 * math.pi / 3 * G_NEWTON * lam
        heat = 8 * math.pi ** 2 / 3 * G_NEWTON ** 2 * lam ** 2 * age * m
        rows.append({"universal_over_local": x, "lambda_kg_m2": lam, "stopping_days_if_u_is_c": C_LIGHT / rate / 86400,
                     "u_min_over_c": rate * age / C_LIGHT, "heat_excess": heat / EARTH["heat_flow_W"]})
    return rows


def sink_bounds() -> dict:
    """Другой вариант того же механизма: давит не поток частиц, а среда, которую тела поглощают
    (стоки в идеальной жидкости плотности ρ). Два стока Q_A, Q_B притягиваются силой
    ρ·Q_A·Q_B/(4πd²) (теорема Лагалли), поэтому Q/m = √(4πG/ρ). Сток, движущийся со скоростью v,
    поглощает неподвижную среду и тормозится: a = ρ·Q·v/m = √(4πGρ)·v. Орбита выживает за
    время t при ρ ≤ 1/(4πG t²); тогда скорость втекания у поверхности Земли ≥ g·t."""
    m, r = EARTH["mass_kg"], EARTH["radius_m"]
    age = EARTH["age_yr"] * YEAR
    rho_max = 1 / (4 * math.pi * G_NEWTON * age ** 2)
    q_per_kg = math.sqrt(4 * math.pi * G_NEWTON / rho_max)
    inflow = m * q_per_kg / (4 * math.pi * r * r)
    g = G_NEWTON * m / r ** 2
    return {
        "rho_max_kg_m3": rho_max,
        "inflow_speed_min_m_s": inflow,
        "inflow_over_c": inflow / C_LIGHT,
        "g_times_age_m_s": g * age,
        "absorbed_mass_rate_per_kg_s": rho_max * q_per_kg,  # = 1/t: масса растёт в e раз за возраст Земли
        "kinetic_power_per_kg_W": inflow ** 2 / (2 * age),
    }


def instant_carriers() -> dict:
    """Тезис заказчика: тяготение действует мгновенно (u → ∞), свет ограничен средой.

    Торможение ∝ 1/u исчезает. Нагрев равен поглощённому импульсу 4π·I·N, умноженному на
    отношение энергии к импульсу носителя E/p: P = 4π·G·λ·M·(E/p). У массивных носителей
    Ньютона E/p = u/2 — нагрев растёт с u. Выжить D может, только если носитель несёт импульс
    почти без энергии; в механике Эйнштейна так ведёт себя частица быстрее света: E/p = c²/u.
    """
    m = EARTH["mass_kg"]
    age = EARTH["age_yr"] * YEAR
    lam = earth_bounds()["lambda_min_kg_m2"]
    per_unit = 4 * math.pi * G_NEWTON * lam * m  # Вт на 1 м/с отношения E/p
    e_over_p_max = EARTH["heat_flow_W"] / per_unit  # даже весь тепловой поток Земли отдан потоку
    u_min = C_LIGHT ** 2 / e_over_p_max
    rate = 4 * math.pi / 3 * G_NEWTON * lam
    return {
        "heat_per_unit_e_over_p_W": per_unit,
        "e_over_p_max_m_s": e_over_p_max,
        "newtonian_carriers_u_max_m_s": 2 * e_over_p_max,  # E/p = u/2: противоречит торможению
        "superluminal_u_min_over_c": u_min / C_LIGHT,
        "orbit_change_over_age": rate * age / u_min,
    }


# Наблюдения, которые мгновенное тяготение должно объяснить
GW170817 = {"distance_Mpc": 40, "gamma_after_gw_s": 1.74,  # Abbott et al. 2017, ApJL 848 L13
            "speed_bound": "−3·10⁻¹⁵ ≤ (v_gw − c)/c ≤ 7·10⁻¹⁶", "f_gw_Hz": 100}
PSR_B1913_16 = {"pbdot_observed_over_gr": 0.9983, "sigma": 0.0016,  # Weisberg & Huang 2016, ApJ 829 55
                "pbdot_gr": -2.40263e-12}
MPC = 3.0857e22  # м


def instant_vs_observations() -> dict:
    """Что предсказывает мгновенное тяготение (D, u → ∞) для двух измерений.

    GW170817: слияние двух нейтронных звёзд. При мгновенном действии изменение поля пришло бы
    одновременно со слиянием, то есть за d/c до света, а не за 1,74 с; кроме того, мгновенное
    приливное поле двойной звезды убывает как 1/d⁵ против 1/d у волны, и на 40 Мпк его
    отношение к измеренному сигналу ≈ (c/(ωd))⁴.
    PSR B1913+16: мгновенное поле без излучения не уносит энергию — период орбиты не
    уменьшается (торможение потоком ∝ 1/u → 0), а измерено уменьшение, совпадающее с
    излучением волн со скоростью c до 0,2 %.
    """
    d = GW170817["distance_Mpc"] * MPC
    omega = 2 * math.pi * GW170817["f_gw_Hz"]
    return {
        "gw170817": {**GW170817, "instant_lead_s": d / C_LIGHT, "instant_lead_years": d / C_LIGHT / YEAR,
                     "instant_tidal_over_wave": (C_LIGHT / (omega * d)) ** 4},
        "psr_b1913_16": {**PSR_B1913_16, "instant_prediction_over_gr": 0.0,
                         "deviation_sigma": PSR_B1913_16["pbdot_observed_over_gr"] / PSR_B1913_16["sigma"]},
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
        "big_and_small": big_and_small(),
        "universal_over_local": universal_over_local(),
        "sink_medium": sink_bounds(),
        "instant_carriers": instant_carriers(),
        "instant_vs_observations": instant_vs_observations(),
    }
