"""Гипотеза заказчика: сила одна, силы отличаются частотой колебания поля (27.10.2026).

Вычисляется в два шага.

1. Сила (интенсивность). Безразмерная сила взаимодействия двух тел α = F·r²/(ħc). Для тяготения
   она точно равна произведению частот колебаний тел, делённому на квадрат одной общей частоты:
   α = f₁·f₂/F², где f = mc²/h — частота колебания массы, F = m_P·c²/h = √(ħc/G)·c²/h.
   Это не подгонка, а тождество: G·m₁·m₂ = ħc·f₁·f₂/F². По тому же правилу каждой силе
   сопоставляется своя частота f = √α·F: заряду — по α = e²/(4πε₀ħc), ядру — по силе, которая
   связывает дейтрон.
2. Радиус. Колебание с частотой f затухает в поле на длине r₀ = c/(2πf): коэффициент
   пространственной передачи k_пр = e^(−r/r₀). У тяготения и электричества затухания нет
   (f = 0). Для ядра частоту можно найти по данным: уравнение Шрёдингера для дейтрона с
   потенциалом −g·e^(−r/r₀)/r при разных r₀; g подбирается по энергии связи 2,2246 МэВ, а
   размер дейтрона и длина рассеяния нейтрона на протоне — предсказания. Лучшее r₀ даёт
   частоту ядерного колебания; её сравниваем с частотой пиона, измеренной отдельно.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

H_PLANCK = 6.62607015e-34  # Дж·с, точно
HBAR = H_PLANCK / (2 * math.pi)
C_LIGHT = 299_792_458.0
G_NEWTON = 6.67430e-11
ALPHA = 7.2973525693e-3  # постоянная тонкой структуры, CODATA 2018
M_PROTON = 1.67262192369e-27
HBARC = 197.3269804  # МэВ·фм
MP_MEV, MN_MEV = 938.27208816, 939.56542052
MU_MEV = MP_MEV * MN_MEV / (MP_MEV + MN_MEV)  # приведённая масса n–p
PION_MEV = 139.57039  # π±, PDG
MEV_TO_HZ = 1.602176634e-13 / H_PLANCK  # частота кванта 1 МэВ
DEUTERON = {
    "binding_MeV": 2.224566,  # из масс CODATA 2018 (см. unified.deuteron)
    "structure_radius_fm": 1.975,  # среднеквадратичный радиус точечных нуклонов
    "scattering_length_fm": 5.4194,  # триплетная длина рассеяния n–p, Hackenburg 2006
    "effective_range_fm": 1.753,
}
K2 = 2 * MU_MEV / HBARC ** 2  # фм⁻²·МэВ⁻¹


@njit(cache=True)
def _integrate(g: float, r0: float, energy: float, r_max: float, h: float) -> np.ndarray:
    """u'' = K2·(V − E)·u, V = −g·e^(−r/r₀)/r, u(0) = 0, u'(0) = 1; метод Рунге — Кутты 4."""
    n = int(r_max / h)
    out = np.empty((n + 1, 3))
    r, u, du = 1e-8, 1e-8, 1.0
    out[0] = (r, u, du)
    for i in range(n):
        k1u, k1d = du, K2 * (-g * math.exp(-r / r0) / r - energy) * u
        rm = r + h / 2
        qm = K2 * (-g * math.exp(-rm / r0) / rm - energy)
        k2u, k2d = du + h / 2 * k1d, qm * (u + h / 2 * k1u)
        k3u, k3d = du + h / 2 * k2d, qm * (u + h / 2 * k2u)
        r1 = r + h
        k4u, k4d = du + h * k3d, K2 * (-g * math.exp(-r1 / r0) / r1 - energy) * (u + h * k3u)
        u += h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        du += h / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        r = r1
        out[i + 1] = (r, u, du)
    return out


def deuteron_fit(r0: float, h: float = 0.002) -> dict:
    """Подбор g по энергии связи и предсказание размера и длины рассеяния."""
    b = DEUTERON["binding_MeV"]
    kappa = math.sqrt(K2 * b)
    r_max = 30.0
    # основное состояние (без узлов): первое g, при котором u(R) меняет знак, растя g шагами по 2 %
    lo, hi = 1.0, 1.0  # g в МэВ·фм
    while _integrate(hi, r0, -b, r_max, h)[-1, 1] > 0:
        lo, hi = hi, hi * 1.02
    for _ in range(80):  # при собственном g решение затухает; меньше — растёт, больше — уходит в минус
        mid = (lo + hi) / 2
        if _integrate(mid, r0, -b, r_max, h)[-1, 1] > 0:
            lo = mid
        else:
            hi = mid
    g = (lo + hi) / 2
    sol = _integrate(g, r0, -b, r_max, h)
    cut = int(22.0 / h)  # хвост после 22 фм — аналитический: u = A·e^(−κr)
    r, u = sol[: cut + 1, 0], sol[: cut + 1, 1]
    a_tail = u[-1] * math.exp(kappa * r[-1])
    rc = r[-1]
    t0 = a_tail ** 2 * math.exp(-2 * kappa * rc) / (2 * kappa)
    t2 = a_tail ** 2 * math.exp(-2 * kappa * rc) * (rc ** 2 / (2 * kappa) + rc / (2 * kappa ** 2) + 1 / (4 * kappa ** 3))
    norm = np.trapezoid(u * u, r) + t0
    r2 = (np.trapezoid(r * r * u * u, r) + t2) / norm
    zero = _integrate(g, r0, 0.0, 40.0, h)[-1]
    a_scat = zero[0] - zero[1] / zero[2]
    r_eff = 2 * (kappa - 1 / a_scat) / kappa ** 2
    return {"r0_fm": r0, "g_MeV_fm": g, "alpha_nuclear": g / HBARC,
            "structure_radius_fm": math.sqrt(r2) / 2, "scattering_length_fm": a_scat,
            "effective_range_fm": r_eff, "frequency_Hz": C_LIGHT / (2 * math.pi * r0 * 1e-15)}


def strength_frequencies(alpha_nuclear: float) -> dict:
    """Частоты, при которых одна формула α = f₁·f₂/F² даёт силу тяготения, электричества и ядра."""
    m_planck = math.sqrt(HBAR * C_LIGHT / G_NEWTON)
    big_f = m_planck * C_LIGHT ** 2 / H_PLANCK
    f_proton = M_PROTON * C_LIGHT ** 2 / H_PLANCK
    alpha_g = G_NEWTON * M_PROTON ** 2 / (HBAR * C_LIGHT)
    return {
        "universal_frequency_F_Hz": big_f,
        "proton_mass_frequency_Hz": f_proton,
        "gravity": {"alpha": alpha_g, "alpha_from_frequencies": (f_proton / big_f) ** 2, "frequency_Hz": f_proton},
        "electric": {"alpha": ALPHA, "frequency_Hz": math.sqrt(ALPHA) * big_f},
        "nuclear": {"alpha": alpha_nuclear, "frequency_Hz": math.sqrt(alpha_nuclear) * big_f},
        "electric_over_gravity": ALPHA / alpha_g,
    }


def experiment() -> dict:
    pion_range = HBARC / PION_MEV
    scan = [deuteron_fit(r0) for r0 in (0.5, 0.75, 1.0, 1.25, pion_range, 1.75, 2.0, 2.5, 3.0)]
    meas = DEUTERON

    def miss(row):
        return math.sqrt(sum(((row[k] - meas[k]) / meas[k]) ** 2
                             for k in ("structure_radius_fm", "scattering_length_fm", "effective_range_fm")) / 3)

    for row in scan:
        row["rms_relative_error"] = miss(row)
    best = min(scan, key=lambda row: row["rms_relative_error"])
    pion = next(row for row in scan if row["r0_fm"] == pion_range)
    # частая сетка без точки пиона: частота ядра только по данным дейтрона
    fine = [deuteron_fit(float(r0)) for r0 in np.arange(1.2, 1.7001, 0.025)]
    for row in fine:
        row["rms_relative_error"] = miss(row)
    fine_best = min(fine, key=lambda row: row["rms_relative_error"])
    return {
        "hypothesis": "сила одна, силы отличаются частотой колебания поля",
        "scope": "model+data",
        "law": "α = f₁·f₂/F² (сила), k_пр = e^(−2πf·r/c) (радиус)",
        "deuteron_measured": meas,
        "range_scan": scan,
        "best": best,
        "fine_scan": fine,
        "fine_best": fine_best,
        "deuteron_frequency_over_pion": fine_best["frequency_Hz"] / (PION_MEV * MEV_TO_HZ),
        "pion": {"mass_MeV": PION_MEV, "frequency_Hz": PION_MEV * MEV_TO_HZ, "range_fm": pion_range,
                 "fit": pion},
        "strength": strength_frequencies(pion["alpha_nuclear"]),
    }
