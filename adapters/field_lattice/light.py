"""Свет — частица, движущаяся в базовом гравитационном поле (тезис заказчика, 27.10.2026).

Сравниваются две формы тезиса на двух измерениях у Солнца:

- «корпускула»: свет — частица, которую поле притягивает как любое тело (Зольднер, 1801).
  Отклонение δ = (1/c²)∫ g⊥ dz = 2GM/(c²b). Падая в поле, частица ускоряется
  (v² = c² + 2GM/r), поэтому сигнал приходит раньше: опережение (GM/c³)·ln(4r₁r₂/b²) в одну сторону;
- «частица в среде-поле»: поле для света — среда, которая его замедляет; показатель
  преломления n = 1 + 2GM/(rc²). Отклонение δ = ∫ ∂n/∂b dz = 4GM/(c²b), задержка
  (1/c)∫(n − 1)dz = (2GM/c³)·ln(4r₁r₂/b²) в одну сторону.

В параметризации γ (отклонение ∝ (1 + γ)/2, задержка ∝ (1 + γ)) корпускула — γ = −1 для
задержки и γ = 0 для отклонения, среда — γ = 1. Интегралы по лучу считаются квадратурой.
"""
from __future__ import annotations

import math

import numpy as np

C_LIGHT = 299_792_458.0
GM_SUN = 1.32712440018e20  # м³/с², IAU
R_SUN = 6.957e8  # м, номинальный радиус IAU 2015
AU = 1.495978707e11  # м, точно
ARCSEC = math.pi / 180 / 3600
CASSINI = {"gamma_minus_1": 2.1e-5, "sigma": 2.3e-5,  # Bertotti, Iess, Tortora 2003, Nature 425, 374
           "source": "Кассини, 2002: эффект задержки радиосигнала у Солнца (по сдвигу частоты)"}
GW170817_GAMMA = "−2,6·10⁻⁷ ≤ γ_волн − γ_света ≤ 1,2·10⁻⁶ (Abbott et al. 2017, ApJL 848 L13)"


def _line_integral(f, b: float, z1: float, z2: float, n: int = 400) -> float:
    """∫ f(z) dz по лучу от z1 до z2 с параметром удара b (замена z = b·sh s для 1/r-особенности)."""
    s1, s2 = math.asinh(z1 / b), math.asinh(z2 / b)
    x, w = np.polynomial.legendre.leggauss(n)
    s = (s2 - s1) / 2 * x + (s2 + s1) / 2
    z = b * np.sinh(s)
    return float(np.sum(w * f(z) * b * np.cosh(s)) * (s2 - s1) / 2)


def deflection(b: float, model: str) -> float:
    """Угол отклонения луча, проходящего мимо Солнца на расстоянии b (радианы)."""
    far = 1e6 * b
    if model == "corpuscle":
        return _line_integral(lambda z: GM_SUN * b / (b * b + z * z) ** 1.5, b, -far, far) / C_LIGHT ** 2
    if model == "medium":
        k = 2 * GM_SUN / C_LIGHT ** 2  # n − 1 = k/r, |∂n/∂b| = k·b/r³
        return _line_integral(lambda z: k * b / (b * b + z * z) ** 1.5, b, -far, far)
    raise ValueError(model)


def delay(b: float, r1: float, r2: float, model: str) -> float:
    """Добавка ко времени пути в одну сторону (с): плюс — задержка, минус — опережение."""
    z1, z2 = -math.sqrt(r1 * r1 - b * b), math.sqrt(r2 * r2 - b * b)
    if model == "corpuscle":
        # v = √(c² + 2GM/r): t = ∫dz/v ≈ ∫(1 − GM/(rc²))dz/c
        return -_line_integral(lambda z: GM_SUN / np.sqrt(b * b + z * z), b, z1, z2) / C_LIGHT ** 3
    if model == "medium":
        return _line_integral(lambda z: 2 * GM_SUN / np.sqrt(b * b + z * z), b, z1, z2) / C_LIGHT ** 3
    raise ValueError(model)


def experiment() -> dict:
    b = R_SUN
    defl = {m: deflection(b, m) / ARCSEC for m in ("corpuscle", "medium")}
    gamma = 1 + CASSINI["gamma_minus_1"]
    # Земля — Солнце — Кассини (≈ 8,4 а. е.) в 2002 г.; луч у края Солнца (b ≈ 1,6 R☉)
    b_c, r_e, r_s = 1.6 * R_SUN, AU, 8.4 * AU
    rt = {m: 2 * delay(b_c, r_e, r_s, m) * 1e6 for m in ("corpuscle", "medium")}  # туда и обратно, мкс
    analytic = 2 * 2 * GM_SUN / C_LIGHT ** 3 * math.log(4 * r_e * r_s / b_c ** 2) * 1e6
    return {
        "thesis": "свет — частица, движущаяся в базовом гравитационном поле",
        "scope": "model+data",
        "deflection_at_limb_arcsec": defl,
        "deflection_observed_arcsec": (1 + gamma) / 2 * defl["medium"],
        "shapiro_round_trip_us": rt,
        "shapiro_medium_analytic_us": analytic,
        "cassini": CASSINI,
        "gamma_by_model": {"corpuscle_deflection": 0.0, "corpuscle_delay": -1.0, "medium": 1.0},
        "corpuscle_deflection_sigma": abs(gamma - 0.0) / CASSINI["sigma"],
        "corpuscle_delay_sigma": abs(gamma + 1.0) / CASSINI["sigma"],
        "medium_sigma": abs(gamma - 1.0) / CASSINI["sigma"],
        "gw_and_light_same_medium": GW170817_GAMMA,
    }
