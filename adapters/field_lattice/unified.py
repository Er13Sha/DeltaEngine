"""Гипотеза единой схемы: магнитные и ядерные силы — эффекты одного поля (тезис заказчика, 27.10.2026).

Проверяется, что даёт формула воздействия W = Q·N (видео Rek8C322Q5k) без новых сил:

1. Магнетизм. Статический закон 1/r² плюс конечная скорость передачи воздействия (c) и
   преобразование Лоренца дают у двух зарядов, летящих параллельно со скоростью v,
   добавочную силу F_маг = (v²/c²)·F_эл. Это и есть магнитная сила: отдельного закона не нужно.
   Вопрос — какое количество N её несёт: масса (тяготение) или заряд.
2. Ядерные силы. Энергии связи 20 моноизотопных ядер (`delta_science.pyramid`) сравниваются
   с моделью «нуклоны в плотной упаковке, пары притягиваются по закону воздействия с
   коэффициентом пространственной передачи k_пр = e^(−r/r₀)» при разных r₀; r₀ = ∞ — сама
   формула видео (k_пр = 1, потенциал 1/r). Кулоновское отталкивание протонов вычитается.

Оценки в плавающей точке; входные данные — CODATA 2018 и массы PubChem.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from adapters.field_lattice.screening import E_CHARGE, G_NEWTON, K_COULOMB, M_PROTON

M_NEUTRON = 1.67492749804e-27  # кг, CODATA 2018
M_ELECTRON = 9.1093837015e-31  # кг, CODATA 2018
MU_0 = 1.25663706212e-6  # Н/А², CODATA 2018
MEV = 1.602176634e-13  # Дж
FM = 1e-15  # м
COULOMB_MEV_FM = 1.43996448  # e²/(4πε₀), МэВ·фм
R0_NUCLEUS = 1.2  # фм, школьная формула радиуса ядра R = r₀·A^(1/3)
PION_RANGE = 1.4138  # фм, ħ/(m_π·c) для π± (139,570 МэВ): независимая оценка радиуса ядерных сил
# дейтрон: масса и радиус, CODATA 2018
M_P_U, M_N_U, M_D_U = Fraction("1.007276466621"), Fraction("1.00866491595"), Fraction("2.013553212745")
MEV_PER_U = Fraction("931.49410242")
DEUTERON_RADIUS = 2.12799  # фм


def magnetic_over_electric(beta: Fraction) -> Fraction:
    """Точно: доля магнитной силы у двух зарядов, летящих параллельно со скоростью β·c.

    В общей системе покоя сила чисто статическая F₀. В лабораторной системе поперечное поле
    растёт в γ раз (F_эл = γF₀), а полная поперечная сила уменьшается в γ раз (F = F₀/γ).
    Разность — магнитная сила: (γ − 1/γ)/γ = 1 − 1/γ² = β².
    """
    gamma_sq = 1 / (1 - beta * beta)
    return 1 - 1 / gamma_sq


def wires_check() -> dict:
    """Два провода с током 1 А на расстоянии 1 м: сила Ампера и тот же расчёт, если
    магнитный эффект несёт масса электронов (тяготение), а не их заряд."""
    ampere = MU_0 / (2 * math.pi)  # Н/м
    # для зарядов: F/L = 2kλ_q²v²/(c²d) = μ₀I²/(2πd); для масс λ_q → λ_m = λ_q·m_e/e, k → G
    ratio = G_NEWTON * M_ELECTRON ** 2 / (K_COULOMB * E_CHARGE ** 2)
    return {"ampere_force_N_per_m": ampere, "mass_carried_force_N_per_m": ampere * ratio,
            "mass_over_charge": ratio}


def fcc_cluster(a_count: int, spacing: float) -> np.ndarray:
    """A узлов гранецентрированной кубической упаковки, ближайших к центру (детерминированно)."""
    n = int(math.ceil((a_count / 4) ** (1 / 3))) + 2
    rng = np.arange(-n, n + 1)
    x, y, z = np.meshgrid(rng, rng, rng, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    pts = pts[(pts.sum(axis=1) % 2) == 0].astype(float)  # fcc: чётная сумма координат
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0], np.einsum("ij,ij->i", pts, pts)))
    return pts[order[:a_count]] * spacing / math.sqrt(2)


def pair_sum(points: np.ndarray, r0: float) -> float:
    """Σ по парам k_пр(r)/r, k_пр = e^(−r/r₀) (r₀ = ∞ — формула видео без затухания), фм⁻¹."""
    diff = points[:, None, :] - points[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))[np.triu_indices(len(points), 1)]
    return float(np.sum(np.exp(-r / r0) / r if math.isfinite(r0) else 1 / r))


def nuclei() -> list[dict]:
    from delta_science.atoms import elements
    from delta_science.pyramid import MONONUCLIDIC, unfold_element

    rows = []
    for z, e in elements().items():
        if e.symbol in MONONUCLIDIC:
            nums = {}
            for step in unfold_element(z):
                nums.update(step.numbers or {})
            rows.append({"symbol": e.symbol, "Z": z, "A": z + nums["neutrons"], "B_MeV": float(nums["binding_mev"])})
    return rows


def nuclear_fit(r0_values=(0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, math.inf)) -> dict:
    rows = nuclei()
    density = 3 / (4 * math.pi * R0_NUCLEUS ** 3)  # нуклонов на фм³
    spacing = (4 / density) ** (1 / 3) / math.sqrt(2)  # ближайшие соседи fcc
    a = np.array([r["A"] for r in rows], dtype=float)
    z = np.array([r["Z"] for r in rows], dtype=float)
    b = np.array([r["B_MeV"] for r in rows])
    coulomb = 0.6 * z * (z - 1) * COULOMB_MEV_FM / (R0_NUCLEUS * a ** (1 / 3))
    data_exponent = float(np.polyfit(np.log(a), np.log(b), 1)[0])
    clusters = {int(n): fcc_cluster(int(n), spacing) for n in a}
    fits = []
    for r0 in r0_values:
        s = np.array([pair_sum(clusters[int(n)], r0) for n in a])
        g = float(np.dot(s, b + coulomb) / np.dot(s, s))  # МНК для B = g·S − E_C
        model = g * s - coulomb
        rel = float(np.sqrt(np.mean(((model - b) / b) ** 2)))
        fits.append({"r0_fm": r0 if math.isfinite(r0) else "inf", "g_MeV_fm": g, "rms_relative_error": rel,
                     "model_exponent": float(np.polyfit(np.log(a), np.log(np.maximum(model, 1e-9)), 1)[0])})
    best = min(fits, key=lambda f: f["rms_relative_error"])
    g_mass = G_NEWTON * M_PROTON * M_NEUTRON / (MEV * FM)  # G·m_p·m_n в МэВ·фм
    return {
        "nuclei": rows, "spacing_fm": spacing, "data_exponent_B_vs_A": data_exponent,
        "fits": fits, "best": best, "pion_range_fm": PION_RANGE,
        "gravity_coupling_MeV_fm": g_mass, "best_coupling_over_gravity": best["g_MeV_fm"] / g_mass,
    }


def deuteron() -> dict:
    """Энергия связи дейтрона из масс CODATA и тяготение протона и нейтрона на его радиусе."""
    binding = float((M_P_U + M_N_U - M_D_U) * MEV_PER_U)
    gravity = G_NEWTON * M_PROTON * M_NEUTRON / (DEUTERON_RADIUS * FM) / MEV
    return {"binding_MeV": binding, "gravity_at_radius_MeV": gravity, "ratio": binding / gravity}


def experiment() -> dict:
    betas = [Fraction(1, 1000), Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)]
    return {
        "hypothesis": "магнитные и ядерные силы — эффекты одного поля (W = Q·N), отдельных сил нет",
        "scope": "model+data",
        "magnetism": {
            "relativistic_ratio": [{"beta": str(b), "F_mag_over_F_el": str(magnetic_over_electric(b))} for b in betas],
            "wires": wires_check(),
        },
        "nuclear": nuclear_fit(),
        "deuteron": deuteron(),
    }
