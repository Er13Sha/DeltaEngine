"""Семейство D: притяжение через экранирование изотропного потока (гипотеза H2, план 9.4).

Предложение заказчика (27.10.2026): одноимённые заряды поместить в пространство, на которое
со всех сторон действует гравитационное воздействие; тогда возникнет притяжение.

Модель (3D, непрерывная геометрия):

- пространство пронизано изотропным потоком носителей воздействия: с каждого направления
  Ω на единицу площади приходит импульс I (на стерадиан);
- тело — шар радиуса r с коэффициентом поглощения μ; луч, проходящий хорду L, теряет
  долю a(L) = 1 − e^(−μL) и передаёт её импульс телу;
- члена притяжения в правиле нет. Сила на тело B из-за тела A — недостаток импульса лучей,
  которые прошли A до B: F_B = −I ∫ a_A · a_B · u dΩ dS (u — направление луча).

Одиночное тело в изотропном потоке получает нулевую силу (симметрия) — отрицательный
контроль. Интеграл считается квадратурой в плавающей точке; это оценка вне доверенного
ядра (протокол, прил. C, п. 3), в сводку попадают числа с указанием метода и сетки.
"""
from __future__ import annotations

import math

import numpy as np


def absorbed(mu: float, chord: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-mu * chord)


def shadow_force(d: float, r_a: float, r_b: float, mu_a: float, mu_b: float, intensity: float = 1.0,
                 n_theta: int = 64, n_phi: int = 48, n_rho: int = 32, n_psi: int = 32) -> float:
    """Сила на B к A (положительная — притяжение) от экранирования потока телом A.

    Лучи, задевающие оба тела, идут в конусе вокруг оси A→B с полууглом asin((r_a + r_b)/d).
    Для каждого направления u интегрируем по точкам p круга радиуса r_b в плоскости ⟂ u.
    """
    if d <= r_a + r_b:
        raise ValueError("тела перекрываются")
    alpha = math.asin(min(1.0, (r_a + r_b) / d))
    # узлы Гаусса—Лежандра по cos θ на [cos α, 1], равномерно по φ, по кругу — полярная сетка
    xs, ws = np.polynomial.legendre.leggauss(n_theta)
    c0 = math.cos(alpha)
    cos_t = (1 - c0) / 2 * xs + (1 + c0) / 2
    w_t = ws * (1 - c0) / 2
    phi = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
    xr, wr = np.polynomial.legendre.leggauss(n_rho)
    rho = (xr + 1) / 2 * r_b
    w_rho = wr * r_b / 2 * rho  # якобиан ρ dρ
    psi = (np.arange(n_psi) + 0.5) * 2 * math.pi / n_psi

    axis = np.array([0.0, 0.0, 1.0])  # ось от A к B; A в (0, 0, −d), B в начале координат
    a_center = np.array([0.0, 0.0, -d])
    total = 0.0
    for ct, wt in zip(cos_t, w_t):
        st = math.sqrt(max(0.0, 1 - ct * ct))
        for ph in phi:
            u = np.array([st * math.cos(ph), st * math.sin(ph), ct])  # направление движения луча
            # базис плоскости ⟂ u
            e1 = np.cross(u, [1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.cross(u, [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            R, P = np.meshgrid(rho, psi, indexing="ij")
            pts = R[..., None] * (np.cos(P)[..., None] * e1 + np.sin(P)[..., None] * e2)
            chord_b = 2 * np.sqrt(np.maximum(0.0, r_b * r_b - R * R))
            rel = a_center - pts  # от точки луча к центру A
            along = rel @ u
            dist2 = np.einsum("ijk,ijk->ij", rel, rel) - along * along
            chord_a = 2 * np.sqrt(np.maximum(0.0, r_a * r_a - dist2))
            upstream = along < 0  # A стоит раньше B по ходу луча
            frac = absorbed(mu_a, chord_a) * absorbed(mu_b, chord_b) * upstream
            dS = w_rho[:, None] * (2 * math.pi / n_psi)
            # недостаток импульса направлен против u; проекция на ось B→A равна u·axis
            total += wt * (2 * math.pi / n_phi) * float(np.sum(frac * dS)) * float(u @ axis)
    return intensity * total


def single_body_force(r: float, mu: float, n_dir: int = 200) -> float:
    """Отрицательный контроль: сила на одиночное тело в изотропном потоке.

    Набор направлений симметричен (каждое вместе с противоположным), поглощённый импульс
    шара от направления не зависит, поэтому сумма должна быть ровно нулевой; иначе
    квадратура несимметрична и опыт недействителен.
    """
    golden = math.pi * (3 - math.sqrt(5))
    idx = np.arange(n_dir) + 0.5
    z = 1 - idx / n_dir  # верхняя полусфера
    rr = np.sqrt(1 - z * z)
    th = golden * idx
    half = np.stack([rr * np.cos(th), rr * np.sin(th), z], axis=1)
    dirs = np.concatenate([half, -half])
    xr, wr = np.polynomial.legendre.leggauss(32)
    rho = (xr + 1) / 2 * r
    per_dir = float(np.sum(wr * r / 2 * rho * 2 * math.pi * absorbed(mu, 2 * np.sqrt(r * r - rho * rho))))
    return float(np.linalg.norm((per_dir * dirs).sum(axis=0)))


def power_law_exponent(ds: list[float], fs: list[float]) -> float:
    """Наклон log F по log d (метод наименьших квадратов)."""
    x = np.log(np.array(ds))
    y = np.log(np.array(fs))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------- опыт D и проверка предложения


# CODATA 2018: e и k точны в СИ (k = 1/(4πε₀) с точностью таблицы), G и m_p — измеренные
E_CHARGE = 1.602176634e-19  # Кл, точно
K_COULOMB = 8.9875517923e9  # Н·м²/Кл²
G_NEWTON = 6.67430e-11  # Н·м²/кг², погрешность 2,2·10⁻⁵ отн.
M_PROTON = 1.67262192369e-27  # кг


def experiment() -> dict:
    grid = dict(n_theta=48, n_phi=36, n_rho=24, n_psi=24)
    control = single_body_force(1.0, 0.1)
    ds = [6, 8, 12, 16, 24, 32, 48]
    fs = [shadow_force(d, 1, 1, 0.05, 0.05, **grid) for d in ds]
    exponent = power_law_exponent(ds, fs)
    mass_rows = []
    for mu in (0.01, 0.1, 1.0, 10.0):
        f = shadow_force(12, 1, 1, mu, mu, **grid)
        m = mu * 4 / 3 * math.pi  # масса ∝ поглощающему веществу
        mass_rows.append({"mu": mu, "F": f, "F_over_m2": f / (m * m)})
    # коэффициент экранирования прозрачных тел: F = Gs·m_A·m_B / d² (μ → 0)
    m_small = 0.01 * 4 / 3 * math.pi
    gs = shadow_force(24, 1, 1, 0.01, 0.01, **grid) * 24 ** 2 / (m_small * m_small)
    # одноимённые: F_net = k q²/d² − Gs m²/d² → притяжение при q/m < √(Gs/k)
    real_ratio = K_COULOMB * E_CHARGE ** 2 / (G_NEWTON * M_PROTON ** 2)
    # формула видео Rek8C322Q5k: F = c1·N_A·N_B/(4πd²), N = h·S·ρ; в модели ρ — поглощение μ,
    # c1 = 4π·I. Отношение измеренной силы к формуле → 1 у прозрачных тел.
    video_rows = []
    for mu in (0.001, 0.01, 0.1, 1.0, 10.0):
        n = mu * 4 / 3 * math.pi  # h·S·ρ шара = μ·V
        for d in (12, 24):
            f = shadow_force(d, 1, 1, mu, mu, **grid)
            video_rows.append({"mu": mu, "d": d, "F": f, "formula": n * n / d ** 2,
                               "ratio": f / (n * n / d ** 2)})
    return {
        "family": "D", "scope": "model", "method": "квадратура Гаусса—Лежандра по конусу направлений и кругу "
                                                    f"сечения, сетка {grid}; оценки в плавающей точке",
        "negative_control_single_body_force": control,
        "distance_law": {"d": ds, "F": fs, "exponent": exponent},
        "mass_proportionality": mass_rows,
        "screening_coefficient_small_opacity": gs,
        "video_formula": {
            "source": "Rek8C322Q5k, «Формула мироздания»: W = Q·N, Q = k·c1/(4πr²), N = h·S·ρ",
            "mapping": "c1 = 4π·I (поток со всех направлений), k_пр = 1, N = μ·V (ρ = поглощение μ)",
            "rows": video_rows,
        },
        "like_charges": {
            "condition": "одноимённые притягиваются тогда и только тогда, когда G_s·m_A·m_B > k·q_A·q_B: "
                         "оба члена убывают как 1/d², поэтому знак не зависит от расстояния",
            "real_world": {
                "proton_pair_coulomb_over_gravity": real_ratio,
                "threshold_charge_to_mass_C_per_kg": math.sqrt(G_NEWTON / K_COULOMB),
                "proton_charge_to_mass_C_per_kg": E_CHARGE / M_PROTON,
                "constants": "CODATA 2018: e, k, G, m_p",
            },
        },
    }
