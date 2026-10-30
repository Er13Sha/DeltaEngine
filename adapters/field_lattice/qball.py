"""E3-заряд: поле с сохраняющимся зарядом (комплексное поле), протокол QB_PROTOCOL записан до расчёта.

Уточнение постулата P3 после этапа E3-волна: частица поля с двумя состояниями распадается,
легче переносчика и не несёт заряда. Здесь поле комплексное, φ = |φ|·e^(iθ):

P3'  модуль поля имеет два равновесных состояния |φ| = 0 и |φ| = 1 одинаковой энергии
     (дельта поля сохраняется), фаза θ свободна; вращение фазы — колебание поля с частотой ω;
P5   заряд — сохраняющаяся величина Q = ∫ 2·Im(φ*·φ_t) d³x («количество вращения» поля).

Уравнение: φ_tt = ∇²φ − φ·(1 − |φ|²)(1 − 3|φ|²), V = |φ|²(1 − |φ|²)². В единицах c = 1,
F = 1 (частота малых колебаний около |φ| = 0) свободных безразмерных параметров нет.

Частица — устойчивый вращающийся сгусток φ = f(r)·e^(iωt) (Q-шар). Его профиль находится
стрельбой: f'' + 2f'/r = f·[(1 − f²)(1 − 3f²) − ω²], f'(0) = 0, f(∞) = 0.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from numba import njit, prange

QB_PROTOCOL = {
    "format": "delta-protocol@0.1",
    "track": "E",
    "stage": "E3-заряд: частица с сохраняющимся зарядом",
    "postulates": {
        "P1": "пространство заполнено полем", "P2": "поле обладает инерцией (конечная скорость c)",
        "P3'": "модуль поля: два равновесных состояния |φ| = 0 и |φ| = 1 одинаковой энергии; фаза свободна",
        "P4": "частица — долгоживущее локализованное колебание (вращение фазы) поля",
        "P5": "заряд Q = ∫ 2·Im(φ*·φ_t) d³x сохраняется",
    },
    "equation": "φ_tt = ∇²φ − φ(1 − |φ|²)(1 − 3|φ|²); c = 1, F = 1",
    "free_dimensionless_parameters": 0,
    "A_family": {"omega_grid": "0,30 … 0,99 с шагом 0,01", "method": "стрельба RK4, dr = 0,002",
                 "outputs": ["f(0)", "Q(ω)", "E(ω)", "E/Q", "проверка dE/dQ = ω"]},
    "elementary_particle": "наименьший заряд среди устойчивых: dQ/dω < 0 (классическая устойчивость) и E < Q "
                           "(не распадается на свободные кванты поля); берётся наибольшее ω сетки с обоими условиями",
    "B_dynamics": {"geometry": "3D, сферическая симметрия", "t_max": 5000, "perturbation": "амплитуда ×1,01",
                   "criterion": "заряд в ядре r < 20 сохраняется в пределах 1 % до t_max",
                   "control": "осциллон E3-волна при тех же условиях распался за ~1700 периодов"},
    "C_pairs": {"separations": [10, 12, 14, 16], "control_separation": 32,
                "cases": {"same_in": "одинаковый заряд, в фазе", "same_anti": "одинаковый заряд, противофаза",
                          "opposite": "противоположные заряды (ω и −ω), при t = 0 фазы совпадают"},
                "geometry": "3D, осевая симметрия, h = 0,1, dt = 0,05, t_max = 150",
                "measurement": "как в oscillon_pair v3: центр энергии в окне 6, подгонка ускорения до смещения 1"},
    "predictions_before_run": {
        "dE/dQ": "= ω (энергия частицы задаётся частотой)",
        "range": "κ = √(1 − ω²)",
    },
    "comparison_after_run": {
        "sign_rule": "природа: электричество — одноимённые отталкиваются, разноимённые притягиваются; "
                     "тяготение и ядро — притяжение",
        "strength": "α_модель = g/S (S = E/ω) — с α_N = 0,355 и α = 1/137",
    },
}


def protocol_hash() -> str:
    return "sha256:" + hashlib.sha256(json.dumps(QB_PROTOCOL, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- A: семейство частиц


@njit(cache=True)
def _shoot(f0, omega, dr, r_max):
    """+1 — перелёт (f < 0), −1 — недолёт (f' > 0), 0 — не решено; плюс профиль до решения."""
    n = int(r_max / dr)
    fs = np.zeros(n + 1)
    f, g = f0, 0.0
    r = 1e-6
    fs[0] = f
    for i in range(n):
        def rhs(rr, ff, gg):
            return gg, ff * ((1 - ff * ff) * (1 - 3 * ff * ff) - omega * omega) - 2 * gg / rr
        k1f, k1g = rhs(r, f, g)
        k2f, k2g = rhs(r + dr / 2, f + dr / 2 * k1f, g + dr / 2 * k1g)
        k3f, k3g = rhs(r + dr / 2, f + dr / 2 * k2f, g + dr / 2 * k2g)
        k4f, k4g = rhs(r + dr, f + dr * k3f, g + dr * k3g)
        f += dr / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)
        g += dr / 6 * (k1g + 2 * k2g + 2 * k3g + k4g)
        r += dr
        fs[i + 1] = f
        if f < 0:
            return 1, fs[: i + 2], i + 1
        if g > 0 and f < 0.5:
            return -1, fs[: i + 2], i + 1
    return 0, fs, n


def qball(omega: float, dr: float = 0.002, r_max: float = 80.0) -> dict:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        verdict, _, _ = _shoot(mid, omega, dr, r_max)
        if verdict > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15:
            break
    _, fs, n = _shoot(lo, omega, dr, r_max)
    # обрезать на минимуме |f| (дальше — расходимость стрельбы), хвост ~ e^(−κr) отбросить
    cut = int(np.argmin(np.abs(fs[: n])))
    f = fs[:cut]
    r = 1e-6 + np.arange(f.size) * dr
    df = np.gradient(f, dr)
    s = f * f
    w = 4 * math.pi * r * r
    charge = float(np.trapezoid(2 * omega * s * w, r))
    energy = float(np.trapezoid((omega * omega * s + df * df + s * (1 - s) ** 2) * w, r))
    return {"omega": omega, "f0": lo, "Q": charge, "E": energy, "E_over_Q": energy / charge, "r": r, "f": f}


def family() -> list[dict]:
    rows = []
    for k in range(30, 100):
        q = qball(k / 100)
        rows.append({key: q[key] for key in ("omega", "f0", "Q", "E", "E_over_Q")})
    for a, b in zip(rows, rows[1:]):
        a["dQ_domega"] = (b["Q"] - a["Q"]) / (b["omega"] - a["omega"])
        a["dE_dQ"] = (b["E"] - a["E"]) / (b["Q"] - a["Q"])
        a["dE_dQ_over_omega_mid"] = a["dE_dQ"] / ((a["omega"] + b["omega"]) / 2)
    return rows


def elementary(rows: list[dict]) -> dict:
    stable = [r for r in rows if "dQ_domega" in r and r["dQ_domega"] < 0 and r["E"] < r["Q"]]
    return max(stable, key=lambda r: r["omega"])


# ---------------------------------------------------------------- B: живёт ли частица


@njit(cache=True)
def _evolve_radial(u, v, pu, pv, dr, dt, steps, absorb_from, core_n, every):
    n = u.shape[0]
    r = (np.arange(n) + 0.5) * dr
    damp = np.zeros(n)
    for i in range(n):
        if r[i] > absorb_from:
            x = (r[i] - absorb_from) / (r[-1] - absorb_from)
            damp[i] = 2.0 * x * x
    out = np.empty(steps // every + 1)
    lu = np.empty(n)
    lv = np.empty(n)
    k = 0
    for step in range(steps + 1):
        for i in range(n):
            ul = u[i - 1] if i > 0 else u[0]
            ur = u[i + 1] if i < n - 1 else 0.0
            vl = v[i - 1] if i > 0 else v[0]
            vr = v[i + 1] if i < n - 1 else 0.0
            lu[i] = (ur - 2 * u[i] + ul) / (dr * dr) + (ur - ul) / (dr * r[i])
            lv[i] = (vr - 2 * v[i] + vl) / (dr * dr) + (vr - vl) / (dr * r[i])
        if step % every == 0:
            q = 0.0
            for i in range(core_n):
                q += 2 * (u[i] * pv[i] - v[i] * pu[i]) * 4 * math.pi * r[i] * r[i] * dr
            out[k] = q
            k += 1
        if step == steps:
            break
        for i in range(n):
            s = u[i] * u[i] + v[i] * v[i]
            force = (1 - s) * (1 - 3 * s)
            pu[i] += dt * (lu[i] - u[i] * force - damp[i] * pu[i])
            pv[i] += dt * (lv[i] - v[i] * force - damp[i] * pv[i])
        for i in range(n):
            u[i] += dt * pu[i]
            v[i] += dt * pv[i]
    return out


def lifetime(q: dict) -> dict:
    cfg = QB_PROTOCOL["B_dynamics"]
    dr, dt = 0.05, 0.02
    n = int(240 / dr)
    r = (np.arange(n) + 0.5) * dr
    f = np.interp(r, q["r"], q["f"], right=0.0) * 1.01
    u, v = f.copy(), np.zeros(n)
    pu, pv = np.zeros(n), q["omega"] * f  # φ = f·e^(iωt): φ_t = iωf
    steps = int(cfg["t_max"] / dt)
    charge = _evolve_radial(u, v, pu, pv, dr, dt, steps, 180.0, int(20 / dr), 500)
    rel = charge / charge[0]
    return {"t_max": cfg["t_max"], "charge_start": float(charge[0]), "charge_end": float(charge[-1]),
            "max_relative_change": float(np.max(np.abs(rel - 1))), "stable": bool(np.max(np.abs(rel - 1)) < 0.01)}


# ---------------------------------------------------------------- C: пары


@njit(cache=True, parallel=True)
def _pair_axisym(u, v, pu, pv, h, dt, damp, steps, n_rho, n_z, z0, out, z_start, window):
    lu = np.empty_like(u)
    lv = np.empty_like(v)
    for step in range(steps):
        for i in prange(n_rho):
            rho = (i + 0.5) * h
            for j in range(n_z):
                for which in range(2):
                    a = u if which == 0 else v
                    c = a[i, j]
                    rl = a[i - 1, j] if i > 0 else a[0, j]
                    rr = a[i + 1, j] if i < n_rho - 1 else 0.0
                    zl = a[i, j - 1] if j > 0 else 0.0
                    zr = a[i, j + 1] if j < n_z - 1 else 0.0
                    lap = (rr - 2 * c + rl + zr - 2 * c + zl) / (h * h) + (rr - rl) / (2 * h * rho)
                    if which == 0:
                        lu[i, j] = lap
                    else:
                        lv[i, j] = lap
        for i in prange(n_rho):
            for j in range(n_z):
                s = u[i, j] ** 2 + v[i, j] ** 2
                force = (1 - s) * (1 - 3 * s)
                pu[i, j] += dt * (lu[i, j] - u[i, j] * force - damp[i, j] * pu[i, j])
                pv[i, j] += dt * (lv[i, j] - v[i, j] * force - damp[i, j] * pv[i, j])
                u[i, j] += dt * pu[i, j]
                v[i, j] += dt * pv[i, j]
        for half in range(2):
            zc = out[step - 1, half] if step > 0 else (z_start if half == 0 else -z_start)
            wsum = 0.0
            zsum = 0.0
            jlo = max(0, int((zc - window) / h) + z0)
            jhi = min(n_z - 1, int((zc + window) / h) + z0 + 1)
            ihi = min(n_rho - 1, int(window / h))
            for i in range(ihi):
                rho = (i + 0.5) * h
                for j in range(jlo, jhi):
                    gu = ((u[i + 1, j] - u[i, j]) ** 2 + (u[i, j + 1] - u[i, j]) ** 2) / (h * h)
                    gv = ((v[i + 1, j] - v[i, j]) ** 2 + (v[i, j + 1] - v[i, j]) ** 2) / (h * h)
                    s = u[i, j] ** 2 + v[i, j] ** 2
                    e = (0.5 * (pu[i, j] ** 2 + pv[i, j] ** 2) + 0.5 * (gu + gv) + 0.5 * s * (1 - s) ** 2) * rho
                    wsum += e
                    zsum += e * (j - z0) * h
            out[step, half] = zsum / wsum


def pair_run(q: dict, d: float, case: str) -> dict:
    h, dt, t_max, window = 0.1, 0.05, 150.0, 6.0
    n_rho, n_z = 500, 1400
    rho = (np.arange(n_rho) + 0.5) * h
    z = (np.arange(n_z) - n_z // 2) * h
    R, Z = np.meshgrid(rho, z, indexing="ij")

    def prof(zc):
        return np.interp(np.sqrt(R ** 2 + (Z - zc) ** 2), q["r"], q["f"], right=0.0)

    f1, f2 = prof(-d / 2), prof(d / 2)
    w = q["omega"]
    # частица 1: φ = f·e^(iωt); частица 2 — по случаю
    sign2 = -1.0 if case == "opposite" else 1.0
    phase2 = math.pi if case == "same_anti" else 0.0
    u = f1 + f2 * math.cos(phase2)
    v = f2 * math.sin(phase2)
    pu = -w * (f2 * sign2 * math.sin(phase2))
    pv = w * f1 + w * sign2 * f2 * math.cos(phase2)
    edge = np.maximum(np.maximum(R - 38.0, 0), np.maximum(np.abs(Z) - 58.0, 0)) / 12.0
    steps = int(t_max / dt)
    out = np.empty((steps, 2))
    _pair_axisym(u, v, pu, pv, h, dt, 2 * edge ** 2, steps, n_rho, n_z, n_z // 2, out, d / 2, window)
    t = (np.arange(steps) + 1) * dt
    sep = out[:, 0] - out[:, 1]
    moved = np.nonzero(np.abs(sep - sep[0]) >= 1.0)[0]
    n_fit = max(int(10 / dt), int(moved[0]) if moved.size else steps)
    a = 2 * np.polyfit(t[:n_fit], sep[:n_fit], 2)[0]
    return {"d": d, "case": case, "D_start": float(sep[0]), "D_end": float(sep[-1]), "merged": bool(sep[-1] < 1.0),
            "trajectory": [round(float(x), 3) for x in sep[::100]], "a": float(a), "force": float(q["E"] * a / 2),
            "verdict": "attraction" if a < 0 else "repulsion"}


def pair_experiment(omega: float = 0.81) -> dict:
    """Этап C по протоколу QB_PROTOCOL['C_pairs']: 3 случая × расстояния + контроль."""
    q = qball(omega)
    cfg = QB_PROTOCOL["C_pairs"]
    rows = [pair_run(q, d, case) for case in cfg["cases"] for d in cfg["separations"]]
    control = [pair_run(q, cfg["control_separation"], case) for case in cfg["cases"]]
    return {"protocol_hash": protocol_hash(), "scope": "model", "omega": omega,
            "particle": {"Q": q["Q"], "E": q["E"], "E_over_Q": q["E_over_Q"]},
            "kappa_predicted": math.sqrt(1 - omega ** 2), "runs": rows, "control": control}


# ---------------------------------------------------------------- D: зависимость от частоты
#
# Протокол записан до расчёта (qball_freq_protocol.json). Вопрос: сохраняется ли картина этапа C
# (знак задаёт фаза, разноимённые ≈ 0, радиус по κ = √(1 − ω²)) на других ω, или она — свойство
# одного ω = 0,81.

FREQ_PROTOCOL = {
    "format": "delta-protocol@0.1",
    "track": "E",
    "stage": "E3-заряд, этап D: зависимость сил от частоты Q-шара",
    "omegas": [0.6, 0.7, 0.9],
    "omega_roles": {
        "0.6": "устойчивый, E < Q (как кандидат), сильнее разнесён по κ",
        "0.7": "устойчивый, E < Q (как кандидат)",
        "0.9": "граница: E > Q, по критерию этапа A не элементарная частица; проверяет, что картина "
               "не зависит от принадлежности к кандидатам",
    },
    "stage_B_check": "для каждого ω — живучесть при t_max = 5000 по протоколу этапа B; только живущие "
                     "участвуют в выводах о силе",
    "pairs": "те же случаи и расстояния, что C_pairs (same_in, same_anti, opposite; d = 10, 12, 14, 16; контроль 32)",
    "predictions_before_run": {
        "kappa": "κ = √(1 − ω²): 0,800; 0,714; 0,436",
        "sign": "знак по фазе — тот же, что при ω = 0,81 (same_in притяжение, same_anti отталкивание)",
        "opposite": "≈ 0 при d ≥ 12 — тот же механизм усреднения 2ω",
        "boundary_0.9": "E > Q: ожидается, что живучесть окажется хуже",
    },
    "criterion_same_picture": "картина совпадает, если знак same_in/same_anti и ≈0 у opposite сохраняются "
                              "при всех ω, а κ отклоняется от √(1 − ω²) не более чем на 10 %",
    "comparison_after_run": "как в C_pairs: α = g/S с α_N = 0,355 и 1/137",
}


def freq_protocol_hash() -> str:
    return "sha256:" + hashlib.sha256(json.dumps(FREQ_PROTOCOL, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
