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
from pathlib import Path

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
    source = f0 * ((1 - f0*f0) * (1 - 3*f0*f0) - omega*omega)
    fs[0] = f0
    # Регулярный ряд в центре: f = f0 + source*r²/6, f' = source*r/3.
    r = dr
    f, g = f0 + source*dr*dr/6, source*dr/3
    fs[1] = f
    for i in range(1, n):
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
        if g > 0:
            return -1, fs[: i + 2], i + 1
    return 0, fs, n


def qball(omega: float, dr: float = 0.002, r_max: float = 80.0) -> dict:
    if not 0 < omega < 1:
        raise ValueError("Частота должна лежать между 0 и 1")
    # Экстремумы эффективного потенциала. Верхний экстремум > 1;
    # ограничение f0 <= 1 обрезает физическую ветвь при малых ω.
    lo = math.sqrt((2 - math.sqrt(1 + 3*omega*omega))/3)
    hi = math.sqrt((2 + math.sqrt(1 + 3*omega*omega))/3) * (1 - 1e-14)
    if _shoot(hi, omega, dr, r_max)[0] != 1:
        raise RuntimeError(f"Не найден перелёт при omega={omega}")
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
    # Стрельба теряет точность в далёком хвосте. Стыкуем с положительным
    # асимптотическим решением, а не обрываем поле на конечном радиусе.
    minimum = int(np.argmin(np.abs(fs[:n])))
    matches = np.flatnonzero(fs[:minimum] < 1e-6 * lo)
    cut = int(matches[0]) + 1 if matches.size else minimum
    f = fs[:cut]
    r = np.arange(f.size) * dr
    df = np.gradient(f, dr)
    s = f * f
    w = 4 * math.pi * r * r
    charge = float(np.trapezoid(2 * omega * s * w, r))
    energy = float(np.trapezoid((omega * omega * s + df * df + s * (1 - s) ** 2) * w, r))
    gradient = float(np.trapezoid(df*df*w, r))
    potential_effective = float(np.trapezoid((s*(1-s)**2 - omega*omega*s)*w, r))
    kappa = math.sqrt(1 - omega*omega)
    rm, fm = float(r[-1]), float(f[-1])
    tail_i2 = 4*math.pi*fm*fm*rm*rm/(2*kappa)
    tail_gradient = 4*math.pi*fm*fm*rm*rm*(kappa/2 + 1/rm)
    charge += 2*omega*tail_i2
    energy += (1+omega*omega)*tail_i2 + tail_gradient
    gradient += tail_gradient
    potential_effective += (1-omega*omega)*tail_i2
    virial = abs(gradient + 3*potential_effective) / gradient
    return {"omega": omega, "f0": lo, "Q": charge, "E": energy, "E_over_Q": energy / charge,
            "virial_relative_error": virial, "tail_fraction": float(f[-1]/lo),
            "tail_log_derivative_error": float(abs((fs[cut]-fs[cut-2])/(2*dr)/fm + kappa + 1/rm)),
            "tail_Q": 2*omega*tail_i2, "kappa": kappa, "r": r, "f": f}


def profile_at(q: dict, radius):
    """Профиль с бесконечным положительным хвостом, без искусственного порога."""
    radius = np.asarray(radius)
    rm, fm = q["r"][-1], q["f"][-1]
    safe_r = np.maximum(radius, rm)
    tail = fm * rm / safe_r * np.exp(-q["kappa"] * (safe_r-rm))
    return np.where(radius > rm, tail, np.interp(radius, q["r"], q["f"]))


def log_profile_at(q: dict, radius):
    """Логарифм хвоста: сохраняет информацию даже при underflow обычного float."""
    radius = np.asarray(radius)
    rm, fm = q["r"][-1], q["f"][-1]
    safe_r = np.maximum(radius, rm)
    tail_log = math.log(fm) + np.log(rm/safe_r) - q["kappa"]*(safe_r-rm)
    return np.where(radius > rm, tail_log, np.log(np.interp(radius, q["r"], q["f"])))


def family() -> list[dict]:
    rows = []
    for k in range(30, 100):
        q = qball(k / 100)
        rows.append({key: q[key] for key in ("omega", "f0", "Q", "E", "E_over_Q", "virial_relative_error", "tail_fraction")})
    for a, b in zip(rows, rows[1:]):
        a["dQ_domega"] = (b["Q"] - a["Q"]) / (b["omega"] - a["omega"])
        a["dE_dQ"] = (b["E"] - a["E"]) / (b["Q"] - a["Q"])
        a["dE_dQ_over_omega_mid"] = a["dE_dQ"] / ((a["omega"] + b["omega"]) / 2)
    return rows


def elementary(rows: list[dict]) -> dict:
    stable = [r for r in rows if "dQ_domega" in r and r["dQ_domega"] < 0 and r["E"] < r["Q"]
              and r["virial_relative_error"] < 1e-3 and r["tail_fraction"] < 1e-4]
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
    out = np.empty((steps // every + 1, 8))
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
            qc, qt, ec, et, norm, moment, peak = 0., 0., 0., 0., 0., 0., 0.
            for i in range(n):
                weight = 4*math.pi*r[i]*r[i]*dr
                s = u[i]*u[i] + v[i]*v[i]
                qi = 2*(u[i]*pv[i]-v[i]*pu[i])*weight
                im, ip = max(0, i-1), min(n-1, i+1)
                gu = (u[ip]-u[im])/((ip-im)*dr)
                gv = (v[ip]-v[im])/((ip-im)*dr)
                ei = (pu[i]**2+pv[i]**2+gu*gu+gv*gv+s*(1-s)**2)*weight
                qt += qi
                et += ei
                if i < core_n:
                    qc += qi
                    ec += ei
                    norm += s*weight
                    moment += s*weight*r[i]*r[i]
                    peak = max(peak, math.sqrt(s))
            out[k, 0] = step*dt
            out[k, 1] = qc
            out[k, 2] = qt
            out[k, 3] = peak
            out[k, 4] = math.sqrt(moment/norm)
            out[k, 5] = ec
            out[k, 6] = et
            out[k, 7] = u[0]
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


def lifetime(q: dict, perturbation: float = 1.01, dr: float = 0.05, dt: float = 0.02) -> dict:
    cfg = QB_PROTOCOL["B_dynamics"]
    n = int(240 / dr)
    r = (np.arange(n) + 0.5) * dr
    f = profile_at(q, r) * perturbation
    u, v = f.copy(), np.zeros(n)
    pu, pv = np.zeros(n), q["omega"] * f  # φ = f·e^(iωt): φ_t = iωf
    steps = int(cfg["t_max"] / dt)
    trace = _evolve_radial(u, v, pu, pv, dr, dt, steps, 180.0, int(20 / dr), int(round(10/dt)))
    charge = trace[:, 1]
    rel = charge / charge[0]
    return {"t_max": cfg["t_max"], "dr": dr, "dt": dt, "perturbation": perturbation,
            "charge_start": float(charge[0]), "charge_end": float(charge[-1]),
            "max_relative_change": float(np.max(np.abs(rel - 1))), "stable": bool(np.max(np.abs(rel - 1)) < 0.01),
            "total_charge_max_relative_change": float(np.max(np.abs(trace[:, 2]/trace[0, 2]-1))),
            "peak_amplitude_range": [float(np.min(trace[:, 3])), float(np.max(trace[:, 3]))],
            "rms_radius_range": [float(np.min(trace[:, 4])), float(np.max(trace[:, 4]))],
            "trace_columns": ["t", "Q_core", "Q_total", "peak_amplitude", "rms_radius", "E_core", "E_total", "u_center"],
            "trace": trace.tolist()}


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


def pair_run(q: dict, d: float, case: str, h: float = 0.1, dt: float = 0.05) -> dict:
    t_max, window = 150.0, 6.0
    n_rho, n_z = round(50/h), 2*round(70/h)
    rho = (np.arange(n_rho) + 0.5) * h
    z = (np.arange(n_z) - n_z // 2) * h
    R, Z = np.meshgrid(rho, z, indexing="ij")

    def prof(zc):
        return profile_at(q, np.sqrt(R ** 2 + (Z - zc) ** 2))

    f1, f2 = prof(-d / 2), prof(d / 2)
    w = q["omega"]
    # частица 1: φ = f·e^(iωt); частица 2 — по случаю
    sign2 = -1.0 if case == "opposite" else 1.0
    phase2 = math.pi if case == "same_anti" else 0.0
    u = f1 + f2 * math.cos(phase2)
    v = f2 * math.sin(phase2)
    pu = -w * (f2 * sign2 * math.sin(phase2))
    pv = w * f1 + w * sign2 * f2 * math.cos(phase2)
    weights = 2*math.pi*R*h*h
    charge_start = float(np.sum(2*(u*pv-v*pu)*weights))
    overlap = float(np.sum(f1*f2*weights) / np.sum(f1*f1*weights))
    edge = np.maximum(np.maximum(R - 38.0, 0), np.maximum(np.abs(Z) - 58.0, 0)) / 12.0
    steps = int(t_max / dt)
    out = np.empty((steps, 2))
    _pair_axisym(u, v, pu, pv, h, dt, 2 * edge ** 2, steps, n_rho, n_z, n_z // 2, out, d / 2, window)
    t = (np.arange(steps) + 1) * dt
    sep = out[:, 0] - out[:, 1]
    moved = np.nonzero(np.abs(sep - sep[0]) >= 1.0)[0]
    n_fit = int(moved[0]) if moved.size else steps
    valid = n_fit*dt >= 10 and d >= 2*window
    a = None
    residual = None
    if n_fit >= 3:
        coef = np.polyfit(t[:n_fit], sep[:n_fit], 2)
        a = float(2*coef[0])
        residual = float(np.sqrt(np.mean((sep[:n_fit]-np.polyval(coef, t[:n_fit]))**2)))
    charge_end = float(np.sum(2*(u*pv-v*pu)*weights))
    return {"d": d, "case": case, "D_start": float(sep[0]), "D_end": float(sep[-1]), "merged": bool(sep[-1] < 1.0),
            "h": h, "dt": dt, "t_fit": n_fit*dt, "fit_valid": valid, "fit_residual": residual,
            "initial_overlap": overlap, "charge_start": charge_start, "charge_end": charge_end,
            "charge_over_two_isolated": charge_start/(2*q["Q"]) if case != "opposite" else None,
            "trajectory_dt": 100*dt, "trajectory": [float(x) for x in sep[::100]],
            "a": a, "force": q["E"]*a/2 if a is not None else None,
            "verdict": ("attraction" if a < 0 else "repulsion") if valid else "invalid_fit"}


def main():
    import argparse
    import platform
    import numba

    parser = argparse.ArgumentParser(description="Q-шары: воспроизводимый расчёт по зарегистрированному протоколу")
    parser.add_argument("--stage", choices=("family", "lifetime", "pairs", "all"), default="family")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--distance", type=float, help="Один прогон из протокола")
    parser.add_argument("--case", choices=("same_in", "same_anti", "opposite"))
    parser.add_argument("--h", type=float, default=.1)
    parser.add_argument("--dt", type=float, default=.05)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[2]
    output = root / "data/experiments/field/qball.json"
    result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    result.update({"scope": "model", "protocol_hash": protocol_hash(),
                   "addendum_hash": "sha256:" + hashlib.sha256((root / "docs/track-e-qball-protocol-addendum.md").read_bytes()).hexdigest(),
                   "code_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "runtime": {"python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__}})

    def save():
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(output)

    if args.stage in ("family", "all"):
        rows = family()
        selected = elementary(rows)
        result["family"] = rows
        result["selected"] = selected
        q = qball(selected["omega"])
        refined = qball(selected["omega"], dr=0.001)
        result["profile_check"] = {"dr": 0.001, "Q_relative_change": abs(refined["Q"]/q["Q"]-1),
                                   "E_relative_change": abs(refined["E"]/q["E"]-1),
                                   "virial_relative_error": refined["virial_relative_error"]}
        save()
        print("Семейство:", len(rows), "кандидат:", selected, "проверка:", result["profile_check"], flush=True)
    if args.stage in ("lifetime", "all"):
        q = qball(result["selected"]["omega"])
        result["lifetime"] = {}
        for label, perturb, dr, dt in (("perturbed", 1.01, .05, .02), ("unperturbed", 1., .05, .02),
                                      ("refined", 1.01, .025, .01)):
            row = lifetime(q, perturb, dr, dt)
            result["lifetime"][label] = row
            save()
            print("Динамика:", label, {k: v for k, v in row.items() if k not in ("trace", "trace_columns")}, flush=True)
    if args.stage in ("pairs", "all"):
        q = qball(result["selected"]["omega"])
        result.setdefault("pairs", [])
        distances = (args.distance,) if args.distance is not None else (*QB_PROTOCOL["C_pairs"]["separations"], QB_PROTOCOL["C_pairs"]["control_separation"])
        cases = (args.case,) if args.case else QB_PROTOCOL["C_pairs"]["cases"]
        for d in distances:
            for case in cases:
                if any(x["d"] == d and x["case"] == case and x["h"] == args.h and x["dt"] == args.dt for x in result["pairs"]):
                    continue
                row = pair_run(q, d, case, h=args.h, dt=args.dt)
                result["pairs"].append(row)
                save()
                print("Пара:", {k: v for k, v in row.items() if k != "trajectory"}, flush=True)


if __name__ == "__main__":
    main()
