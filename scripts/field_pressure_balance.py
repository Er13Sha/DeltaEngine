"""Равновесие радиуса при наружном давлении. Протокол записан отдельно.

Камеры нет: в правой части нет члена, который возвращал бы R к фиксированному потолку.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

OUT = Path("data/experiments/field/field_pressure_balance_2026-10-27.json")
K = 1.0
Q = 1.0
MU = 1.0
GAMMA = 1.0
PRESSURES = (1.0, 0.5, 0.25, 0.0625)
T_RELAX = 40.0
T_FREE = 5.0


def radius_equilibrium(pressure: float, family: str) -> float:
    if pressure <= 0.0:
        return math.inf
    if family == "shell":
        return (K * Q * Q / (8.0 * math.pi * pressure)) ** 0.25
    if family == "ball":
        return ((3.0 / 5.0) * K * Q * Q / (4.0 * math.pi * pressure)) ** 0.25
    raise KeyError(family)


def outward_energy_derivative(radius: float, family: str) -> float:
    """−dE/dR, сила, раздувающая радиус."""
    if family == "shell":
        return K * Q * Q / (2.0 * radius * radius)
    if family == "ball":
        return (3.0 / 5.0) * K * Q * Q / (radius * radius)
    raise KeyError(family)


def acceleration(radius: float, speed: float, pressure: float, family: str) -> float:
    force = outward_energy_derivative(radius, family) - 4.0 * math.pi * radius * radius * pressure
    return (force - GAMMA * speed) / MU


def relax(pressure: float, family: str, radius0: float, speed0: float = 0.0, t_end: float = T_RELAX) -> dict[str, float]:
    def ode(_t: float, y: np.ndarray) -> list[float]:
        radius, speed = float(y[0]), float(y[1])
        return [speed, acceleration(radius, speed, pressure, family)]

    solution = solve_ivp(
        ode,
        (0.0, t_end),
        (radius0, speed0),
        rtol=1e-8,
        atol=1e-8,
        dense_output=False,
        max_step=0.05,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    radius, speed = (float(v) for v in solution.y[:, -1])
    return {"t": float(solution.t[-1]), "radius": radius, "speed": speed, "n_eval": int(solution.nfev)}


def shell_quadrature(radius: float, order: int = 64) -> dict[str, float]:
    """Сила и потенциал тонкой оболочки. Внутри сила — интеграл, не обнуление порогом."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    phi = (nodes + 1.0) * math.pi
    weight_phi = weights * math.pi
    u_nodes, u_weights = np.polynomial.legendre.leggauss(order)
    force_error = 0.0
    potential_error = 0.0
    outside_error = 0.0
    probes_in = (0.0, 0.25 * radius, 0.5 * radius, 0.9 * radius)
    for probe in probes_in:
        # точка на оси z = probe
        # r_vec от пробы к элементу: (R sinθ cosφ, R sinθ sinφ, R cosθ − probe)
        potential = 0.0
        fz = 0.0
        for u, wu in zip(u_nodes, u_weights, strict=True):
            sin_theta = math.sqrt(max(0.0, 1.0 - u * u))
            z = radius * u
            dz = z - probe
            for p, wp in zip(phi, weight_phi, strict=True):
                # площадь: R^2 dφ du, заряд K не нужен отдельно: σ dA = Q du dφ / (4π)
                dcharge = Q * wu * wp / (4.0 * math.pi)
                dx = radius * sin_theta * math.cos(p)
                dy = radius * sin_theta * math.sin(p)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                potential += dcharge / dist
                fz += dcharge * dz / dist**3
        expected_potential = Q / radius
        force_error = max(force_error, abs(fz))
        potential_error = max(potential_error, abs(potential - expected_potential) / expected_potential)
    for factor in (1.5, 2.0, 4.0):
        probe = factor * radius
        fz = 0.0
        for u, wu in zip(u_nodes, u_weights, strict=True):
            sin_theta = math.sqrt(max(0.0, 1.0 - u * u))
            dz = radius * u - probe
            for p, wp in zip(phi, weight_phi, strict=True):
                dcharge = Q * wu * wp / (4.0 * math.pi)
                dx = radius * sin_theta * math.cos(p)
                dy = radius * sin_theta * math.sin(p)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                fz += dcharge * dz / dist**3
        expected = Q / probe**2
        outside_error = max(outside_error, abs(abs(fz) - expected) / expected)
    return {
        "order": order,
        "max_abs_force_inside": force_error,
        "max_rel_potential_inside": potential_error,
        "max_rel_force_outside": outside_error,
    }


def run_family(family: str) -> dict[str, object]:
    rows = []
    for pressure in PRESSURES:
        predicted = radius_equilibrium(pressure, family)
        start = 0.5 * radius_equilibrium(1.0, family)
        done = relax(pressure, family, start)
        rel = abs(done["radius"] - predicted) / predicted
        rows.append(
            {
                "pressure": pressure,
                "predicted_radius": predicted,
                "radius": done["radius"],
                "speed": done["speed"],
                "relative_error": rel,
                "accepted": rel < 1e-3 and abs(done["speed"]) < 1e-3,
            }
        )
    settled = relax(1.0, family, 0.5 * radius_equilibrium(1.0, family))
    dropped = relax(1.0 / 16.0, family, settled["radius"], settled["speed"])
    predicted_drop = radius_equilibrium(1.0 / 16.0, family)
    growth = dropped["radius"] / settled["radius"]
    free = relax(0.0, family, settled["radius"], 0.0, t_end=T_FREE)
    return {
        "family": family,
        "rows": rows,
        "drop": {
            "radius_before": settled["radius"],
            "radius_after": dropped["radius"],
            "predicted_after": predicted_drop,
            "growth": growth,
            "predicted_growth": 2.0,
            "relative_error": abs(dropped["radius"] - predicted_drop) / predicted_drop,
            "accepted": abs(growth - 2.0) / 2.0 < 1e-3 and abs(dropped["speed"]) < 1e-3,
        },
        "zero_pressure": {
            "radius_start": settled["radius"],
            "radius_end": free["radius"],
            "growth": free["radius"] / settled["radius"],
            "unbounded_record": free["radius"] > 2.0 * settled["radius"],
        },
    }


def main() -> None:
    quadrature = {
        "order_64": shell_quadrature(radius_equilibrium(1.0, "shell"), order=64),
        "order_128": shell_quadrature(radius_equilibrium(1.0, "shell"), order=128),
    }
    payload = {
        "scope": "model",
        "protocol": "docs/field-pressure-balance-protocol-2026-10-27.md",
        "chamber": False,
        "k": K,
        "Q": Q,
        "families": [run_family("shell"), run_family("ball")],
        "shell_quadrature_at_P1": quadrature,
    }
    coarse = quadrature["order_64"]
    fine = quadrature["order_128"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for family in payload["families"]:
        print(family["family"])
        for row in family["rows"]:
            print(
                f"  P={row['pressure']:.4g} R={row['radius']:.6g} "
                f"pred={row['predicted_radius']:.6g} err={row['relative_error']:.3g} ok={row['accepted']}"
            )
        drop = family["drop"]
        print(
            f"  drop x16 growth={drop['growth']:.6g} err={drop['relative_error']:.3g} ok={drop['accepted']}"
        )
        zero = family["zero_pressure"]
        print(f"  P=0 growth={zero['growth']:.6g} unbounded_record={zero['unbounded_record']}")
    print(
        "quadrature64 inside_force",
        f"{coarse['max_abs_force_inside']:.3g}",
        "potential",
        f"{coarse['max_rel_potential_inside']:.3g}",
    )
    print(
        "quadrature128 inside_force",
        f"{fine['max_abs_force_inside']:.3g}",
        "potential",
        f"{fine['max_rel_potential_inside']:.3g}",
        "outside",
        f"{fine['max_rel_force_outside']:.3g}",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
