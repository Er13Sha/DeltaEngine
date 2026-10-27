"""Сводка сохранённых прогонов E-COMP v1.

Протокол и исходные ряды не изменяются. P2 здесь не запускается:
она допустима только после кандидата удержания без существенного
изменения плотности окружения.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/field/external_compression_v1"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _relative(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1e-6)


def _window(trace, key, start=5.0):
    values = [row[key] for row in trace if row["t"] >= start]
    return _mean(values)


def _campaign_row(path: Path):
    data = _load(path)
    trace = data["trace"]
    unbound = next(row["t"] for row in trace if row["max_radius"] > 1.0)
    parameters = data["parameters"]
    return {
        "file": path.name,
        "radius": parameters["radius"],
        "seed": parameters["seed"],
        "omega": parameters["omega"],
        "nodes": data["nodes"],
        "status": data["status"],
        "quality": data["quality"],
        "candidate": data["candidate"],
        "bounded": data["bounded"],
        "stationary_environment": data["stationary_environment"],
        "energy_relative_drift": data["energy_relative_drift"],
        "normalized_momentum_drift": data["normalized_momentum_drift"],
        "external_density_relative_drift": data["external_density_relative_drift"],
        "radius_window_max_over_min": data["radius_window_max_over_min"],
        "rms_end": trace[-1]["rms_radius"],
        "rms_window_mean": _window(trace, "rms_radius"),
        "c_ext_initial": trace[0]["C_ext"],
        "c_ext_window_mean": _window(trace, "C_ext"),
        "first_unbound_sample_time": unbound,
        "runtime": data["runtime"],
        "code_sha256": data["code_sha256"],
        "protocol_sha256": data["protocol_sha256"],
    }


def _grouped(rows):
    groups = []
    for omega in (0.0, 0.5, 1.0):
        item = {"omega": omega, "radii": {}}
        for radius in (4, 6, 8):
            selected = [row for row in rows if row["omega"] == omega and row["radius"] == radius]
            item["radii"][str(radius)] = {
                "runs": len(selected),
                "rms_window_mean": _mean(row["rms_window_mean"] for row in selected),
                "c_ext_window_mean": _mean(row["c_ext_window_mean"] for row in selected),
                "rms_end_mean": _mean(row["rms_end"] for row in selected),
            }
        left = item["radii"]["6"]
        right = item["radii"]["8"]
        rms_difference = _relative(left["rms_window_mean"], right["rms_window_mean"])
        c_difference = _relative(left["c_ext_window_mean"], right["c_ext_window_mean"])
        item["radius6_vs_radius8"] = {
            "rms_relative_difference": rms_difference,
            "c_ext_relative_difference": c_difference,
            "passed": rms_difference <= 0.05 and c_difference <= 0.05,
        }
        groups.append(item)
    return groups


def _timestep_checks():
    checks = []
    for radius, omega in ((4, 0.0), (4, 0.5), (4, 1.0), (8, 0.0), (8, 0.5), (8, 1.0)):
        name = f"mobile-R{radius}-seed1-omega{omega:g}"
        coarse = {round(row["t"], 6): row["rms_radius"] for row in _load(OUT / "campaign" / f"{name}-dt0.002.json")["trace"]}
        fine = {round(row["t"], 6): row["rms_radius"] for row in _load(OUT / "controls" / f"{name}-dt0.001.json")["trace"]}
        times = sorted(set(coarse) & set(fine))
        difference = max(abs(coarse[time] - fine[time]) / max(abs(coarse[time]), abs(fine[time]), 1e-12) for time in times)
        checks.append({
            "radius": radius,
            "seed": 1,
            "omega": omega,
            "samples": len(times),
            "max_rms_relative_difference": difference,
            "passed": difference < 0.01,
        })
    return checks


def _control_overview(prefix):
    rows = []
    for path in sorted((OUT / "controls").glob(f"{prefix}-*-dt0.002.json")):
        data = _load(path)
        rows.append({
            "file": path.name,
            "status": data["status"],
            "quality": data["quality"],
            "candidate": data["candidate"],
            "final_time": data["trace"][-1]["t"],
            "rms_end": data["trace"][-1]["rms_radius"],
        })
    return rows


def build_summary():
    campaign = [_campaign_row(path) for path in sorted((OUT / "campaign").glob("mobile-*.json"))]
    groups = _grouped(campaign)
    timesteps = _timestep_checks()
    frozen = _control_overview("frozen")
    isolated = _control_overview("isolated")
    p0 = _load(OUT / "p0.json")
    pilot = _load(OUT / "pilot/mobile-R4-seed1-omega0-dt0.002.json")
    repeated = _load(OUT / "campaign/mobile-R4-seed1-omega0-dt0.002.json")
    pilot_difference = max(abs(left["rms_radius"] - right["rms_radius"]) for left, right in zip(pilot["trace"], repeated["trace"]))
    candidates = [row for row in campaign if row["candidate"]]
    expanded = all(row["external_density_relative_drift"] > 0.1 for row in campaign)
    same_class = all(not row["candidate"] for row in campaign)
    summary = {
        "scope": "model",
        "protocol_id": "E-COMP-v1",
        "empirical_status_changed": False,
        "p0_passed": bool(p0["passed"]),
        "protocol_sha256": campaign[0]["protocol_sha256"],
        "code_sha256": campaign[0]["code_sha256"],
        "campaign_runs": len(campaign),
        "quality_runs": sum(row["quality"] for row in campaign),
        "candidate_count": len(candidates),
        "bounded_count": sum(row["bounded"] for row in campaign),
        "stationary_environment_count": sum(row["stationary_environment"] for row in campaign),
        "finite_environment_expanded": expanded,
        "same_qualitative_classification": same_class,
        "qualitative_classification": "no_retention",
        "first_unbound_sample_time": {
            "min": min(row["first_unbound_sample_time"] for row in campaign),
            "max": max(row["first_unbound_sample_time"] for row in campaign),
        },
        "c_ext_initial": {
            "min": min(row["c_ext_initial"] for row in campaign),
            "max": max(row["c_ext_initial"] for row in campaign),
        },
        "external_density_relative_drift": {
            "min": min(row["external_density_relative_drift"] for row in campaign),
            "max": max(row["external_density_relative_drift"] for row in campaign),
        },
        "energy_relative_drift_max": max(row["energy_relative_drift"] for row in campaign),
        "normalized_momentum_drift_max": max(row["normalized_momentum_drift"] for row in campaign),
        "rms_end": {
            "min": min(row["rms_end"] for row in campaign),
            "max": max(row["rms_end"] for row in campaign),
        },
        "radius_window_max_over_min": {
            "min": min(row["radius_window_max_over_min"] for row in campaign),
            "max": max(row["radius_window_max_over_min"] for row in campaign),
        },
        "groups": groups,
        "convergence_passed": all(item["radius6_vs_radius8"]["passed"] for item in groups) and same_class,
        "timestep_checks": timesteps,
        "timestep_passed": all(item["passed"] for item in timesteps),
        "pilot_repeat_rms_abs_difference": pilot_difference,
        "isolated_controls": isolated,
        "frozen_controls": {
            "complete": sum(row["status"] == "complete" for row in frozen),
            "unresolved_close_pair": [row["file"] for row in frozen if row["status"] != "complete"],
            "candidate_count": sum(row["candidate"] for row in frozen),
            "rms_end_complete": {
                "min": min(row["rms_end"] for row in frozen if row["status"] == "complete"),
                "max": max(row["rms_end"] for row in frozen if row["status"] == "complete"),
            },
        },
        "p2_executed": False,
        "found_particle": False,
        "refutes_general_scheme": False,
        "runs": campaign,
    }
    summary["decision"] = (
        "P0 пройден. P1 не дал удержания: минимальная модель парного отталкивания "
        "не сохраняет размер локальной группы в проверенных условиях. Конечное окружение "
        "существенно расширяется, поэтому результат не опровергает схему бесконечного окружения "
        "и не объявляет частицу. P2 не запускалась."
    )
    return summary


def campaign_record(summary):
    return {
        "id": "E-COMP-v1-campaign",
        "protocol_id": summary["protocol_id"],
        "protocol_sha256": summary["protocol_sha256"],
        "code_sha256": summary["code_sha256"],
        "acceptance": "data/experiments/field/external_compression_protocol_v1.acceptance.json",
        "simulation_started": True,
        "p0_passed": summary["p0_passed"],
        "campaign_runs": summary["campaign_runs"],
        "candidate_count": summary["candidate_count"],
        "p2_executed": summary["p2_executed"],
        "finite_environment_expanded": summary["finite_environment_expanded"],
        "convergence_passed": summary["convergence_passed"],
        "timestep_passed": summary["timestep_passed"],
        "found_particle": False,
        "refutes_general_scheme": False,
        "empirical_status_changed": False,
        "scope": "model",
        "summary": "data/experiments/field/external_compression_v1/summary.json",
        "report": "docs/external-compression-v1-result.md",
        "decision": summary["decision"],
    }


def write_outputs():
    summary = build_summary()
    record = campaign_record(summary)
    summary_path = OUT / "summary.json"
    record_path = OUT.parent / "external_compression_campaign_record.json"
    for path, payload in ((summary_path, summary), (record_path, record)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path, record_path


if __name__ == "__main__":
    print("\n".join(str(path) for path in write_outputs()))
