"""Сводка уже посчитанных прогонов E-COMP v1 для вердикта P1. Динамику не запускает."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/field/external_compression_v1"
OUT = ROOT / "data/experiments/field/external_compression_v1/p1_summary.json"
R0 = 0.5
HOLD_RADIUS = 2 * R0
WINDOW = (5.0, 10.0)
ENERGY_MAX = 1e-3
MOMENTUM_MAX = 1e-8
RMS_RATIO_MAX = 1.1
DENSITY_FLAG = 0.1
DT_REL_MAX = 0.01
CONV_REL_MAX = 0.05
CONV_FLOOR = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs() -> list[dict]:
    files = sorted(BASE.glob("controls/*.json")) + sorted(BASE.glob("pilot/*.json"))
    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "data": data})
    return rows


def window_rows(trace: list[dict]) -> list[dict]:
    return [x for x in trace if WINDOW[0] <= x["t"] <= WINDOW[1]]


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize(path: str, data: dict) -> dict:
    params = data["parameters"]
    trace = data["trace"]
    t_end = trace[-1]["t"]
    window = window_rows(trace)
    rms_w = [x["rms_radius"] for x in window]
    c_w = [x["C_ext"] for x in window]
    max_all = max(x["max_radius"] for x in trace)
    max_window = max((x["max_radius"] for x in window), default=None)
    bounded = all(x["max_radius"] <= HOLD_RADIUS for x in trace)
    ratio = max(rms_w) / min(rms_w) if rms_w and min(rms_w) > 0 else None
    holding = (
        data["status"] == "complete"
        and bounded
        and ratio is not None
        and ratio <= RMS_RATIO_MAX
    )
    first, last = trace[0], trace[-1]
    momentum_ok = params["mode"] == "frozen" or data["normalized_momentum_drift"] < MOMENTUM_MAX
    energy_ok = data["energy_relative_drift"] < ENERGY_MAX
    numerics_ok = data["status"] == "complete" and energy_ok and momentum_ok
    density_drift = data["external_density_relative_drift"]
    return {
        "path": path,
        "folder": "pilot" if "/pilot/" in path else "controls",
        "mode": params["mode"],
        "R": params["radius"],
        "seed": params["seed"],
        "omega": params["omega"],
        "dt": params["dt"],
        "tmax": params["tmax"],
        "nodes": data["nodes"],
        "active_nodes": data["active_nodes"],
        "status": data["status"],
        "t_end": t_end,
        "protocol_sha256": data["protocol_sha256"],
        "code_sha256": data["code_sha256"],
        "energy_relative_drift": data["energy_relative_drift"],
        "normalized_momentum_drift": data["normalized_momentum_drift"],
        "energy_ok": energy_ok,
        "momentum_ok": momentum_ok,
        "numerics_ok": numerics_ok,
        "quality_recorded": data["quality"],
        "bounded_recorded": data["bounded"],
        "bounded": bounded,
        "max_radius_all": max_all,
        "max_radius_window": max_window,
        "rms_end": last["rms_radius"],
        "max_radius_end": last["max_radius"],
        "C_ext_start": first["C_ext"],
        "C_ext_end": last["C_ext"],
        "window_n": len(window),
        "window_mean_rms": mean(rms_w),
        "window_mean_C_ext": mean(c_w),
        "radius_window_max_over_min": ratio,
        "radius_ratio_recorded": data["radius_window_max_over_min"],
        "holding": holding,
        "candidate_recorded": data["candidate"],
        "density_relative_drift": density_drift,
        "density_substantial": None if density_drift is None else density_drift > DENSITY_FLAG,
        "r95_start": first.get("external_r95"),
        "r95_end": last.get("external_r95"),
        "density_start": first.get("external_density"),
        "density_end": last.get("external_density"),
        "min_pair_end": last["minimum_pair_distance"],
        "histogram_start": first["external_radial_counts"],
        "histogram_end": last["external_radial_counts"],
    }


def matching_rms_difference(a: dict, b: dict) -> dict:
    ta = {round(x["t"], 8): x["rms_radius"] for x in a["trace"]}
    tb = {round(x["t"], 8): x["rms_radius"] for x in b["trace"]}
    times = sorted(set(ta) & set(tb))
    rel = []
    for t in times:
        denom = max(abs(ta[t]), abs(tb[t]), CONV_FLOOR)
        rel.append(abs(ta[t] - tb[t]) / denom)
    return {
        "n_times": len(times),
        "t_max_compared": max(times) if times else None,
        "max_relative_rms_difference": max(rel) if rel else None,
        "pass": bool(rel) and max(rel) < DT_REL_MAX,
    }


def seed_average(rows: list[dict], radius: float, omega: float, mode: str) -> dict | None:
    group = [
        r for r in rows
        if r["mode"] == mode and r["R"] == radius and r["omega"] == omega and r["dt"] == 0.002
        and r["folder"] == "controls" and r["status"] == "complete"
        and r["window_mean_rms"] is not None
    ]
    seeds = sorted(r["seed"] for r in group)
    if not group:
        return None
    return {
        "mode": mode,
        "R": radius,
        "omega": omega,
        "n": len(group),
        "seeds": seeds,
        "mean_rms": mean([r["window_mean_rms"] for r in group]),
        "mean_C_ext": mean([r["window_mean_C_ext"] for r in group]),
        "all_holding": all(r["holding"] for r in group),
        "any_unresolved_excluded": True,
    }


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), CONV_FLOOR)


def convergence(averages: list[dict]) -> list[dict]:
    out = []
    for omega in (0.0, 0.5, 1.0):
        for mode in ("frozen", "mobile"):
            six = next((x for x in averages if x and x["R"] == 6 and x["omega"] == omega and x["mode"] == mode), None)
            eight = next((x for x in averages if x and x["R"] == 8 and x["omega"] == omega and x["mode"] == mode), None)
            if not six or not eight:
                out.append({
                    "mode": mode, "omega": omega, "available": False,
                    "reason": "нет полного тройного среднего по зёрнам для R=6 и R=8",
                })
                continue
            rms_rel = rel_diff(six["mean_rms"], eight["mean_rms"])
            c_rel = rel_diff(six["mean_C_ext"], eight["mean_C_ext"])
            out.append({
                "mode": mode,
                "omega": omega,
                "available": True,
                "R6_n": six["n"],
                "R8_n": eight["n"],
                "R6_mean_rms": six["mean_rms"],
                "R8_mean_rms": eight["mean_rms"],
                "R6_mean_C_ext": six["mean_C_ext"],
                "R8_mean_C_ext": eight["mean_C_ext"],
                "rms_relative_difference": rms_rel,
                "C_ext_relative_difference": c_rel,
                "rms_pass": rms_rel <= CONV_REL_MAX,
                "C_ext_pass": c_rel <= CONV_REL_MAX,
            })
    return out


def p0_excerpt(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    fine = [x for x in data["rows"] if x["order"] == 128]
    interior = [x for x in fine if x["r"] < 1]
    exterior = [x for x in fine if x["r"] > 1]
    return {
        "passed": data["passed"],
        "protocol_sha256": data["protocol_sha256"],
        "code_sha256": data["code_sha256"],
        "max_interior_abs_error": max(x["absolute_error"] for x in interior),
        "max_exterior_rel_error": max(x["relative_error"] for x in exterior),
        "interior": [{"r": x["r"], "force": x["force"], "absolute_error": x["absolute_error"]} for x in interior],
        "exterior": [{"r": x["r"], "force": x["force"], "expected": x["expected"], "relative_error": x["relative_error"]} for x in exterior],
    }


def expected_missing() -> dict:
    main = [
        f"mobile-R{R}-seed{seed}-omega{omega:g}-dt0.002"
        for R in (4, 6, 8) for seed in (1, 2, 3) for omega in (0.0, 0.5, 1.0)
    ]
    half = [
        f"mobile-R{R}-seed1-omega{omega:g}-dt0.001"
        for R in (4, 8) for omega in (0.0, 0.5, 1.0)
    ]
    return {"main_mobile_dt002": main, "half_step_mobile": half}


def main() -> None:
    loaded = load_runs()
    rows = [summarize(item["path"], item["data"]) for item in loaded]
    by_key = {
        f"{r['mode']}-R{r['R']}-seed{r['seed']}-omega{r['omega']:g}-dt{r['dt']:g}": (r, item["data"])
        for r, item in zip(rows, loaded)
    }
    present = set(by_key)
    missing = expected_missing()
    missing_main = [k for k in missing["main_mobile_dt002"] if k not in present]
    missing_half = [k for k in missing["half_step_mobile"] if k not in present]
    dt_pairs = []
    pair_key = ("mobile-R4-seed1-omega0-dt0.002", "mobile-R4-seed1-omega0-dt0.001")
    if pair_key[0] in by_key and pair_key[1] in by_key:
        cmp = matching_rms_difference(by_key[pair_key[0]][1], by_key[pair_key[1]][1])
        cmp["coarse"] = pair_key[0]
        cmp["fine"] = pair_key[1]
        dt_pairs.append(cmp)
    averages = []
    for mode in ("frozen", "mobile", "isolated"):
        for radius in (4, 6, 8):
            for omega in (0.0, 0.5, 1.0):
                item = seed_average(rows, radius, omega, mode)
                if item:
                    averages.append(item)
    hashes = sorted({(r["protocol_sha256"], r["code_sha256"]) for r in rows})
    families = {}
    for mode in ("frozen", "mobile", "isolated"):
        subset = [r for r in rows if r["mode"] == mode]
        families[mode] = {
            "n": len(subset),
            "complete": sum(r["status"] == "complete" for r in subset),
            "unresolved_close_pair": sum(r["status"] == "unresolved_close_pair" for r in subset),
            "holding": sum(r["holding"] for r in subset),
            "numerics_ok": sum(r["numerics_ok"] for r in subset),
            "density_substantial": sum(r["density_substantial"] is True for r in subset),
        }
    qualitative = {}
    for mode in ("frozen", "mobile", "isolated"):
        by_r = {}
        for radius in (4, 6, 8):
            group = [r for r in rows if r["mode"] == mode and r["R"] == radius and r["dt"] == 0.002]
            if not group:
                by_r[str(radius)] = "нет прогонов"
                continue
            labels = sorted({"удержание" if r["holding"] else "нет удержания" for r in group if r["status"] == "complete"})
            unresolved = sum(r["status"] != "complete" for r in group)
            by_r[str(radius)] = {
                "complete_labels": labels,
                "unresolved": unresolved,
                "n": len(group),
            }
        qualitative[mode] = by_r
    result = {
        "scope": "model",
        "protocol_id": "E-COMP-v1",
        "script": "scripts/summarize_ecomp_p1.py",
        "hold_radius": HOLD_RADIUS,
        "p0": p0_excerpt(BASE / "p0.json"),
        "file_hashes": {
            "p0": sha256(BASE / "p0.json"),
            "protocol_json": sha256(ROOT / "data/experiments/field/external_compression_protocol_v1.json"),
            "protocol_md": sha256(ROOT / "docs/external-compression-protocol-v1.md"),
            "code": sha256(ROOT / "adapters/field_lattice/external_compression.py"),
        },
        "run_hashes": hashes,
        "n_runs_read": len(rows),
        "families": families,
        "missing_main_mobile_dt002": missing_main,
        "missing_half_step_mobile": missing_half,
        "dt_comparison": dt_pairs,
        "seed_averages": averages,
        "convergence_R6_R8": convergence(averages),
        "qualitative_by_radius": qualitative,
        "runs": [{k: v for k, v in r.items()} for r in rows],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT).as_posix())
    print("n_runs", len(rows))
    print("families", json.dumps(families, ensure_ascii=False))
    print("missing_main", len(missing_main), "missing_half", len(missing_half))
    print("dt_pairs", json.dumps(dt_pairs, ensure_ascii=False))
    print("holding_any", any(r["holding"] for r in rows))
    print("unresolved", sum(r["status"] == "unresolved_close_pair" for r in rows))


if __name__ == "__main__":
    main()
