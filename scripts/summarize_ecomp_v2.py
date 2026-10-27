"""Сводка прогонов E-COMP v2. Динамику не запускает."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/field/external_compression_v2"
OUT = BASE / "p1_summary.json"
HOLD_RADIUS = 1.0
WINDOW = (5.0, 10.0)
ENERGY_MAX = 1e-3
MOMENTUM_MAX = 1e-8
RMS_RATIO_MAX = 1.1
DENSITY_FLAG = 0.1
DT_REL_MAX = 0.01
SATURATION = 0.1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def load_runs() -> list[dict]:
    files = sorted(BASE.glob("controls/*.json")) + sorted(BASE.glob("pilot/*.json")) + sorted(BASE.glob("runs/*.json"))
    return [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "data": json.loads(path.read_text(encoding="utf-8"))} for path in files]


def summarize(path: str, data: dict) -> dict:
    params = data["parameters"]
    trace = data["trace"]
    window = [x for x in trace if WINDOW[0] <= x["t"] <= WINDOW[1]]
    rms_w = [x["rms_radius"] for x in window]
    c_w = [x["C_ext"] for x in window]
    inner_w = [x["belt_inner_boundary_fraction"] for x in window]
    outer_w = [x["belt_outer_boundary_fraction"] for x in window]
    ratio = max(rms_w) / min(rms_w) if rms_w and min(rms_w) > 0 else None
    bounded = all(x["max_radius"] <= HOLD_RADIUS for x in trace)
    holding = data["status"] == "complete" and bounded and ratio is not None and ratio <= RMS_RATIO_MAX
    first, last = trace[0], trace[-1]
    return {
        "path": path,
        "folder": path.split("/")[-2],
        "family": params["family"],
        "seed": params["seed"],
        "omega": params["omega"],
        "dt": params["dt"],
        "tmax": params["tmax"],
        "nodes": data["nodes"],
        "active_nodes": data["active_nodes"],
        "n_inf": data["n_inf"],
        "status": data["status"],
        "t_end": last["t"],
        "elapsed_seconds": data["elapsed_seconds"],
        "protocol_sha256": data["protocol_sha256"],
        "code_sha256": data["code_sha256"],
        "energy_relative_drift": data["energy_relative_drift"],
        "normalized_momentum_drift": data["normalized_momentum_drift"],
        "energy_ok": data["energy_relative_drift"] < ENERGY_MAX,
        "momentum_ok": params["family"] == "frozen" or data["normalized_momentum_drift"] < MOMENTUM_MAX,
        "numerics_ok": data["numerics_ok"],
        "quality_recorded": data["quality"],
        "constraint_hit_fraction": data["constraint_hit_fraction"],
        "constraint_saturated": data["constraint_saturated"],
        "bounded": bounded,
        "max_radius_all": max(x["max_radius"] for x in trace),
        "rms_end": last["rms_radius"],
        "C_ext_start": first["C_ext"],
        "C_ext_end": last["C_ext"],
        "window_n": len(window),
        "window_mean_rms": mean(rms_w),
        "window_mean_C_ext": mean(c_w),
        "window_mean_inner": mean(inner_w),
        "window_mean_outer": mean(outer_w),
        "radius_window_max_over_min": ratio,
        "holding": holding,
        "candidate_recorded": data["candidate"],
        "band_density_relative_drift": data["belt_band_density_relative_drift"],
        "r95_density_relative_drift": data["belt_r95_density_relative_drift"],
        "r95_start": first.get("belt_r95"),
        "r95_end": last.get("belt_r95"),
        "density_start": first.get("belt_r95_density"),
        "density_end": last.get("belt_r95_density"),
        "outer_end": last["belt_outer_boundary_fraction"],
        "min_pair_end": last["minimum_pair_distance"],
    }


def matching_rms_difference(a: dict, b: dict) -> dict:
    ta = {round(x["t"], 8): x["rms_radius"] for x in a["trace"]}
    tb = {round(x["t"], 8): x["rms_radius"] for x in b["trace"]}
    times = sorted(set(ta) & set(tb))
    rel = [abs(ta[t] - tb[t]) / max(abs(ta[t]), abs(tb[t]), 1e-6) for t in times]
    return {
        "n_times": len(times),
        "t_max_compared": max(times) if times else None,
        "max_relative_rms_difference": max(rel) if rel else None,
        "pass": bool(rel) and max(rel) < DT_REL_MAX,
    }


def main() -> None:
    loaded = load_runs()
    rows = [summarize(item["path"], item["data"]) for item in loaded]
    by_key = {}
    for row, item in zip(rows, loaded):
        key = f"{row['family']}-seed{row['seed']}-omega{row['omega']:g}-dt{row['dt']:g}"
        if row["folder"] == "pilot":
            key = "pilot-" + key
        by_key[key] = (row, item["data"])
    dt_pairs = []
    for omega in (0.0, 0.5, 1.0):
        coarse = f"mobile_reservoir-seed1-omega{omega:g}-dt0.002"
        fine = f"mobile_reservoir-seed1-omega{omega:g}-dt0.001"
        if coarse in by_key and fine in by_key:
            cmp = matching_rms_difference(by_key[coarse][1], by_key[fine][1])
            cmp["coarse"] = coarse
            cmp["fine"] = fine
            dt_pairs.append(cmp)
    averages = []
    for family in ("frozen", "mobile_reservoir"):
        for omega in (0.0, 0.5, 1.0):
            group = [
                r for r in rows
                if r["family"] == family and r["omega"] == omega and r["dt"] == 0.002 and r["folder"] in {"controls", "runs"}
                and r["window_mean_rms"] is not None
            ]
            if not group:
                continue
            averages.append({
                "family": family,
                "omega": omega,
                "n": len(group),
                "seeds": sorted(r["seed"] for r in group),
                "mean_rms": mean([r["window_mean_rms"] for r in group]),
                "mean_C_ext": mean([r["window_mean_C_ext"] for r in group]),
                "all_holding": all(r["holding"] for r in group),
                "all_saturated": all(r["constraint_saturated"] for r in group),
            })
    families = {}
    for family in ("isolated", "frozen", "mobile_open", "mobile_reservoir"):
        subset = [r for r in rows if r["family"] == family and r["folder"] != "pilot"]
        families[family] = {
            "n": len(subset),
            "complete": sum(r["status"] == "complete" for r in subset),
            "unresolved_close_pair": sum(r["status"] == "unresolved_close_pair" for r in subset),
            "constraint_saturated": sum(r["status"] == "constraint_saturated" for r in subset),
            "holding": sum(r["holding"] for r in subset),
            "candidate": sum(r["candidate_recorded"] for r in subset),
            "numerics_ok": sum(r["numerics_ok"] for r in subset),
        }
    t0 = json.loads((BASE / "t0.json").read_text(encoding="utf-8"))
    result = {
        "scope": "model",
        "protocol_id": "E-COMP-v2",
        "script": "scripts/summarize_ecomp_v2.py",
        "hold_radius": HOLD_RADIUS,
        "t0": {"passed": t0["passed"], "protocol_sha256": t0["protocol_sha256"], "code_sha256": t0["code_sha256"], "rows": t0["rows"]},
        "file_hashes": {
            "t0": sha256(BASE / "t0.json"),
            "protocol_json": sha256(ROOT / "data/experiments/field/external_compression_protocol_v2.json"),
            "protocol_md": sha256(ROOT / "docs/external-compression-protocol-v2.md"),
            "code": sha256(ROOT / "adapters/field_lattice/external_compression_v2.py"),
            "acceptance": sha256(ROOT / "data/experiments/field/external_compression_protocol_v2.acceptance.json"),
        },
        "run_hashes": sorted({(r["protocol_sha256"], r["code_sha256"]) for r in rows}),
        "n_runs_read": len(rows),
        "families": families,
        "dt_comparison": dt_pairs,
        "seed_averages": averages,
        "holding_any": any(r["holding"] for r in rows),
        "candidate_any": any(r["candidate_recorded"] for r in rows),
        "unresolved": sum(r["status"] == "unresolved_close_pair" for r in rows),
        "runs": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT).as_posix())
    print("n_runs", len(rows))
    print("families", json.dumps(families, ensure_ascii=False))
    print("holding_any", result["holding_any"], "candidate_any", result["candidate_any"], "unresolved", result["unresolved"])
    print("dt", json.dumps([{k: x[k] for k in ("coarse", "max_relative_rms_difference", "pass")} for x in dt_pairs], ensure_ascii=False))


if __name__ == "__main__":
    main()
