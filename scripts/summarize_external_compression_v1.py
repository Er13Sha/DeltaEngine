"""Состояние прежней, неполной кампании v1 после уточнения требований v2."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/field/external_compression_v1"


def main():
    records = {}
    hashes = []
    for folder in ("pilot", "runs", "controls"):
        paths = sorted((OUT/folder).glob("*.json"))
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        records[folder] = rows
        hashes += [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    rows = records["runs"]
    assert rows
    comparisons = []
    for coarse in rows:
        p = coarse["parameters"]
        matches = [r for r in records["controls"] if all(r["parameters"][k] == p[k] for k in ("radius", "seed", "omega", "mode"))
                   and r["parameters"]["dt"] == p["dt"]/2]
        for fine in matches:
            tc = np.array([r["t"] for r in coarse["trace"]])
            tf = np.array([r["t"] for r in fine["trace"]])
            if len(tc) != len(tf) or not np.allclose(tc, tf, atol=1e-12):
                continue
            rc = np.array([r["rms_radius"] for r in coarse["trace"]])
            rf = np.array([r["rms_radius"] for r in fine["trace"]])
            delta = float(np.max(np.abs(rc/rf-1)))
            comparisons.append({"radius": p["radius"], "seed": p["seed"], "omega": p["omega"],
                                "max_relative_radius_difference": delta, "passes_1_percent": delta < .01})
    final = [r["trace"][-1]["rms_radius"] for r in rows]
    result = {
        "scope": "model", "status": "partial_campaign_superseded_by_v2_requirements",
        "expected_main_runs": 27, "saved_main_runs": len(rows), "complete_main_runs": sum(r["status"] == "complete" for r in rows),
        "main_quality_passes_before_missing_timestep_checks": sum(r["quality"] for r in rows),
        "candidate_count": sum(r["candidate"] for r in rows), "initial_rms_radius": .5,
        "final_rms_radius_range": [min(final), max(final)],
        "max_energy_drift": max(r["energy_relative_drift"] for r in rows),
        "max_normalized_momentum_drift": max(r["normalized_momentum_drift"] for r in rows),
        "environment_density_relative_drift_range": [min(r["external_density_relative_drift"] for r in rows), max(r["external_density_relative_drift"] for r in rows)],
        "saved_controls": len(records["controls"]),
        "controls_with_unresolved_close_pairs": [{"parameters": r["parameters"], "last_sample_t": r["trace"][-1]["t"]}
                                                for r in records["controls"] if r["status"] != "complete"],
        "timestep_checks_available": comparisons,
        "interpretation": "No bound central group among saved runs. Finite surroundings expand; this does not simulate the subsequently required constant universal complex or attenuation by intervening bodies.",
        "full_protocol_completed": False, "new_model_simulated": False,
        "P2_status": "not_started_no_P1_candidate", "source_files": hashes,
    }
    output = OUT/"partial_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
