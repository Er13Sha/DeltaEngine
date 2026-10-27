"""Анализ сохранённых результатов, без повторного запуска динамики."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/experiments/field"


def fit(rows, kappa):
    d = np.array([x["d"] for x in rows])
    force = np.array([abs(x["force"]) for x in rows])
    log_g = np.log(force*d*d) + kappa*d - np.log1p(kappa*d)
    return {"kappa": float(kappa), "g": float(np.exp(log_g.mean())),
            "log_rms": float(np.std(log_g)), "separations": d.tolist()}


def main():
    source = DATA / "qball.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    q = saved["selected"]
    kappa = math.sqrt(1-q["omega"]**2)
    result = {"scope": "model", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256((ROOT / "adapters/field_lattice/qball.py").read_bytes()).hexdigest(),
              "kappa_predicted": kappa, "fits": {}, "refinement": {}}
    for case in ("same_in", "same_anti", "opposite"):
        rows = sorted([r for r in saved["pairs"] if r["case"] == case and r["h"] == .1
                       and r["fit_valid"] and r["d"] < 32], key=lambda x: x["d"])
        fixed = fit(rows, kappa)
        free = min((fit(rows, k) for k in np.linspace(.01, 1.5, 1491)), key=lambda x: x["log_rms"])
        fixed["internal_g_over_E_over_omega"] = fixed["g"]/(q["E"]/q["omega"])
        result["fits"][case] = {"fixed": fixed, "free": free,
                                "relative_kappa_difference": free["kappa"]/kappa-1}
        coarse = next(x for x in rows if x["d"] == 16)
        fine = next(x for x in saved["pairs"] if x["case"] == case and x["d"] == 16 and x["h"] == .075)
        result["refinement"][case] = {"force_coarse": coarse["force"], "force_fine": fine["force"],
                                     "relative_change": fine["force"]/coarse["force"]-1}
    same_charge = [r for r in saved["pairs"] if r["fit_valid"] and r["d"] < 32
                   and r["case"] in ("same_in", "same_anti")]
    result["K1_pass"] = bool(same_charge) and all(r["force"] > 0 for r in same_charge)
    result["absolute_alpha_comparison_valid"] = False
    (DATA / "qball_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
