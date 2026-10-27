"""Опыты трека E на решётчатой модели поля: протокол, тест существования, проверка K1.

    python -m adapters.field_lattice.experiments            # все опыты, таблица прогонов и сводка

Протокол (`data/experiments/field/protocol.json`) записывается и хэшируется до запуска;
каждый прогон — запись `delta-experiment@0.1` с хэшем протокола, параметрами, зерном,
хэшем начальной конфигурации, наблюдаемыми и вердиктом. Все утверждения — `scope = model`:
опыт показывает, что возможно в модели, а не что верно в природе (спецификация, прил. C).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adapters.field_lattice.model import (acceptance, clusters, energy, kawasaki, kernel, periodic_distance, topple)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "experiments" / "field"

PROTOCOL = {
    "format": "delta-protocol@0.1",
    "track": "E",
    "stage": "E1 (шип)",
    "status": "определения — вариант ассистента (план, 9.3), ждут утверждения заказчиком",
    "premises": {
        "H0.P1": "пространство заполнено полем (Hypothesis)",
        "H0.P2": "уплотнённое поле — частица (Hypothesis)",
        "K1": "одноимённые отталкиваются (Constatation для электрического заряда; условие модели)",
    },
    "definitions": {
        "lattice": "квадрат L × L с периодическими границами; шип в 2D, 3D — следующий этап",
        "field": "целое отклонение от фона δ = φ − φ₀",
        "rule": "сохраняет сумму δ (обмен между соседями или осыпание)",
        "densification": "связная по 4 соседям область одного знака с |δ| > θ",
        "particle": "уплотнение не меньше s_min ячеек, которое через T свипов находится рядом (≤ d_match) "
                    "и сохраняет размер в пределах множителя 2",
        "charge": "Q = Σ δ по области",
        "interaction_energy": "E_int(d) = E(пара на расстоянии d) − E(первая отдельно) − E(вторая отдельно), "
                              "целое число; капли — диски радиуса r",
        "repulsion": "E_int(d) ≥ 0 при всех зазорах (включая касание), > 0 хотя бы при одном и не возрастает "
                     "с ростом d (сила направлена от партнёра; ноль за радиусом ядра — нет силы), "
                     "и в динамике у контакта (зазор 1 ячейка) пара не сливается ни в одном зерне. Отклонение от "
                     "плана 9.3 (там — относительное ускорение): динамика модели передемпфирована, капли почти "
                     "неподвижны, поэтому знак силы берётся из точной энергии взаимодействия",
        "attraction": "E_int(d) < 0 хотя бы при одном d или слияние у контакта",
    },
    "families": {
        "A": {"rule": "осыпание", "theta_top": 3, "role": "отрицательный контроль"},
        "B": {"rule": "Кавасаки, сцепление J", "J": 40, "K": 0, "T": "8", "role": "отрицательный контроль K1"},
        "C": {"rule": "Кавасаки, сцепление J + отталкивание одноимённых K·⌊R/r⌋", "J": 40, "K": 1, "R": 8, "T": "8",
              "role": "кандидат"},
        "parameter_choice": "J, K, R, T выбраны по пилотным прогонам: при J = 4 и G(1) = 6 сцепление слабее "
                            "отталкивания, капли C не собираются; при T/J ≥ 1/4 капли испаряются",
    },
    "existence": {"L": 48, "density_each_sign": "8/100", "warmup": 300, "T_life": 300, "s_min": 4, "d_match": 4,
                  "seeds": [1, 2, 3]},
    "pair": {"L": 64, "radius": 3, "gaps": [0, 1, 2, 3, 4, 6, 8, 10, 12, 16], "orientations": ["x", "diagonal"],
             "contact_sweeps": 2000, "seeds": [1, 2, 3, 4, 5, 6]},
    "criteria": {
        "admitted": "тест существования пройден хотя бы в одном зерне и K1 выполнен при всех расстояниях и ориентациях",
        "negative_controls": "A и B обязаны провалить K1 или тест существования, иначе проверка нечувствительна",
        "scope": "model",
    },
}


def protocol_hash() -> str:
    return "sha256:" + hashlib.sha256(json.dumps(PROTOCOL, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(s: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(s.astype(np.int64).tobytes()).hexdigest()[:16]


@dataclass
class Family:
    name: str
    J: int = 0
    K: int = 0
    R: int = 6
    T: tuple[int, int] = (2, 1)

    def run(self, s: np.ndarray, sweeps: int, seed: int) -> None:
        kern = kernel(self.R)
        kawasaki(s, self.J, self.K, kern, acceptance(self.T[0], self.T[1], 400), sweeps, seed)


FAMILIES = {
    "B": Family("B", J=40, K=0, T=(8, 1)),
    "C": Family("C", J=40, K=1, R=8, T=(8, 1)),
}


def _record(kind: str, family: str, params: dict, seed: int, init: np.ndarray, observables: dict,
            verdict: str) -> dict[str, Any]:
    return {
        "format": "delta-experiment@0.1", "run_id": f"{kind}-{family}-{params.get('d', '')}{params.get('o', '')}-s{seed}",
        "protocol_hash": protocol_hash(), "model_family": family, "kind": kind, "params": params, "seed": seed,
        "code_version": code_version(), "config_hash": config_hash(init), "observables": observables,
        "verdict": verdict, "scope": "model",
    }


# ---------------------------------------------------------------- существование


def existence_run(family: str, seed: int) -> dict[str, Any]:
    cfg = PROTOCOL["existence"]
    L = cfg["L"]
    rng = np.random.default_rng(seed)
    if family == "A":
        s = np.zeros((L, L), dtype=np.int64)
        # сгустки поля: в случайных ячейках отклонения ±(4…12)
        idx = rng.choice(L * L, size=L * L // 10, replace=False)
        s.flat[idx] = rng.choice([-1, 1], size=idx.size) * rng.integers(4, 13, size=idx.size)
        init = s.copy()
        steps = topple(s, PROTOCOL["families"]["A"]["theta_top"], 100_000)
        dens = clusters(s, theta=PROTOCOL["families"]["A"]["theta_top"])
        obs = {"topple_steps": int(steps), "densifications_after": len(dens), "sum_conserved": int(s.sum()) == int(init.sum())}
        return _record("existence", "A", {"L": L}, seed, init, obs, "particles" if dens else "no_particles")
    p = float(cfg["density_each_sign"].split("/")[0]) / float(cfg["density_each_sign"].split("/")[1])
    s = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=(L, L), p=[p, 1 - 2 * p, p])
    init = s.copy()
    fam = FAMILIES[family]
    fam.run(s, cfg["warmup"], seed)
    before = [c for c in clusters(s) if c["cells"] >= cfg["s_min"]]
    fam.run(s, cfg["T_life"], seed + 1000)
    after = [c for c in clusters(s) if c["cells"] >= cfg["s_min"]]
    persistent = []
    for c in before:
        near = [d for d in after if (d["charge"] > 0) == (c["charge"] > 0)
                and periodic_distance(c["center"], d["center"], L) <= cfg["d_match"]
                and c["cells"] / 2 <= d["cells"] <= c["cells"] * 2]
        if near:
            persistent.append(c["charge"])
    sizes = sorted(abs(c["charge"]) for c in after)
    obs = {
        "clusters_before": len(before), "clusters_after": len(after), "persistent": len(persistent),
        "charges_after": sizes, "sum_conserved": int(s.sum()) == int(init.sum()),
    }
    return _record("existence", family, {"L": L, "J": fam.J, "K": fam.K, "R": fam.R, "T": f"{fam.T[0]}/{fam.T[1]}"},
                   seed, init, obs, "particles" if persistent else "no_particles")


# ---------------------------------------------------------------- пары: K1 и разноимённые


def _disc(L: int, center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cx, cy = center
    return [((cx + dx) % L, (cy + dy) % L) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius]


def _place(L: int, radius: int, gap: int, orientation: str, signs: tuple[int, int]):
    """Два диска с зазором gap ячеек между краями (по оси x или по диагонали)."""
    c1 = (L // 4, L // 4)
    if orientation == "x":
        c2 = (c1[0] + 2 * radius + 1 + gap, c1[1])
    else:
        step = math.ceil((2 * radius + 1 + gap) / math.sqrt(2))
        c2 = (c1[0] + step, c1[1] + step)
    one = np.zeros((L, L), dtype=np.int64)
    two = np.zeros((L, L), dtype=np.int64)
    for cell in _disc(L, c1, radius):
        one[cell] = signs[0]
    for cell in _disc(L, c2, radius):
        two[cell] = signs[1]
    if np.any((one != 0) & (two != 0)):
        raise ValueError("диски перекрываются")
    return one, two, math.dist(c1, c2)


def interaction_energy(family: str, signs: tuple[int, int], gap: int, orientation: str) -> tuple[int, float]:
    fam = FAMILIES[family]
    cfg = PROTOCOL["pair"]
    one, two, d = _place(cfg["L"], cfg["radius"], gap, orientation, signs)
    kern = kernel(fam.R)
    e = lambda s: energy(s, fam.J, fam.K, kern)  # noqa: E731
    return e(one + two) - e(one) - e(two), d


def pair_static(family: str, signs: tuple[int, int]) -> dict[str, Any]:
    cfg = PROTOCOL["pair"]
    profile = {}
    for o in cfg["orientations"]:
        profile[o] = [{"gap": g, "d": round(d, 3), "E_int": ei}
                      for g in cfg["gaps"] for ei, d in [interaction_energy(family, signs, g, o)]]
    values = [row["E_int"] for rows in profile.values() for row in rows]
    monotone = all(a["E_int"] >= b["E_int"] for rows in profile.values() for a, b in zip(rows, rows[1:]))
    if any(v < 0 for v in values):
        verdict = "attraction"
    elif any(v > 0 for v in values) and monotone:
        verdict = "repulsion"
    elif all(v == 0 for v in values):
        verdict = "no_interaction"
    else:
        verdict = "mixed"
    return {"profile": profile, "monotone": monotone, "verdict": verdict}


def contact_run(family: str, signs: tuple[int, int], orientation: str, seed: int) -> dict[str, Any]:
    cfg = PROTOCOL["pair"]
    one, two, d = _place(cfg["L"], cfg["radius"], 1, orientation, signs)
    s = one + two
    init = s.copy()
    FAMILIES[family].run(s, cfg["contact_sweeps"], seed)
    big = sorted((c for c in clusters(s) if c["cells"] >= 4), key=lambda c: -c["cells"])
    n0 = int(np.abs(one).sum())
    same = signs[0] == signs[1]
    if same and big and big[0]["cells"] > n0 * 3 // 2:
        outcome = "merged"
    elif len(big) >= 2:
        outcome = "separate"
    else:
        outcome = "dissolved"
    obs = {"d0": round(d, 3), "outcome": outcome, "clusters": [c["charge"] for c in big],
           "sum_conserved": int(s.sum()) == int(init.sum())}
    kind = "contact_like" if same else "contact_opposite"
    return _record(kind, family, {"gap": 1, "o": orientation}, seed, init, obs, outcome)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps({**PROTOCOL, "hash": protocol_hash()}, ensure_ascii=False, indent=1)
                                       + "\n", encoding="utf-8")
    runs: list[dict] = []
    summary: dict[str, Any] = {"protocol_hash": protocol_hash(), "scope": "model", "families": {}}
    for family in ("A", "B", "C"):
        ex = [existence_run(family, seed) for seed in PROTOCOL["existence"]["seeds"]]
        runs += ex
        fam_summary: dict[str, Any] = {"existence": [r["verdict"] for r in ex],
                                       "existence_observables": [r["observables"] for r in ex]}
        if family != "A" and any(r["verdict"] == "particles" for r in ex):
            cfg = PROTOCOL["pair"]
            for signs, kind in (((1, 1), "like"), ((1, -1), "opposite")):
                static = pair_static(family, signs)
                contact = [contact_run(family, signs, o, seed) for o in cfg["orientations"] for seed in cfg["seeds"]]
                runs += contact
                merges = sum(r["verdict"] == "merged" for r in contact)
                fam_summary[kind] = {"static": static, "contact_outcomes": [r["verdict"] for r in contact],
                                     "merges": merges}
                print(family, kind, static["verdict"], "слияний у контакта:", merges, flush=True)
        exists = any(v == "particles" for v in fam_summary["existence"])
        like = fam_summary.get("like", {})
        k1 = like.get("static", {}).get("verdict") == "repulsion" and like.get("merges", 1) == 0
        fam_summary["K1"] = k1
        fam_summary["admitted"] = exists and k1
        summary["families"][family] = fam_summary
        print(family, "existence", fam_summary["existence"], "admitted", fam_summary["admitted"], flush=True)
    controls_ok = not summary["families"]["A"]["admitted"] and not summary["families"]["B"]["admitted"]
    summary["negative_controls_failed_as_required"] = controls_ok
    with (OUT / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("контроли провалили, как требуется:", controls_ok, "прогонов:", len(runs))
    return 0


if __name__ == "__main__" and "--scan" not in sys.argv:
    sys.exit(main())


# ---------------------------------------------------------------- перебор параметров (E4, упрощённо)

SCAN = {"J": [10, 20, 40], "K": [0, 1, 2], "T": [4, 8, 16], "R": 8, "seeds": [1, 2]}
SCAN_SCHEMA = {
    "name": "FieldModelParams",
    "version": "0.1.0",
    "attributes": [
        {"name": "J", "type": {"kind": "int"}, "doc": "сцепление соседних ячеек одного знака"},
        {"name": "K", "type": {"kind": "int"}, "doc": "сила отталкивания одноимённых"},
        {"name": "T", "type": {"kind": "int"}, "doc": "температура динамики"},
        {"name": "t_ratio", "type": {"kind": "rat"}, "doc": "T / J — тепловое движение против сцепления"},
        {"name": "k_ratio", "type": {"kind": "rat"}, "doc": "K·G(1) / J — отталкивание соседей против сцепления"},
    ],
}


def scan() -> dict[str, Any]:
    """Классы исхода по параметрам: частицы с отталкиванием одноимённых / частицы без K1 / нет частиц."""
    from fractions import Fraction

    from delta_tools.induction import Budget, Inductor, predict

    examples, rows = [], []
    for J in SCAN["J"]:
        for K in SCAN["K"]:
            FAMILIES["S"] = Family("S", J=J, K=K, R=SCAN["R"])
            static = pair_static("S", (1, 1))["verdict"]
            for T in SCAN["T"]:
                FAMILIES["S"] = Family("S", J=J, K=K, R=SCAN["R"], T=(T, 1))
                exists = any(existence_run("S", seed)["verdict"] == "particles" for seed in SCAN["seeds"])
                label = ("частицы, одноимённые отталкиваются" if exists and static == "repulsion"
                         else "частицы, K1 не выполнен" if exists else "частиц нет")
                record = {"J": J, "K": K, "T": T, "t_ratio": Fraction(T, J), "k_ratio": Fraction(K * SCAN["R"], J)}
                examples.append((record, label))
                rows.append({**record, "t_ratio": str(record["t_ratio"]), "k_ratio": str(record["k_ratio"]),
                             "static_like": static, "particles": exists, "label": label})
                print(J, K, T, label, flush=True)
    FAMILIES.pop("S", None)
    inductor = Inductor(SCAN_SCHEMA, Budget(max_depth=8, conjunctions=False), tree_id="field_model", version="0.1.0")
    tree = inductor.fit(examples)
    errors = sum(predict(tree, r)[0] != label for r, label in examples)
    data = inductor.tree_dict()
    data["root"]["label"] = "Поле (модель трека E, scope = model)"
    (ROOT / "data" / "trees" / "field_model.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                                                             encoding="utf-8")
    result = {"protocol_hash": protocol_hash(), "scope": "model", "scan": SCAN, "rows": rows,
              "tree_forks": inductor.report.forks, "tree_errors": errors}
    (OUT / "scan.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("вилок", inductor.report.forks, "ошибок", errors, "из", len(examples))
    return result


if __name__ == "__main__" and "--scan" in sys.argv:
    scan()
