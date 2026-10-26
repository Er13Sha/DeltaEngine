"""Эксперименты индуктора E1–E3 и обучение типа вопроса (docs/report-induction.md).

    python scripts/experiments_induction.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_kernel import Tree  # noqa: E402
from delta_text import learn_intent  # noqa: E402
from delta_tools.bridge_matter import matter_tree_from_deltas  # noqa: E402
from delta_tools.featurize import segment_schema, select_boundaries  # noqa: E402
from delta_tools.induction import Budget, Inductor, predict  # noqa: E402


def e1() -> None:
    src = Tree.from_dict(matter_tree_from_deltas(json.loads((ROOT / "data" / "matter-deltas.json").read_text(encoding="utf-8"))))
    traits = ["continuous", "grouped", "planetary", "preservesShape", "preservesVolume", "ionized"]
    schema = {"name": "m", "version": "1", "attributes": [{"name": t, "type": {"kind": "bool"}} for t in traits]}
    examples = []
    for bits in itertools.product([False, True], repeat=len(traits)):
        record = dict(zip(traits, bits))
        examples.append((record, src.classify(record).leaf))
    ind = Inductor(schema)
    tree = ind.fit(examples)
    same = all(predict(tree, r)[0] == label for r, label in examples)
    print(f"E1 дерево материи: {len(examples)} примеров, {ind.report.forks} вилок, {ind.report.leaves} листьев, "
          f"совпадение с исходным: {same}")


def e2_e3() -> None:
    data = json.loads((ROOT / "data" / "confirmed-concepts.json").read_text(encoding="utf-8"))
    for feats in (["z"], ["z", "mass"]):
        attrs = [{"name": "z", "type": {"kind": "int"}}]
        if "mass" in feats:
            attrs.append({"name": "mass", "type": {"kind": "rat"}})
        schema = {"name": "e", "version": "1", "attributes": attrs}
        examples = [({"z": e["atomic_number"], **({"mass": e["atomic_mass"]} if "mass" in feats else {})},
                     e["group_block"]) for e in data["elements"]]
        for min_support in (1, 2):
            budget = Budget(min_support=min_support)
            t0 = time.time()
            correct = unknown = 0
            for i in range(len(examples)):
                tree = Inductor(schema, budget).fit(examples[:i] + examples[i + 1:])
                label, _ = predict(tree, examples[i][0])
                correct += label == examples[i][1]
                unknown += label is None
            full = Inductor(schema, budget)
            full.fit(examples)
            n = len(examples)
            print(f"E2/E3 group_block, признаки {feats}, min_support={min_support}: "
                  f"верно {correct / n:.3f}, UNKNOWN {unknown / n:.3f}, ошибок {(n - correct - unknown) / n:.3f}, "
                  f"листьев {full.report.leaves}, {time.time() - t0:.0f} с")


def e2_features() -> None:
    """E2 с признаками, построенными индуктором: границы отбираются внутри каждого прогона."""
    data = json.loads((ROOT / "data" / "confirmed-concepts.json").read_text(encoding="utf-8"))
    examples = [({"z": e["atomic_number"]}, e["group_block"]) for e in data["elements"]]
    base = {"name": "e", "version": "1", "attributes": [{"name": "z", "type": {"kind": "int"}}]}
    t0 = time.time()
    correct = unknown = 0
    for i in range(len(examples)):
        train = examples[:i] + examples[i + 1:]
        bounds = select_boundaries([(r["z"], label) for r, label in train])
        tree = Inductor(segment_schema(base, "z", bounds, 1, 118), Budget()).fit(train)
        label, _ = predict(tree, examples[i][0])
        correct += label == examples[i][1]
        unknown += label is None
    bounds = select_boundaries([(r["z"], label) for r, label in examples])
    full = Inductor(segment_schema(base, "z", bounds, 1, 118), Budget())
    full.fit(examples)
    n = len(examples)
    print(f"E2 с построенными признаками: границы {bounds}, верно {correct / n:.3f}, UNKNOWN {unknown}, "
          f"листьев {full.report.leaves}, {time.time() - t0:.0f} с")


def intent() -> None:
    """Тип вопроса и смысл слова: 10-кратная проверка, сохранение деревьев."""
    learn_intent.main()


if __name__ == "__main__":
    e1()
    e2_e3()
    e2_features()
    intent()
