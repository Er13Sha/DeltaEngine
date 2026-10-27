"""Дерево классов химических элементов по строению атома (уровень «Атом» пирамиды).

Признаки атома выводятся только из заряда ядра Z по правилу Клечковского
(`delta_science.atoms.madelung`): электроны внешнего слоя, период, блок,
завершённость внешнего слоя, d-электроны предвнешнего слоя, f-электроны.
Индуктор открывает вилки по этим признакам; классы PubChem служат только метками
для обучения и проверки. Результат — `data/trees/atoms.json`.

    python scripts/build_atoms_tree.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_science.atoms import elements  # noqa: E402
from delta_science.pyramid import ATOM_SCHEMA, atom_features  # noqa: E402
from delta_tools.induction import Budget, Inductor, predict  # noqa: E402

OUT = ROOT / "data" / "trees" / "atoms.json"
BUDGET = Budget(max_depth=20, conjunctions=False)


def examples() -> list[tuple[dict, str]]:
    return [(atom_features(z), e.group_block) for z, e in sorted(elements().items())]


def main() -> int:
    data = examples()
    inductor = Inductor(ATOM_SCHEMA, BUDGET, tree_id="atoms", version="0.1.0")
    tree = inductor.fit(data)
    errors = [(r, label) for r, label in data if predict(tree, r)[0] != label]
    loo = 0
    for i in range(len(data)):
        t = Inductor(ATOM_SCHEMA, BUDGET).fit(data[:i] + data[i + 1:])
        loo += predict(t, data[i][0])[0] == data[i][1]
    root = inductor.tree_dict()
    root["root"]["label"] = "Атом"
    OUT.write_text(json.dumps(root, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"вилок {inductor.report.forks}, листьев {inductor.report.leaves}, ошибок на всех {len(errors)} из "
          f"{len(data)}, leave-one-out {loo}/{len(data)}, проверки ядра: {tree.check() or 'ok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
