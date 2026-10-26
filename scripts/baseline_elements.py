"""Бейзлайн: насколько пороговые деревья восстанавливают классы 118 элементов.

Жадное дерево на чистом Python (критерий Джини на Fraction, пороги — середины между соседними
значениями), перекрёстная проверка с исключением по одному (leave-one-out).
Результаты на 27.10.2026 зафиксированы в docs/delta-engine-plan.md, раздел 5.1:
group_block (номер, масса): 0.619 при 45 листьях полной подгонки; с вручную заданными
«период, позиция» 0.788 (31 лист); мажоритарный класс 0.322; standard_state: 0.864.

Запуск: python scripts/baseline_elements.py [data/confirmed-concepts.json] [--out результат.txt]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


def gini(rows, target):
    n = len(rows)
    if n == 0:
        return Fraction(0)
    counts = Counter(r[target] for r in rows)
    return 1 - sum(Fraction(v, n) ** 2 for v in counts.values())


def best_split(rows, features, target):
    base = gini(rows, target)
    n = len(rows)
    best = None
    for feature in features:
        values = sorted(set(r[feature] for r in rows))
        for a, b in zip(values, values[1:]):
            threshold = Fraction(a + b, 2) if isinstance(a, int) else (a + b) / 2
            left = [r for r in rows if r[feature] <= threshold]
            right = [r for r in rows if r[feature] > threshold]
            if not left or not right:
                continue
            weighted = Fraction(len(left), n) * gini(left, target) + Fraction(len(right), n) * gini(right, target)
            gain = base - weighted
            key = (gain, feature, -threshold)
            if best is None or key > best[0]:
                best = (key, feature, threshold, left, right)
    return best


def build(rows, features, target, depth=0, max_depth=99):
    counts = Counter(r[target] for r in rows)
    if len(counts) == 1 or depth >= max_depth:
        return ("leaf", counts.most_common(1)[0][0])
    split = best_split(rows, features, target)
    if split is None or split[0][0] <= 0:
        return ("leaf", counts.most_common(1)[0][0])
    _, feature, threshold, left, right = split
    return (
        "node",
        feature,
        threshold,
        build(left, features, target, depth + 1, max_depth),
        build(right, features, target, depth + 1, max_depth),
    )


def predict(tree, row):
    while tree[0] == "node":
        _, feature, threshold, left, right = tree
        tree = left if row[feature] <= threshold else right
    return tree[1]


def leaves(tree):
    return 1 if tree[0] == "leaf" else leaves(tree[3]) + leaves(tree[4])


def depth_of(tree):
    return 0 if tree[0] == "leaf" else 1 + max(depth_of(tree[3]), depth_of(tree[4]))


def loo(rows, features, target, max_depth=99):
    correct = 0
    for i in range(len(rows)):
        train = rows[:i] + rows[i + 1:]
        tree = build(train, features, target, 0, max_depth)
        correct += predict(tree, rows[i]) == rows[i][target]
    return correct / len(rows)


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for e in data["elements"]:
        mass = e.get("atomic_mass")
        rows.append({
            "n": e["atomic_number"],
            "mass": Fraction(mass.split("(")[0].strip()) if mass else Fraction(0),
            "state": e.get("standard_state") or "Unknown",
            "block": e.get("group_block") or "Unknown",
        })
    return rows


def run(path: Path) -> list[str]:
    rows = load_rows(path)
    out = [f"elements: {len(rows)}"]
    out.append("group_block classes: " + str(Counter(r["block"] for r in rows).most_common()))
    out.append("standard_state classes: " + str(Counter(r["state"] for r in rows).most_common()))

    features = ["n", "mass"]
    full = build(rows, features, "block")
    out.append(f"group_block full-fit tree on (n, mass): leaves={leaves(full)} depth={depth_of(full)}")
    out.append(f"group_block LOO accuracy (n, mass): {loo(rows, features, 'block'):.3f}")
    out.append(f"group_block LOO accuracy (n only): {loo(rows, ['n'], 'block'):.3f}")
    for max_depth in (3, 5, 8):
        out.append(f"group_block LOO accuracy (n, mass), max_depth={max_depth}: {loo(rows, features, 'block', max_depth):.3f}")

    # Подсказка: границы периодов заданы вручную; индуктор должен находить такие признаки сам.
    bounds = [0, 2, 10, 18, 36, 54, 86, 118]
    for r in rows:
        period = next(i for i in range(1, len(bounds)) if r["n"] <= bounds[i])
        r["period"] = period
        r["pos"] = r["n"] - bounds[period - 1]
    derived = ["period", "pos"]
    tree = build(rows, derived, "block")
    out.append(f"group_block with derived (period,pos): leaves={leaves(tree)} depth={depth_of(tree)} LOO={loo(rows, derived, 'block'):.3f}")

    state_tree = build(rows, features, "state")
    out.append(f"standard_state full-fit tree on (n, mass): leaves={leaves(state_tree)}  LOO={loo(rows, features, 'state'):.3f}")
    majority = Counter(r["block"] for r in rows).most_common(1)[0]
    out.append(f"majority baseline for group_block: {majority[1] / len(rows):.3f} ({majority[0]})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("concepts", nargs="?", default="data/confirmed-concepts.json")
    parser.add_argument("--out", help="сохранить результат в файл")
    args = parser.parse_args()
    lines = run(Path(args.concepts))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
