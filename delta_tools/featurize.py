"""Построение признаков индуктором: отрезки числовой оси (план, раздел 5.1; спецификация, M5).

Пороговые вилки на сыром числе не видят повторяющейся структуры. Пример — классы
химических элементов: они повторяются по периодам, но периоды разной длины, поэтому
ни порог, ни `z mod k` их не выражают. Нужны признаки «номер отрезка», «позиция от
начала отрезка» и «позиция от конца отрезка» при правильно выбранных границах.

Границы выбираются жадно по критерию согласованности: каждый пример предсказывается
по другим примерам с той же позицией (от начала или от конца отрезка); одиночка в
группе — ошибка. Граница добавляется, пока она уменьшает число ошибок больше чем на
`penalty` (штраф за сложность, как MDL).

Построенные признаки записываются в схему как таблицы и производные атрибуты DSL,
поэтому их вычисляет и проверяет ядро, а не внешний код.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Iterable


def positions(value: int, bounds: list[int], lo: int, hi: int) -> tuple[int, int, int]:
    """(номер отрезка, позиция от начала, позиция от конца); граница b — последний элемент отрезка."""
    edges = [lo - 1] + bounds + [hi]
    k = sum(1 for b in bounds if value > b)
    return k, value - edges[k], edges[k + 1] - value


def consistency_errors(pairs: list[tuple[int, str]], bounds: list[int], lo: int, hi: int) -> int:
    by_start: dict[int, Counter] = defaultdict(Counter)
    by_end: dict[int, Counter] = defaultdict(Counter)
    keys = []
    for value, label in pairs:
        _, start, end = positions(value, bounds, lo, hi)
        by_start[start][label] += 1
        by_end[end][label] += 1
        keys.append((start, end, label))

    def predicted(group: Counter, label: str) -> bool:
        others = group.copy()
        others[label] -= 1
        total = sum(others.values())
        if total == 0:
            return False
        top = max(others.values())
        return others[label] == top and list(others.values()).count(top) == 1

    return sum(1 for s, e, label in keys if not (predicted(by_start[s], label) or predicted(by_end[e], label)))


def select_boundaries(pairs: Iterable[tuple[int, str]], penalty: int = 1, max_bounds: int = 20) -> list[int]:
    pairs = sorted(pairs)
    values = sorted({v for v, _ in pairs})
    lo, hi = values[0], values[-1]
    bounds: list[int] = []
    current = consistency_errors(pairs, bounds, lo, hi)
    while len(bounds) < max_bounds:
        best = None
        for b in values[:-1]:
            if b in bounds:
                continue
            errors = consistency_errors(pairs, sorted(bounds + [b]), lo, hi)
            if best is None or errors < best[0]:
                best = (errors, b)
        if best is None or current - best[0] <= penalty:
            break
        current = best[0]
        bounds = sorted(bounds + [best[1]])
    return bounds


def segment_schema(schema: dict[str, Any], attr: str, bounds: list[int], lo: int, hi: int) -> dict[str, Any]:
    """Схема с таблицами и производными атрибутами `<attr>_seg`, `<attr>_off`, `<attr>_off_end`.

    Таблицы конечны: заданы на диапазоне [lo, hi]; вне его значение по умолчанию −1.
    """
    out = copy.deepcopy(schema)
    tables = out.setdefault("tables", [])
    derived = out.setdefault("derived", [])
    names = ("seg", "off", "off_end")
    for i, suffix in enumerate(names):
        table = f"{attr}_{suffix}_table"
        tables.append({
            "name": table,
            "key": {"kind": "int"},
            "value": {"kind": "int"},
            "mapping": [[v, positions(v, bounds, lo, hi)[i]] for v in range(lo, hi + 1)],
            "default": -1,
        })
        derived.append({
            "name": f"{attr}_{suffix}",
            "expr": f"{table}({attr})",
            "doc": f"Построено индуктором: границы отрезков {bounds} на [{lo}, {hi}]",
        })
    return out
