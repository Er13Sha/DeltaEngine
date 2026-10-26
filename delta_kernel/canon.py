"""Каноническая сериализация и хэши (инвариант I4 спецификации).

Одинаковые данные всегда дают одинаковую строку и одинаковый хэш:
ключи отсортированы, пробелов нет, дроби записаны как "числитель/знаменатель".
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return value.numerator
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalize(v) for v in value]
        return sorted(items, key=lambda v: canonical_json(v))
    raise TypeError(f"Тип {type(value).__name__} не поддерживается канонической сериализацией")


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
