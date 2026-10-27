"""Доверенное ядро Delta Engine.

Правила границы доверия (docs/delta-engine-plan.md, раздел 3):

- только стандартная библиотека Python;
- только точная арифметика (int, Fraction), без float;
- детерминизм: фиксированный порядок, канонический JSON, хэши;
- статус `verified` выдаётся только при наличии сертификата ядра.

Модули:

- `schema` — схема данных Σ: типы атрибутов, таблицы, производные атрибуты;
- `dsl` — язык предикатов вилок: разбор, типизация, трёхзначное вычисление,
  статический анализ тотальности, каноническая форма;
- `tree` — дерево вилок: проверки, классификация с путём, сертификаты;
- `canon` — каноническая сериализация и хэши.

Эти правила проверяются тестами в tests/architecture/.
"""
from delta_kernel.canon import canonical_json, digest
from delta_kernel.dsl import OUT_OF_DOMAIN, BudgetExceeded, DslError, Language, Predicate
from delta_kernel.schema import Schema, SchemaError
from delta_kernel.tree import INVALID, PROVEN, UNKNOWN, Classification, Tree, TreeError

__all__ = [
    "INVALID",
    "OUT_OF_DOMAIN",
    "PROVEN",
    "UNKNOWN",
    "BudgetExceeded",
    "Classification",
    "DslError",
    "Language",
    "Predicate",
    "Schema",
    "SchemaError",
    "Tree",
    "TreeError",
    "canonical_json",
    "digest",
]
