"""Схема данных Σ (спецификация, раздел 5).

Атрибут имеет тип, признак обязательности и признак «определяющий / описательный».
Типы: bool, int, rat (Fraction), enum (конечный набор строк), str, set (конечное
множество значений enum или str). Чисел с плавающей точкой нет.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from delta_kernel.canon import digest

SCALAR_KINDS = ("bool", "int", "rat", "enum", "str")


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class AttrType:
    kind: str  # bool | int | rat | enum | str | set
    values: tuple[str, ...] = ()  # для enum и для set с элементами enum
    element: str = ""  # для set: "enum" или "str"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.values:
            out["values"] = list(self.values)
        if self.element:
            out["element"] = self.element
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AttrType":
        kind = data["kind"]
        values = tuple(data.get("values", ()))
        element = data.get("element", "")
        if kind not in SCALAR_KINDS + ("set",):
            raise SchemaError(f"Неизвестный тип атрибута: {kind}")
        if kind == "enum" and not values:
            raise SchemaError("У enum должен быть непустой список значений")
        if kind == "set":
            if element not in ("enum", "str"):
                raise SchemaError("Элемент set должен быть enum или str")
            if element == "enum" and not values:
                raise SchemaError("У set из enum должен быть список значений")
        if len(set(values)) != len(values):
            raise SchemaError(f"Повторяющиеся значения enum: {values}")
        return AttrType(kind, values, element)


@dataclass(frozen=True)
class Attribute:
    name: str
    type: AttrType
    required: bool = True
    defining: bool = True  # False — описательный (S-14), запрещён в определениях
    doc: str = ""
    # Условная обязательность: предикат DSL; если он истинен, атрибут обязан быть.
    required_when: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.to_dict(),
            "required": self.required,
            "required_when": self.required_when,
            "defining": self.defining,
            "doc": self.doc,
        }


@dataclass(frozen=True)
class Table:
    """Таблица значений: конечное отображение с обязательным значением по умолчанию."""

    name: str
    key: AttrType
    value: AttrType
    mapping: tuple[tuple[Any, Any], ...]
    default: Any

    def lookup(self, key: Any) -> Any:
        for k, v in self.mapping:
            if k == key:
                return v
        return self.default

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
            "mapping": [[k, v] for k, v in self.mapping],
            "default": self.default,
        }


@dataclass(frozen=True)
class Derived:
    """Производный атрибут: имя := выражение DSL. Порядок объявления исключает циклы."""

    name: str
    expr: str
    doc: str = ""


@dataclass
class Schema:
    name: str
    version: str
    attributes: dict[str, Attribute]
    tables: dict[str, Table] = field(default_factory=dict)
    derived: list[Derived] = field(default_factory=list)
    budget: int = 100_000  # предел шагов вычисления одного предиката

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Schema":
        attributes: dict[str, Attribute] = {}
        for raw in data["attributes"]:
            name = raw["name"]
            if name in attributes:
                raise SchemaError(f"Атрибут объявлен дважды: {name}")
            _check_ident(name)
            if raw.get("required_when") and raw.get("required", True):
                raise SchemaError(f"{name}: required_when имеет смысл только при required=false")
            attributes[name] = Attribute(
                name=name,
                type=AttrType.from_dict(raw["type"]),
                required=raw.get("required", True),
                defining=raw.get("defining", True),
                doc=raw.get("doc", ""),
                required_when=raw.get("required_when", ""),
            )
        tables: dict[str, Table] = {}
        for raw in data.get("tables", []):
            key = AttrType.from_dict(raw["key"])
            value = AttrType.from_dict(raw["value"])
            if "default" not in raw:
                raise SchemaError(f"У таблицы {raw['name']} нет значения по умолчанию")
            mapping = tuple(
                (coerce_value(key, k, f"ключ {raw['name']}"), coerce_value(value, v, f"значение {raw['name']}"))
                for k, v in raw["mapping"]
            )
            keys = [k for k, _ in mapping]
            if len(set(keys)) != len(keys):
                raise SchemaError(f"Повторяющиеся ключи в таблице {raw['name']}")
            tables[raw["name"]] = Table(
                name=raw["name"],
                key=key,
                value=value,
                mapping=mapping,
                default=coerce_value(value, raw["default"], f"default {raw['name']}"),
            )
        derived = [Derived(d["name"], d["expr"], d.get("doc", "")) for d in data.get("derived", [])]
        names = set(attributes) | set(tables)
        for d in derived:
            _check_ident(d.name)
            if d.name in names:
                raise SchemaError(f"Имя производного атрибута занято: {d.name}")
            names.add(d.name)
        return Schema(
            name=data["name"],
            version=data["version"],
            attributes=attributes,
            tables=tables,
            derived=derived,
            budget=data.get("budget", 100_000),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "attributes": [a.to_dict() for a in self.attributes.values()],
            "tables": [t.to_dict() for t in self.tables.values()],
            "derived": [{"name": d.name, "expr": d.expr, "doc": d.doc} for d in self.derived],
            "budget": self.budget,
        }

    def hash(self) -> str:
        return digest(self.to_dict())

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """Ошибки записи. Пустой список — запись допустима."""
        errors = []
        for name in record:
            if name not in self.attributes:
                errors.append(f"неизвестный атрибут: {name}")
        for attr in self.attributes.values():
            if attr.name not in record:
                if attr.required:
                    errors.append(f"нет обязательного атрибута: {attr.name}")
                continue
            try:
                coerce_value(attr.type, record[attr.name], attr.name)
            except SchemaError as exc:
                errors.append(str(exc))
        return errors

    def coerce_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Приводит значения записи к типам схемы. Бросает SchemaError при ошибке."""
        errors = self.validate_record(record)
        if errors:
            raise SchemaError("; ".join(errors))
        return {
            name: coerce_value(self.attributes[name].type, value, name)
            for name, value in record.items()
        }


def coerce_value(t: AttrType, value: Any, where: str) -> Any:
    """Проверяет значение по типу и приводит его к внутреннему виду."""
    if t.kind == "bool":
        if type(value) is not bool:
            raise SchemaError(f"{where}: ожидался bool, получено {value!r}")
        return value
    if t.kind == "int":
        if type(value) is not int:
            raise SchemaError(f"{where}: ожидался int, получено {value!r}")
        return value
    if t.kind == "rat":
        return parse_rat(value, where)
    if t.kind == "enum":
        if value not in t.values:
            raise SchemaError(f"{where}: значение {value!r} не входит в {list(t.values)}")
        return value
    if t.kind == "str":
        if not isinstance(value, str):
            raise SchemaError(f"{where}: ожидалась строка, получено {value!r}")
        return value
    if t.kind == "set":
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise SchemaError(f"{where}: ожидалось множество, получено {value!r}")
        element = AttrType(t.element, t.values)
        return frozenset(coerce_value(element, v, where) for v in value)
    raise SchemaError(f"{where}: неизвестный тип {t.kind}")


def parse_rat(value: Any, where: str = "значение") -> Fraction:
    """Точное рациональное: int, Fraction или строка "a/b" / "1.25". Без float."""
    if type(value) is int or isinstance(value, Fraction):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            pass
    raise SchemaError(f"{where}: ожидалось точное рациональное число, получено {value!r}")


def _check_ident(name: str) -> None:
    if not name or not (name[0].isalpha() or name[0] == "_") or not all(c.isalnum() or c == "_" for c in name):
        raise SchemaError(f"Недопустимое имя: {name!r}")
