"""DSL предикатов вилок (спецификация, раздел 6).

Язык намеренно ограничен: нет рекурсии, циклов, float и внешних вызовов, поэтому
любое вычисление завершается. Выражение компилируется против схемы: имена и типы
проверяются до вычисления, опечатка в значении enum — ошибка компиляции.

Семантика отсутствующих атрибутов — трёхзначная (Клини):
- обращение к отсутствующему атрибуту даёт OUT_OF_DOMAIN, а не false;
- `false and X` = false, `true or X` = true при любом X, в том числе OUT_OF_DOMAIN;
- `has(a)` никогда не даёт OUT_OF_DOMAIN и служит охраной: `has(a) and a > 3`.

Статический анализ `unguarded` находит атрибуты, отсутствие которых может дать
OUT_OF_DOMAIN. На нём строится проверка тотальности вилок в дереве.

Грамматика (приоритет по возрастанию):
    expr    := or
    or      := and ('or' and)*
    and     := not ('and' not)*
    not     := 'not' not | cmp
    cmp     := sum (('=='|'!='|'<'|'<='|'>'|'>=') sum | 'in' '{' lit (',' lit)* '}')?
    sum     := prod (('+'|'-') prod)*
    prod    := unary (('*') unary | ('mod'|'div') INT)*
    unary   := '-' unary | primary
    primary := lit | ident | ident '(' expr ')' | 'has' '(' ident ')'
             | ('all'|'any') ident 'in' ident ':' expr | '(' expr ')'
    lit     := INT | RAT ("3/2", "1.25") | STRING ("...") | 'true' | 'false'
"""
from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

from delta_kernel.canon import digest
from delta_kernel.schema import AttrType, Schema


class DslError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class _OutOfDomainType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "OUT_OF_DOMAIN"

    def __bool__(self) -> bool:
        raise TypeError("OUT_OF_DOMAIN нельзя использовать как bool")


OUT_OF_DOMAIN = _OutOfDomainType()

KEYWORDS = {"and", "or", "not", "in", "all", "any", "true", "false", "mod", "div", "has"}
CMP_OPS = ("==", "!=", "<=", ">=", "<", ">")
BOOL = AttrType("bool")
INT = AttrType("int")
RAT = AttrType("rat")
STR = AttrType("str")


# ---------------------------------------------------------------- токенизатор


@dataclass(frozen=True)
class Token:
    kind: str  # num | str | ident | kw | op | end
    value: Any
    pos: int


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and (text[j].isdigit() or text[j] in "./"):
                j += 1
            raw = text[i:j]
            if raw.count("/") + raw.count(".") > 1 or raw.endswith(("/", ".")):
                raise DslError(f"Некорректное число {raw!r} в позиции {i}")
            try:
                value = int(raw) if raw.isdigit() else Fraction(raw)
            except (ValueError, ZeroDivisionError):
                raise DslError(f"Некорректное число {raw!r} в позиции {i}") from None
            tokens.append(Token("num", value, i))
            i = j
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                buf.append(text[j])
                j += 1
            if j >= n:
                raise DslError(f"Незакрытая строка в позиции {i}")
            tokens.append(Token("str", "".join(buf), i))
            i = j + 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            tokens.append(Token("kw" if word in KEYWORDS else "ident", word, i))
            i = j
            continue
        two = text[i : i + 2]
        if two in ("==", "!=", "<=", ">="):
            tokens.append(Token("op", two, i))
            i += 2
            continue
        if c in "<>+-*(){},:":
            tokens.append(Token("op", c, i))
            i += 1
            continue
        raise DslError(f"Неожиданный символ {c!r} в позиции {i}")
    tokens.append(Token("end", None, n))
    return tokens


# ---------------------------------------------------------------- AST


@dataclass(frozen=True)
class Node:
    kind: str
    value: Any = None
    args: tuple["Node", ...] = ()
    type: AttrType | None = None


class _Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def take(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def accept(self, kind: str, value: Any = None) -> Token | None:
        tok = self.peek()
        if tok.kind == kind and (value is None or tok.value == value):
            self.i += 1
            return tok
        return None

    def expect(self, kind: str, value: Any = None) -> Token:
        tok = self.accept(kind, value)
        if tok is None:
            got = self.peek()
            want = value if value is not None else kind
            raise DslError(f"Ожидалось {want!r}, получено {got.value!r} в позиции {got.pos}")
        return tok

    def parse(self) -> Node:
        node = self.expr()
        if self.peek().kind != "end":
            tok = self.peek()
            raise DslError(f"Лишний текст {tok.value!r} в позиции {tok.pos}")
        return node

    def expr(self) -> Node:
        return self.or_()

    def or_(self) -> Node:
        items = [self.and_()]
        while self.accept("kw", "or"):
            items.append(self.and_())
        return items[0] if len(items) == 1 else Node("or", args=tuple(items))

    def and_(self) -> Node:
        items = [self.not_()]
        while self.accept("kw", "and"):
            items.append(self.not_())
        return items[0] if len(items) == 1 else Node("and", args=tuple(items))

    def not_(self) -> Node:
        if self.accept("kw", "not"):
            return Node("not", args=(self.not_(),))
        return self.cmp()

    def cmp(self) -> Node:
        left = self.sum_()
        tok = self.peek()
        if tok.kind == "op" and tok.value in CMP_OPS:
            self.take()
            right = self.sum_()
            nxt = self.peek()
            if nxt.kind == "op" and nxt.value in CMP_OPS:
                raise DslError(f"Цепочки сравнений запрещены (позиция {nxt.pos}); используйте and")
            return Node("cmp", tok.value, (left, right))
        if self.accept("kw", "in"):
            self.expect("op", "{")
            lits = [self.literal()]
            while self.accept("op", ","):
                lits.append(self.literal())
            self.expect("op", "}")
            return Node("in", tuple(lits), (left,))
        return left

    def literal(self) -> Any:
        tok = self.take()
        if tok.kind in ("num", "str"):
            return tok.value
        if tok.kind == "op" and tok.value == "-":
            num = self.expect("num")
            return -num.value
        if tok.kind == "kw" and tok.value in ("true", "false"):
            return tok.value == "true"
        raise DslError(f"Ожидался литерал, получено {tok.value!r} в позиции {tok.pos}")

    def sum_(self) -> Node:
        node = self.prod()
        while True:
            tok = self.peek()
            if tok.kind == "op" and tok.value in "+-":
                self.take()
                node = Node("add" if tok.value == "+" else "sub", args=(node, self.prod()))
            else:
                return node

    def prod(self) -> Node:
        node = self.unary()
        while True:
            tok = self.peek()
            if tok.kind == "op" and tok.value == "*":
                self.take()
                node = Node("mul", args=(node, self.unary()))
            elif tok.kind == "kw" and tok.value in ("mod", "div"):
                self.take()
                sign = -1 if self.accept("op", "-") else 1
                num = self.expect("num")
                if type(num.value) is not int or num.value == 0:
                    raise DslError(f"{tok.value} допускается только на ненулевой целый литерал (позиция {num.pos})")
                node = Node(tok.value, sign * num.value, (node,))
            else:
                return node

    def unary(self) -> Node:
        if self.accept("op", "-"):
            return Node("neg", args=(self.unary(),))
        return self.primary()

    def primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "num":
            self.take()
            return Node("lit", tok.value)
        if tok.kind == "str":
            self.take()
            return Node("lit", tok.value)
        if tok.kind == "kw" and tok.value in ("true", "false"):
            self.take()
            return Node("lit", tok.value == "true")
        if tok.kind == "kw" and tok.value == "has":
            self.take()
            self.expect("op", "(")
            name = self.expect("ident").value
            self.expect("op", ")")
            return Node("has", name)
        if tok.kind == "kw" and tok.value in ("all", "any"):
            self.take()
            var = self.expect("ident").value
            self.expect("kw", "in")
            domain = self.expect("ident").value
            self.expect("op", ":")
            body = self.expr()
            return Node(tok.value, (var, domain), (body,))
        if tok.kind == "ident":
            self.take()
            if self.accept("op", "("):
                arg = self.expr()
                self.expect("op", ")")
                return Node("call", tok.value, (arg,))
            return Node("name", tok.value)
        if self.accept("op", "("):
            node = self.expr()
            self.expect("op", ")")
            return node
        raise DslError(f"Неожиданный токен {tok.value!r} в позиции {tok.pos}")


def parse(text: str) -> Node:
    return _Parser(text).parse()


# ---------------------------------------------------------------- типизация


def _is_num(t: AttrType) -> bool:
    return t.kind in ("int", "rat")


def _lit_type(value: Any) -> AttrType:
    if type(value) is bool:
        return BOOL
    if type(value) is int:
        return INT
    if isinstance(value, Fraction):
        return RAT
    return STR


def _describe(t: AttrType) -> str:
    return f"enum{list(t.values)}" if t.kind == "enum" else t.kind


class Language:
    """Скомпилированная схема: типы атрибутов, таблицы и производные атрибуты."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.derived: dict[str, Node] = {}
        self.derived_unguarded: dict[str, frozenset[str]] = {}
        for d in schema.derived:
            node = self._check(parse(d.expr), {})
            self.derived[d.name] = node
            self.derived_unguarded[d.name] = unguarded(node, self)
        self.conditions: dict[str, Predicate] = {}
        for attr in schema.attributes.values():
            if attr.required_when:
                try:
                    self.conditions[attr.name] = self.compile(attr.required_when)
                except DslError as exc:
                    raise DslError(f"required_when атрибута {attr.name}: {exc}") from None

    def condition_errors(self, record: dict[str, Any]) -> list[str]:
        """Нарушения условной обязательности (required_when) в приведённой записи."""
        errors = []
        for name, cond in self.conditions.items():
            if name not in record and cond.evaluate(record).value is True:
                errors.append(f"нет атрибута {name}, обязательного при {cond.text}")
        return errors

    def compile(self, text: str) -> "Predicate":
        node = self._check(parse(text), {})
        if node.type != BOOL:
            raise DslError(f"Предикат должен быть bool, получен {_describe(node.type)}")
        return Predicate(self, node)

    def compile_term(self, text: str) -> Node:
        return self._check(parse(text), {})

    # Возвращает узел с проставленным типом; поля enum-литералов приводятся к enum.
    def _check(self, node: Node, scope: dict[str, AttrType]) -> Node:
        k = node.kind
        if k == "lit":
            return Node("lit", node.value, (), _lit_type(node.value))
        if k == "name":
            name = node.value
            if name in scope:
                return Node("var", name, (), scope[name])
            if name in self.schema.attributes:
                t = self.schema.attributes[name].type
                if t.kind == "set":
                    raise DslError(f"Множество {name} можно использовать только в all/any или has")
                return Node("attr", name, (), t)
            if name in self.derived:
                return Node("derived", name, (), self.derived[name].type)
            raise DslError(f"Неизвестное имя: {name}")
        if k == "has":
            if node.value not in self.schema.attributes:
                raise DslError(f"has(): неизвестный атрибут {node.value}")
            return Node("has", node.value, (), BOOL)
        if k == "call":
            table = self.schema.tables.get(node.value)
            if table is None:
                raise DslError(f"Неизвестная таблица: {node.value}")
            arg = self._coerce_to(self._check(node.args[0], scope), table.key, f"ключ таблицы {table.name}")
            return Node("call", node.value, (arg,), table.value)
        if k in ("and", "or"):
            args = tuple(self._check(a, scope) for a in node.args)
            for a in args:
                if a.type != BOOL:
                    raise DslError(f"Операнд {k} должен быть bool, получен {_describe(a.type)}")
            return Node(k, None, args, BOOL)
        if k == "not":
            a = self._check(node.args[0], scope)
            if a.type != BOOL:
                raise DslError(f"Операнд not должен быть bool, получен {_describe(a.type)}")
            return Node("not", None, (a,), BOOL)
        if k == "neg":
            a = self._check(node.args[0], scope)
            if not _is_num(a.type):
                raise DslError("Унарный минус только для чисел")
            return Node("neg", None, (a,), a.type)
        if k in ("add", "sub", "mul"):
            a, b = (self._check(x, scope) for x in node.args)
            if not (_is_num(a.type) and _is_num(b.type)):
                raise DslError(f"Операция {k} только для чисел")
            t = INT if a.type == INT and b.type == INT else RAT
            return Node(k, None, (a, b), t)
        if k in ("mod", "div"):
            a = self._check(node.args[0], scope)
            if a.type != INT:
                raise DslError(f"{k} только для целых")
            return Node(k, node.value, (a,), INT)
        if k == "cmp":
            a, b = (self._check(x, scope) for x in node.args)
            op = node.value
            if op in ("<", "<=", ">", ">="):
                if not (_is_num(a.type) and _is_num(b.type)):
                    raise DslError(f"Сравнение {op} только для чисел")
            else:
                a, b = self._unify(a, b)
            return Node("cmp", op, (a, b), BOOL)
        if k == "in":
            a = self._check(node.args[0], scope)
            lits = []
            for v in node.value:
                lit = self._coerce_to(Node("lit", v, (), _lit_type(v)), a.type, "элемент множества")
                lits.append(lit.value)
            if len(set(lits)) != len(lits):
                raise DslError("Повторяющиеся элементы в множестве in {...}")
            return Node("in", tuple(lits), (a,), BOOL)
        if k in ("all", "any"):
            var, domain = node.value
            attr = self.schema.attributes.get(domain)
            if attr is None or attr.type.kind != "set":
                raise DslError(f"{k}: {domain} не является атрибутом-множеством")
            if var in scope or var in self.schema.attributes or var in self.derived:
                raise DslError(f"{k}: имя переменной {var} уже занято")
            elem = AttrType(attr.type.element, attr.type.values)
            body = self._check(node.args[0], {**scope, var: elem})
            if body.type != BOOL:
                raise DslError(f"Тело {k} должно быть bool")
            return Node(k, (var, domain), (body,), BOOL)
        raise DslError(f"Неизвестный узел: {k}")

    def _coerce_to(self, node: Node, t: AttrType, where: str) -> Node:
        if node.type == t:
            return node
        if _is_num(t) and _is_num(node.type):
            if t == INT and node.type == RAT:
                raise DslError(f"{where}: ожидалось целое")
            return node
        if t.kind == "enum" and node.kind == "lit" and node.type == STR:
            if node.value not in t.values:
                raise DslError(f"{where}: значение {node.value!r} не входит в {list(t.values)}")
            return Node("lit", node.value, (), t)
        raise DslError(f"{where}: тип {_describe(node.type)} не совпадает с {_describe(t)}")

    def _unify(self, a: Node, b: Node) -> tuple[Node, Node]:
        if _is_num(a.type) and _is_num(b.type):
            return a, b
        if a.type == b.type:
            return a, b
        if a.type.kind == "enum" and b.kind == "lit":
            return a, self._coerce_to(b, a.type, "сравнение")
        if b.type.kind == "enum" and a.kind == "lit":
            return self._coerce_to(a, b.type, "сравнение"), b
        raise DslError(f"Нельзя сравнивать {_describe(a.type)} и {_describe(b.type)}")


# ---------------------------------------------------------------- вычисление


class _Ctx:
    def __init__(self, lang: Language, record: dict[str, Any], budget: int):
        self.lang = lang
        self.record = record
        self.budget = budget
        self.steps = 0
        self.missing: set[str] = set()
        self.derived_cache: dict[str, Any] = {}

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.budget:
            raise BudgetExceeded(f"Превышен бюджет вычисления: {self.budget} шагов")


def _eval(node: Node, ctx: _Ctx, env: dict[str, Any]) -> Any:
    ctx.tick()
    k = node.kind
    if k == "lit":
        return node.value
    if k == "var":
        return env[node.value]
    if k == "attr":
        if node.value not in ctx.record:
            ctx.missing.add(node.value)
            return OUT_OF_DOMAIN
        return ctx.record[node.value]
    if k == "derived":
        if node.value not in ctx.derived_cache:
            ctx.derived_cache[node.value] = _eval(ctx.lang.derived[node.value], ctx, {})
        return ctx.derived_cache[node.value]
    if k == "has":
        return node.value in ctx.record
    if k == "call":
        key = _eval(node.args[0], ctx, env)
        if key is OUT_OF_DOMAIN:
            return OUT_OF_DOMAIN
        return ctx.lang.schema.tables[node.value].lookup(key)
    if k == "and":
        values = [_eval(a, ctx, env) for a in node.args]
        if any(v is False for v in values):
            return False
        if any(v is OUT_OF_DOMAIN for v in values):
            return OUT_OF_DOMAIN
        return True
    if k == "or":
        values = [_eval(a, ctx, env) for a in node.args]
        if any(v is True for v in values):
            return True
        if any(v is OUT_OF_DOMAIN for v in values):
            return OUT_OF_DOMAIN
        return False
    if k == "not":
        v = _eval(node.args[0], ctx, env)
        return v if v is OUT_OF_DOMAIN else not v
    if k in ("all", "any"):
        var, domain = node.value
        if domain not in ctx.record:
            ctx.missing.add(domain)
            return OUT_OF_DOMAIN
        values = [
            _eval(node.args[0], ctx, {**env, var: item})
            for item in sorted(ctx.record[domain], key=repr)
        ]
        decisive = False if k == "all" else True
        if any(v is decisive for v in values):
            return decisive
        if any(v is OUT_OF_DOMAIN for v in values):
            return OUT_OF_DOMAIN
        return not decisive
    args = [_eval(a, ctx, env) for a in node.args]
    if any(v is OUT_OF_DOMAIN for v in args):
        return OUT_OF_DOMAIN
    if k == "neg":
        return -args[0]
    if k == "add":
        return args[0] + args[1]
    if k == "sub":
        return args[0] - args[1]
    if k == "mul":
        return args[0] * args[1]
    if k == "mod":
        return args[0] % node.value
    if k == "div":
        return args[0] // node.value
    if k == "cmp":
        a, b = args
        op = node.value
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        return a >= b
    if k == "in":
        return args[0] in node.value
    raise DslError(f"Неизвестный узел при вычислении: {k}")


# ---------------------------------------------------------------- статический анализ


def unguarded(node: Node, lang: Language) -> frozenset[str]:
    """Атрибуты, отсутствие которых может сделать выражение OUT_OF_DOMAIN."""
    k = node.kind
    if k in ("lit", "var", "has"):
        return frozenset()
    if k == "attr":
        return frozenset({node.value})
    if k == "derived":
        return lang.derived_unguarded[node.value]
    if k in ("all", "any"):
        var, domain = node.value
        return frozenset({domain}) | unguarded(node.args[0], lang)
    inner = frozenset().union(*(unguarded(a, lang) for a in node.args)) if node.args else frozenset()
    if k == "and":
        return inner - frozenset().union(*(absent_makes(a, False) for a in node.args))
    if k == "or":
        return inner - frozenset().union(*(absent_makes(a, True) for a in node.args))
    return inner


def absent_makes(node: Node, value: bool) -> frozenset[str]:
    """Атрибуты, при отсутствии которых выражение гарантированно равно value."""
    k = node.kind
    if k == "has":
        return frozenset({node.value}) if value is False else frozenset()
    if k == "not":
        return absent_makes(node.args[0], not value)
    if k in ("and", "or"):
        parts = [absent_makes(a, value) for a in node.args]
        # and=false (or=true) решается одним операндом; and=true (or=false) — всеми
        decided_by_one = (k == "and") != value
        if decided_by_one:
            return frozenset().union(*parts)
        return frozenset.intersection(*parts)
    return frozenset()


def present_if(node: Node, value: bool, lang: Language) -> frozenset[str]:
    """Атрибуты, которые гарантированно присутствуют, если выражение равно value."""
    k = node.kind
    if k == "has":
        return frozenset({node.value}) if value else frozenset()
    if k == "not":
        return present_if(node.args[0], not value, lang)
    if k in ("and", "or"):
        parts = [present_if(a, value, lang) for a in node.args]
        all_operands_hold = (k == "and") == value
        if all_operands_hold:
            return frozenset().union(*parts)
        return frozenset.intersection(*parts)
    if k in ("all", "any"):
        return frozenset({node.value[1]})
    # Сравнение и прочие листья: раз значение определено, все их атрибуты на месте.
    return unguarded(node, lang)


# ---------------------------------------------------------------- каноническая форма


@lru_cache(maxsize=65536)
def to_text(node: Node) -> str:
    node = _effective(node)
    k = node.kind
    if k == "lit":
        return _lit_text(node.value)
    if k in ("attr", "var", "derived", "name"):
        return node.value
    if k == "has":
        return f"has({node.value})"
    if k == "call":
        return f"{node.value}({to_text(node.args[0])})"
    if k in ("and", "or"):
        return f" {k} ".join(sorted({_wrap(a, k) for a in _flatten(node, k)}))
    if k == "not":
        return f"not {_wrap(node.args[0], 'not')}"
    if k == "neg":
        return f"-{_wrap(node.args[0], 'neg')}"
    if k in ("add", "sub", "mul"):
        sym = {"add": "+", "sub": "-", "mul": "*"}[k]
        return f"{_wrap(node.args[0], k)} {sym} {_wrap(node.args[1], k, right=True)}"
    if k in ("mod", "div"):
        return f"{_wrap(node.args[0], k)} {k} {node.value}"
    if k == "cmp":
        return f"{_wrap(node.args[0], 'cmp')} {node.value} {_wrap(node.args[1], 'cmp')}"
    if k == "in":
        items = sorted((_lit_text(v) for v in node.value))
        return f"{_wrap(node.args[0], 'cmp')} in {{{', '.join(items)}}}"
    if k in ("all", "any"):
        var, domain = node.value
        return f"{k} {var} in {domain}: {to_text(node.args[0])}"
    raise DslError(f"Неизвестный узел: {k}")


_PREC = {"or": 1, "and": 2, "not": 3, "cmp": 4, "in": 4, "add": 5, "sub": 5, "mul": 6, "mod": 6, "div": 6, "neg": 7}


def _effective(node: Node) -> Node:
    """and/or, у которого после удаления повторов остался один операнд, равен этому операнду."""
    while node.kind in ("and", "or"):
        unique = {to_text(a): a for a in _flatten(node, node.kind)}
        if len(unique) != 1:
            break
        node = next(iter(unique.values()))
    return node


def _wrap(node: Node, parent: str, right: bool = False) -> str:
    node = _effective(node)
    text = to_text(node)
    if node.kind in ("all", "any"):
        return f"({text})"  # тело квантора жадное, поэтому квантор всегда в скобках
    p = _PREC.get(node.kind)
    if p is None:
        return text
    pp = _PREC[parent]
    if p < pp:
        return f"({text})"
    if p == pp and (parent == "cmp" or right):
        return f"({text})"  # сравнения не ассоциативны; правый операнд того же уровня — в скобках
    return text


def _flatten(node: Node, kind: str) -> Iterable[Node]:
    for a in node.args:
        if a.kind == kind:
            yield from _flatten(a, kind)
        else:
            yield a


def _lit_text(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------- предикат


@dataclass(frozen=True)
class Evaluation:
    value: Any  # True | False | OUT_OF_DOMAIN
    missing: tuple[str, ...] = ()
    steps: int = 0


@dataclass(frozen=True)
class Predicate:
    language: Language = field(repr=False, compare=False)
    node: Node

    @property
    def text(self) -> str:
        return to_text(self.node)

    @property
    def hash(self) -> str:
        return digest({"predicate": self.text, "schema": self.language.schema.hash()})

    def unguarded(self) -> frozenset[str]:
        return unguarded(self.node, self.language)

    def present_if(self, value: bool) -> frozenset[str]:
        return present_if(self.node, value, self.language)

    def evaluate(self, record: dict[str, Any]) -> Evaluation:
        """record — запись, уже приведённая к схеме (Schema.coerce_record)."""
        ctx = _Ctx(self.language, record, self.language.schema.budget)
        value = _eval(self.node, ctx, {})
        missing = tuple(sorted(ctx.missing)) if value is OUT_OF_DOMAIN else ()
        return Evaluation(value, missing, ctx.steps)
