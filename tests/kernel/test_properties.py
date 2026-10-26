"""Свойства ядра на случайных выражениях (план, этап M1: ≥ 10⁵ случаев).

- детерминизм: повторное вычисление даёт тот же результат;
- каноническая форма: разбор канонического текста даёт тот же текст и то же значение;
- трёхзначная семантика сверяется с независимым эталоном;
- корректность статического анализа: если все атрибуты из `unguarded` на месте,
  результат никогда не OUT_OF_DOMAIN.
"""
import random
import unittest

from delta_kernel.dsl import OUT_OF_DOMAIN, Language
from delta_kernel.schema import Schema

SCHEMA = Schema.from_dict({
    "name": "random",
    "version": "1",
    "attributes": [
        {"name": "a", "type": {"kind": "int"}, "required": False},
        {"name": "b", "type": {"kind": "int"}, "required": False},
        {"name": "p", "type": {"kind": "bool"}, "required": False},
        {"name": "q", "type": {"kind": "bool"}, "required": False},
        {"name": "c", "type": {"kind": "enum", "values": ["x", "y", "z"]}, "required": False},
    ],
})
LANG = Language(SCHEMA)
ATTRS = ["a", "b", "p", "q", "c"]


def rand_term(rng, depth):
    if depth <= 0 or rng.random() < 0.4:
        return rng.choice(["a", "b", str(rng.randint(-3, 3))])
    op = rng.choice(["+", "-", "*", "mod", "div"])
    left = rand_term(rng, depth - 1)
    if op in ("mod", "div"):
        return f"({left} {op} {rng.choice([2, 3, -2])})"
    return f"({left} {op} {rand_term(rng, depth - 1)})"


def rand_bool(rng, depth):
    if depth <= 0 or rng.random() < 0.3:
        kind = rng.randrange(5)
        if kind == 0:
            return rng.choice(["p", "q", "true", "false"])
        if kind == 1:
            return f"has({rng.choice(ATTRS)})"
        if kind == 2:
            return f"c {rng.choice(['==', '!='])} \"{rng.choice('xyz')}\""
        if kind == 3:
            items = rng.sample("xyz", rng.randint(1, 3))
            return "c in {" + ", ".join(f'"{v}"' for v in items) + "}"
        return f"{rand_term(rng, 2)} {rng.choice(['==', '!=', '<', '<=', '>', '>='])} {rand_term(rng, 2)}"
    kind = rng.randrange(3)
    if kind == 0:
        return f"not ({rand_bool(rng, depth - 1)})"
    op = "and" if kind == 1 else "or"
    return f"({rand_bool(rng, depth - 1)}) {op} ({rand_bool(rng, depth - 1)})"


def rand_record(rng):
    record = {}
    if rng.random() < 0.8:
        record["a"] = rng.randint(-5, 5)
    if rng.random() < 0.8:
        record["b"] = rng.randint(-5, 5)
    if rng.random() < 0.8:
        record["p"] = rng.random() < 0.5
    if rng.random() < 0.8:
        record["q"] = rng.random() < 0.5
    if rng.random() < 0.8:
        record["c"] = rng.choice("xyz")
    return record


def oracle(node, record):
    """Независимый эталон трёхзначной семантики (без бюджета и кэшей)."""
    k = node.kind
    if k == "lit":
        return node.value
    if k == "attr":
        return record.get(node.value, OUT_OF_DOMAIN)
    if k == "has":
        return node.value in record
    if k == "not":
        v = oracle(node.args[0], record)
        return OUT_OF_DOMAIN if v is OUT_OF_DOMAIN else not v
    if k in ("and", "or"):
        vals = [oracle(a, record) for a in node.args]
        dominant = k == "or"
        if dominant in vals:
            return dominant
        return OUT_OF_DOMAIN if OUT_OF_DOMAIN in vals else not dominant
    vals = [oracle(a, record) for a in node.args]
    if OUT_OF_DOMAIN in vals:
        return OUT_OF_DOMAIN
    ops = {
        "neg": lambda x: -x, "add": lambda x, y: x + y, "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y, "mod": lambda x: x % node.value, "div": lambda x: x // node.value,
        "in": lambda x: x in node.value,
    }
    if k == "cmp":
        x, y = vals
        return {"==": x == y, "!=": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[node.value]
    return ops[k](*vals)


class PropertyTests(unittest.TestCase):
    def test_random_predicates(self):
        rng = random.Random(20261026)
        cases = 0
        for _ in range(2500):
            pred = LANG.compile(rand_bool(rng, 4))
            canon = LANG.compile(pred.text)
            self.assertEqual(canon.text, pred.text)
            guard = pred.unguarded()
            for _ in range(40):
                record = rand_record(rng)
                value = pred.evaluate(record).value
                cases += 1
                self.assertIs(pred.evaluate(record).value, value)
                self.assertIs(canon.evaluate(record).value, value, pred.text)
                self.assertIs(oracle(pred.node, record), value, pred.text)
                if guard <= set(record):
                    self.assertIsNot(value, OUT_OF_DOMAIN, f"{pred.text} on {record}")
                if value is not OUT_OF_DOMAIN:
                    self.assertLessEqual(pred.present_if(value), set(record), pred.text)
        self.assertGreaterEqual(cases, 100_000)


if __name__ == "__main__":
    unittest.main()
