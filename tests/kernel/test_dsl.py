import unittest
from fractions import Fraction

from delta_kernel.dsl import OUT_OF_DOMAIN, BudgetExceeded, DslError, Language, parse, to_text
from delta_kernel.schema import Schema, SchemaError

SCHEMA = {
    "name": "element",
    "version": "1",
    "attributes": [
        {"name": "z", "type": {"kind": "int"}},
        {"name": "mass", "type": {"kind": "rat"}, "defining": False},
        {"name": "block", "type": {"kind": "enum", "values": ["s", "p", "d", "f"]}},
        {"name": "symbol", "type": {"kind": "str"}},
        {"name": "radioactive", "type": {"kind": "bool"}, "required": False},
        {"name": "phases", "type": {"kind": "set", "element": "enum", "values": ["solid", "liquid", "gas"]}, "required": False},
    ],
    "tables": [
        {"name": "period_of", "key": {"kind": "int"}, "value": {"kind": "int"},
         "mapping": [[1, 1], [2, 1], [3, 2], [11, 3]], "default": 0},
    ],
    "derived": [
        {"name": "even_z", "expr": "z mod 2 == 0"},
        {"name": "period", "expr": "period_of(z)"},
    ],
}

SODIUM = {"z": 11, "mass": "22.99", "block": "s", "symbol": "Na", "phases": ["solid"]}


class DslTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema.from_dict(SCHEMA)
        self.lang = Language(self.schema)
        self.na = self.schema.coerce_record(SODIUM)

    def value(self, text, record=None):
        return self.lang.compile(text).evaluate(self.na if record is None else record).value

    def test_basic_comparisons(self):
        self.assertIs(self.value("z == 11"), True)
        self.assertIs(self.value("z > 11"), False)
        self.assertIs(self.value("mass < 23"), True)
        self.assertIs(self.value("mass == 2299/100"), True)
        self.assertIs(self.value("block == \"s\" and symbol == \"Na\""), True)
        self.assertIs(self.value("block in {\"d\", \"f\"}"), False)

    def test_exact_arithmetic(self):
        self.assertIs(self.value("mass * 100 == 2299"), True)
        self.assertIs(self.value("z div 2 == 5 and z mod 2 == 1"), True)
        self.assertIs(self.value("-z + 11 == 0"), True)

    def test_tables_and_derived(self):
        self.assertIs(self.value("period == 3"), True)
        self.assertIs(self.value("period_of(z + 1) == 0"), True)  # default
        self.assertIs(self.value("even_z"), False)

    def test_quantifiers(self):
        self.assertIs(self.value("any p in phases: p == \"solid\""), True)
        self.assertIs(self.value("all p in phases: p == \"gas\""), False)

    def test_missing_attribute_is_out_of_domain_not_false(self):
        self.assertIs(self.value("radioactive"), OUT_OF_DOMAIN)
        self.assertIs(self.value("not radioactive"), OUT_OF_DOMAIN)
        ev = self.lang.compile("radioactive or z > 100").evaluate(self.na)
        self.assertEqual(ev.missing, ("radioactive",))

    def test_kleene_guards(self):
        self.assertIs(self.value("has(radioactive) and radioactive"), False)
        self.assertIs(self.value("not has(radioactive) or radioactive"), True)
        self.assertIs(self.value("z == 11 or radioactive"), True)

    def test_type_errors_are_compile_errors(self):
        bad = [
            "z",  # не bool
            "block == \"x\"",  # значения нет в enum
            "symbol > 3",
            "z == \"11\"",
            "z mod 0 == 1",
            "z mod 2.5 == 1",
            "unknown_attr",
            "phases",
            "a < b < c",
            "all p in z: p == 1",
            "period_of(\"x\") == 1",
            "z == 1 and",
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(DslError):
                self.lang.compile(text)

    def test_no_float_accepted(self):
        with self.assertRaises(SchemaError):
            self.schema.coerce_record({**SODIUM, "mass": 22.99})

    def test_canonical_form_is_stable_and_order_independent(self):
        a = self.lang.compile("symbol == \"Na\" and (z > 3 or block == \"s\")")
        b = self.lang.compile("(block == \"s\" or z > 3) and symbol == \"Na\"")
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.hash, b.hash)
        self.assertEqual(self.lang.compile(a.text).text, a.text)

    def test_canonical_form_keeps_meaning_of_parentheses(self):
        cases = ["z - (z - 1) == 1", "(z + 1) * 2 == 24", "not (z == 1 or z == 2)", "-(z - 12) == 1"]
        for text in cases:
            with self.subTest(text=text):
                p = self.lang.compile(text)
                self.assertIs(p.evaluate(self.na).value, True)
                self.assertIs(self.lang.compile(p.text).evaluate(self.na).value, True)

    def test_budget(self):
        schema = Schema.from_dict({**SCHEMA, "budget": 5})
        with self.assertRaises(BudgetExceeded):
            Language(schema).compile("z == 1 or z == 2 or z == 3").evaluate(self.na)

    def test_unguarded_analysis(self):
        self.assertEqual(self.lang.compile("radioactive").unguarded(), {"radioactive"})
        self.assertEqual(self.lang.compile("has(radioactive) and radioactive").unguarded(), frozenset())
        self.assertEqual(self.lang.compile("not has(radioactive) or radioactive").unguarded(), frozenset())
        self.assertEqual(self.lang.compile("has(radioactive) or radioactive").unguarded(), {"radioactive"})

    def test_rational_literals(self):
        self.assertEqual(parse("3/2").value, Fraction(3, 2))
        self.assertEqual(to_text(parse("1.25")), "5/4")


if __name__ == "__main__":
    unittest.main()
