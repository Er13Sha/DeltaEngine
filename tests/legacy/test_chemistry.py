import unittest

from logical_delta_matter.chemistry import capacity_for_requirement, hill_formula, parse_formula, shortage_for_requirement


class ChemistryTests(unittest.TestCase):
    def test_parse_formula_with_parentheses(self):
        self.assertEqual(parse_formula("Ca3(PO4)2"), {"Ca": 3, "P": 2, "O": 8})

    def test_parse_formula_rejects_unmatched_group(self):
        with self.assertRaises(ValueError):
            parse_formula("Ca3(PO4")

    def test_hill_formula_orders_carbon_first(self):
        self.assertEqual(hill_formula({"O": 6, "H": 12, "C": 6}), "C6H12O6")

    def test_requirement_capacity_and_shortage_are_integer_exact(self):
        requirement = {"protons": 12, "neutrons": 14, "electrons": 12}
        stock = {"protons": 30, "neutrons": 20, "electrons": 100}
        self.assertEqual(capacity_for_requirement(requirement, stock), 1)
        self.assertEqual(shortage_for_requirement(requirement, {"protons": 10, "neutrons": 20, "electrons": 4}), {
            "protons": 2,
            "neutrons": 0,
            "electrons": 8,
        })


if __name__ == "__main__":
    unittest.main()
