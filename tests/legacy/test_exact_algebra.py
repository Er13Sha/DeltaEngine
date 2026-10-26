import unittest

from logical_delta_matter.exact_algebra import build_space_formatting
from logical_delta_matter.smt_certificates import prove_excluded_middle


class ExactAlgebraTests(unittest.TestCase):
    def test_space_formatting_defines_l_and_t_symbolically(self):
        formatting = build_space_formatting().to_dict()
        self.assertEqual(formatting["definitions"], "L = R*n; T = R/n")
        self.assertIn(formatting["backend"], {"fallback", "sympy"})
        self.assertEqual(formatting["L"].replace(" ", ""), "R*n")
        self.assertEqual(formatting["T"].replace(" ", ""), "R/n")

    def test_smt_certificate_is_machine_readable_without_z3(self):
        certificate = prove_excluded_middle("Continuous").to_dict()
        self.assertEqual(certificate["solver"], "z3")
        self.assertIn(certificate["status"], {"proved", "failed", "unavailable"})
        self.assertIn("total", certificate["claims"])


if __name__ == "__main__":
    unittest.main()
