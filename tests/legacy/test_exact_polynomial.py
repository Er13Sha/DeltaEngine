from fractions import Fraction
import unittest

from logical_delta_matter.exact_polynomial import LaurentPolynomial


class ExactPolynomialTests(unittest.TestCase):
    def test_laurent_polynomial_exact_arithmetic(self):
        n = LaurentPolynomial.n()
        n_inv = LaurentPolynomial.n_inv()
        one = LaurentPolynomial.constant(1)
        self.assertEqual(n * n_inv, one)
        self.assertEqual((n + n_inv).to_dict(), {"-1": "1", "1": "1"})

    def test_rejects_float_coefficients(self):
        with self.assertRaises(TypeError):
            LaurentPolynomial({0: 1.25})

    def test_fraction_coefficients_are_allowed(self):
        poly = LaurentPolynomial({2: Fraction(3, 7)})
        self.assertEqual(poly.to_dict(), {"2": "3/7"})


if __name__ == "__main__":
    unittest.main()
