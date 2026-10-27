import unittest
from fractions import Fraction

from delta_science.action import C1, E_CHARGE, G_NEWTON, M_PROTON, PI, coupling, force, quality, quantity, textbook_check


class ActionFormulaTests(unittest.TestCase):
    """Формула воздействия из видео Rek8C322Q5k: W = Q·N, Q = k·c1/(4πr²), N = h·S·ρ."""

    def test_gravity_coupling_is_newton_constant_exactly(self):
        c = coupling(C1["тяготение"])
        self.assertEqual(c.power, 0)  # 4π сократились точно
        self.assertEqual(c.coef, G_NEWTON)

    def test_force_reduces_to_newton_law(self):
        f = force(C1["тяготение"], 2, M_PROTON, M_PROTON)
        self.assertEqual(f.coef, G_NEWTON * M_PROTON ** 2 / 4)
        self.assertEqual(f.power, 0)

    def test_coulomb_constant_matches_table(self):
        self.assertAlmostEqual(float(coupling(C1["электричество"])) / 8.9875517923e9, 1.0, delta=1e-10)

    def test_inverse_square_and_quantity(self):
        self.assertEqual(quality(C1["тяготение"], 1).coef / quality(C1["тяготение"], 3).coef, 9)
        self.assertEqual(quantity(Fraction(1, 2), 4, 3), 6)  # масса кружки: высота × дно × плотность

    def test_textbook_check(self):
        check = textbook_check()
        self.assertTrue(check["gravity_equals_newton_exactly"])
        self.assertAlmostEqual(check["coulomb_over_gravity_protons"] / 1.2355516e36, 1.0, delta=1e-6)
        self.assertEqual(PI.power, 1)
        self.assertGreater(float(force(C1["электричество"], 1, E_CHARGE, E_CHARGE)), 0)
