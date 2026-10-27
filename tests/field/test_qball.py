import json
import unittest
from pathlib import Path

import numpy as np
from adapters.field_lattice.qball import _evolve_radial, log_profile_at, profile_at, protocol_hash, qball


class QballTests(unittest.TestCase):
    def test_original_protocol_preserved(self):
        path = Path(__file__).resolve().parents[2] / "data/experiments/field/qball_protocol.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["hash"], protocol_hash())

    def test_profile_reaches_vacuum_and_satisfies_virial(self):
        for omega in (0.3, 0.8, 0.95):
            with self.subTest(omega=omega):
                q = qball(omega)
                self.assertLess(q["f"][-1] / q["f0"], 1e-4)
                self.assertLess(q["virial_relative_error"], 1e-3)

    def test_tail_has_no_finite_cutoff(self):
        q = qball(0.81)
        radii = np.array([40., 80., 160., 240.])
        values = profile_at(q, radii)
        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(np.diff(values) < 0))
        self.assertAlmostEqual(values[1]/values[0], 0.5*np.exp(-q["kappa"]*40), places=20)
        self.assertLess(q["tail_log_derivative_error"], 1e-4)

    def test_log_tail_remains_finite_at_large_distance(self):
        q = qball(0.81)
        self.assertTrue(np.isfinite(log_profile_at(q, 1e6)))
        self.assertAlmostEqual(float(log_profile_at(q, 40)), float(np.log(profile_at(q, 40))), places=12)

    def test_energy_charge_derivative(self):
        a, b = qball(.8099), qball(.8101)
        self.assertAlmostEqual((b["E"]-a["E"])/(b["Q"]-a["Q"]), .81, delta=1e-5)

    def test_both_charge_signs_are_conserved_without_absorber(self):
        q = qball(.81)
        r = (np.arange(400)+.5)*.1
        f = profile_at(q, r)
        for sign in (-1, 1):
            trace = _evolve_radial(f.copy(), np.zeros(400), np.zeros(400), sign*.81*f,
                                   .1, .01, 500, 100., 200, 100)
            self.assertEqual(np.sign(trace[0, 2]), sign)
            self.assertLess(np.max(np.abs(trace[:, 2]/trace[0, 2]-1)), 1e-10)


if __name__ == "__main__":
    unittest.main()
