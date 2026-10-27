"""Дешёвые проверки постановки E-COMP v2. Сетка P1 не запускается."""
import itertools
from fractions import Fraction
import unittest

import numba
import numpy as np

from adapters.field_lattice.external_compression_v2 import (
    N_INF_CAMPAIGN,
    R_INF,
    apply_reservoir_constraint,
    belt_nodes_no_jitter,
    campaign_n_infinity,
    forces,
    initial_state,
    integrate,
    pair_force_exact,
    protocol_registration_ok,
    run,
    t0_checks,
    thick_shell_force,
)


class ExternalCompressionV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(2)

    def test_registration_matches_protocol_files(self):
        self.assertTrue(protocol_registration_ok())

    def test_campaign_counts_match_the_card(self):
        self.assertEqual(campaign_n_infinity(), N_INF_CAMPAIGN)
        self.assertEqual(campaign_n_infinity(), 33144)
        self.assertEqual(belt_nodes_no_jitter(), 164)

    def test_pair_repulsion_has_no_attractive_term(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        force, distance = forces(positions, 2, n_inf=0.0)
        np.testing.assert_allclose(force, [[-0.25, 0, 0], [0.25, 0, 0]], atol=1e-15)
        self.assertEqual(distance, 2.0)
        self.assertLess(force[0, 0], 0.0)
        self.assertGreater(force[1, 0], 0.0)
        positions[1, 0] = 4.0
        force2, _ = forces(positions, 2, n_inf=0.0)
        np.testing.assert_allclose(force2, force / 4, atol=1e-15)

    def test_exact_pair_law_conserves_r_squared_force(self):
        for exponent in (0, 1, 2, 6, 12):
            radius = Fraction(10 ** exponent)
            force = pair_force_exact(radius)
            self.assertGreater(force, 0)
            self.assertEqual(force * radius * radius, 1)
            self.assertEqual(pair_force_exact(2 * radius) / force, Fraction(1, 4))

    def test_no_central_spring_on_a_lone_node(self):
        probe = np.array([[0.2, 0.0, 0.0]])
        force, _ = forces(probe, 1, n_inf=0.0)
        np.testing.assert_allclose(force, 0, atol=1e-15)
        inside, _ = forces(probe, 1, n_inf=float(N_INF_CAMPAIGN), r_inf=R_INF)
        np.testing.assert_allclose(inside, 0, atol=1e-15)

    def test_pair_force_is_along_separation_not_toward_origin(self):
        positions = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
        force, _ = forces(positions, 2, n_inf=0.0)
        np.testing.assert_allclose(force[:, 1], 0, atol=1e-15)
        np.testing.assert_allclose(force[:, 2], 0, atol=1e-15)
        self.assertLess(force[0, 0], 0.0)
        self.assertGreater(force[1, 0], 0.0)

    def test_far_spherical_replacement_is_zero_inside_inverse_square_outside(self):
        for radius in (0.0, 5.0, 19.9):
            probe = np.array([[radius, 0.0, 0.0]])
            force, _ = forces(probe, 1, n_inf=1.0, r_inf=R_INF)
            np.testing.assert_allclose(force, 0, atol=1e-15)
        for radius in (25.0, 40.0):
            probe = np.array([[radius, 0.0, 0.0]])
            force, _ = forces(probe, 1, n_inf=1.0, r_inf=R_INF)
            np.testing.assert_allclose(force[0], [1 / radius ** 2, 0, 0], atol=1e-15)
            self.assertAlmostEqual(float(np.log(abs(force[0, 0])) + 2 * np.log(radius)), 0.0, places=12)

    def test_t0_static_checks(self):
        passed, rows = t0_checks()
        self.assertTrue(passed, rows)
        self.assertLess(abs(thick_shell_force(0.0)), 1e-8)

    def test_isolated_octet_still_expands(self):
        result = run(family="isolated", seed=1, omega=0.0, dt=0.002, tmax=0.1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["nodes"], 8)
        self.assertEqual(result["n_inf"], 0.0)
        self.assertGreater(result["trace"][-1]["rms_radius"], result["trace"][0]["rms_radius"])

    def test_reservoir_wall_does_not_act_on_octet_or_origin(self):
        origin = np.zeros(3)
        belt = 3.5 * np.array(
            [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
        )
        positions = np.vstack([origin, belt])
        velocities = np.zeros_like(positions)
        reservoir = np.array([False, True, True, True, True, True, True, False][: len(positions)])
        force, _ = forces(positions, len(positions), n_inf=0.0)
        np.testing.assert_allclose(force[0], 0, atol=1e-14)
        hit = apply_reservoir_constraint(positions, velocities, reservoir)
        self.assertFalse(hit)
        np.testing.assert_allclose(positions[0], 0)
        escaped = np.array([[5.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
        escaped_v = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        apply_reservoir_constraint(escaped, escaped_v, np.array([False, True]))
        np.testing.assert_allclose(escaped[0], [5.0, 0.0, 0.0])
        np.testing.assert_allclose(escaped_v[0], [1.0, 0.0, 0.0])

    def test_inner_wall_projects_reservoir_outward_not_inward(self):
        positions = np.array([[0.2, 0.0, 0.0], [2.9, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        reservoir = np.array([False, True])
        self.assertTrue(apply_reservoir_constraint(positions, velocities, reservoir))
        np.testing.assert_allclose(positions[0], [0.2, 0.0, 0.0])
        np.testing.assert_allclose(positions[1], [3.0, 0.0, 0.0])
        np.testing.assert_allclose(velocities[0], 0)
        np.testing.assert_allclose(velocities[1], 0)

    def test_frozen_belt_stays_fixed_reservoir_can_move(self):
        central = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * (0.5 / np.sqrt(3))
        belt = np.array([[3.5, 0.0, 0.0], [-3.5, 0.0, 0.0], [0.0, 3.5, 0.0]])
        positions = np.vstack([central, belt])
        velocities = np.zeros_like(positions)
        reservoir = np.zeros(len(positions), dtype=bool)
        reservoir[8:] = True
        frozen = integrate(
            positions, velocities, family="frozen", reservoir=reservoir, n_inf=0.0, dt=0.002, tmax=0.02
        )
        mobile = integrate(
            positions, velocities, family="mobile_reservoir", reservoir=reservoir, n_inf=0.0, dt=0.002, tmax=0.02
        )
        np.testing.assert_allclose(frozen["positions"][8:], belt)
        self.assertFalse(np.allclose(mobile["positions"][8:], belt))
        self.assertGreater(np.linalg.norm(mobile["positions"][8:] - frozen["positions"][8:]), 0.0)

    def test_initial_belt_stays_out_of_the_cavity(self):
        positions, _, reservoir, n_inf = initial_state(1, 0.0, "mobile_reservoir")
        self.assertEqual(int(n_inf), N_INF_CAMPAIGN)
        self.assertGreater(int(reservoir.sum()), 100)
        radii = np.linalg.norm(positions[8:], axis=1)
        self.assertTrue(np.all(radii >= 3.0 - 1e-12))
        self.assertTrue(np.all(radii <= 4.0 + 1e-12))
        self.assertTrue(np.all(np.linalg.norm(positions[:8], axis=1) < 1.5))

    def test_run_requires_explicit_tmax(self):
        with self.assertRaises(TypeError):
            run(family="isolated", seed=1, omega=0.0, dt=0.002)


if __name__ == "__main__":
    unittest.main()
