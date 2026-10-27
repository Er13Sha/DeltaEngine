import json
import math
import unittest
from pathlib import Path

import numpy as np

from adapters.field_lattice.external_compression_v2 import (
    B_CUBE,
    ball_quadrature,
    cavity_forces_group,
    cube_positions,
    cube_radius,
    cube_scale,
    evaluate_stage_a,
    evaluate_stage_b,
    load_protocol,
    medium_positions,
    pair_forces_numpy,
    rms_radius,
    spring,
)

ROOT = Path(__file__).resolve().parents[2]


class ExternalCompressionV2Tests(unittest.TestCase):
    def test_cube_equilibrium_matches_direct_force(self):
        for rho in (1.0, 0.5):
            scale = cube_scale(rho)
            positions = cube_positions(scale)
            force, _ = cavity_forces_group(positions, rho)
            self.assertLess(np.linalg.norm(force), 1e-10)
            self.assertAlmostEqual(cube_radius(1.0), 0.838291753609, delta=1e-12)
            self.assertAlmostEqual(cube_scale(rho / 2) / scale, 2 ** (1 / 3), delta=1e-12)

    def test_equilibrium_constant_matches_formula(self):
        expected = 0.25 + 1 / (4 * math.sqrt(2)) + 1 / (12 * math.sqrt(3))
        self.assertAlmostEqual(B_CUBE, expected, delta=1e-15)
        self.assertAlmostEqual(cube_scale(1.0) ** 3, 3 * B_CUBE / (4 * math.pi), delta=1e-12)

    def test_cavity_coefficient_matches_ball_quadrature(self):
        radius, rho = 0.4, 1.0
        measured = ball_quadrature(radius, rho, order=128)
        self.assertLess(abs(measured / (spring(rho) * radius) - 1), 1e-3)

    def test_zero_density_is_pure_repulsion(self):
        positions = cube_positions(0.5)
        cavity, _ = cavity_forces_group(positions, 0.0)
        pair, _ = pair_forces_numpy(positions)
        np.testing.assert_allclose(cavity, pair, atol=1e-12)
        self.assertGreater(rms_radius(positions), 0)

    def test_initial_medium_is_separated(self):
        for spacing in (1.0, 0.5):
            positions, hole, rho = medium_positions(spacing, 4.0)
            delta = positions[:, None, :] - positions[None, :, :]
            distance = np.sqrt(np.sum(delta * delta, axis=2))
            np.fill_diagonal(distance, np.inf)
            self.assertGreater(distance.min(), 0.2 * spacing)
            self.assertAlmostEqual(hole, (6 / (math.pi * rho)) ** (1 / 3), delta=1e-12)
            self.assertEqual(len(positions) > 8, True)

    def test_recorded_decisions_follow_the_protocol(self):
        protocol = load_protocol()
        stage_a = json.loads((ROOT / "data/experiments/field/external_compression_v2/stage_a.json").read_text(encoding="utf-8"))
        stage_b = json.loads((ROOT / "data/experiments/field/external_compression_v2/stage_b.json").read_text(encoding="utf-8"))
        self.assertEqual(evaluate_stage_a(stage_a, protocol), stage_a["decision"])
        self.assertEqual(evaluate_stage_b(stage_b, protocol), stage_b["decision"])
        self.assertTrue(stage_a["decision"]["passed"])
        self.assertFalse(stage_b["decision"]["passed"])
