import unittest

from logical_delta_matter.galaxy_shadow import (
    ShadowConfig,
    cupy_raw_kernel_source,
    direct_shadow_acceleration_soa_cpu,
    direct_shadow_forces_cpu,
    velocity_verlet_step_soa_cpu,
)


class GalaxyShadowTests(unittest.TestCase):
    def test_direct_shadow_forces_are_pairwise_balanced(self):
        forces = direct_shadow_forces_cpu(
            [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            [1.0, 1.0],
            ShadowConfig(cosmic_pressure=2.0, shadow_coeff=3.0, softening=0.0),
        )
        self.assertGreater(forces[0][0], 0.0)
        self.assertAlmostEqual(forces[0][0], -forces[1][0])
        self.assertEqual(forces[0][1], 0.0)
        self.assertEqual(forces[0][2], 0.0)

    def test_cuda_kernel_source_is_direct_n_body(self):
        source = cupy_raw_kernel_source()
        self.assertIn("for (int j = 0; j < n; ++j)", source)
        self.assertIn("shadow_forces", source)

    def test_soa_acceleration_has_confinement_component(self):
        ax, ay, az = direct_shadow_acceleration_soa_cpu(
            [2.0],
            [0.0],
            [0.0],
            [1.0],
            ShadowConfig(local_g=0.0, pressure_lambda=3.0, confinement_scale=1.0, softening=0.0),
        )
        self.assertLess(ax[0], 0.0)
        self.assertEqual(ay[0], 0.0)
        self.assertEqual(az[0], 0.0)

    def test_velocity_verlet_step_updates_soa_state(self):
        x = [-1.0, 1.0]
        y = [0.0, 0.0]
        z = [0.0, 0.0]
        vx = [0.0, 0.0]
        vy = [0.1, -0.1]
        vz = [0.0, 0.0]
        velocity_verlet_step_soa_cpu(x, y, z, vx, vy, vz, [1.0, 1.0], 0.01, ShadowConfig(local_g=1.0, softening=1e-6))
        self.assertNotEqual(x, [-1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
