import unittest

import numba
import numpy as np

from adapters.field_lattice.external_compression import forces, initial_state, potential, shell_force


class ExternalCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numba.set_num_threads(2)

    def test_pair_repulsion_reciprocity_and_inverse_square(self):
        p = np.array([[0., 0., 0.], [2., 0., 0.]])
        f, d = forces(p, 2)
        np.testing.assert_allclose(f, [[-.25, 0, 0], [.25, 0, 0]], atol=1e-15)
        self.assertEqual(d, 2.)
        p[1, 0] = 4
        f2, _ = forces(p, 2)
        np.testing.assert_allclose(f2, f/4, atol=1e-15)

    def test_force_is_negative_gradient_of_energy(self):
        p = np.array([[.1, .2, -.1], [2., .4, .5], [-1., .5, 1.]])
        f, _ = forces(p, 3)
        for i in range(3):
            for axis in range(3):
                a, b = p.copy(), p.copy()
                a[i, axis] += 1e-6
                b[i, axis] -= 1e-6
                self.assertAlmostEqual(f[i, axis], -(potential(a)-potential(b))/2e-6, delta=1e-9)
        np.testing.assert_allclose(f.sum(axis=0), 0, atol=1e-14)

    def test_shell_cancellation_and_nonzero_external_field(self):
        for r in (0., .25, .5, .75):
            self.assertLess(abs(shell_force(r, 128)), 1e-8)
        for r in (1.25, 2.):
            self.assertAlmostEqual(shell_force(r, 128), 1/r**2, delta=1e-8)

    def test_nested_sources_and_prescribed_central_radius(self):
        small, _ = initial_state(4, 1, .5)
        large, velocity = initial_state(8, 1, .5)
        subset = large[8:][np.linalg.norm(large[8:], axis=1) <= 4]
        np.testing.assert_array_equal(small[8:], subset)
        np.testing.assert_allclose(np.linalg.norm(large[:8], axis=1), .5)
        np.testing.assert_allclose(velocity[:8].sum(axis=0), 0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
