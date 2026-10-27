from fractions import Fraction
import unittest

from logical_delta_matter.formal_domains import (
    DiscretenessTraits,
    MatterTraits,
    Plane,
    classify_discreteness,
    classify_matter,
    classify_plane_pair,
)


class FormalDomainsTests(unittest.TestCase):
    def test_matter_hierarchy_reaches_liquid_leaf(self):
        decision = classify_matter(
            MatterTraits(
                continuous=False,
                grouped=True,
                planetary=False,
                preserves_shape=False,
                preserves_volume=True,
                ionized=False,
            )
        )
        self.assertEqual(decision.leaf, "Liquid")
        self.assertEqual([step["delta_id"] for step in decision.trace][-1], "fluid.preserves_volume")

    def test_discreteness_has_fourth_state(self):
        decision = classify_discreteness(DiscretenessTraits(bounded=True, internally_discontinuous=True))
        self.assertEqual(decision.leaf, "InternalExternalDiscrete")
        self.assertEqual(len(decision.trace), 2)

    def test_plane_pair_detects_perpendicular_intersection(self):
        decision = classify_plane_pair(Plane.from_ints(1, 0, 0, 0), Plane.from_ints(0, 1, 0, 0))
        self.assertEqual(decision.leaf, "Perpendicular")

    def test_plane_pair_detects_coincident(self):
        decision = classify_plane_pair(
            Plane((Fraction(1), Fraction(2), Fraction(3)), Fraction(4)),
            Plane((Fraction(2), Fraction(4), Fraction(6)), Fraction(8)),
        )
        self.assertEqual(decision.leaf, "Coincident")

    def test_plane_pair_detects_parallel_without_common_points(self):
        decision = classify_plane_pair(Plane.from_ints(1, 0, 0, 0), Plane.from_ints(1, 0, 0, 1))
        self.assertEqual(decision.leaf, "Parallel")


if __name__ == "__main__":
    unittest.main()
