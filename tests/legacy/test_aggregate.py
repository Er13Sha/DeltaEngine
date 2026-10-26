import unittest

from logical_delta_matter.aggregate import compute_signature, detect_clusters
from logical_delta_matter.world_exact import WorldState


class AggregateTests(unittest.TestCase):
    def test_detects_connected_components(self):
        state = WorldState(
            positions=[(0, 0, 0), (1, 0, 0), (5, 5, 5)],
            types=["proto_A", "proto_A", "proto_B"],
            occupied={(0, 0, 0): 0, (1, 0, 0): 1, (5, 5, 5): 2},
            step=0,
        )
        clusters = detect_clusters(state)
        sizes = sorted(cluster["size"] for cluster in clusters)
        self.assertEqual(sizes, [1, 2])

    def test_signature_is_integer(self):
        state = WorldState(
            positions=[(0, 0, 0), (1, 0, 0)],
            types=["proto_A", "proto_A"],
            occupied={(0, 0, 0): 0, (1, 0, 0): 1},
            step=0,
        )
        signature = compute_signature(detect_clusters(state)[0], state)
        self.assertEqual(signature["size"], 2)
        self.assertIsInstance(signature["compactness_permille"], int)


if __name__ == "__main__":
    unittest.main()
