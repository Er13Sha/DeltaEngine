import unittest

from logical_delta_matter.world_exact import WorldConfig, init_world, run_world, state_hash, step_world_exact


class WorldExactTests(unittest.TestCase):
    def cfg(self):
        return WorldConfig(
            grid_shape=(8, 8, 8),
            max_steps=5,
            seed=7,
            neighborhood_radius=2,
            checkpoint_every=2,
            particle_counts={"proto_A": 20, "proto_B": 12},
            interaction_weights={"proto_A": 3, "proto_B": 2},
        )

    def test_init_has_unique_occupied_cells(self):
        state = init_world(self.cfg())
        self.assertEqual(len(state.positions), len(set(state.positions)))
        self.assertEqual(len(state.occupied), len(state.positions))

    def test_step_preserves_particle_count(self):
        cfg = self.cfg()
        state = init_world(cfg)
        after = step_world_exact(state, cfg)
        self.assertEqual(len(after.positions), len(state.positions))
        self.assertEqual(len(after.occupied), len(state.positions))

    def test_reproducible_hash(self):
        cfg = self.cfg()
        left = [state_hash(state) for state in run_world(cfg)]
        right = [state_hash(state) for state in run_world(cfg)]
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
