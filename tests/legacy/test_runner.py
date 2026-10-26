import json
import tempfile
import unittest
from pathlib import Path

from logical_delta_matter.runner import run_simulation


class RunnerTests(unittest.TestCase):
    def test_run_exports_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = {
                "world": {
                    "grid_shape": [8, 8, 8],
                    "seed": 3,
                    "max_steps": 3,
                    "neighborhood_radius": 1,
                    "checkpoint_every": 2,
                },
                "particle_counts": {"proto_A": 10},
                "interaction_weights": {"proto_A": 1},
            }
            config_path = tmp_path / "config.json"
            export_path = tmp_path / "exact-latest.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            summary = run_simulation(config_path, tmp_path / "run", export_path)
            self.assertTrue(export_path.exists())
            self.assertEqual(summary["metrics"]["step"], 3)
            self.assertIn("state_hash", summary["metrics"])


if __name__ == "__main__":
    unittest.main()
