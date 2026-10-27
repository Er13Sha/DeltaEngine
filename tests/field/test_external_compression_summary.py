import json
import unittest
from pathlib import Path

from adapters.field_lattice.external_compression import provenance
from adapters.field_lattice.external_compression_summary import build_summary

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data/experiments/field/external_compression_v1/summary.json"


class ExternalCompressionSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provenance()
        cls.fresh = build_summary()
        cls.saved = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_saved_summary_matches_runs(self):
        self.assertEqual(self.fresh, self.saved)

    def test_protocol_decision(self):
        summary = self.fresh
        self.assertTrue(summary["p0_passed"])
        self.assertEqual(summary["campaign_runs"], 27)
        self.assertEqual(summary["quality_runs"], 27)
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(summary["bounded_count"], 0)
        self.assertTrue(summary["finite_environment_expanded"])
        self.assertTrue(summary["convergence_passed"])
        self.assertTrue(summary["timestep_passed"])
        self.assertLess(summary["energy_relative_drift_max"], 1e-3)
        self.assertLess(summary["normalized_momentum_drift_max"], 1e-8)
        self.assertEqual(summary["pilot_repeat_rms_abs_difference"], 0.0)
        self.assertFalse(summary["p2_executed"])
        self.assertFalse(summary["found_particle"])
        self.assertFalse(summary["refutes_general_scheme"])
        self.assertFalse(summary["empirical_status_changed"])
        self.assertEqual(summary["scope"], "model")
