import json
import tempfile
import unittest
from pathlib import Path

from logical_delta_matter.compound_space import compute_compound_space


class CompoundSpaceTests(unittest.TestCase):
    def test_computes_possible_and_impossible_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concepts_path = root / "concepts.json"
            config_path = root / "config.json"
            out_path = root / "computed.json"
            concepts_path.write_text(json.dumps(minimal_concepts()), encoding="utf-8")
            config_path.write_text(json.dumps(minimal_config()), encoding="utf-8")

            output = compute_compound_space(concepts_path, config_path, out_path)
            by_formula = {row["formula"]: row for row in output["candidates"]}

            self.assertTrue(out_path.exists())
            self.assertEqual(by_formula["NaO"]["status"], "possible")
            self.assertEqual(by_formula["H2"]["status"], "impossible")
            self.assertIn("same_repulsion_homonuclear_without_binding_mechanism", by_formula["H2"]["reasons"])
            self.assertIn("compound.known_elements", [row["delta_id"] for row in by_formula["NaO"]["decision_trace"]])
            self.assertGreater(output["summary"]["possible"], 0)
            self.assertGreater(output["summary"]["impossible"], 0)
            self.assertEqual(output["decision_manifest"][0]["domain"], "CompoundCandidate")


def minimal_concepts():
    return {
        "schema_version": "confirmed-concepts@test",
        "elements": [
            element("H", "Hydrogen", 1, 0, "Nonmetal"),
            element("O", "Oxygen", 8, 8, "Nonmetal"),
            element("Na", "Sodium", 11, 12, "Alkali metal"),
        ],
        "compounds": [],
    }


def element(symbol, name, atomic_number, neutrons, group_block):
    return {
        "symbol": symbol,
        "name_en": name,
        "atomic_number": atomic_number,
        "atomic_mass": str(atomic_number + neutrons),
        "standard_state": "Gas",
        "group_block": group_block,
        "aliases": [symbol, name],
        "verified": {"source_id": "test"},
        "model": {"neutrons": neutrons, "phase": "gas"},
    }


def minimal_config():
    return {
        "schema_version": "compound-space-config@test",
        "rule_pack_id": "katyushchik_hypothesis",
        "particle_stock": {"protons": 100, "neutrons": 100, "electrons": 100},
        "generation": {
            "elements": ["H", "O", "Na"],
            "max_total_atoms": 3,
            "max_distinct_elements": 2,
            "max_count_per_element": 2,
            "include_confirmed": False,
        },
        "rules": {
            "min_atoms_for_compound": 2,
            "same_repulsion_homonuclear_impossible": True,
            "min_repulsion_classes_for_generated": 2,
            "noble_gas_requires_confirmation": True,
            "resource_limited": True,
        },
    }


if __name__ == "__main__":
    unittest.main()
