import copy
import itertools
import json
import unittest
from pathlib import Path

from delta_kernel import INVALID, PROVEN, UNKNOWN, Tree, TreeError
from delta_tools.bridge_matter import matter_tree_from_deltas
from logical_delta_matter.delta_core import classify_delta, compile_delta_tree

ROOT = Path(__file__).resolve().parents[2]
TRAITS = ["continuous", "grouped", "planetary", "preservesShape", "preservesVolume", "ionized"]

STATES = {
    "format": "delta-tree@0.1",
    "id": "states",
    "version": "1",
    "schema": {
        "name": "substance",
        "version": "1",
        "attributes": [
            {"name": "keeps_shape", "type": {"kind": "bool"}},
            {"name": "keeps_volume", "type": {"kind": "bool"}, "required": False},
            {"name": "mass_g", "type": {"kind": "rat"}, "defining": False},
        ],
    },
    "root": {
        "id": "shape",
        "predicate": "keeps_shape",
        "true": {"id": "solid", "label": "Твёрдое"},
        "false": {
            "id": "volume",
            "predicate": "keeps_volume",
            "true": {"id": "liquid", "label": "Жидкость"},
            "false": {"id": "gas", "label": "Газ"},
        },
    },
}


def matter_tree() -> Tree:
    data = json.loads((ROOT / "data" / "matter-deltas.json").read_text(encoding="utf-8"))
    return Tree.from_dict(matter_tree_from_deltas(data))


class TreeTests(unittest.TestCase):
    def test_static_totality_detects_optional_attribute(self):
        issues = Tree.from_dict(STATES).check()
        self.assertEqual([(i.code, i.node) for i in issues], [("non_total", "volume")])

    def test_has_guard_makes_fork_total(self):
        data = copy.deepcopy(STATES)
        data["root"]["false"]["predicate"] = "has(keeps_volume) and keeps_volume"
        self.assertEqual(Tree.from_dict(data).check(), [])

    def test_required_when_makes_fork_total(self):
        data = copy.deepcopy(STATES)
        data["schema"]["attributes"][1]["required_when"] = "not keeps_shape"
        tree = Tree.from_dict(data)
        self.assertEqual(tree.check(), [])
        self.assertEqual(tree.classify({"keeps_shape": False, "mass_g": 1}).status, INVALID)
        self.assertEqual(tree.classify({"keeps_shape": True, "mass_g": 1}).leaf, "solid")

    def test_unknown_names_missing_attribute(self):
        tree = Tree.from_dict(STATES)
        result = tree.classify({"keeps_shape": False, "mass_g": "1/2"})
        self.assertEqual(result.status, UNKNOWN)
        self.assertIsNone(result.leaf)
        self.assertEqual(result.missing, ("keeps_volume",))
        self.assertEqual(result.stopped_at, "volume")

    def test_descriptive_attribute_warning(self):
        data = copy.deepcopy(STATES)
        data["root"]["predicate"] = "mass_g > 10"
        data["schema"]["attributes"][1]["required"] = True
        codes = [i.code for i in Tree.from_dict(data).check()]
        self.assertEqual(codes, ["descriptive_attr"])

    def test_degenerate_branch_on_samples(self):
        data = copy.deepcopy(STATES)
        data["schema"]["attributes"][1]["required"] = True
        tree = Tree.from_dict(data)
        samples = [{"keeps_shape": True, "keeps_volume": True, "mass_g": 1}] * 3
        self.assertEqual({i.code for i in tree.check(samples)}, {"degenerate"})

    def test_hypothesis_on_path_is_not_proven(self):
        data = copy.deepcopy(STATES)
        data["schema"]["attributes"][1]["required"] = True
        data["root"]["false"]["knowledge"] = "hypothesis"
        tree = Tree.from_dict(data)
        result = tree.classify({"keeps_shape": False, "keeps_volume": True, "mass_g": 1})
        self.assertEqual((result.status, result.leaf, result.assumptions), (UNKNOWN, "liquid", ("volume",)))
        self.assertEqual(tree.classify({"keeps_shape": True, "keeps_volume": True, "mass_g": 1}).status, PROVEN)

    def test_malformed_trees_rejected(self):
        broken = copy.deepcopy(STATES)
        del broken["root"]["false"]["false"]
        duplicate = copy.deepcopy(STATES)
        duplicate["root"]["true"]["id"] = "gas"
        bad_predicate = copy.deepcopy(STATES)
        bad_predicate["root"]["predicate"] = "keeps_shape == 3"
        for data in (broken, duplicate, bad_predicate):
            with self.assertRaises(TreeError):
                Tree.from_dict(data)

    def test_certificate_roundtrip_and_tamper(self):
        tree = Tree.from_dict(STATES)
        record = {"keeps_shape": False, "keeps_volume": False, "mass_g": 2}
        cert = tree.classify(record).certificate
        self.assertEqual(tree.verify_certificate(record, cert), (True, "ok"))
        forged = {**cert, "leaf": "liquid"}
        self.assertFalse(tree.verify_certificate(record, forged)[0])
        self.assertFalse(tree.verify_certificate({**record, "keeps_volume": True}, cert)[0])

    def test_tree_hash_is_deterministic(self):
        self.assertEqual(Tree.from_dict(STATES).hash, Tree.from_dict(copy.deepcopy(STATES)).hash)

    def test_separating_fork(self):
        tree = Tree.from_dict(STATES)
        a, b = tree.separating_fork("liquid", "gas")
        self.assertEqual((a.node, a.branch, b.branch), ("volume", True, False))
        self.assertIsNone(tree.separating_fork("volume", "gas"))


class MatterBridgeTests(unittest.TestCase):
    def test_bridge_tree_is_valid(self):
        tree = matter_tree()
        self.assertEqual(tree.check(), [])
        self.assertEqual(tree.nodes["Plasma"].label, "Плазма")

    def test_bridge_matches_legacy_classifier_on_all_trait_combinations(self):
        tree = matter_tree()
        legacy_root = compile_delta_tree(_legacy_tree(tree))
        predicates = {t: (lambda name: lambda obj: obj[name])(t) for t in TRAITS}
        for bits in itertools.product([False, True], repeat=len(TRAITS)):
            record = dict(zip(TRAITS, bits))
            with self.subTest(record=record):
                result = tree.classify(record)
                self.assertEqual(result.status, PROVEN)
                self.assertEqual(result.leaf, classify_delta(legacy_root, record, predicates)[-1])

    def test_bridge_only_requires_traits_on_the_path(self):
        tree = matter_tree()
        self.assertEqual(tree.classify({"continuous": True}).leaf, "Field")
        self.assertEqual(tree.classify({"continuous": False, "grouped": False}).leaf, "FreeParticles")


def _legacy_tree(tree: Tree) -> dict:
    def dump(node):
        if node.is_leaf:
            return {"name": node.id}
        return {"name": node.id, "predicate": node.predicate.text,
                "yes": dump(node.if_true), "no": dump(node.if_false)}
    return dump(tree.root)


if __name__ == "__main__":
    unittest.main()
