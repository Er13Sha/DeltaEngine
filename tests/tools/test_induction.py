import copy
import itertools
import json
import unittest
from pathlib import Path

from delta_kernel import PROVEN, UNKNOWN, Tree
from delta_tools.bridge_matter import matter_tree_from_deltas
from delta_tools.induction import Budget, Inductor, predict
from delta_tools.review import set_status

ROOT = Path(__file__).resolve().parents[2]
TRAITS = ["continuous", "grouped", "planetary", "preservesShape", "preservesVolume", "ionized"]
BOOL_SCHEMA = {"name": "m", "version": "1", "attributes": [{"name": t, "type": {"kind": "bool"}} for t in TRAITS]}


def matter_examples():
    src = Tree.from_dict(matter_tree_from_deltas(json.loads((ROOT / "data" / "matter-deltas.json").read_text(encoding="utf-8"))))
    out = []
    for bits in itertools.product([False, True], repeat=len(TRAITS)):
        record = dict(zip(TRAITS, bits))
        out.append((record, src.classify(record).leaf))
    return src, out


class InductionTests(unittest.TestCase):
    def test_e1_recovers_matter_tree_exactly(self):
        src, examples = matter_examples()
        ind = Inductor(BOOL_SCHEMA)
        tree = ind.fit(examples)
        self.assertEqual(tree.check(), [])
        self.assertEqual((ind.report.forks, ind.report.leaves), (6, 7))
        for record, label in examples:
            self.assertEqual(predict(tree, record)[0], label)

        def shape(node):
            return (node.predicate.text, shape(node.if_true), shape(node.if_false)) if not node.is_leaf else node.meta["class"]

        def src_shape(node):
            return (node.predicate.text, src_shape(node.if_true), src_shape(node.if_false)) if not node.is_leaf else node.id

        self.assertEqual(shape(tree.root), src_shape(src.root))

    def test_induced_forks_are_hypotheses_until_accepted(self):
        _, examples = matter_examples()
        ind = Inductor(BOOL_SCHEMA)
        tree = ind.fit(examples)
        record = examples[0][0]
        result = tree.classify(record)
        self.assertEqual(result.status, UNKNOWN)
        self.assertTrue(result.assumptions)
        data = ind.tree_dict()
        for fork_id in result.assumptions:
            set_status(data, fork_id, "accepted", "test")
        self.assertEqual(Tree.from_dict(data).classify(record).status, PROVEN)
        self.assertEqual(Inductor(BOOL_SCHEMA, auto_accept=True).fit(examples).classify(record).status, PROVEN)

    def test_exact_thresholds(self):
        schema = {"name": "n", "version": "1", "attributes": [{"name": "x", "type": {"kind": "rat"}}]}
        tree = Inductor(schema).fit([({"x": "1/3"}, "small"), ({"x": "1/2"}, "big")])
        self.assertEqual(tree.root.predicate.text, "x <= 5/12")

    def test_conflict_gives_unknown_leaf_not_majority(self):
        schema = {"name": "c", "version": "1", "attributes": [{"name": "a", "type": {"kind": "bool"}}]}
        ind = Inductor(schema)
        tree = ind.fit([({"a": True}, "x"), ({"a": True}, "y"), ({"a": True}, "x"), ({"a": False}, "z")])
        label, result = predict(tree, {"a": True})
        self.assertIsNone(label)
        self.assertEqual(tree.nodes[result.leaf].meta["unknown"], "одинаковые записи с разными метками")
        self.assertEqual(ind.report.quarantined[0]["labels"], {"x": 2, "y": 1})

    def test_optional_attribute_is_guarded_by_has(self):
        schema = {"name": "o", "version": "1", "attributes": [
            {"name": "kind", "type": {"kind": "enum", "values": ["a", "b"]}, "required": False}]}
        examples = [({"kind": "a"}, "A"), ({"kind": "b"}, "B"), ({}, "none")]
        tree = Inductor(schema).fit(examples)
        self.assertEqual(tree.check(), [])
        for record, label in examples:
            self.assertEqual(predict(tree, record)[0], label)

    def test_set_attributes(self):
        schema = {"name": "s", "version": "1", "attributes": [
            {"name": "words", "type": {"kind": "set", "element": "str"}}]}
        examples = [({"words": ["чем", "отличаться"]}, "DIFF"), ({"words": ["что", "такой"]}, "DEF"),
                    ({"words": ["разница", "между"]}, "DIFF"), ({"words": ["что", "означать"]}, "DEF")]
        tree = Inductor(schema).fit(examples)
        for record, label in examples:
            self.assertEqual(predict(tree, record)[0], label)

    def test_min_support_turns_thin_leaves_into_unknown(self):
        schema = {"name": "n", "version": "1", "attributes": [{"name": "x", "type": {"kind": "int"}}]}
        examples = [({"x": 1}, "a"), ({"x": 2}, "a"), ({"x": 3}, "b")]
        tree = Inductor(schema, Budget(min_support=2)).fit(examples)
        self.assertEqual(predict(tree, {"x": 1})[0], "a")
        self.assertIsNone(predict(tree, {"x": 3})[0])

    def test_incremental_add_rebuilds_only_conflicting_leaf(self):
        _, examples = matter_examples()
        ind = Inductor(BOOL_SCHEMA)
        before = ind.fit(examples[:40])
        after = before
        for record, label in examples[40:]:
            after = ind.add(record, label)  # внутри проверяется G5
        for record, label in examples:
            self.assertEqual(predict(after, record)[0], label)

    def test_deterministic(self):
        _, examples = matter_examples()
        shuffled = list(reversed(examples))
        a = Inductor(BOOL_SCHEMA).fit(examples)
        b = Inductor(BOOL_SCHEMA).fit(copy.deepcopy(examples))
        self.assertEqual(a.hash, b.hash)
        self.assertEqual(Inductor(BOOL_SCHEMA).fit(shuffled).root.predicate.text, a.root.predicate.text)


class FeaturizeTests(unittest.TestCase):
    def test_boundaries_reveal_repeating_structure(self):
        from delta_tools.featurize import consistency_errors, select_boundaries

        # метки повторяются отрезками разной длины: 1–3, 4–8, 9–15
        pattern = ["a", "b", "c", "a", "x", "y", "b", "c", "a", "x", "y", "x", "y", "b", "c"]
        pairs = list(zip(range(1, 16), pattern))
        bounds = select_boundaries(pairs)
        self.assertTrue(bounds)
        self.assertLess(consistency_errors(pairs, bounds, 1, 15), consistency_errors(pairs, [], 1, 15))
        self.assertLess(consistency_errors(pairs, [3, 8], 1, 15), consistency_errors(pairs, [], 1, 15))

    def test_constructed_features_are_kernel_derived_attributes(self):
        from delta_tools.featurize import segment_schema

        base = {"name": "e", "version": "1", "attributes": [{"name": "z", "type": {"kind": "int"}}]}
        pattern = ["a", "b", "c", "a", "x", "y", "b", "c", "a", "x", "y", "x", "y", "b", "c"]
        examples = [({"z": z}, label) for z, label in zip(range(1, 16), pattern)]
        schema = segment_schema(base, "z", [3, 8], 1, 15)
        ind = Inductor(schema)
        tree = ind.fit(examples)
        self.assertEqual(tree.check(), [])
        self.assertTrue(any(f.predicate.text.startswith("z_off") for f in tree.forks()))
        for record, label in examples:
            self.assertEqual(predict(tree, record)[0], label)
        # запись содержит только z: производные признаки вычисляет ядро
        self.assertEqual(tree.classify({"z": 4}).path[0].predicate.split()[0][:5], "z_off")


class IntentLearningTests(unittest.TestCase):
    def test_learned_trees_are_valid_and_reproducible(self):
        """Деревья в data/trees воспроизводятся обучением с нуля (детерминизм индуктора)."""
        from delta_text import learn_intent as li
        from delta_text.knowledge import default_knowledge

        kb = default_knowledge()
        budget = li.BUDGET
        for schema, corpus, path, tree_id in (
            (li.INTENT_SCHEMA, li.intent_corpus(kb), li.LEARNED, "intent.learned"),
            (li.SENSE_SCHEMA, li.sense_corpus(kb), li.SENSE, "sense.learned"),
        ):
            with self.subTest(tree=tree_id):
                _, tree = li.train(schema, corpus, budget, tree_id)
                self.assertEqual(tree.check(), [])
                saved = Tree.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.assertEqual(tree.hash, saved.hash)


if __name__ == "__main__":
    unittest.main()
