import unittest

from logical_delta_matter.delta_core import classify_delta, compile_delta_tree, validate_delta_tree


class DeltaCoreTests(unittest.TestCase):
    def setUp(self):
        self.predicates = {
            "is_discrete": lambda obj: bool(obj["is_discrete"]),
            "is_grouped": lambda obj: bool(obj["is_grouped"]),
        }
        self.tree = compile_delta_tree(
            {
                "name": "matter",
                "predicate": "is_discrete",
                "yes": {
                    "name": "particles",
                    "predicate": "is_grouped",
                    "yes": {"name": "substance"},
                    "no": {"name": "free_particles"},
                },
                "no": {"name": "field"},
            }
        )

    def test_validate_tree(self):
        validate_delta_tree(self.tree, self.predicates)

    def test_unique_path_to_leaf(self):
        path = classify_delta(self.tree, {"is_discrete": True, "is_grouped": False}, self.predicates)
        self.assertEqual(path, ["matter:+", "particles:-", "free_particles"])

    def test_reject_non_bool_predicate(self):
        bad = {"is_discrete": lambda obj: 1, "is_grouped": lambda obj: True}
        with self.assertRaises(TypeError):
            classify_delta(self.tree, {"is_discrete": True, "is_grouped": True}, bad)


if __name__ == "__main__":
    unittest.main()
