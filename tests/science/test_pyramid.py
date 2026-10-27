import itertools
import json
import unittest
from fractions import Fraction
from pathlib import Path

from delta_kernel import Tree
from delta_science.atoms import elements
from delta_science.pyramid import atom_features, matter_tree, tree, unfold_compound, unfold_element

ROOT = Path(__file__).resolve().parents[2]


class CharacteristicTests(unittest.TestCase):
    def test_partition_identity_on_matter_tree(self):
        """Σ χ_лист(x) = 1: каждый объект попадает ровно в один лист, и χ сверяется с проходом."""
        t = matter_tree()
        traits = ["continuous", "grouped", "planetary", "preservesShape", "preservesVolume", "ionized"]
        leaves = [leaf.id for leaf in t.leaves()]
        for bits in itertools.product([False, True], repeat=len(traits)):
            record = dict(zip(traits, bits))
            ones = [leaf for leaf in leaves if t.characteristic(leaf).evaluate(record).value is True]
            self.assertEqual(ones, [t.classify(record).leaf])

    def test_characteristic_of_root_is_true(self):
        self.assertEqual(tree("bonds").characteristic("bond.metallic").text, "true")


class PyramidTests(unittest.TestCase):
    def test_atom_classes_unfold_from_nuclear_charge_without_errors(self):
        atoms = tree("atoms")
        self.assertEqual(atoms.check(), [])
        wrong = [e.symbol for z, e in elements().items()
                 if atoms.nodes[atoms.classify(atom_features(z), certify=False).leaf].meta["class"] != e.group_block]
        self.assertEqual(wrong, [])

    def test_atoms_tree_is_reproducible(self):
        import scripts.build_atoms_tree as build
        from delta_tools.induction import Inductor

        inductor = Inductor(build.ATOM_SCHEMA, build.BUDGET, tree_id="atoms", version="0.1.0")
        inductor.fit(build.examples())
        data = inductor.tree_dict()
        data["root"]["label"] = "Атом"
        self.assertEqual(Tree.from_dict(data).hash, tree("atoms").hash)

    def test_sodium_unfolds_with_exact_numbers(self):
        steps = {s.level: s for s in unfold_element(11)}
        self.assertEqual(steps["particles"].numbers, {"protons": 11, "neutrons": 12, "electrons": 11})
        self.assertEqual(steps["nucleus"].numbers["nucleon_mass_u"], Fraction("23.184020124231"))
        # энергия связи ядра ²³Na: справочное значение 186,56 МэВ
        self.assertEqual(round(steps["atoms"].numbers["binding_mev"], 2), Fraction(18656, 100))
        self.assertEqual(steps["bonds"].node, "MetallicBond")
        self.assertEqual(steps["substances"].node, "MetalSubstance")
        self.assertEqual(steps["states"].node, "Solid")

    def test_every_element_unfolds(self):
        for z in elements():
            with self.subTest(z=z):
                self.assertGreaterEqual(len(unfold_element(z)), 5)

    def test_compound_unfolds(self):
        steps = {s.level: s for s in unfold_compound("[Na+].[Cl-]", "ClNa", "solid")}
        self.assertEqual(steps["particles"].numbers["protons"], 28)
        self.assertEqual(steps["substances"].node, "IonicSubstance")

    def test_page_is_generated_from_kernel_trees(self):
        from delta_app.pyramid_view import build

        data = build()
        self.assertEqual(data["atomsCheck"], "118 из 118")
        self.assertEqual(len(data["objects"]), 200)
        self.assertEqual({l["id"] for l in data["levelsList"]}, {"matter", "particles", "atoms", "bonds", "substances"})
