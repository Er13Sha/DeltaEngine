import json
import unittest
from pathlib import Path

from delta_kernel import Tree
from delta_science.atoms import elements, format_configuration, madelung, structure
from delta_science.bonds import compound_structure, simple_substance
from delta_science.causes import class_record, compound_record, element_record, evaluate, rules
from delta_science.smiles import parse, parse_formula

ROOT = Path(__file__).resolve().parents[2]
TEXTBOOK_ANOMALIES = {"Cr", "Cu", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "La", "Ce", "Gd", "Pt", "Au",
                      "Ac", "Th", "Pa", "U", "Np", "Cm"}


class AtomTests(unittest.TestCase):
    def test_configuration_unfolds_from_nuclear_charge(self):
        """У 99 элементов конфигурация выводится из Z; расходятся только известные «провалы» электрона."""
        anomalies = {elements()[z].symbol for z in elements() if structure(z).anomaly}
        self.assertEqual(anomalies, TEXTBOOK_ANOMALIES)
        self.assertEqual(len(elements()) - len(anomalies), 99)

    def test_sodium(self):
        s = structure(11)
        self.assertEqual((s.protons, s.neutrons, s.electrons), (11, 12, 11))
        self.assertEqual(s.shells, (2, 8, 1))
        self.assertEqual((s.outer_electrons, s.period, s.group, s.block), (1, 3, 1, "s"))
        self.assertEqual(format_configuration(s.configuration), "1s2 2s2 2p6 3s1")

    def test_groups_and_periods(self):
        expected = {"He": (1, 18), "C": (2, 14), "Cl": (3, 17), "Fe": (4, 8), "Cu": (4, 11), "Pd": (5, 10),
                    "Og": (7, 18)}
        by_symbol = {e.symbol: z for z, e in elements().items()}
        for sym, (period, group) in expected.items():
            with self.subTest(sym=sym):
                s = structure(by_symbol[sym])
                self.assertEqual((s.period, s.group), (period, group))

    def test_noble_gases_have_full_outer_shell(self):
        full = {elements()[z].symbol for z in elements() if structure(z).outer_full}
        self.assertEqual(full, {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"})

    def test_madelung_capacity(self):
        self.assertEqual(sum(k for _, _, k in madelung(118)), 118)


class SmilesTests(unittest.TestCase):
    def test_all_compound_structures_match_pubchem_formulas(self):
        data = json.loads((ROOT / "data" / "sources" / "pubchem" / "compound-structures.json").read_text(encoding="utf-8"))
        for row in data["compounds"]:
            with self.subTest(cid=row["CID"]):
                self.assertEqual(parse(row["SMILES"]).formula(), +parse_formula(row["MolecularFormula"]))

    def test_rings_branches_ions(self):
        benzene = parse("C1=CC=CC=C1")
        self.assertEqual(dict(benzene.formula()), {"C": 6, "H": 6})
        salt = parse("[Na+].[Cl-]")
        self.assertEqual(len(salt.components()), 2)
        self.assertEqual([a.charge for a in salt.atoms], [1, -1])


class BondTests(unittest.TestCase):
    def test_bond_tree_is_valid(self):
        tree = Tree.from_dict(json.loads((ROOT / "data" / "trees" / "bonds.json").read_text(encoding="utf-8")))
        self.assertEqual(tree.check(), [])

    def test_simple_substances(self):
        self.assertEqual(simple_substance(11).kind, "metallic")
        self.assertEqual(simple_substance(17).kind, "covalent_nonpolar")
        self.assertIsNone(simple_substance(2))  # гелий: отдельные атомы

    def test_compounds(self):
        kinds = lambda smiles: {(b.a, b.b): b.kind for b in compound_structure(smiles)[1]}
        self.assertEqual(kinds("O"), {("H", "O"): "covalent_polar"})
        self.assertEqual(kinds("[Na+].[Cl-]"), {("Cl⁻", "Na⁺"): "ionic"})
        self.assertEqual(kinds("O=[Mg]"), {("Mg", "O"): "ionic"})  # ΔЭО 2,13 > 1,7
        self.assertEqual(kinds("ClCl"), {("Cl", "Cl"): "covalent_nonpolar"})


class CauseTests(unittest.TestCase):
    def fired(self, sub):
        return {r.id for r in rules() if evaluate(r, sub).value is True}

    def test_rules_compile(self):
        self.assertEqual(len(rules()), 15)

    def test_elements(self):
        self.assertEqual(self.fired(element_record(11)),
                         {"solid.keeps_shape", "metal.conducts", "metal.ductile_luster", "alkali.reactive"})
        self.assertIn("noble.inert", self.fired(element_record(2)))
        self.assertIn("halogen.oxidizer", self.fired(element_record(17)))
        self.assertNotIn("alkali.reactive", self.fired(element_record(1)))  # водород не металл
        self.assertNotIn("alkaline_earth.reactive", self.fired(element_record(26)))  # железо — d-элемент

    def test_compounds_and_classes(self):
        self.assertIn("ionic.conducts_in_melt", self.fired(compound_record("[Na+].[Cl-]", "solid", False)))
        self.assertEqual(self.fired(class_record("matter:Liquid")), {"liquid.keeps_volume"})


if __name__ == "__main__":
    unittest.main()
