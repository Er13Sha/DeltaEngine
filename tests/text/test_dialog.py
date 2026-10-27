import unittest
from functools import lru_cache

from delta_kernel import PROVEN
from delta_text.dialog import respond
from delta_text.knowledge import HAND_INTENT, KnowledgeBase, default_knowledge
from delta_text.morph import name_pattern
from delta_text.understand import levenshtein, understand


@lru_cache(maxsize=1)
def hand() -> KnowledgeBase:
    """Ручное дерево типа вопроса без дерева смысла: строгие статусы проверяются на нём."""
    return KnowledgeBase(intent_tree=HAND_INTENT, sense_tree=False)


class UnderstandingTests(unittest.TestCase):
    def setUp(self):
        self.kb = hand()

    def test_multiword_names_win_over_single_words(self):
        u = understand("Что такое благородные газы?", self.kb)
        self.assertEqual([m.candidates[0].concept for m in u.mentions], ["elements:element.noble_gas"])

    def test_inflected_forms_and_symbols(self):
        u = understand("Чем натрия отличается от Cl?", self.kb)
        self.assertEqual([m.candidates[0].concept for m in u.mentions], ["element:Na", "element:Cl"])

    def test_typo_is_an_assumption(self):
        u = understand("натри", self.kb)
        [mention] = u.mentions
        self.assertEqual([c.concept for c in mention.candidates], ["element:Na"])
        self.assertIn("опечатка", mention.candidates[0].assumption)

    def test_intent_tree_is_a_kernel_tree(self):
        self.assertEqual(self.kb.intent.check(), [])
        h = understand("Чем жидкость отличается от газа?", self.kb).hypotheses[0]
        self.assertEqual((h.intent, h.assumptions), ("DIFFERENCE", []))
        self.assertEqual(h.intent_certificate["status"], PROVEN)

    def test_levenshtein(self):
        self.assertEqual(levenshtein("натри", "натрий", 2), 1)
        self.assertEqual(levenshtein("кошка", "натрий", 2), 3)


class DialogTests(unittest.TestCase):
    def test_define_element(self):
        r = respond("Что такое натрий?", hand())
        self.assertEqual(r.status, "proven")
        self.assertIn("щелочной металл", r.text)
        self.assertIn("твёрдое вещество", r.text)
        self.assertEqual([c["status"] for c in r.certificates if "status" in c], ["proven"] * 3)

    def test_difference_of_classes_uses_separating_fork(self):
        r = respond("Чем жидкость отличается от газа?", hand())
        self.assertEqual(r.status, "proven")
        self.assertIn("жидкость — сохраняет объём", r.text)
        self.assertIn("газ — не сохраняет объём", r.text)

    def test_is_a(self):
        self.assertTrue(respond("Является ли натрий металлом?", hand()).text.startswith("Да"))
        no = respond("Является ли хлор металлом?", hand())
        self.assertTrue(no.text.startswith("Нет"))
        self.assertIn("галоген", no.text)

    def test_unclear_input_is_answered_with_named_assumptions(self):
        r = respond("натри", hand())
        self.assertEqual(r.status, "assumed")
        self.assertTrue(r.text.startswith("Понял так: что такое «натрий»"))
        self.assertEqual(len(r.assumptions), 2)
        r = respond("жидкость газ", hand())
        self.assertEqual(r.status, "assumed")
        self.assertIn("чем отличаются", r.text)
        r = respond("натрий металл", hand())
        self.assertEqual(r.status, "assumed")
        self.assertIn("Да", r.text)

    def test_unverified_data_is_a_caveat_not_a_proof(self):
        r = respond("Что такое вода?", hand())
        self.assertEqual(r.status, "assumed")
        self.assertIn("Оговорка", r.text)
        r = respond("Что такое оганесон?", hand())
        self.assertEqual(r.status, "assumed")
        self.assertIn("ожидаемое", r.text)

    def test_unknown_topic(self):
        r = respond("Как приготовить борщ?", hand())
        self.assertEqual(r.status, "unknown")
        self.assertIn("«борщ»", r.text)  # известное существительное — не опечатка в «бор»
        self.assertEqual(r.certificates, [])

    def test_not_applicable_task_is_explained(self):
        r = respond("Чем натрий отличается от хлора и аргона?", hand())
        self.assertEqual(r.status, "unknown")
        self.assertIn("ровно два понятия", r.text)

    def test_ambiguous_name_produces_one_clarifying_question(self):
        kb = KnowledgeBase(intent_tree=HAND_INTENT, sense_tree=False)
        kb.patterns.append((name_pattern("натрий"), "element:K"))
        r = respond("Что такое натрий?", kb)
        self.assertEqual(r.status, "clarify")
        self.assertEqual(r.text, "Уточните: «натрий» — это «натрий» или «калий»?")

    def test_answer_mentions_only_known_concepts(self):
        """Ложных фактов нет: каждое понятие в кавычках есть в базе знаний."""
        import re

        kb = hand()
        names = {n for c in kb.concepts.values() for n in c.names} | {c.name for c in kb.concepts.values()}
        for q in ["Что такое галоген?", "Чем металл отличается от неметалла?", "Что такое материя?",
                  "Чем натрий отличается от хлора?", "Является ли хлор металлом?"]:
            for quoted in re.findall(r"«([^»]+)»", respond(q, hand()).text):
                with self.subTest(q=q, quoted=quoted):
                    self.assertIn(quoted.lower(), {n.lower() for n in names})




class LearnedTreesTests(unittest.TestCase):
    """По умолчанию тип вопроса и смысл слова определяют деревья, обученные на настоящих вопросах."""

    def test_other_sense_is_explained(self):
        r = respond("что происходит в германии?")
        self.assertEqual(r.status, "unknown")
        self.assertIn("не о физике или химии", r.text)
        self.assertIn("германий", r.text)

    def test_other_question_type_is_declined(self):
        r = respond("где можно купить воду?")
        self.assertEqual(r.status, "unknown")
        self.assertIn("«где»", r.text)

    def test_learned_intent_is_an_assumption(self):
        r = respond("Что такое натрий?")
        self.assertEqual(r.status, "assumed")
        self.assertIn("щелочной металл", r.text)
        self.assertIn("обученным деревом", r.text)

    def test_falls_back_to_hand_tree_when_learned_tree_does_not_know(self):
        r = respond("жидкость газ")
        self.assertIn("жидкость — сохраняет объём", r.text)
        self.assertIn("тип вопроса угадан", r.text)

    def test_default_knowledge_uses_learned_trees(self):
        kb = default_knowledge()
        self.assertEqual((kb.intent.id, kb.sense.id), ("intent.learned", "sense.learned"))
        self.assertEqual(kb.intent.check(), [])
        self.assertEqual(kb.sense.check(), [])


if __name__ == "__main__":
    unittest.main()


class WhyTests(unittest.TestCase):
    """«Почему» и «как отличить» запрограммированы: правила и объяснение по пути в дереве."""

    def test_rules_tree_is_a_kernel_tree_without_hypotheses(self):
        rules = default_knowledge().rules
        self.assertEqual(rules.check(), [])
        self.assertTrue(all(f.knowledge == "constatation" for f in rules.forks()))

    def test_why_member_shows_fork_conditions_and_data(self):
        r = respond("Почему натрий металл?")
        self.assertEqual(r.status, "proven")
        self.assertIn('group_block = «Alkali metal»', r.text)

    def test_why_with_false_premise(self):
        r = respond("почему хлор металл?")
        self.assertTrue(r.text.startswith("Посылка вопроса неверна."))

    def test_why_property_is_a_definition_not_a_cause(self):
        r = respond("Почему газ не сохраняет объём?")
        self.assertEqual(r.status, "proven")
        self.assertIn("вилкой «не сохраняет объём»", r.text)
        self.assertIn("причинных знаний", r.text)

    def test_why_property_of_instance_through_concept_name(self):
        r = respond("почему натрий твёрдый?")
        self.assertIn("сохраняет форму («твёрдое вещество»)", r.text)

    def test_why_without_known_property_is_declined(self):
        r = respond("почему вода мокрая?")
        self.assertEqual(r.status, "unknown")
        self.assertIn("причинных знаний нет", r.text)

    def test_how_to_distinguish(self):
        r = respond("Как отличить жидкость от газа?")
        self.assertEqual(r.status, "proven")
        self.assertIn("газ — не сохраняет объём", r.text)
