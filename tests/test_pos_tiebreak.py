"""Who is allowed to break a suffix/verbal tie.

A caller holds the sentence. `analyze` holds one word. A tag this module
obtains by tagging that bare word has no more context than the built-in
heuristic, and on the forms where the two tables collide it is wrong: with a
tagger installed it calls "cantado" and "aprovado" nominal, and "geometria"
verbal. Trusting it lost the participle to the -ado adjective suffix, the
future to the -ao augmentative, and -metria to a verbal -ria.
"""
import unittest

from tugamorph import PortugueseMorphAnalyzer, is_past_participle


class TestSelfObtainedTagDoesNotTiebreak(unittest.TestCase):
    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_the_participle_survives_a_nominal_tag(self):
        r = self.a.analyze("cantado")
        self.assertIsNotNone(r.verbal, "cantado: no verbal analysis")
        self.assertEqual(r.verbal.tense_mood, "participle")
        self.assertTrue(is_past_participle("cantado"))

    def test_the_future_survives_a_nominal_tag(self):
        r = self.a.analyze("cantarão")
        self.assertIsNotNone(r.verbal, "cantarão: no verbal analysis")
        self.assertEqual(r.verbal.tense_mood, "fut_ind")

    def test_the_long_suffix_survives_a_verbal_tag(self):
        r = self.a.analyze("geometria")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "metria")

    def test_the_tag_is_still_reported(self):
        """Not trusting the tag for tiebreaking is not the same as dropping it."""
        self.assertIsNotNone(self.a.analyze("cantado").pos_tag)


class TestCallerTagStillTiebreaks(unittest.TestCase):
    """A caller with a sentence keeps full control of the tie."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_caller_verb_forces_the_verbal_reading(self):
        r = self.a.analyze("geometria", pos_tag="VERB")
        self.assertIsNotNone(r.verbal)
        self.assertIsNone(r.suffix)

    def test_caller_noun_forces_the_suffix_reading(self):
        r = self.a.analyze("cantado", pos_tag="NOUN")
        self.assertIsNone(r.verbal)
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "ado")

    def test_the_two_disagree_on_the_same_word(self):
        """If they agreed, neither test above would be proving anything."""
        as_verb = self.a.analyze("cantado", pos_tag="VERB")
        as_noun = self.a.analyze("cantado", pos_tag="NOUN")
        self.assertIsNotNone(as_verb.verbal)
        self.assertIsNone(as_noun.verbal)


if __name__ == "__main__":
    unittest.main()
