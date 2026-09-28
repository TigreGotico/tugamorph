"""The -ia class: nouns that end like a conditional or an imperfect.

The strings -aria/-eria/-ia carry the conditional, the 2nd/3rd conjugation
imperfect, AND a large class of Portuguese nouns. These tests assert the
analysis, never only that an analysis exists.
"""
import unittest

import tugamorph
from tugamorph import PortugueseMorphAnalyzer


def _tense(analysis):
    if analysis.verbal is None:
        return None
    return analysis.verbal.tense_mood


class TestIrregularStemGuard(unittest.TestCase):
    """A stem match is not evidence on its own."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_vida_is_not_a_preterite_of_ver(self):
        # "vi" is the preterite stem of *ver*, but no form of *ver* is "vida".
        self.assertIsNone(_tense(self.a.analyze('vida')))

    def test_other_nouns_on_the_vi_stem(self):
        for word in ('vidro', 'vinho', 'vila'):
            with self.subTest(word=word):
                self.assertIsNone(_tense(self.a.analyze(word)))

    def test_real_irregular_forms_survive_the_guard(self):
        self.assertEqual(_tense(self.a.analyze('fizemos')), 'pret_perf')
        self.assertEqual(_tense(self.a.analyze('tivemos')), 'pret_perf')


class TestConditionalOnAFutureStem(unittest.TestCase):
    """A future stem also builds the conditional."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_faria_is_the_conditional_of_fazer(self):
        r = self.a.analyze('faria')
        self.assertEqual(_tense(r), 'conditional')
        self.assertEqual(r.verbal.lemma_guess, 'fazer')

    def test_diria_is_the_conditional_of_dizer(self):
        r = self.a.analyze('diria')
        self.assertEqual(_tense(r), 'conditional')
        self.assertEqual(r.verbal.lemma_guess, 'dizer')

    def test_the_futures_stay_futures(self):
        for word in ('farei', 'fará', 'farão', 'faremos'):
            with self.subTest(word=word):
                r = self.a.analyze(word)
                self.assertEqual(_tense(r), 'fut_ind')
                self.assertEqual(r.verbal.lemma_guess, 'fazer')

    def test_the_other_conditional_persons(self):
        for word in ('farias', 'fariam', 'faríamos'):
            with self.subTest(word=word):
                self.assertEqual(_tense(self.a.analyze(word)), 'conditional')


class TestConditionalNeedsAnInfinitive(unittest.TestCase):
    """The conditional is INFINITIVE + ia. This needs no lexicon."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_regular_conditionals_are_kept(self):
        r = self.a.analyze('cantaria')
        self.assertEqual(_tense(r), 'conditional')
        self.assertEqual(r.verbal.lemma_guess, 'cantar')

    def test_nouns_whose_stem_is_no_infinitive_lose_the_conditional(self):
        # "aleg", "teo", "sabedo" and "geomet" are not infinitives.
        for word in ('alegria', 'teoria', 'sabedoria'):
            with self.subTest(word=word):
                self.assertNotEqual(_tense(self.a.analyze(word)), 'conditional')


@unittest.skipUnless(tugamorph._HAS_LEXICON, 'tugalex is not installed')
class TestLexiconSettlesTheTie(unittest.TestCase):
    """Shape cannot separate "padaria" from "cantaria"; the lexicon can."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_place_nouns_are_not_conditionals(self):
        for word in ('padaria', 'cafeteria', 'sapataria', 'pastelaria'):
            with self.subTest(word=word):
                self.assertIsNone(_tense(self.a.analyze(word)))

    def test_abstract_ia_nouns_are_not_imperfects(self):
        for word in ('energia', 'ironia', 'agonia', 'cirurgia',
                     'harmonia', 'melodia', 'poesia'):
            with self.subTest(word=word):
                self.assertIsNone(_tense(self.a.analyze(word)))

    def test_real_imperfects_are_kept(self):
        for word, lemma in (('comia', 'comer'), ('bebia', 'beber'),
                            ('partia', 'partir'), ('vendia', 'vender')):
            with self.subTest(word=word):
                self.assertEqual(_tense(self.a.analyze(word)), 'imperf')

    def test_genuinely_ambiguous_words_stay_verbal(self):
        # "livraria" is a bookshop AND the conditional of *livrar*;
        # "bateria" is a battery AND the conditional of *bater*.
        for word in ('livraria', 'bateria'):
            with self.subTest(word=word):
                self.assertEqual(_tense(self.a.analyze(word)), 'conditional')


class TestNoLexiconKeepsTheOldBehaviour(unittest.TestCase):
    """Without a lexicon nothing that used to work may break."""

    def test_imperfects_survive_without_the_lexicon(self):
        from tugamorph import AnalysisConfig
        a = PortugueseMorphAnalyzer(AnalysisConfig(use_lexicon=False))
        for word in ('comia', 'bebia', 'partia', 'vendia'):
            with self.subTest(word=word):
                self.assertIsNotNone(_tense(a.analyze(word)))


if __name__ == '__main__':
    unittest.main()
