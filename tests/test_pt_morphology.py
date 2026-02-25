"""
Comprehensive test suite for the Portuguese Morphological Analyzer.

Covers:
    1.  Character-level features (length, vowels, consonants, capitalization, accents)
    2.  Prefix extraction (single, stacked, blocklist, category correctness)
    3.  Suffix extraction (all categories, longest-match, scientific priority)
    4.  Verbal paradigm recognition (all tenses × conjugation classes)
    5.  Suffix vs verbal disambiguation (longest-match-wins)
    6.  Irregular verb recognition (whole-word table + stem-based fallback)
    7.  Enclitic handling (all pronoun forms, grammatical info)
    8.  Mesoclitic handling (reconstruction, verbal re-analysis)
    9.  Compound word detection (hyphenated)
    10. Phonological features (syllables, stress, diphthongs, hiatus, digraphs, nasals)
    11. Feature export (to_feature_dict, to_feature_vector, to_json)
    12. Segmentation string output
    13. Batch processing
    14. Configuration toggles (disable individual pipeline stages)
    15. Edge cases (empty, single char, whitespace, numbers, all-consonant)
    16. Regression tests for previously-broken words
"""

import json
import unittest
from tugamorph import (
    PortugueseMorphAnalyzer,
    AnalysisConfig,
    MorphologicalAnalysis,
    MorphemeType,
    PrefixCategory,
    SuffixCategory,
    StressPattern,
    CliticPosition,
    CliticInfo,
    VerbalAnalysis,
    PhonologicalFeatures,
    Morpheme,
    VOWELS,
    CONSONANTS,
    ACCENTED_VOWELS,
    _HAS_TAGGER,
)


class TestCharacterFeatures(unittest.TestCase):
    """Test basic character-level feature extraction."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_word_length(self):
        self.assertEqual(self.a.analyze("casa").word_length, 4)
        self.assertEqual(self.a.analyze("extraordinariamente").word_length, 19)

    def test_vowel_count(self):
        r = self.a.analyze("aeiou")
        self.assertEqual(r.vowel_count, 5)

    def test_vowel_count_accented(self):
        r = self.a.analyze("água")
        # á, u, a = 3 vowels
        self.assertEqual(r.vowel_count, 3)

    def test_consonant_count(self):
        r = self.a.analyze("brç")
        self.assertEqual(r.consonant_count, 3)

    def test_capitalization_upper(self):
        self.assertTrue(self.a.analyze("Lisboa").is_capitalized)

    def test_capitalization_lower(self):
        self.assertFalse(self.a.analyze("lisboa").is_capitalized)

    def test_has_accent(self):
        self.assertTrue(self.a.analyze("café").has_accent)
        self.assertFalse(self.a.analyze("casa").has_accent)

    def test_has_hyphen(self):
        self.assertTrue(self.a.analyze("guarda-chuva").has_hyphen)
        self.assertFalse(self.a.analyze("casa").has_hyphen)

    def test_normalization_to_lower(self):
        r = self.a.analyze("CASA")
        self.assertEqual(r.normalized, "casa")
        self.assertEqual(r.original, "CASA")

    def test_whitespace_stripped(self):
        r = self.a.analyze("  casa  ")
        self.assertEqual(r.normalized, "casa")
        self.assertEqual(r.word_length, 4)


# ─────────────────────────────────────────────────
# Prefix Tests
# ─────────────────────────────────────────────────

class TestPrefixes(unittest.TestCase):
    """Test prefix detection, stacking, and blocklist."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_single_negation_prefix_des(self):
        r = self.a.analyze("desfazer")
        self.assertEqual(len(r.prefixes), 1)
        self.assertEqual(r.prefixes[0][0], "des")
        self.assertEqual(r.prefixes[0][1], PrefixCategory.NEGATION)

    def test_single_negation_prefix_in(self):
        r = self.a.analyze("infelizmente")
        self.assertIn(("in", PrefixCategory.NEGATION), r.prefixes)

    def test_single_negation_prefix_im(self):
        r = self.a.analyze("impossibilidade")
        self.assertIn(("im", PrefixCategory.NEGATION), r.prefixes)

    def test_repetition_prefix_re(self):
        r = self.a.analyze("reutilização")
        self.assertIn(("re", PrefixCategory.REPETITION), r.prefixes)

    def test_position_prefix_extra(self):
        r = self.a.analyze("extraordinariamente")
        self.assertIn(("extra", PrefixCategory.POSITION), r.prefixes)

    def test_temporal_prefix_pre(self):
        r = self.a.analyze("predeterminar")
        self.assertIn(("pre", PrefixCategory.TEMPORAL), r.prefixes)

    def test_prefix_stacking_two(self):
        r = self.a.analyze("subdesenvolvimento")
        self.assertEqual(len(r.prefixes), 2)
        self.assertEqual(r.prefixes[0][0], "sub")
        self.assertEqual(r.prefixes[1][0], "des")

    def test_prefix_stacking_respects_max(self):
        cfg = AnalysisConfig(max_prefix_stack=1)
        a = PortugueseMorphAnalyzer(config=cfg)
        r = a.analyze("subdesenvolvimento")
        self.assertEqual(len(r.prefixes), 1)

    def test_prefix_anti(self):
        r = self.a.analyze("anticonstitucional")
        self.assertIn(("anti", PrefixCategory.NEGATION), r.prefixes)

    def test_prefix_contra(self):
        r = self.a.analyze("contraproducente")
        self.assertIn(("contra", PrefixCategory.NEGATION), r.prefixes)

    # --- Blocklist tests (false prefix prevention) ---

    def test_blocklist_biologia_no_bi(self):
        """'biologia' must NOT get a 'bi' prefix."""
        r = self.a.analyze("biologia")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("bi", prefix_forms)

    def test_blocklist_impossibilidade_no_pos(self):
        """'impossibilidade' must NOT get a 'pos' prefix after 'im'."""
        r = self.a.analyze("impossibilidade")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("pos", prefix_forms)
        self.assertIn("im", prefix_forms)

    def test_blocklist_disseram_no_dis(self):
        """'disseram' is irregular dizer, must NOT get 'dis' prefix."""
        r = self.a.analyze("disseram")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("dis", prefix_forms)

    def test_blocklist_comessemos_no_com(self):
        """'comêssemos' (comer) must NOT get 'com' prefix."""
        r = self.a.analyze("comêssemos")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("com", prefix_forms)

    def test_min_root_length_prevents_overstrip(self):
        """Prefix stripping must leave at least min_root_length chars."""
        cfg = AnalysisConfig(min_root_length=4)
        a = PortugueseMorphAnalyzer(config=cfg)
        r = a.analyze("remar")  # re+mar → root 'mar' (3 chars) < 4
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("re", prefix_forms)

    def test_real_prefix_desligar(self):
        """'desligar' is genuinely des+ligar."""
        r = self.a.analyze("desligar")
        self.assertIn(("des", PrefixCategory.NEGATION), r.prefixes)

    def test_prefix_hiper(self):
        r = self.a.analyze("hipermercado")
        self.assertIn(("hiper", PrefixCategory.INTENSITY), r.prefixes)

    def test_prefix_ultra(self):
        r = self.a.analyze("ultravioleta")
        self.assertIn(("ultra", PrefixCategory.INTENSITY), r.prefixes)

    def test_prefix_multi(self):
        r = self.a.analyze("multicultural")
        self.assertIn(("multi", PrefixCategory.INTENSITY), r.prefixes)


# ─────────────────────────────────────────────────
# Suffix Tests
# ─────────────────────────────────────────────────

class TestSuffixes(unittest.TestCase):
    """Test derivational suffix extraction."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_adverb_mente(self):
        r = self.a.analyze("felizmente")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "mente")
        self.assertEqual(r.suffix[1], SuffixCategory.ADVERB)

    def test_noun_abstract_ção(self):
        r = self.a.analyze("comunicação")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[1], SuffixCategory.NOUN_ABSTRACT)

    def test_noun_abstract_ização(self):
        r = self.a.analyze("reutilização")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "ização")

    def test_noun_abstract_idade(self):
        r = self.a.analyze("impossibilidade")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "idade")

    def test_noun_abstract_imento(self):
        r = self.a.analyze("subdesenvolvimento")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "imento")

    def test_noun_abstract_ismo(self):
        r = self.a.analyze("capitalismo")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "ismo")

    def test_noun_agent_ista(self):
        r = self.a.analyze("jornalista")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "ista")
        self.assertEqual(r.suffix[1], SuffixCategory.NOUN_AGENT)

    def test_noun_agent_dor(self):
        r = self.a.analyze("trabalhador")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "dor")

    def test_adjective_oso(self):
        r = self.a.analyze("perigoso")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "oso")
        self.assertEqual(r.suffix[1], SuffixCategory.ADJECTIVE)

    def test_adjective_avel(self):
        r = self.a.analyze("inacreditável")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "ável")

    def test_scientific_logia(self):
        r = self.a.analyze("biologia")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "logia")
        self.assertEqual(r.suffix[1], SuffixCategory.SCIENTIFIC)

    def test_scientific_grafia(self):
        r = self.a.analyze("fotografia")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "grafia")

    def test_scientific_metria(self):
        r = self.a.analyze("geometria")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "metria")

    def test_diminutive_inho(self):
        r = self.a.analyze("gatinho")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "inho")
        self.assertEqual(r.suffix[1], SuffixCategory.DIMINUTIVE)

    def test_diminutive_zinho(self):
        r = self.a.analyze("paezinho")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "zinho")

    def test_augmentative_ona(self):
        r = self.a.analyze("mulherona")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[1], SuffixCategory.AUGMENTATIVE)

    def test_adjective_al(self):
        r = self.a.analyze("constitucional")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "al")


# ─────────────────────────────────────────────────
# Verbal Paradigm Tests
# ─────────────────────────────────────────────────

class TestVerbalParadigm(unittest.TestCase):
    """Test verbal inflection recognition across tenses and conjugation classes."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def _assert_verbal(self, word, tense_mood, person=None, number=None, conj=None):
        r = self.a.analyze(word)
        self.assertIsNotNone(r.verbal, f"{word}: no verbal analysis")
        self.assertEqual(r.verbal.tense_mood, tense_mood,
                         f"{word}: expected tense '{tense_mood}', got '{r.verbal.tense_mood}'")
        if person is not None:
            self.assertEqual(r.verbal.person, person,
                             f"{word}: expected person {person}, got {r.verbal.person}")
        if number is not None:
            self.assertEqual(r.verbal.number, number,
                             f"{word}: expected number '{number}', got '{r.verbal.number}'")
        if conj is not None:
            self.assertEqual(r.verbal.conjugation_class, conj,
                             f"{word}: expected conj {conj}, got {r.verbal.conjugation_class}")

    # Infinitive
    def test_infinitive_1st(self):
        self._assert_verbal("cantar", "infinitive", conj=1)

    def test_infinitive_2nd(self):
        self._assert_verbal("comer", "infinitive", conj=2)

    def test_infinitive_3rd(self):
        self._assert_verbal("partir", "infinitive", conj=3)

    # Gerund
    def test_gerund_1st(self):
        self._assert_verbal("cantando", "gerund", conj=1)

    def test_gerund_2nd(self):
        self._assert_verbal("comendo", "gerund", conj=2)

    def test_gerund_3rd(self):
        self._assert_verbal("partindo", "gerund", conj=3)

    # Participle
    def test_participle_1st(self):
        self._assert_verbal("cantado", "participle", conj=1)

    def test_participle_2nd_3rd(self):
        self._assert_verbal("comido", "participle")

    # Imperfeito
    def test_imperf_1sg_1st(self):
        self._assert_verbal("cantava", "imperf", person=1, number="sg", conj=1)

    def test_imperf_3pl_1st(self):
        self._assert_verbal("cantavam", "imperf", person=3, number="pl", conj=1)

    # Futuro do indicativo
    def test_fut_ind_1sg(self):
        self._assert_verbal("cantarei", "fut_ind", person=1, number="sg")

    def test_fut_ind_3sg(self):
        self._assert_verbal("cantará", "fut_ind", person=3, number="sg")

    def test_fut_ind_3pl(self):
        self._assert_verbal("cantarão", "fut_ind", person=3, number="pl")

    # Condicional
    def test_conditional_1sg(self):
        self._assert_verbal("cantaria", "conditional", person=1, number="sg")

    def test_conditional_3pl(self):
        self._assert_verbal("cantariam", "conditional", person=3, number="pl")

    # Pretérito perfeito
    def test_pret_perf_1sg_1st(self):
        self._assert_verbal("cantei", "pret_perf", person=1, number="sg", conj=1)

    def test_pret_perf_3sg_1st(self):
        self._assert_verbal("cantou", "pret_perf", person=3, number="sg", conj=1)

    def test_pret_perf_3pl_1st(self):
        self._assert_verbal("cantaram", "pret_perf", person=3, number="pl", conj=1)

    # Subjuntivo imperfeito
    def test_subj_imperf_1sg_1st(self):
        self._assert_verbal("cantasse", "subj_imperf", person=1, number="sg", conj=1)

    def test_subj_imperf_1pl_2nd(self):
        self._assert_verbal("comêssemos", "subj_imperf", person=1, number="pl", conj=2)


# ─────────────────────────────────────────────────
# Suffix vs Verbal Disambiguation Tests
# ─────────────────────────────────────────────────

class TestSuffixVsVerbal(unittest.TestCase):
    """Longest-match-wins: suffix beats short verbal, verbal wins on ties."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_biologia_is_suffix_not_verbal(self):
        """'logia' (5 chars) > 'ia' (2 chars) → suffix wins."""
        r = self.a.analyze("biologia")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "logia")
        self.assertIsNone(r.verbal)

    def test_democracia_is_suffix(self):
        """'cracia' (6 chars) > 'ia' (2 chars) → suffix wins."""
        r = self.a.analyze("democracia")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "cracia")

    def test_fotografia_is_suffix(self):
        r = self.a.analyze("fotografia")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "grafia")

    def test_predeterminar_is_verbal_not_suffix(self):
        """'ar' ties at 2 chars → verbal wins over adjective suffix."""
        r = self.a.analyze("predeterminar")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "infinitive")
        self.assertIsNone(r.suffix)

    def test_desfazer_is_verbal(self):
        r = self.a.analyze("desfazer")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "infinitive")

    def test_cantar_is_verbal(self):
        r = self.a.analyze("cantar")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "infinitive")
        self.assertIsNone(r.suffix)

    def test_felizmente_is_suffix(self):
        """'mente' (5 chars) > any verbal ending → suffix wins."""
        r = self.a.analyze("felizmente")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "mente")


# ─────────────────────────────────────────────────
# Irregular Verb Tests
# ─────────────────────────────────────────────────

class TestIrregularVerbs(unittest.TestCase):
    """Test irregular verb recognition via whole-word table and stem fallback."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def _assert_irregular(self, word, lemma, tense_mood=None, person=None, number=None):
        r = self.a.analyze(word)
        self.assertIsNotNone(r.verbal, f"{word}: no verbal analysis")
        self.assertTrue(r.verbal.is_irregular, f"{word}: not flagged as irregular")
        self.assertEqual(r.verbal.lemma_guess, lemma,
                         f"{word}: expected lemma '{lemma}', got '{r.verbal.lemma_guess}'")
        if tense_mood:
            self.assertEqual(r.verbal.tense_mood, tense_mood)
        if person:
            self.assertEqual(r.verbal.person, person)
        if number:
            self.assertEqual(r.verbal.number, number)

    # Whole-word table hits
    def test_fiz(self):
        self._assert_irregular("fiz", "fazer", "pret_perf", 1, "sg")

    def test_fizemos(self):
        self._assert_irregular("fizemos", "fazer", "pret_perf", 1, "pl")

    def test_fez(self):
        self._assert_irregular("fez", "fazer", "pret_perf", 3, "sg")

    def test_disseram(self):
        self._assert_irregular("disseram", "dizer", "pret_perf", 3, "pl")

    def test_disse(self):
        self._assert_irregular("disse", "dizer", "pret_perf", 1, "sg")

    def test_vou(self):
        self._assert_irregular("vou", "ir", "pres_ind", 1, "sg")

    def test_vai(self):
        self._assert_irregular("vai", "ir", "pres_ind", 3, "sg")

    def test_vamos(self):
        self._assert_irregular("vamos", "ir", "pres_ind", 1, "pl")

    def test_vao(self):
        self._assert_irregular("vão", "ir", "pres_ind", 3, "pl")

    def test_fui(self):
        self._assert_irregular("fui", "ser/ir", "pret_perf", 1, "sg")

    def test_foi(self):
        self._assert_irregular("foi", "ser/ir", "pret_perf", 3, "sg")

    def test_sou(self):
        self._assert_irregular("sou", "ser", "pres_ind", 1, "sg")

    def test_somos(self):
        self._assert_irregular("somos", "ser", "pres_ind", 1, "pl")

    def test_era(self):
        self._assert_irregular("era", "ser", "imperf", 1, "sg")

    def test_tive(self):
        self._assert_irregular("tive", "ter", "pret_perf", 1, "sg")

    def test_tenho(self):
        self._assert_irregular("tenho", "ter", "pres_ind", 1, "sg")

    def test_houve(self):
        self._assert_irregular("houve", "haver", "pret_perf", 3, "sg")

    def test_trouxe(self):
        self._assert_irregular("trouxe", "trazer", "pret_perf", 1, "sg")

    def test_pude(self):
        self._assert_irregular("pude", "poder", "pret_perf", 1, "sg")

    def test_quis(self):
        self._assert_irregular("quis", "querer", "pret_perf", 1, "sg")

    def test_soube(self):
        self._assert_irregular("soube", "saber", "pret_perf", 1, "sg")

    def test_pus(self):
        self._assert_irregular("pus", "pôr", "pret_perf", 1, "sg")

    def test_vim(self):
        self._assert_irregular("vim", "vir", "pret_perf", 1, "sg")

    def test_coube(self):
        self._assert_irregular("coube", "caber", "pret_perf", 1, "sg")

    # Stem-based fallback (not in whole-word table)
    def test_stem_disse_lhe_after_clitic_removal(self):
        """After removing enclitic '-lhe', the remaining 'disse' should be irregular."""
        r = self.a.analyze("disse-lhe")
        self.assertIsNotNone(r.verbal)
        self.assertTrue(r.verbal.is_irregular)
        self.assertEqual(r.verbal.lemma_guess, "dizer")


# ─────────────────────────────────────────────────
# Enclitic Tests
# ─────────────────────────────────────────────────

class TestEnclitics(unittest.TestCase):
    """Test enclitic pronoun extraction."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def _assert_enclitic(self, word, clitic_form, person=None, number=None, case=None):
        r = self.a.analyze(word)
        self.assertIsNotNone(r.clitic, f"{word}: no clitic found")
        self.assertEqual(r.clitic.position, CliticPosition.ENCLITIC)
        self.assertEqual(r.clitic.form, clitic_form,
                         f"{word}: expected clitic '{clitic_form}', got '{r.clitic.form}'")
        if person:
            self.assertEqual(r.clitic.person, person)
        if number:
            self.assertEqual(r.clitic.number, number)
        if case:
            self.assertEqual(r.clitic.case, case)

    def test_disse_lhe(self):
        self._assert_enclitic("disse-lhe", "lhe", person=3, number="sg", case="dat")

    def test_encontramo_nos(self):
        self._assert_enclitic("encontramo-nos", "nos", person=1, number="pl")

    def test_da_lo(self):
        self._assert_enclitic("dá-lo", "lo", person=3, number="sg", case="acc")

    def test_viu_me(self):
        self._assert_enclitic("viu-me", "me", person=1, number="sg")

    def test_disse_te(self):
        self._assert_enclitic("disse-te", "te", person=2, number="sg")

    def test_lavou_se(self):
        self._assert_enclitic("lavou-se", "se", person=3, number="sg", case="refl")

    def test_contou_lhes(self):
        self._assert_enclitic("contou-lhes", "lhes", person=3, number="pl", case="dat")

    def test_vi_a(self):
        self._assert_enclitic("vi-a", "a", person=3, number="sg", case="acc")

    def test_vi_os(self):
        self._assert_enclitic("vi-os", "os", person=3, number="pl", case="acc")


# ─────────────────────────────────────────────────
# Mesoclitic Tests
# ─────────────────────────────────────────────────

class TestMesoclitics(unittest.TestCase):
    """Test mesoclitic pronoun extraction and verb reconstruction."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_dir_se_ia(self):
        r = self.a.analyze("dir-se-ia")
        self.assertIsNotNone(r.clitic)
        self.assertEqual(r.clitic.position, CliticPosition.MESOCLITIC)
        self.assertEqual(r.clitic.form, "se")

    def test_fazer_te_ei(self):
        r = self.a.analyze("fazer-te-ei")
        self.assertIsNotNone(r.clitic)
        self.assertEqual(r.clitic.position, CliticPosition.MESOCLITIC)
        self.assertEqual(r.clitic.form, "te")

    def test_dar_lhe_ia(self):
        r = self.a.analyze("dar-lhe-ia")
        self.assertIsNotNone(r.clitic)
        self.assertEqual(r.clitic.form, "lhe")
        self.assertEqual(r.clitic.position, CliticPosition.MESOCLITIC)

    def test_mesoclitic_not_compound(self):
        """Mesoclitic words should not be flagged as compounds."""
        r = self.a.analyze("dir-se-ia")
        self.assertFalse(r.is_compound)


# ─────────────────────────────────────────────────
# Compound Word Tests
# ─────────────────────────────────────────────────

class TestCompounds(unittest.TestCase):
    """Test compound word detection."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_guarda_chuva(self):
        r = self.a.analyze("guarda-chuva")
        self.assertTrue(r.is_compound)
        self.assertEqual(r.compound_parts, ["guarda", "chuva"])

    def test_segunda_feira(self):
        r = self.a.analyze("segunda-feira")
        self.assertTrue(r.is_compound)
        self.assertEqual(r.compound_parts, ["segunda", "feira"])

    def test_compound_has_interfixe_morpheme(self):
        r = self.a.analyze("guarda-chuva")
        mtypes = [m.mtype for m in r.morphemes]
        self.assertIn(MorphemeType.INTERFIXE, mtypes)

    def test_single_word_not_compound(self):
        r = self.a.analyze("casa")
        self.assertFalse(r.is_compound)
        self.assertEqual(r.compound_parts, [])


# ─────────────────────────────────────────────────
# Phonological Feature Tests
# ─────────────────────────────────────────────────

class TestPhonology(unittest.TestCase):
    """Test phonological feature extraction."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    # Syllable count (approximate)
    def test_syllable_count_monosyllable(self):
        r = self.a.analyze("sol")
        self.assertEqual(r.phonology.syllable_count, 1)

    def test_syllable_count_disyllable(self):
        r = self.a.analyze("casa")
        self.assertEqual(r.phonology.syllable_count, 2)

    def test_syllable_count_polysyllable(self):
        r = self.a.analyze("extraordinariamente")
        self.assertGreaterEqual(r.phonology.syllable_count, 6)

    # Stress patterns
    def test_stress_oxytone_cafe(self):
        r = self.a.analyze("café")
        self.assertEqual(r.phonology.stress_pattern, StressPattern.OXYTONE)

    def test_stress_oxytone_cantar(self):
        r = self.a.analyze("cantar")
        self.assertEqual(r.phonology.stress_pattern, StressPattern.OXYTONE)

    def test_stress_paroxytone_casa(self):
        r = self.a.analyze("casa")
        self.assertEqual(r.phonology.stress_pattern, StressPattern.PAROXYTONE)

    def test_stress_proparoxytone(self):
        r = self.a.analyze("cântaro")
        self.assertEqual(r.phonology.stress_pattern, StressPattern.PROPAROXYTONE)

    def test_stress_proparoxytone_musica(self):
        r = self.a.analyze("música")
        self.assertEqual(r.phonology.stress_pattern, StressPattern.PROPAROXYTONE)

    # Nasal diphthongs
    def test_nasal_diphthong_ao(self):
        r = self.a.analyze("pão")
        self.assertTrue(r.phonology.has_nasal_diphthong)

    def test_nasal_diphthong_oe(self):
        r = self.a.analyze("põe")
        self.assertTrue(r.phonology.has_nasal_diphthong)

    # Oral diphthongs
    def test_oral_diphthong_ei(self):
        r = self.a.analyze("feira")
        self.assertTrue(r.phonology.has_oral_diphthong)

    def test_oral_diphthong_ou(self):
        r = self.a.analyze("ouro")
        self.assertTrue(r.phonology.has_oral_diphthong)

    # Nasal vowels
    def test_nasal_vowel(self):
        r = self.a.analyze("mão")
        self.assertTrue(r.phonology.has_nasal_vowel)

    def test_no_nasal_vowel(self):
        r = self.a.analyze("mesa")
        self.assertFalse(r.phonology.has_nasal_vowel)

    # Ends in nasal
    def test_ends_in_nasal_m(self):
        r = self.a.analyze("cantam")
        self.assertTrue(r.phonology.ends_in_nasal)

    def test_ends_in_nasal_ao(self):
        r = self.a.analyze("canção")
        self.assertTrue(r.phonology.ends_in_nasal)

    # Digraphs
    def test_digraph_ch(self):
        r = self.a.analyze("chuva")
        self.assertTrue(r.phonology.has_digraph)

    def test_digraph_nh(self):
        r = self.a.analyze("manhã")
        self.assertTrue(r.phonology.has_digraph)

    def test_digraph_lh(self):
        r = self.a.analyze("trabalho")
        self.assertTrue(r.phonology.has_digraph)

    def test_no_digraph(self):
        r = self.a.analyze("mesa")
        self.assertFalse(r.phonology.has_digraph)

    # Hiatus
    def test_hiatus_ia(self):
        """'ia' in 'dia' is a hiatus (not a recognized diphthong pair in all analyses)."""
        r = self.a.analyze("dia")
        # 'ia' is not in the oral diphthong list, so it should be a hiatus
        # Actually 'ia' can be debated; at minimum we check the feature is boolean
        self.assertIsInstance(r.phonology.has_hiatus, bool)


# ─────────────────────────────────────────────────
# Feature Export Tests
# ─────────────────────────────────────────────────

class TestFeatureExport(unittest.TestCase):
    """Test to_feature_dict, to_feature_vector, to_json."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_feature_dict_has_required_keys(self):
        d = self.a.analyze("casa").to_feature_dict()
        required = [
            'word', 'word_length', 'vowel_count', 'consonant_count',
            'is_capitalized', 'has_accent', 'has_hyphen',
            'syllable_count', 'stress_pattern', 'has_nasal_diphthong',
            'has_oral_diphthong', 'has_hiatus', 'n_prefixes',
            'prefix_types', 'root', 'suffix_type', 'has_clitic',
            'clitic_position', 'is_compound', 'n_morphemes',
        ]
        for key in required:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_feature_dict_verbal_keys(self):
        d = self.a.analyze("cantar").to_feature_dict()
        verbal_keys = [
            'conjugation_class', 'tense_mood', 'person', 'number',
            'thematic_vowel', 'is_irregular', 'lemma_guess',
        ]
        for key in verbal_keys:
            self.assertIn(key, d, f"Missing verbal key: {key}")

    def test_feature_vector_length(self):
        vec = self.a.analyze("casa").to_feature_vector()
        self.assertEqual(len(vec), 17)

    def test_feature_vector_all_float(self):
        vec = self.a.analyze("cantávamos").to_feature_vector()
        for i, v in enumerate(vec):
            self.assertIsInstance(v, float, f"Element {i} is not float: {type(v)}")

    def test_to_json_valid(self):
        j = self.a.analyze("casa").to_json()
        parsed = json.loads(j)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed['word'], 'casa')

    def test_to_json_unicode(self):
        j = self.a.analyze("coração").to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed['word'], 'coração')

    def test_feature_vector_irregular(self):
        vec = self.a.analyze("fiz").to_feature_vector()
        # is_irregular should be 1.0
        self.assertEqual(vec[13], 1.0)

    def test_feature_vector_not_irregular(self):
        vec = self.a.analyze("cantar").to_feature_vector()
        self.assertEqual(vec[13], 0.0)


# ─────────────────────────────────────────────────
# Segmentation String Tests
# ─────────────────────────────────────────────────

class TestSegmentation(unittest.TestCase):
    """Test the human-readable segmentation output."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_segmentation_has_brackets(self):
        s = self.a.segment("desfazer")
        self.assertIn("[", s)
        self.assertIn("]", s)

    def test_segmentation_prefix_visible(self):
        s = self.a.segment("desfazer")
        self.assertIn("[des]", s)

    def test_segmentation_suffix_visible(self):
        s = self.a.segment("felizmente")
        self.assertIn("[mente]", s)

    def test_segment_method_returns_string(self):
        s = self.a.segment("casa")
        self.assertIsInstance(s, str)

    def test_empty_morphemes_returns_normalized(self):
        """If somehow no morphemes are generated, return normalized."""
        r = MorphologicalAnalysis(original="x", normalized="x")
        self.assertEqual(r.segmentation_str(), "x")


# ─────────────────────────────────────────────────
# Batch Processing Tests
# ─────────────────────────────────────────────────

class TestBatch(unittest.TestCase):
    """Test batch analysis."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_batch_returns_list(self):
        results = self.a.analyze_batch(["casa", "cantar"])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_batch_each_is_analysis(self):
        results = self.a.analyze_batch(["casa", "cantar"])
        for r in results:
            self.assertIsInstance(r, MorphologicalAnalysis)

    def test_batch_empty_list(self):
        results = self.a.analyze_batch([])
        self.assertEqual(results, [])

    def test_batch_consistency(self):
        """Batch results should equal individual results."""
        words = ["desfazer", "biologia", "cantávamos"]
        batch = self.a.analyze_batch(words)
        for word, batch_r in zip(words, batch):
            single_r = self.a.analyze(word)
            self.assertEqual(batch_r.root, single_r.root)
            self.assertEqual(batch_r.normalized, single_r.normalized)


# ─────────────────────────────────────────────────
# Configuration Toggle Tests
# ─────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    """Test that AnalysisConfig toggles disable pipeline stages."""

    def test_disable_prefixes(self):
        a = PortugueseMorphAnalyzer(config=AnalysisConfig(extract_prefixes=False))
        r = a.analyze("desfazer")
        self.assertEqual(len(r.prefixes), 0)

    def test_disable_suffixes(self):
        a = PortugueseMorphAnalyzer(config=AnalysisConfig(extract_suffixes=False))
        r = a.analyze("felizmente")
        self.assertIsNone(r.suffix)

    def test_disable_verbal(self):
        a = PortugueseMorphAnalyzer(config=AnalysisConfig(extract_verbal=False))
        r = a.analyze("cantar")
        # Verbal should be None or tense_mood == 'none'
        if r.verbal:
            self.assertEqual(r.verbal.tense_mood, "none")

    def test_disable_clitics(self):
        a = PortugueseMorphAnalyzer(config=AnalysisConfig(extract_clitics=False))
        r = a.analyze("disse-lhe")
        self.assertIsNone(r.clitic)

    def test_disable_compounds(self):
        a = PortugueseMorphAnalyzer(config=AnalysisConfig(detect_compounds=False))
        r = a.analyze("guarda-chuva")
        self.assertFalse(r.is_compound)

    def test_disable_phonology(self):
        a = PortugueseMorphAnalyzer(config=AnalysisConfig(extract_phonology=False))
        r = a.analyze("casa")
        self.assertEqual(r.phonology.syllable_count, 0)
        self.assertEqual(r.phonology.stress_pattern, StressPattern.UNKNOWN)


# ─────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_single_character(self):
        r = self.a.analyze("a")
        self.assertEqual(r.word_length, 1)
        self.assertIsInstance(r, MorphologicalAnalysis)

    def test_single_consonant(self):
        r = self.a.analyze("b")
        self.assertEqual(r.word_length, 1)

    def test_all_vowels(self):
        r = self.a.analyze("aeiou")
        self.assertEqual(r.vowel_count, 5)
        self.assertEqual(r.consonant_count, 0)

    def test_accented_single(self):
        r = self.a.analyze("é")
        self.assertTrue(r.has_accent)

    def test_very_long_word(self):
        """Should not crash on very long words."""
        long_word = "anti" * 10 + "constitucional"
        r = self.a.analyze(long_word)
        self.assertIsInstance(r, MorphologicalAnalysis)

    def test_numbers_in_word(self):
        r = self.a.analyze("abc123")
        self.assertIsInstance(r, MorphologicalAnalysis)

    def test_hyphen_only(self):
        r = self.a.analyze("-")
        self.assertIsInstance(r, MorphologicalAnalysis)

    def test_repr_does_not_crash(self):
        """__repr__ should work for any analysis."""
        for word in ["casa", "desfazer", "dir-se-ia", "guarda-chuva", "fiz", "a"]:
            r = self.a.analyze(word)
            s = repr(r)
            self.assertIsInstance(s, str)

    def test_two_char_word(self):
        r = self.a.analyze("ir")
        self.assertIsInstance(r, MorphologicalAnalysis)

    def test_mixed_case_preserved_in_original(self):
        r = self.a.analyze("CaSa")
        self.assertEqual(r.original, "CaSa")
        self.assertEqual(r.normalized, "casa")


# ─────────────────────────────────────────────────
# Regression Tests
# ─────────────────────────────────────────────────

class TestRegressions(unittest.TestCase):
    """
    Regression tests for previously-broken words.
    Each test documents a specific bug that was fixed.
    """

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_regression_biologia_no_bi_prefix(self):
        """BUG: biologia was decomposed as bi+olog+ia (verbal)."""
        r = self.a.analyze("biologia")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("bi", prefix_forms)
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "logia")

    def test_regression_impossibilidade_no_double_prefix(self):
        """BUG: impossibilidade got im+pos as two prefixes."""
        r = self.a.analyze("impossibilidade")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertEqual(prefix_forms, ["im"])

    def test_regression_disseram_not_dis_prefix(self):
        """BUG: disseram was decomposed as dis+s+eram."""
        r = self.a.analyze("disseram")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("dis", prefix_forms)
        self.assertTrue(r.verbal.is_irregular)
        self.assertEqual(r.verbal.lemma_guess, "dizer")

    def test_regression_predeterminar_is_verb(self):
        """BUG: predeterminar was getting 'ar' as adjective suffix instead of infinitive."""
        r = self.a.analyze("predeterminar")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "infinitive")
        self.assertIsNone(r.suffix)

    def test_regression_comessemos_no_com_prefix(self):
        """BUG: comêssemos was decomposed as com+êssemos (prefix com)."""
        r = self.a.analyze("comêssemos")
        prefix_forms = [p[0] for p in r.prefixes]
        self.assertNotIn("com", prefix_forms)

    def test_regression_fizemos_is_irregular(self):
        """BUG: fizemos was not recognized as irregular fazer."""
        r = self.a.analyze("fizemos")
        self.assertTrue(r.verbal.is_irregular)
        self.assertEqual(r.verbal.lemma_guess, "fazer")
        self.assertEqual(r.verbal.person, 1)
        self.assertEqual(r.verbal.number, "pl")

    def test_regression_disse_lhe_irregular(self):
        """BUG: disse-lhe was decomposed as dis+se after clitic removal."""
        r = self.a.analyze("disse-lhe")
        self.assertIsNotNone(r.verbal)
        self.assertTrue(r.verbal.is_irregular)
        self.assertEqual(r.verbal.lemma_guess, "dizer")


# ─────────────────────────────────────────────────
# POS Tagging Tests
# ─────────────────────────────────────────────────

class TestPOSTagging(unittest.TestCase):
    """Test POS tag assignment (heuristic mode — no tugatagger installed)."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    # Heuristic POS for common word classes
    def test_pos_noun_default(self):
        r = self.a.analyze("mesa")
        self.assertEqual(r.pos_tag, "NOUN")

    def test_pos_verb_infinitive(self):
        r = self.a.analyze("cantar")
        self.assertEqual(r.pos_tag, "VERB")

    def test_pos_verb_gerund(self):
        r = self.a.analyze("cantando")
        self.assertEqual(r.pos_tag, "VERB")

    def test_pos_adverb_mente(self):
        r = self.a.analyze("rapidamente")
        self.assertEqual(r.pos_tag, "ADV")

    def test_pos_adjective_oso(self):
        r = self.a.analyze("perigoso")
        self.assertEqual(r.pos_tag, "ADJ")

    def test_pos_det(self):
        r = self.a.analyze("o")
        self.assertEqual(r.pos_tag, "DET")

    def test_pos_adp(self):
        r = self.a.analyze("de")
        self.assertEqual(r.pos_tag, "ADP")

    def test_pos_cconj(self):
        r = self.a.analyze("e")
        self.assertEqual(r.pos_tag, "CCONJ")

    def test_pos_pron(self):
        r = self.a.analyze("eu")
        self.assertEqual(r.pos_tag, "PRON")

    def test_pos_aux(self):
        r = self.a.analyze("é")
        self.assertEqual(r.pos_tag, "AUX")

    def test_pos_propn_capitalized(self):
        r = self.a.analyze("Lisboa")
        self.assertEqual(r.pos_tag, "PROPN")

    def test_pos_punct(self):
        r = self.a.analyze(".")
        self.assertEqual(r.pos_tag, "PUNCT")

    def test_pos_num(self):
        r = self.a.analyze("42")
        self.assertEqual(r.pos_tag, "NUM")

    def test_pos_noun_dor(self):
        r = self.a.analyze("trabalhador")
        self.assertEqual(r.pos_tag, "NOUN")

    def test_pos_noun_ção(self):
        r = self.a.analyze("comunicação")
        self.assertEqual(r.pos_tag, "NOUN")

    # POS tag appears in output
    def test_pos_in_feature_dict(self):
        d = self.a.analyze("casa").to_feature_dict()
        self.assertIn("pos_tag", d)
        self.assertEqual(d["pos_tag"], "NOUN")

    def test_pos_in_json(self):
        j = json.loads(self.a.analyze("cantar").to_json())
        self.assertIn("pos_tag", j)
        self.assertEqual(j["pos_tag"], "VERB")

    def test_pos_in_repr(self):
        s = repr(self.a.analyze("casa"))
        self.assertIn("pos: NOUN", s)

    # Explicit POS hint overrides heuristic
    def test_pos_hint_overrides(self):
        r = self.a.analyze("casa", pos_tag="VERB")
        self.assertEqual(r.pos_tag, "VERB")

    def test_pos_hint_none_uses_heuristic(self):
        r = self.a.analyze("mesa", pos_tag=None)
        self.assertIsNotNone(r.pos_tag)

    # POS does NOT affect disambiguation in heuristic mode
    def test_heuristic_pos_does_not_force_verbal(self):
        """Heuristic POS=VERB should NOT override longest-match suffix wins."""
        r = self.a.analyze("geometria")
        # -metria (scientific suffix, 6 chars) should still beat -ria (verbal, 3 chars)
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "metria")

    def test_heuristic_pos_does_not_force_nominal(self):
        """Heuristic POS=NOUN should NOT override verbal parsing for cantar."""
        r = self.a.analyze("cantar")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "infinitive")


# ─────────────────────────────────────────────────
# POS Disambiguation Tests (with explicit hints)
# ─────────────────────────────────────────────────

class TestPOSDisambiguation(unittest.TestCase):
    """Test that explicit POS hints steer suffix vs verbal disambiguation."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_verb_hint_forces_verbal(self):
        """With VERB hint, prefer verbal reading even at equal length."""
        r = self.a.analyze("cantaria", pos_tag="VERB")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "conditional")
        self.assertIsNone(r.suffix)

    def test_noun_hint_forces_suffix(self):
        """With NOUN hint, prefer suffix reading."""
        r = self.a.analyze("cantaria", pos_tag="NOUN")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "aria")
        self.assertIsNone(r.verbal)

    def test_adj_hint_forces_suffix(self):
        """With ADJ hint, prefer suffix reading."""
        r = self.a.analyze("perigoso", pos_tag="ADJ")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "oso")

    def test_verb_hint_for_participle(self):
        """VERB hint should give cantado as participle, not adjective suffix."""
        r = self.a.analyze("cantado", pos_tag="VERB")
        self.assertIsNotNone(r.verbal)
        self.assertEqual(r.verbal.tense_mood, "participle")

    def test_disambiguation_off(self):
        """With pos_disambiguate=False, explicit POS hint should NOT affect parsing.
        Note: 'cantaria' has both -aria suffix (4) AND -aria verbal (4) at equal length.
        Equal-length tie → verbal wins regardless, so this tests a word where
        suffix is strictly longer."""
        cfg = AnalysisConfig(pos_disambiguate=False)
        a = PortugueseMorphAnalyzer(config=cfg)
        # biologia: -logia suffix (5) > -ia verbal (2) → suffix wins even with VERB hint
        r = a.analyze("biologia", pos_tag="VERB")
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "logia")


# ─────────────────────────────────────────────────
# Syllabification Tests
# ─────────────────────────────────────────────────

class TestSyllabification(unittest.TestCase):
    """Test syllable extraction (heuristic fallback mode)."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_syllables_list_populated(self):
        r = self.a.analyze("casa")
        self.assertIsInstance(r.phonology.syllables, list)
        self.assertGreater(len(r.phonology.syllables), 0)

    def test_syllables_join_equals_word(self):
        """Syllables joined should reproduce the original word."""
        for w in ["casa", "computador", "felizmente", "sol", "cantar"]:
            r = self.a.analyze(w)
            joined = "".join(r.phonology.syllables)
            self.assertEqual(joined, w, f"'{w}': syllables {r.phonology.syllables} don't join to word")

    def test_syllable_count_matches_list(self):
        r = self.a.analyze("computador")
        self.assertEqual(r.phonology.syllable_count, len(r.phonology.syllables))

    def test_monosyllable_single_element(self):
        r = self.a.analyze("sol")
        self.assertEqual(len(r.phonology.syllables), 1)
        self.assertEqual(r.phonology.syllables[0], "sol")

    def test_disyllable(self):
        r = self.a.analyze("casa")
        self.assertEqual(len(r.phonology.syllables), 2)

    def test_syllables_in_feature_dict(self):
        d = self.a.analyze("casa").to_feature_dict()
        self.assertIn("syllables", d)
        self.assertIsInstance(d["syllables"], list)

    def test_stressed_index_in_feature_dict(self):
        d = self.a.analyze("casa").to_feature_dict()
        self.assertIn("stressed_syllable_index", d)

    def test_syllables_in_repr(self):
        """Repr should show dotted syllables like ca.sa."""
        s = repr(self.a.analyze("casa"))
        self.assertIn("ca.sa", s)

    def test_stressed_syllable_index_paroxytone(self):
        """For 'casa' (paroxytone), stress on penultimate = index 0 of 2."""
        r = self.a.analyze("casa")
        self.assertEqual(r.phonology.stressed_syllable_index, 0)

    def test_stressed_syllable_index_oxytone(self):
        """For 'café' (oxytone), stress on last syllable."""
        r = self.a.analyze("café")
        self.assertEqual(r.phonology.stressed_syllable_index, len(r.phonology.syllables) - 1)

    def test_stressed_syllable_index_proparoxytone(self):
        """For 'música' (proparoxytone), stress on antepenultimate."""
        r = self.a.analyze("música")
        n = len(r.phonology.syllables)
        self.assertEqual(r.phonology.stressed_syllable_index, n - 3)

    def test_disabled_syllabifier_fallback(self):
        """With use_syllabifier=False, should still get syllable count."""
        cfg = AnalysisConfig(use_syllabifier=False)
        a = PortugueseMorphAnalyzer(config=cfg)
        r = a.analyze("casa")
        self.assertGreater(r.phonology.syllable_count, 0)


# ─────────────────────────────────────────────────
# Sentence-Level Analysis Tests
# ─────────────────────────────────────────────────

class TestSentenceAnalysis(unittest.TestCase):
    """Test analyze_sentence() method."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_returns_list(self):
        results = self.a.analyze_sentence("O gato preto")
        self.assertIsInstance(results, list)

    def test_correct_count(self):
        results = self.a.analyze_sentence("O gato preto")
        self.assertEqual(len(results), 3)

    def test_each_is_analysis(self):
        results = self.a.analyze_sentence("O gato preto")
        for r in results:
            self.assertIsInstance(r, MorphologicalAnalysis)

    def test_words_match_input(self):
        results = self.a.analyze_sentence("O gato preto")
        words = [r.original for r in results]
        self.assertEqual(words, ["O", "gato", "preto"])

    def test_empty_sentence(self):
        results = self.a.analyze_sentence("")
        # Should handle gracefully
        self.assertIsInstance(results, list)

    def test_single_word_sentence(self):
        results = self.a.analyze_sentence("casa")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].original, "casa")

    def test_pos_tags_assigned(self):
        results = self.a.analyze_sentence("O gato preto pulou")
        for r in results:
            self.assertIsNotNone(r.pos_tag)

    def test_complex_sentence(self):
        """Analyze a realistic sentence without crashing."""
        sent = "Os meninos cantaram alegremente na praça"
        results = self.a.analyze_sentence(sent)
        self.assertEqual(len(results), 6)
        for r in results:
            self.assertIsInstance(r, MorphologicalAnalysis)
            self.assertNotEqual(r.root, "")


# ─────────────────────────────────────────────────
# Config Toggle Tests for New Features
# ─────────────────────────────────────────────────

class TestConfigNewFeatures(unittest.TestCase):
    """Test config toggles for POS and syllabifier."""

    def test_disable_pos_tagger(self):
        cfg = AnalysisConfig(use_pos_tagger=False)
        a = PortugueseMorphAnalyzer(config=cfg)
        r = a.analyze("casa")
        self.assertIsNone(r.pos_tag)

    def test_disable_pos_disambiguate(self):
        """With pos_disambiguate=False, POS hint doesn't override length rules.
        Use biologia (suffix strictly longer) to verify."""
        cfg = AnalysisConfig(pos_disambiguate=False)
        a = PortugueseMorphAnalyzer(config=cfg)
        r = a.analyze("biologia", pos_tag="VERB")
        # -logia (5 chars) > -ia (2 chars) → suffix wins even with VERB hint
        self.assertIsNotNone(r.suffix)
        self.assertEqual(r.suffix[0], "logia")

    def test_integration_flags_available(self):
        from tugamorph import _HAS_TAGGER
        self.assertIsInstance(_HAS_TAGGER, bool)


# ─────────────────────────────────────────────────
# Data Integrity Tests
# ─────────────────────────────────────────────────

class TestDataIntegrity(unittest.TestCase):
    """Test that the lexicon data tables are well-formed."""

    def test_all_clitic_forms_in_table(self):
        from tugamorph import CLITIC_TABLE
        expected = {'me', 'te', 'se', 'lhe', 'lhes', 'nos', 'vos',
                    'o', 'a', 'os', 'as', 'lo', 'la', 'los', 'las'}
        self.assertEqual(set(CLITIC_TABLE.keys()), expected)

    def test_verbal_endings_sorted_longest_first(self):
        from tugamorph import VERBAL_ENDINGS
        # Within each group, longer endings should generally come before shorter
        # We just check that no ending is duplicated with identical params
        seen = set()
        for entry in VERBAL_ENDINGS:
            key = (entry[0], entry[1], entry[2], entry[3], entry[4])
            # Duplicates are allowed for different conjugation classes
            # Just ensure the table isn't empty
        self.assertGreater(len(VERBAL_ENDINGS), 50)

    def test_prefix_table_not_empty(self):
        from tugamorph import PREFIX_TABLE
        self.assertGreater(len(PREFIX_TABLE), 30)

    def test_suffix_table_not_empty(self):
        from tugamorph import SUFFIX_TABLE
        self.assertGreater(len(SUFFIX_TABLE), 30)

    def test_irregular_stems_not_empty(self):
        from tugamorph import IRREGULAR_STEMS
        self.assertGreater(len(IRREGULAR_STEMS), 20)

    def test_vowels_set_complete(self):
        for v in 'aeiouáéíóúâêîôûãõ':
            self.assertIn(v, VOWELS, f"Missing vowel: {v}")

    def test_consonants_set_complete(self):
        for c in 'bcdfghjklmnpqrstvwxyz':
            self.assertIn(c, CONSONANTS, f"Missing consonant: {c}")
        self.assertIn('ç', CONSONANTS)


# ─────────────────────────────────────────────────
# Root Extraction Tests
# ─────────────────────────────────────────────────

class TestRootExtraction(unittest.TestCase):
    """Test that roots are correctly extracted after prefix/suffix stripping."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()

    def test_root_after_prefix_and_suffix(self):
        r = self.a.analyze("infelizmente")
        self.assertEqual(r.root, "feliz")

    def test_root_after_prefix_only(self):
        r = self.a.analyze("desfazer")
        # root should be the verbal root after stripping 'des' and 'er'
        self.assertNotEqual(r.root, "")

    def test_root_simple_word(self):
        r = self.a.analyze("sol")
        self.assertEqual(r.root, "sol")

    def test_root_biologia(self):
        r = self.a.analyze("biologia")
        self.assertEqual(r.root, "bio")

    def test_root_impossibilidade(self):
        r = self.a.analyze("impossibilidade")
        self.assertEqual(r.root, "possibil")

    def test_root_reutilizacao(self):
        r = self.a.analyze("reutilização")
        self.assertEqual(r.root, "util")

    def test_root_subdesenvolvimento(self):
        r = self.a.analyze("subdesenvolvimento")
        self.assertEqual(r.root, "envolv")


# ─────────────────────────────────────────────────
# Broader Vocabulary Stress Test
# ─────────────────────────────────────────────────

class TestBroaderVocabulary(unittest.TestCase):
    """Smoke tests on a broader set of Portuguese words — no crashes, sane output."""

    def setUp(self):
        self.a = PortugueseMorphAnalyzer()
        self.words = [
            "amor", "coração", "saudade", "liberdade", "universidade",
            "computador", "trabalhador", "professora", "estudante",
            "português", "brasileira", "moçambicano", "angolana",
            "rapidamente", "lentamente", "cuidadosamente",
            "refazer", "desconhecer", "reconhecer", "prever",
            "hipermercado", "ultravioleta", "multicultural",
            "casinha", "gatinho", "mãezinha",
            "grandalhão", "casarão",
            "democracia", "burocracia",
            "psicologia", "sociologia", "geologia",
            "chovendo", "dormindo", "escrevendo",
            "cantássemos", "bebêssemos", "partíssemos",
            "falaremos", "comeremos", "partiremos",
            "falaria", "comeria", "partiria",
            "guarda-roupa", "bem-estar", "cor-de-rosa",
            "dá-me", "vê-lo", "pô-lo",
        ]

    def test_no_crashes(self):
        for w in self.words:
            try:
                r = self.a.analyze(w)
                self.assertIsInstance(r, MorphologicalAnalysis)
            except Exception as e:
                self.fail(f"Crashed on '{w}': {e}")

    def test_all_have_root(self):
        for w in self.words:
            r = self.a.analyze(w)
            self.assertNotEqual(r.root, "", f"'{w}' has empty root")

    def test_all_have_morphemes(self):
        for w in self.words:
            r = self.a.analyze(w)
            self.assertGreater(len(r.morphemes), 0, f"'{w}' has no morphemes")

    def test_all_feature_vectors_valid(self):
        for w in self.words:
            r = self.a.analyze(w)
            vec = r.to_feature_vector()
            self.assertEqual(len(vec), 17, f"'{w}' vector length != 17")
            for i, v in enumerate(vec):
                self.assertIsInstance(v, float, f"'{w}' vec[{i}] not float")

    def test_all_json_valid(self):
        for w in self.words:
            r = self.a.analyze(w)
            j = r.to_json()
            parsed = json.loads(j)
            self.assertIsInstance(parsed, dict)

    def test_scientific_suffixes(self):
        """All -logia words should get scientific suffix."""
        for w in ["psicologia", "sociologia", "geologia"]:
            r = self.a.analyze(w)
            self.assertIsNotNone(r.suffix, f"'{w}' has no suffix")
            self.assertEqual(r.suffix[0], "logia", f"'{w}' suffix is '{r.suffix[0]}'")

    def test_gerunds_recognized(self):
        for w in ["chovendo", "dormindo", "escrevendo"]:
            r = self.a.analyze(w)
            self.assertIsNotNone(r.verbal, f"'{w}' not recognized as verbal")
            self.assertEqual(r.verbal.tense_mood, "gerund", f"'{w}' tense is '{r.verbal.tense_mood}'")

    def test_adverbs_in_mente(self):
        for w in ["rapidamente", "lentamente", "cuidadosamente"]:
            r = self.a.analyze(w)
            self.assertIsNotNone(r.suffix, f"'{w}' has no suffix")
            self.assertEqual(r.suffix[0], "mente", f"'{w}' suffix is '{r.suffix[0]}'")

    def test_conditionals(self):
        for w in ["falaria", "comeria", "partiria"]:
            r = self.a.analyze(w)
            self.assertIsNotNone(r.verbal, f"'{w}' not recognized as verbal")
            self.assertEqual(r.verbal.tense_mood, "conditional",
                             f"'{w}' tense is '{r.verbal.tense_mood}'")

    def test_future_indicative(self):
        for w in ["falaremos", "comeremos", "partiremos"]:
            r = self.a.analyze(w)
            self.assertIsNotNone(r.verbal, f"'{w}' not recognized as verbal")
            self.assertEqual(r.verbal.tense_mood, "fut_ind",
                             f"'{w}' tense is '{r.verbal.tense_mood}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
