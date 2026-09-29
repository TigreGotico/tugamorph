"""An irregular stem fires only before an ending of its own paradigm.

The stem table maps a beginning, not a word: ``vi`` opens ``viu`` and
``vida`` alike. Before the ending table, every word that began with a stem
and was short enough was read as that verb, so ``soma`` was ser, ``inveja``
was ver and ``decidido`` was dar. The stem now needs the rest of the word to
be an ending that verb takes.
"""
import pytest

from tugamorph import PortugueseMorphAnalyzer, IRREGULAR_STEMS, IRREGULAR_STEM_ENDINGS


@pytest.fixture(scope="module")
def analyzer():
    return PortugueseMorphAnalyzer()


NOT_A_VERB = [
    "decidido",   # read as dar, through the stem "de"
    "debatido",   # read as dar
    "detida",     # read as dar
    "vida",       # read as ver, through the stem "vi"
    "soma",       # read as ser, through the stem "som"
    "abreviado",  # read as ver, after "ab" and "re" were stripped
    "direita",    # read as dizer, through the stem "dir"
    "conquista",  # read as querer, through the stem "quis"
    "aperitivo",  # read as ter, through the stem "tiv"
    "dissabor",   # read as dizer, through the stem "diss"
    "afogado",    # read as ser/ir, through the stem "fo"
    "estivador",  # read as estar, through the stem "estiv"
    "coqueiro",   # read as querer, through the stem "queir"
    "saibro",     # read as saber, through the stem "saib"
]

STILL_IRREGULAR = [
    ("fosses", "ser/ir"),
    ("for", "ser/ir"),
    ("sejam", "ser"),
    ("eras", "ser"),
    ("estiveste", "estar"),
    ("tenham", "ter"),
    ("haja", "haver"),
    ("fizeste", "fazer"),
    ("faça", "fazer"),
    ("farei", "fazer"),
    ("direi", "dizer"),
    ("traga", "trazer"),
    ("pudeste", "poder"),
    ("queira", "querer"),
    ("soubeste", "saber"),
    ("saiba", "saber"),
    ("puseste", "pôr"),
    ("ponha", "pôr"),
    ("vieste", "vir"),
    ("venha", "vir"),
    ("viu", "ver"),
    ("viram", "ver"),
    ("veja", "ver"),
    ("dei", "dar"),
    ("deu", "dar"),
    ("caiba", "caber"),
]


@pytest.mark.parametrize("word", NOT_A_VERB)
def test_a_word_that_only_starts_like_a_verb_is_not_that_verb(analyzer, word):
    result = analyzer.analyze(word)
    lemma = result.verbal.lemma_guess if result.verbal else None
    irregular = bool(result.verbal and result.verbal.is_irregular)
    assert not irregular, f"{word}: read as the irregular verb {lemma!r}"


@pytest.mark.parametrize("word,lemma", STILL_IRREGULAR)
def test_a_real_irregular_form_still_resolves(analyzer, word, lemma):
    result = analyzer.analyze(word)
    assert result.verbal is not None, f"{word}: no verbal analysis"
    assert result.verbal.is_irregular, f"{word}: not flagged as irregular"
    assert result.verbal.lemma_guess == lemma


def test_every_stem_has_an_ending_set():
    """A stem with no ending set can never fire, so none may be missing."""
    missing = sorted(set(IRREGULAR_STEMS) - set(IRREGULAR_STEM_ENDINGS))
    assert missing == [], f"stems with no ending set: {missing}"


def test_no_ending_set_is_orphaned():
    orphan = sorted(set(IRREGULAR_STEM_ENDINGS) - set(IRREGULAR_STEMS))
    assert orphan == [], f"ending sets for unknown stems: {orphan}"
