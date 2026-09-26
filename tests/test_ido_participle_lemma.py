"""The -ido/-ida participle must not invent an infinitive.

The ending ``-ido/-ida/-idos/-idas`` is shared by the 2nd conjugation
(``comer`` -> ``comida``) and the 3rd (``partir`` -> ``partida``), so the
ending alone cannot give the lemma. The analyzer reads the class off the
stem, and emits no lemma when neither candidate infinitive is attested.
"""
import pytest

from tugamorph import PortugueseMorphAnalyzer
from tugamorph._ido_infinitives import ATTESTED_INFINITIVES


@pytest.fixture(scope="module")
def analyzer():
    return PortugueseMorphAnalyzer()


THIRD_CONJUGATION = [
    ("partida", "partir"),
    ("partido", "partir"),
    ("partidas", "partir"),
    ("medida", "medir"),
    ("medido", "medir"),
    ("dormida", "dormir"),
    ("vestido", "vestir"),
    ("sentida", "sentir"),
    ("servida", "servir"),
    ("pedido", "pedir"),
    ("ferida", "ferir"),
    ("admitido", "admitir"),
]

SECOND_CONJUGATION = [
    ("comida", "comer"),
    ("comido", "comer"),
    ("bebida", "beber"),
    ("vendida", "vender"),
    ("batido", "bater"),
    ("conhecida", "conhecer"),
    ("aprendido", "aprender"),
    ("erguido", "erguer"),
]


@pytest.mark.parametrize("word,lemma", THIRD_CONJUGATION)
def test_the_third_conjugation_participle_keeps_its_ir_lemma(analyzer, word, lemma):
    result = analyzer.analyze(word)
    assert result.verbal is not None, f"{word}: no verbal analysis"
    assert result.verbal.lemma_guess == lemma
    assert result.verbal.conjugation_class == 3
    assert result.lemma_guess == lemma


@pytest.mark.parametrize("word,lemma", SECOND_CONJUGATION)
def test_the_second_conjugation_participle_keeps_its_er_lemma(analyzer, word, lemma):
    result = analyzer.analyze(word)
    assert result.verbal is not None, f"{word}: no verbal analysis"
    assert result.verbal.lemma_guess == lemma
    assert result.verbal.conjugation_class == 2
    assert result.lemma_guess == lemma


@pytest.mark.parametrize("word", ["avenida", "bandido", "apelido", "aminoácido"])
def test_a_noun_in_ido_gets_no_invented_lemma(analyzer, word):
    """These are nouns. Neither candidate infinitive exists, so no lemma."""
    result = analyzer.analyze(word)
    lemma = result.verbal.lemma_guess if result.verbal else None
    assert lemma is None, f"{word}: invented the lemma {lemma!r}"


@pytest.mark.parametrize("word", [w for w, _ in THIRD_CONJUGATION + SECOND_CONJUGATION]
                         + ["avenida", "bandido", "apelido", "aminoácido"])
def test_every_emitted_lemma_is_an_attested_infinitive(analyzer, word):
    result = analyzer.analyze(word)
    lemma = result.verbal.lemma_guess if result.verbal else None
    if lemma is not None:
        assert lemma in ATTESTED_INFINITIVES, f"{word}: {lemma!r} is not a word"


def test_the_ar_participle_is_untouched(analyzer):
    """Only the -id* endings are ambiguous; -ado/-ada still lemmatizes."""
    result = analyzer.analyze("cantada")
    assert result.verbal is not None
    assert result.verbal.lemma_guess == "cantar"
    assert result.verbal.conjugation_class == 1
