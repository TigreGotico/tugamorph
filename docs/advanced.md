# Advanced

Recipes, the configuration knobs, and how to extend the lexicon.

## Configuration

`AnalysisConfig` toggles pipeline stages and tunes their parameters. Pass it to
the constructor:

```python
from tugamorph import PortugueseMorphAnalyzer, AnalysisConfig

config = AnalysisConfig(
    extract_prefixes=True,
    extract_suffixes=True,
    extract_verbal=True,
    extract_clitics=True,
    extract_phonology=True,
    detect_compounds=True,
    max_prefix_stack=3,     # cap nested prefixes, e.g. sub-des-envolvimento
    min_root_length=2,      # never strip a root shorter than this
    use_pos_tagger=True,    # wire up tugatagger if importable
    tagger_engine="auto",   # "auto" / "spacy" / "brill" / "lexicon" / "dummy"
    pos_disambiguate=True,  # let POS break verbal-vs-suffix ties
    use_syllabifier=True,   # use silabificador for syllables
)

analyzer = PortugueseMorphAnalyzer(config)
```

Turn a stage off and the corresponding fields stay at their defaults. For example
with `extract_prefixes=False`, `result.prefixes` is always `[]` and the prefix is
left on the root.

## Clitics

Pronouns attached to verbs are pulled off before the rest of the analysis. The
analyzer handles enclitics (after the verb) and mesoclitics (inside a future or
conditional form):

```python
analyzer.analyze("disse-me").clitic
# CliticInfo(form='me', position=<CliticPosition.ENCLITIC>, person=1, number='sg', case='acc/dat')

r = analyzer.analyze("dir-se-ia")
r.clitic.position     # <CliticPosition.MESOCLITIC>
r.verbal.lemma_guess  # 'dizer'  — the remaining 'diria' is recognised as irregular
```

`CliticInfo` carries the pronoun `form`, its `position`, and grammatical
`person` / `number` / `case`.

## Compounds

Hyphenated words are split: the non-final parts become compound morphemes and the
head (last part) is analyzed normally.

```python
r = analyzer.analyze("guarda-chuva")
r.is_compound        # True
r.compound_parts     # ['guarda', 'chuva']
```

## Verbal-vs-suffix disambiguation

Many endings are ambiguous between a derivational suffix and a verbal inflection.
The default rule is **longest match wins**, and equal-length ties go to the verbal
reading. A POS tag overrides this:

```python
# Without a hint, length heuristics decide.
analyzer.analyze("cantaria")

# A trusted tag forces the reading.
analyzer.analyze("cantaria", pos_tag="NOUN")   # suffix reading
analyzer.analyze("cantaria", pos_tag="VERB")   # verbal reading
```

When `tugatagger` is installed and `pos_disambiguate=True`, this happens
automatically. `analyze_sentence` is the strongest form — it tags the whole
sentence so each word's POS reflects its context:

```python
for r in analyzer.analyze_sentence("O gato comeria o peixe"):
    print(f"{r.original:10} {r.pos_tag or '?':6} {r.segmentation_str()}")
```

## Extending the lexicon

The lexical tables are plain module-level structures. Append to them before
building an analyzer:

```python
from tugamorph import PREFIX_TABLE, SUFFIX_TABLE, PrefixCategory, SuffixCategory

PREFIX_TABLE.append(('proto', PrefixCategory.TEMPORAL))
SUFFIX_TABLE.append(('teca', SuffixCategory.NOUN_PLACE))

from tugamorph import PortugueseMorphAnalyzer
analyzer = PortugueseMorphAnalyzer()   # picks up the additions on construction
```

Two per-instance dicts let you patch behaviour after construction:

```python
analyzer = PortugueseMorphAnalyzer()

# Register a whole irregular form: word -> (lemma, tense_mood, person, number)
analyzer._irregular_whole_words['deram'] = ('dar', 'pret_perf', 3, 'pl')

# Stop a prefix from being stripped off a specific stem.
analyzer._prefix_block_stems['recon'] = {'re'}   # keep "reconhecer" intact
```

The prefix blocklist prevents false decompositions — it is what keeps `biologia`
from splitting as `bi-ologia`. It is manually curated, so uncommon words may need
a new entry here.

## Feeding an ML pipeline

`to_feature_dict()` rows drop straight into a DataFrame; `to_feature_vector()`
gives a fixed 17-dim numeric vector for models that want tensors:

```python
import csv
words = ["desfazer", "infelizmente", "biologia", "cantaria", "casinha"]
rows = [analyzer.analyze(w).to_feature_dict() for w in words]
# rows is a list of flat dicts with identical core keys
```

## Sibling libraries

- [silabificador](https://github.com/TigreGotico/silabificador) supplies the
  syllabification used by the phonology stage.
- [tugatagger](https://github.com/TigreGotico/tugatagger) supplies the POS tags
  used for disambiguation.
- [tugalex](https://github.com/TigreGotico/tugalex) is a phonetic lexicon with IPA
  transcriptions and dialect mappings, complementary for downstream phonetics.

## Where next

- [quickstart.md](quickstart.md) — the basics
- [api.md](api.md) — full type and method reference
- [phonology.md](phonology.md) — syllables, stress, and diphthong detection
