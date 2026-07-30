# API reference

Everything you import lives in the top-level `tugamorph` package.

```python
from tugamorph import (
    PortugueseMorphAnalyzer,
    AnalysisConfig,
    MorphologicalAnalysis,
    Morpheme, CliticInfo, VerbalAnalysis, PhonologicalFeatures,
    MorphemeType, PrefixCategory, SuffixCategory, StressPattern, CliticPosition,
    PREFIX_TABLE, SUFFIX_TABLE, IRREGULAR_STEMS,
)
```

## `PortugueseMorphAnalyzer`

The orchestrator. Build one and reuse it.

```python
PortugueseMorphAnalyzer(config: Optional[AnalysisConfig] = None)
```

`config` defaults to a fresh `AnalysisConfig()` with every stage enabled. The
constructor pre-sorts the lexical tables for greedy longest-first matching. If
`tugatagger` is importable and `config.use_pos_tagger` is true, it wires up a
POS tagger. Otherwise it falls back silently to heuristics.

### Methods

| Method | Returns | Purpose |
|---|---|---|
| `analyze(word, pos_tag=None)` | `MorphologicalAnalysis` | Full analysis of one word |
| `analyze_batch(words)` | `List[MorphologicalAnalysis]` | `analyze` over a list |
| `segment(word)` | `str` | Just the `[a]-[b]-[c]` segmentation string |
| `analyze_sentence(sentence)` | `List[MorphologicalAnalysis]` | Tag the sentence, then analyze each word with its POS |

```python
analyze(word: str, pos_tag: Optional[str] = None) -> MorphologicalAnalysis
```

`pos_tag` is an optional trusted UPOS hint (`"VERB"`, `"NOUN"`, `"ADJ"`, `"ADV"`,
`"AUX"`, `"PROPN"`, …). When supplied it skips internal tagging and drives
verbal-vs-suffix disambiguation directly. `analyze_sentence` uses this internally:
it tags the full sentence with `tugatagger` (when available) so each word is
analyzed in context.

## `MorphologicalAnalysis`

The result dataclass. Fields:

| Field | Type | Notes |
|---|---|---|
| `original` | `str` | The input as given |
| `normalized` | `str` | Lower-cased, stripped |
| `morphemes` | `List[Morpheme]` | Ordered spans (prefixes, root, suffix, clitic, …) |
| `prefixes` | `List[Tuple[str, PrefixCategory]]` | Stacked prefixes, outermost first |
| `root` | `str` | The remaining stem |
| `suffix` | `Optional[Tuple[str, SuffixCategory]]` | Derivational suffix, if any |
| `clitic` | `Optional[CliticInfo]` | Attached pronoun, if any |
| `verbal` | `Optional[VerbalAnalysis]` | Verbal decomposition, if a verb |
| `phonology` | `PhonologicalFeatures` | Syllables / stress / diphthongs |
| `pos_tag` | `Optional[str]` | UPOS tag used or guessed |
| `word_length`, `vowel_count`, `consonant_count` | `int` | Character counts |
| `is_capitalized`, `has_accent`, `has_hyphen` | `bool` | Character flags |
| `is_compound`, `compound_parts` | `bool`, `List[str]` | Hyphenated compounds |

### Result methods

```python
segmentation_str() -> str          # "[des]-[faz]-[er]"
to_feature_dict() -> Dict[str, Any]
to_feature_vector() -> List[float] # 17-dim
to_json(indent: int = 2) -> str
```

`to_feature_dict()` is flat and JSON-safe. It always carries the character,
phonology, prefix, suffix, clitic, and compound fields. When the word is verbal,
it also adds `conjugation_class`, `tense_mood`, `person`, `number`, `thematic_vowel`,
`is_irregular`, and `lemma_guess`.

`to_feature_vector()` returns these 17 floats in order:

```
word_length, vowel_count, consonant_count, is_capitalized, has_accent,
has_hyphen, syllable_count, has_nasal_diphthong, has_oral_diphthong,
has_hiatus, n_prefixes, n_morphemes, is_compound, is_irregular,
conjugation_class, person, number
```

(`number` is encoded `sg=1.0`, `pl=2.0`, otherwise `0.0`.)

## Component dataclasses

```python
Morpheme(form, mtype: MorphemeType, label="", start=0, end=0)
```

```python
CliticInfo(form, position: CliticPosition, person=None,
           number=None, case=None)   # number 'sg'/'pl'; case 'acc'/'dat'/'refl'
```

```python
VerbalAnalysis(conjugation_class=None, tense_mood="none", person=None,
               number=None, thematic_vowel=None, is_irregular=False,
               lemma_guess=None, allomorph=None)
```

`conjugation_class` is `1` (`-ar`), `2` (`-er`), or `3` (`-ir`). `lemma_guess` is
populated for irregular forms only.

```python
PhonologicalFeatures(syllable_count=0, syllables=[], stressed_syllable_index=None,
                     stress_pattern=StressPattern.UNKNOWN,
                     has_nasal_diphthong=False, has_oral_diphthong=False,
                     has_hiatus=False, has_nasal_vowel=False,
                     ends_in_nasal=False, has_digraph=False)
```

## Enums

| Enum | Members |
|---|---|
| `MorphemeType` | `PREFIX`, `ROOT`, `THEMATIC_VOWEL`, `TENSE_MOOD`, `PERSON_NUMBER`, `SUFFIX`, `CLITIC`, `INTERFIXE` |
| `PrefixCategory` | `NEGATION`, `REPETITION`, `POSITION`, `INTENSITY`, `CONNECTION`, `DUALITY`, `QUANTITY`, `SIZE`, `TEMPORAL` |
| `SuffixCategory` | `ADVERB`, `NOUN_ABSTRACT`, `NOUN_AGENT`, `NOUN_ACTION`, `NOUN_PLACE`, `ADJECTIVE`, `DIMINUTIVE`, `AUGMENTATIVE`, `PEJORATIVE`, `SCIENTIFIC`, `GENTILICO`, `COLLECTIVE` |
| `StressPattern` | `OXYTONE`, `PAROXYTONE`, `PROPAROXYTONE`, `UNKNOWN` |
| `CliticPosition` | `PROCLITIC`, `ENCLITIC`, `MESOCLITIC`, `NONE` |

Each enum has a readable `__str__` (e.g. `str(StressPattern.PAROXYTONE)` is
`"paroxytone"`), which is what the feature dict and JSON use.

## `AnalysisConfig`

See [advanced.md](advanced.md) for the full toggle list and what each one changes.

## Lexical tables

`PREFIX_TABLE`, `SUFFIX_TABLE`, and `IRREGULAR_STEMS` are module-level lists/dicts
intended to be extended in place. See [advanced.md](advanced.md#extending-the-lexicon).

---
[← Quickstart](quickstart.md) · [Home](../README.md) · [Advanced →](advanced.md)
