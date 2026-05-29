# Phonology

Every analysis carries a `PhonologicalFeatures` object on `result.phonology`,
estimated from orthography. When [silabificador](https://github.com/TigreGotico/silabificador)
is installed (the default), syllabification comes from its rule-based engine;
otherwise a built-in heuristic syllabifier is used.

## The fields

```python
r = analyzer.analyze("extraordinariamente")
p = r.phonology

p.syllable_count            # 9
p.syllables                 # ['ex', 'tra', 'or', 'di', 'na', 'ri', 'a', 'men', 'te']
p.stressed_syllable_index   # 0-indexed position of the stressed syllable
p.stress_pattern            # <StressPattern.PAROXYTONE>
p.has_nasal_diphthong       # ão / ãe / õe present?
p.has_oral_diphthong        # ai / ei / ou / …?
p.has_hiatus                # adjacent vowels not forming a diphthong?
p.has_nasal_vowel           # ã / õ present?
p.ends_in_nasal             # word ends in a nasal sound?
p.has_digraph               # ch / lh / nh / rr / ss / qu / gu present?
```

## Stress

`stress_pattern` is one of `OXYTONE` (last syllable), `PAROXYTONE` (penultimate —
the Portuguese default), `PROPAROXYTONE` (antepenultimate), or `UNKNOWN`.

Assignment follows standard rules: an explicit accent mark wins; otherwise words
ending in `-a`, `-e`, `-o`, `-am`, `-em` default to paroxytone, and words ending
in `-r`, `-l`, `-z`, `-i`, `-u` default to oxytone.

```python
analyzer.analyze("casa").phonology.stress_pattern      # paroxytone (default)
analyzer.analyze("cantar").phonology.stress_pattern    # oxytone (ends in -r)
analyzer.analyze("rápido").phonology.stress_pattern    # proparoxytone (accent)
```

## Diphthongs and hiatus

These flags feed phonological feature engineering directly. Use real Portuguese to
see them light up:

```python
analyzer.analyze("pão").phonology.has_nasal_diphthong   # True  (ão)
analyzer.analyze("pai").phonology.has_oral_diphthong    # True  (ai)
analyzer.analyze("saída").phonology.has_hiatus          # True  (a-í)
```

## In the feature exports

The phonology fields surface in `to_feature_dict()` as `syllable_count`,
`syllables`, `stressed_syllable_index`, `stress_pattern` (as a string),
`has_nasal_diphthong`, `has_oral_diphthong`, and `has_hiatus`; the numeric vector
includes `syllable_count`, `has_nasal_diphthong`, `has_oral_diphthong`, and
`has_hiatus`. See [api.md](api.md) for the full ordering.

## Where next

- [quickstart.md](quickstart.md) — the basics
- [api.md](api.md) — full reference
- [advanced.md](advanced.md) — config, clitics, lexicon extension
