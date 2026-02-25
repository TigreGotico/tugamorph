# TugaMorph 🇵🇹

A rule-based morphological analyzer for Portuguese. Segments words into prefixes, roots, suffixes, verbal inflections, and clitics, and produces structured analyses suitable for NLP pipelines, linguistic research, and language documentation.

No dependencies beyond the Python standard library (3.8+). Optional integrations with [silabificador](https://github.com/TigreGotico/silabificador) and [tugatagger](https://github.com/TigreGotico/tugatagger) for improved syllabification and POS-informed disambiguation.

## Installation

```bash
pip install tugamorph
```

Optional extras for enhanced accuracy:

```bash
pip install git+https://github.com/TigreGotico/silabificador   # rule-based syllabifier (~99.6% accuracy)
pip install git+https://github.com/TigreGotico/tugatagger       # POS tagger (spaCy / Brill / heuristic)
```

## Quick start

```python
from tugamorph import PortugueseMorphAnalyzer

analyzer = PortugueseMorphAnalyzer()

result = analyzer.analyze("extraordinariamente")
print(result)
# MorphAnalysis('extraordinariamente')
#   segmentation: [extra]-[ordinaria]-[mente]
#   prefixes: extra (position)
#   root: 'ordinaria'
#   suffix: mente (adverb)
#   phonology: ex.tra.or.di.na.ri.a.men.te, paroxytone

# Segmentation only
print(analyzer.segment("impossibilidade"))
# [im]-[possibil]-[idade]

# Batch processing
results = analyzer.analyze_batch(["desfazer", "infelizmente", "disseram"])
```

## Coverage

| Component | Entries | Details |
|---|---|---|
| Prefixes | 57 | 9 categories — negation, position, intensity, temporal, repetition, connection, duality, quantity, size |
| Derivational suffixes | 89 | 12 categories — scientific, abstract noun, agent, action, place, adjective, diminutive, augmentative, pejorative, gentílico, collective, adverb |
| Verbal endings | 95 | All tenses, moods, persons, and conjugation classes (1st `-ar`, 2nd `-er`, 3rd `-ir`) |
| Irregular verbs | 40 stem allomorphs + 58 whole-word lookups | ser, ir, ter, haver, fazer, dizer, poder, querer, saber, pôr, trazer, estar, … |
| Clitics | 15 | Pronoun forms with person, number, and case annotation |

## Analysis pipeline

```
Input word
  │
  ├─ 0. Irregular whole-word lookup (short-circuit)
  ├─ 1. Clitic extraction (mesoclitic → enclitic)
  ├─ 2. Compound detection (hyphenated)
  ├─ 3. Prefix stacking (longest-first, up to N layers)
  ├─ 4. Irregular stem check
  ├─ 5. Suffix vs verbal ending (longest-match-wins)
  ├─ 6. Root + thematic vowel extraction
  └─ 7. Phonological feature estimation
```

Key design decisions:

- **Longest-match-wins** between suffixes and verbal endings — `-logia` (5 chars, scientific) beats `-ia` (2 chars, imperfeito) for "biologia", but equal-length ties go to verbal for words like "cantaria".
- **Prefix blocklist** prevents false decomposition — "biologia" won't split as `bi-`, "impossibilidade" won't strip a second `pos-` prefix.
- **Whole-word irregular table** short-circuits the pipeline for ~60 common suppletive forms (`disseram`, `fizemos`, `vou`, …).
- **POS-informed disambiguation** — when a tagger is available, VERB/AUX tags force verbal analysis and NOUN/ADJ tags force suffix analysis, overriding length heuristics.

## Output formats

### Structured object

Every call to `analyze()` returns a `MorphologicalAnalysis` dataclass:

```python
result = analyzer.analyze("dir-se-ia")

result.clitic          # CliticInfo(form='se', position=MESOCLITIC, person=3, number='sg', case='refl')
result.verbal          # VerbalAnalysis(tense_mood='fut_ind', is_irregular=True, lemma_guess='dizer', ...)
result.prefixes        # []
result.root            # 'diria'
result.phonology       # PhonologicalFeatures(syllable_count=3, stress_pattern=PAROXYTONE, ...)
result.morphemes       # [<CLITIC: 'se' (mesoclitic)>, <ROOT: 'diria' (irregular:dizer)>]
```

### Feature dictionary

Flat dictionary suitable for tabular ML:

```python
result.to_feature_dict()
# {
#   'word': 'dir-se-ia',
#   'word_length': 9,
#   'vowel_count': 3,
#   'root': 'diria',
#   'has_clitic': True,
#   'clitic_position': 'mesoclitic',
#   'is_irregular': True,
#   'lemma_guess': 'dizer',
#   'tense_mood': 'fut_ind',
#   ...
# }
```

### Numerical vector (17-dim)

```python
result.to_feature_vector()
# [9.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0]
```

Dimensions: word_length, vowel_count, consonant_count, is_capitalized, has_accent, has_hyphen, syllable_count, has_nasal_diphthong, has_oral_diphthong, has_hiatus, n_prefixes, n_morphemes, is_compound, is_irregular, conjugation_class, person, number.

### JSON

```python
print(result.to_json())
```

## Sentence-level analysis

When tugatagger is installed, `analyze_sentence` tags the full sentence first and feeds each word's POS into the morphological analyzer for context-aware disambiguation:

```python
results = analyzer.analyze_sentence("O gato comeria o peixe")
for r in results:
    print(f"{r.original:12} {r.pos_tag:6} {r.segmentation_str()}")
# O            DET    [o]
# gato         NOUN   [gato]
# comeria      VERB   [com]-[eria]
# o            DET    [o]
# peixe        NOUN   [peixe]
```

## Configuration

Toggle pipeline stages or tune parameters via `AnalysisConfig`:

```python
from tugamorph import PortugueseMorphAnalyzer, AnalysisConfig

config = AnalysisConfig(
    extract_prefixes=True,
    extract_suffixes=True,
    extract_verbal=True,
    extract_clitics=True,
    extract_phonology=True,
    detect_compounds=True,
    max_prefix_stack=3,     # max nested prefixes (e.g., sub-des-envolvimento)
    min_root_length=2,      # won't strip a root below this length
    use_pos_tagger=True,    # use tugatagger if installed
    tagger_engine="auto",   # "auto", "spacy", "brill", "lexicon", "dummy"
    pos_disambiguate=True,  # use POS to break verbal vs suffix ties
    use_syllabifier=True,   # use silabificador if installed
)

analyzer = PortugueseMorphAnalyzer(config)
```

## Phonological features

Estimated from orthography (accurate syllabification when silabificador is installed):

| Feature | Example |
|---|---|
| Syllable count | "extraordinariamente" → 9 |
| Stress pattern | oxytone / paroxytone / proparoxytone |
| Nasal diphthong | "ão", "ãe", "õe" |
| Oral diphthong | "ai", "ei", "ou", … |
| Hiatus | adjacent vowels that aren't diphthongs |
| Nasal vowel | "ã", "õ" |
| Digraphs | "ch", "lh", "nh", "rr", "ss", "qu", "gu" |

Stress assignment: explicit accent marks override; otherwise standard Portuguese rules apply (words ending in `-a`, `-e`, `-o`, `-am`, `-em` default paroxytone; `-r`, `-l`, `-z`, `-i`, `-u` default oxytone).

## Extending the lexicon

All data tables are module-level lists and dicts:

```python
from tugamorph import PREFIX_TABLE, SUFFIX_TABLE, IRREGULAR_STEMS, PrefixCategory, SuffixCategory

# Add a prefix
PREFIX_TABLE.append(('proto', PrefixCategory.TEMPORAL))

# Add a suffix
SUFFIX_TABLE.append(('teca', SuffixCategory.NOUN_PLACE))
```

To add whole irregular forms or block prefixes from specific stems:

```python
analyzer = PortugueseMorphAnalyzer()

# Irregular whole-word form
analyzer._irregular_whole_words['deram'] = ('dar', 'pret_perf', 3, 'pl')

# Block a prefix on a specific stem
analyzer._prefix_block_stems['recon'] = {'re'}   # don't strip re- from "reconhecer"
```

## Related projects

- **[silabificador](https://github.com/TigreGotico/silabificador)** — rule-based Portuguese syllabifier (~99.6% accuracy on 53k words)
- **[tugatagger](https://github.com/TigreGotico/tugatagger)** — unified POS tagging interface (spaCy / Brill / lexicon / heuristic fallback)
- **[tugalex](https://github.com/TigreGotico/tugalex)** — Portuguese phonetic lexicon with IPA transcriptions, syllable segmentations, and AO1990 mappings for 5 dialect regions

## Limitations

This is a rule-based heuristic analyzer, not a statistical model.

- **Ambiguity is not modeled.** Returns a single best parse. Words like "canto" (noun "corner" vs. verb "I sing") match whichever pattern fires first without sentence context.
- **The prefix blocklist is manually curated.** Uncommon words may get false prefix splits — extend `_prefix_block_stems` as needed.
- **Syllable count is estimated** from orthography. Some hiatus vs. diphthong distinctions require phonological knowledge beyond spelling (mitigated when silabificador is installed).
- **No lemmatization.** The analyzer provides `lemma_guess` for irregular verbs but does not perform full lemmatization.
- **European Portuguese focus.** Suffix, prefix, and verbal tables reflect EP norms. Brazilian Portuguese verbal forms are covered by the endings table, but BP-specific morphological patterns are not specially annotated.

## Testing

250 tests covering character features, prefix/suffix extraction, the full verbal paradigm, irregular verbs, clitics (enclitic + mesoclitic), compounds, phonology, feature export, POS disambiguation, and regression cases:

```bash
python -m unittest discover tests -v
```

## License

MIT