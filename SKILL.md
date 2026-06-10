# tugamorph — Agent Skill

**tugamorph** is a pure-Python, zero-dependency morphological analyser for European Portuguese.
It segments words into their constituent morphemes, identifies verbal conjugation, and guesses lemmas.

## Accessibility framing

For blind and voice-only users, knowing a word's morphology removes ambiguity in spoken output:

- A TTS pipeline needs to stress the right syllable — stressing the verb "FALAR" vs the noun "FALA" differently.
- A screen reader benefits from knowing that "cantavam" is the 3rd-person plural imperfect of "cantar", so it can announce "verb — cantar, past tense" rather than reading raw text.
- Lemmatization lets a voice assistant normalise a query — "disseram" → "dizer" — so a dictionary or knowledge base lookup succeeds.

## Verified API

```python
from tugamorph import PortugueseMorphAnalyzer, lemmatize_word, is_past_participle

analyzer = PortugueseMorphAnalyzer()

# Analyse a single word
result = analyzer.analyze("cantavam")
result.verbal.tense_mood        # "imperf"
result.verbal.conjugation_class # 1  (1st conj = -ar)
result.lemma_guess              # "cantar"

# Convenience helpers (module-level)
lemmatize_word("disseram")      # "dizer"
lemmatize_word("aprovado")      # "aprovar"
is_past_participle("aprovado")  # True
is_past_participle("correndo")  # False

# Sentence-level (uses tugatagger when installed, heuristic otherwise)
analyses = analyzer.analyze_sentence("Ela cantava muito bem.")
for a in analyses:
    print(a.normalized, "→", a.lemma_guess or a.root)
```

## How to speak results

Given `analyzer.analyze("cantavam")`:

> "The word *cantavam* is the third-person plural imperfect indicative of *cantar* — 'they were singing'."

Given `is_past_participle("aprovado") == True`:

> "The word *aprovado* is a past participle, the adjectival form of *aprovar* — 'approved'."

## Optional dependencies

| Extra | Installs | Effect |
|-------|----------|--------|
| `tugatagger` | `tugatagger` | Full sentence-level POS context for disambiguation |
| `silabificador` | `pt-silabificador` | Syllabification and stress-pattern features |

Install: `pip install tugamorph[tugatagger]`
