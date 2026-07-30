# Quickstart

`tugamorph` takes a Portuguese word and tells you how it is built: which prefixes
were stacked on, what the root is, which suffix or verbal ending closes it off,
whether a clitic pronoun is attached, and what its syllables and stress look like.
One class, one method, one structured result.

## 1. Install

```bash
pip install tugamorph
```

The analyzer runs on the standard library plus
[silabificador](https://github.com/TigreGotico/silabificador) for syllabification.
One optional integration sharpens disambiguation:

```bash
pip install git+https://github.com/TigreGotico/tugatagger   # POS tagging
```

When `tugatagger` is importable the analyzer uses POS tags to break ties between
a verbal reading and a derivational-suffix reading. Without it everything still
works on length-based heuristics.

## 2. The one thing to understand

You build a `PortugueseMorphAnalyzer` once and call `analyze(word)` on it. Each
call returns a single `MorphologicalAnalysis`: the best parse for that word,
holding every layer the pipeline found.

```python
from tugamorph import PortugueseMorphAnalyzer

analyzer = PortugueseMorphAnalyzer()
result = analyzer.analyze("extraordinariamente")
print(result)
# MorphAnalysis('extraordinariamente')
#   segmentation: [extra]-[ordinaria]-[mente]
#   pos: ADV
#   prefixes: extra (position)
#   root: 'ordinaria'
#   suffix: mente (adverb)
#   phonology: ex.tra.or.di.na.ri.a.men.te, paroxytone
```

The `result` is a dataclass. Reach into any layer directly:

```python
result.root                 # 'ordinaria'
result.prefixes             # [('extra', <PrefixCategory.POSITION>)]
result.suffix               # ('mente', <SuffixCategory.ADVERB>)
result.phonology.stress_pattern   # <StressPattern.PAROXYTONE>
result.morphemes            # ordered list of Morpheme spans
```

## 3. Just the segmentation

When you only want the dashed breakdown, skip the object:

```python
analyzer.segment("impossibilidade")
# '[im]-[possibil]-[idade]'
```

## 4. A whole batch

```python
results = analyzer.analyze_batch(["desfazer", "infelizmente", "disseram"])
for r in results:
    print(r.original, "->", r.segmentation_str())
# desfazer -> [des]-[faz]-[er]
# infelizmente -> [in]-[feliz]-[mente]
# disseram -> [disseram]
```

`disseram` comes back as a single morpheme because it is a suppletive form caught
by the whole-word irregular table. The analyzer recognizes it as the
preterite of `dizer` rather than trying to peel it apart.

## 5. Export for ML

Every result speaks three downstream formats:

```python
r = analyzer.analyze("dir-se-ia")
r.to_feature_dict()      # flat dict, good for a DataFrame row
r.to_feature_vector()    # 17-dim list[float]
r.to_json()              # JSON string of the feature dict
```

---
[Home](../README.md) · [API reference →](api.md)
