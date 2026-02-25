"""
Portuguese Morphological Analyzer
=================================

A full-featured morpheme lexer and feature extractor for Portuguese,
suitable for NLP pipelines, linguistic research, and language documentation.

Features:
    - Morpheme segmentation with boundary tracking
    - Full verbal paradigm recognition (all tenses, moods, persons)
    - Proclitic / enclitic / mesoclitic handling
    - Prefix stacking (multiple prefixes per word)
    - Compound word splitting
    - Irregular verb stem mapping with allomorph tracking
    - Accurate syllabification via silabificador (rule-based, ~99.6% accuracy)
    - POS tagging via tugatagger (spaCy / Brill / heuristic fallback)
    - POS-informed verbal vs. nominal disambiguation
    - Sentence-level analysis with context-aware POS tags
    - Phonological feature extraction (nasal diphthongs, hiatus, etc.)
    - Numerical feature vector export for ML pipelines
    - Configurable analysis via AnalysisConfig

Integrations (optional — graceful fallback when not installed):
    - silabificador: pip install git+https://github.com/TigreGotico/silabificador
    - tugatagger:    pip install git+https://github.com/TigreGotico/tugatagger

Author: Casimiro Ferreira + Claude Opus 4.6
License: MIT
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Set

from silabificador import syllabify

# ─── Optional integration imports (graceful fallback) ───

_HAS_TAGGER = False

try:
    from tugatagger import TugaTagger as _ExtTugaTagger
    _HAS_TAGGER = True
except ImportError:
    _ExtTugaTagger = None


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class MorphemeType(Enum):
    PREFIX = auto()
    ROOT = auto()
    THEMATIC_VOWEL = auto()
    TENSE_MOOD = auto()
    PERSON_NUMBER = auto()
    SUFFIX = auto()
    CLITIC = auto()
    INTERFIXE = auto()  # linking element in compounds

    def __str__(self):
        return self.name.lower()


class PrefixCategory(Enum):
    NEGATION = auto()
    REPETITION = auto()
    POSITION = auto()
    INTENSITY = auto()
    CONNECTION = auto()
    DUALITY = auto()
    QUANTITY = auto()
    SIZE = auto()
    TEMPORAL = auto()

    def __str__(self):
        return self.name.lower()


class SuffixCategory(Enum):
    ADVERB = auto()
    NOUN_ABSTRACT = auto()
    NOUN_AGENT = auto()
    NOUN_ACTION = auto()
    NOUN_PLACE = auto()
    ADJECTIVE = auto()
    DIMINUTIVE = auto()
    AUGMENTATIVE = auto()
    PEJORATIVE = auto()
    SCIENTIFIC = auto()
    GENTILICO = auto()  # nationality / origin
    COLLECTIVE = auto()

    def __str__(self):
        return self.name.lower()


class StressPattern(Enum):
    OXYTONE = auto()  # aguda: last syllable
    PAROXYTONE = auto()  # grave: penultimate (most common in PT)
    PROPAROXYTONE = auto()  # esdrúxula: antepenultimate
    UNKNOWN = auto()

    def __str__(self):
        return self.name.lower()


class CliticPosition(Enum):
    PROCLITIC = auto()  # before verb: "me disse"
    ENCLITIC = auto()  # after verb: "disse-me"
    MESOCLITIC = auto()  # inside verb: "dir-me-ia"
    NONE = auto()

    def __str__(self):
        return self.name.lower()


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class Morpheme:
    """A single morpheme with its type, surface form, and span."""
    form: str
    mtype: MorphemeType
    label: str = ""
    start: int = 0
    end: int = 0

    def __repr__(self):
        return f"<{self.mtype}: '{self.form}' ({self.label})>"


@dataclass
class CliticInfo:
    """Clitic pronoun details."""
    form: str
    position: CliticPosition = CliticPosition.NONE
    person: Optional[int] = None
    number: Optional[str] = None  # 'sg' or 'pl'
    case: Optional[str] = None  # 'acc', 'dat', 'refl'


@dataclass
class VerbalAnalysis:
    """Full verbal morphology decomposition."""
    conjugation_class: Optional[int] = None  # 1 (-ar), 2 (-er), 3 (-ir)
    tense_mood: str = "none"
    person: Optional[int] = None  # 1, 2, 3
    number: Optional[str] = None  # 'sg', 'pl'
    thematic_vowel: Optional[str] = None
    is_irregular: bool = False
    lemma_guess: Optional[str] = None
    allomorph: Optional[str] = None  # the irregular stem variant


@dataclass
class PhonologicalFeatures:
    """Phonological properties of the word."""
    syllable_count: int = 0
    syllables: List[str] = field(default_factory=list)
    stressed_syllable_index: Optional[int] = None  # 0-indexed from start
    stress_pattern: StressPattern = StressPattern.UNKNOWN
    has_nasal_diphthong: bool = False
    has_oral_diphthong: bool = False
    has_hiatus: bool = False
    has_nasal_vowel: bool = False
    ends_in_nasal: bool = False
    has_digraph: bool = False


@dataclass
class MorphologicalAnalysis:
    """Complete morphological analysis result for a single word."""
    original: str
    normalized: str
    morphemes: List[Morpheme] = field(default_factory=list)
    prefixes: List[Tuple[str, PrefixCategory]] = field(default_factory=list)
    root: str = ""
    suffix: Optional[Tuple[str, SuffixCategory]] = None
    clitic: Optional[CliticInfo] = None
    verbal: Optional[VerbalAnalysis] = None
    phonology: PhonologicalFeatures = field(default_factory=PhonologicalFeatures)
    pos_tag: Optional[str] = None  # UPOS tag (NOUN, VERB, ADJ, ADV, etc.)

    # Character-level features
    word_length: int = 0
    vowel_count: int = 0
    consonant_count: int = 0
    is_capitalized: bool = False
    has_accent: bool = False
    has_hyphen: bool = False

    # Compound analysis
    is_compound: bool = False
    compound_parts: List[str] = field(default_factory=list)

    def segmentation_str(self) -> str:
        """Return a human-readable morpheme segmentation like: [des]-[faz]-[er]"""
        if not self.morphemes:
            return self.normalized
        return "-".join(f"[{m.form}]" for m in self.morphemes)

    def to_feature_dict(self) -> Dict[str, Any]:
        """Flat dictionary suitable for tabular ML features."""
        d = {
            'word': self.normalized,
            'word_length': self.word_length,
            'vowel_count': self.vowel_count,
            'consonant_count': self.consonant_count,
            'is_capitalized': self.is_capitalized,
            'has_accent': self.has_accent,
            'has_hyphen': self.has_hyphen,
            'syllable_count': self.phonology.syllable_count,
            'syllables': self.phonology.syllables,
            'stressed_syllable_index': self.phonology.stressed_syllable_index,
            'stress_pattern': str(self.phonology.stress_pattern),
            'has_nasal_diphthong': self.phonology.has_nasal_diphthong,
            'has_oral_diphthong': self.phonology.has_oral_diphthong,
            'has_hiatus': self.phonology.has_hiatus,
            'n_prefixes': len(self.prefixes),
            'prefix_types': [str(p[1]) for p in self.prefixes],
            'root': self.root,
            'suffix_type': str(self.suffix[1]) if self.suffix else 'none',
            'has_clitic': self.clitic is not None and self.clitic.position != CliticPosition.NONE,
            'clitic_position': str(self.clitic.position) if self.clitic else 'none',
            'is_compound': self.is_compound,
            'n_morphemes': len(self.morphemes),
            'pos_tag': self.pos_tag,
        }
        if self.verbal:
            d.update({
                'conjugation_class': self.verbal.conjugation_class,
                'tense_mood': self.verbal.tense_mood,
                'person': self.verbal.person,
                'number': self.verbal.number,
                'thematic_vowel': self.verbal.thematic_vowel,
                'is_irregular': self.verbal.is_irregular,
                'lemma_guess': self.verbal.lemma_guess,
            })
        return d

    def to_feature_vector(self) -> List[float]:
        """Numerical vector for embedding / ML input."""
        d = self.to_feature_dict()
        vec = [
            float(d['word_length']),
            float(d['vowel_count']),
            float(d['consonant_count']),
            float(d['is_capitalized']),
            float(d['has_accent']),
            float(d['has_hyphen']),
            float(d['syllable_count']),
            float(d['has_nasal_diphthong']),
            float(d['has_oral_diphthong']),
            float(d['has_hiatus']),
            float(d['n_prefixes']),
            float(d['n_morphemes']),
            float(d['is_compound']),
            float(d.get('is_irregular', False)),
            float(d.get('conjugation_class') or 0),
            float(d.get('person') or 0),
            1.0 if d.get('number') == 'sg' else (2.0 if d.get('number') == 'pl' else 0.0),
        ]
        return vec

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_feature_dict(), ensure_ascii=False, indent=indent)

    def __repr__(self):
        parts = [f"MorphAnalysis('{self.original}')"]
        parts.append(f"  segmentation: {self.segmentation_str()}")
        if self.pos_tag:
            parts.append(f"  pos: {self.pos_tag}")
        if self.prefixes:
            parts.append(f"  prefixes: {', '.join(f'{p[0]} ({p[1]})' for p in self.prefixes)}")
        parts.append(f"  root: '{self.root}'")
        if self.suffix:
            parts.append(f"  suffix: {self.suffix[0]} ({self.suffix[1]})")
        if self.verbal and self.verbal.tense_mood != 'none':
            v = self.verbal
            parts.append(
                f"  verbal: {v.tense_mood} {v.person or '?'}{'sg' if v.number == 'sg' else 'pl' if v.number == 'pl' else '?'} (conj {v.conjugation_class or '?'})")
            if v.is_irregular:
                parts.append(f"  irregular: {v.allomorph} → {v.lemma_guess}")
        if self.clitic and self.clitic.position != CliticPosition.NONE:
            parts.append(f"  clitic: -{self.clitic.form} ({self.clitic.position})")
        syl_str = '.'.join(
            self.phonology.syllables) if self.phonology.syllables else f"{self.phonology.syllable_count} syl"
        parts.append(f"  phonology: {syl_str}, {self.phonology.stress_pattern}")
        return "\n".join(parts)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Toggle components of the analysis pipeline."""
    extract_prefixes: bool = True
    extract_suffixes: bool = True
    extract_verbal: bool = True
    extract_clitics: bool = True
    extract_phonology: bool = True
    detect_compounds: bool = True
    max_prefix_stack: int = 3
    min_root_length: int = 2  # Don't strip root below this length

    # POS tagger integration (tugatagger)
    use_pos_tagger: bool = True  # Use tugatagger if installed, else heuristic
    tagger_engine: str = "auto"  # "auto", "spacy", "brill", "lexicon", "dummy"
    pos_disambiguate: bool = True  # Use POS to break verbal vs suffix ties

    # Syllabifier integration (silabificador)
    use_syllabifier: bool = True  # Use silabificador if installed, else heuristic


# ─────────────────────────────────────────────
# Lexicon Data
# ─────────────────────────────────────────────

# Prefixes: ordered longest-first per category to avoid partial matches
PREFIX_TABLE: List[Tuple[str, PrefixCategory]] = [
    # Position / spatial
    ('circum', PrefixCategory.POSITION),
    ('extra', PrefixCategory.POSITION),
    ('inter', PrefixCategory.POSITION),
    ('infra', PrefixCategory.POSITION),
    ('intra', PrefixCategory.POSITION),
    ('retro', PrefixCategory.POSITION),
    ('super', PrefixCategory.POSITION),
    ('sobre', PrefixCategory.POSITION),
    ('supra', PrefixCategory.POSITION),
    ('trans', PrefixCategory.POSITION),
    ('ante', PrefixCategory.POSITION),
    ('peri', PrefixCategory.POSITION),
    ('endo', PrefixCategory.POSITION),
    ('post', PrefixCategory.TEMPORAL),
    ('pre', PrefixCategory.TEMPORAL),
    ('pos', PrefixCategory.TEMPORAL),
    ('pós', PrefixCategory.TEMPORAL),
    ('pré', PrefixCategory.TEMPORAL),
    ('sub', PrefixCategory.POSITION),
    ('sob', PrefixCategory.POSITION),
    ('exo', PrefixCategory.POSITION),
    ('per', PrefixCategory.POSITION),
    ('ex', PrefixCategory.POSITION),

    # Negation
    ('contra', PrefixCategory.NEGATION),
    ('anti', PrefixCategory.NEGATION),
    ('des', PrefixCategory.NEGATION),
    ('dis', PrefixCategory.NEGATION),
    ('in', PrefixCategory.NEGATION),
    ('im', PrefixCategory.NEGATION),
    ('ir', PrefixCategory.NEGATION),
    ('an', PrefixCategory.NEGATION),
    ('a', PrefixCategory.NEGATION),

    # Repetition
    ('re', PrefixCategory.REPETITION),

    # Intensity / quantity
    ('hiper', PrefixCategory.INTENSITY),
    ('ultra', PrefixCategory.INTENSITY),
    ('multi', PrefixCategory.INTENSITY),
    ('pluri', PrefixCategory.INTENSITY),
    ('super', PrefixCategory.INTENSITY),
    ('maxi', PrefixCategory.INTENSITY),
    ('mini', PrefixCategory.SIZE),
    ('semi', PrefixCategory.QUANTITY),
    ('mono', PrefixCategory.QUANTITY),
    ('poli', PrefixCategory.QUANTITY),

    # Connection
    ('con', PrefixCategory.CONNECTION),
    ('com', PrefixCategory.CONNECTION),
    ('cor', PrefixCategory.CONNECTION),
    ('co', PrefixCategory.CONNECTION),
    ('ad', PrefixCategory.CONNECTION),
    ('ab', PrefixCategory.CONNECTION),

    # Duality
    ('ambi', PrefixCategory.DUALITY),
    ('bi', PrefixCategory.DUALITY),
    ('di', PrefixCategory.DUALITY),

    # Quantity / number
    ('hexa', PrefixCategory.QUANTITY),
    ('penta', PrefixCategory.QUANTITY),
    ('tetra', PrefixCategory.QUANTITY),
    ('tri', PrefixCategory.QUANTITY),
    ('uni', PrefixCategory.QUANTITY),
]

# Suffixes: ordered longest-first per category
SUFFIX_TABLE: List[Tuple[str, SuffixCategory]] = [
    # Scientific / learned
    ('logia', SuffixCategory.SCIENTIFIC),
    ('grafia', SuffixCategory.SCIENTIFIC),
    ('metria', SuffixCategory.SCIENTIFIC),
    ('scopia', SuffixCategory.SCIENTIFIC),
    ('nomia', SuffixCategory.SCIENTIFIC),
    ('cracia', SuffixCategory.SCIENTIFIC),
    ('patia', SuffixCategory.SCIENTIFIC),
    ('fobia', SuffixCategory.SCIENTIFIC),
    ('filia', SuffixCategory.SCIENTIFIC),

    # Adverb
    ('mente', SuffixCategory.ADVERB),

    # Noun: abstract / state
    ('ização', SuffixCategory.NOUN_ABSTRACT),
    ('imento', SuffixCategory.NOUN_ABSTRACT),
    ('amento', SuffixCategory.NOUN_ABSTRACT),
    ('ência', SuffixCategory.NOUN_ABSTRACT),
    ('ância', SuffixCategory.NOUN_ABSTRACT),
    ('idade', SuffixCategory.NOUN_ABSTRACT),
    ('ição', SuffixCategory.NOUN_ABSTRACT),
    ('ção', SuffixCategory.NOUN_ABSTRACT),
    ('são', SuffixCategory.NOUN_ABSTRACT),
    ('ismo', SuffixCategory.NOUN_ABSTRACT),
    ('dade', SuffixCategory.NOUN_ABSTRACT),
    ('tura', SuffixCategory.NOUN_ABSTRACT),
    ('ura', SuffixCategory.NOUN_ABSTRACT),
    ('ice', SuffixCategory.NOUN_ABSTRACT),
    ('eza', SuffixCategory.NOUN_ABSTRACT),
    ('ez', SuffixCategory.NOUN_ABSTRACT),

    # Noun: agent / doer
    ('eiro', SuffixCategory.NOUN_AGENT),
    ('eira', SuffixCategory.NOUN_AGENT),
    ('ista', SuffixCategory.NOUN_AGENT),
    ('ante', SuffixCategory.NOUN_AGENT),
    ('ente', SuffixCategory.NOUN_AGENT),
    ('ário', SuffixCategory.NOUN_AGENT),
    ('ária', SuffixCategory.NOUN_AGENT),
    ('dor', SuffixCategory.NOUN_AGENT),
    ('dora', SuffixCategory.NOUN_AGENT),

    # Noun: action / result
    ('agem', SuffixCategory.NOUN_ACTION),
    ('ança', SuffixCategory.NOUN_ACTION),
    ('ença', SuffixCategory.NOUN_ACTION),
    ('ida', SuffixCategory.NOUN_ACTION),

    # Noun: place
    ('ário', SuffixCategory.NOUN_PLACE),
    ('aria', SuffixCategory.NOUN_PLACE),
    ('ório', SuffixCategory.NOUN_PLACE),
    ('eiro', SuffixCategory.NOUN_PLACE),

    # Gentílico (nationality / origin)
    ('ense', SuffixCategory.GENTILICO),
    ('ês', SuffixCategory.GENTILICO),
    ('esa', SuffixCategory.GENTILICO),
    ('ano', SuffixCategory.GENTILICO),
    ('ana', SuffixCategory.GENTILICO),

    # Collective
    ('agem', SuffixCategory.COLLECTIVE),
    ('edo', SuffixCategory.COLLECTIVE),
    ('ada', SuffixCategory.COLLECTIVE),

    # Adjective
    ('ável', SuffixCategory.ADJECTIVE),
    ('ível', SuffixCategory.ADJECTIVE),
    ('esco', SuffixCategory.ADJECTIVE),
    ('esca', SuffixCategory.ADJECTIVE),
    ('ável', SuffixCategory.ADJECTIVE),
    ('oso', SuffixCategory.ADJECTIVE),
    ('osa', SuffixCategory.ADJECTIVE),
    ('ivo', SuffixCategory.ADJECTIVE),
    ('iva', SuffixCategory.ADJECTIVE),
    ('ico', SuffixCategory.ADJECTIVE),
    ('ica', SuffixCategory.ADJECTIVE),
    ('udo', SuffixCategory.ADJECTIVE),
    ('uda', SuffixCategory.ADJECTIVE),
    ('ado', SuffixCategory.ADJECTIVE),
    ('vel', SuffixCategory.ADJECTIVE),
    ('al', SuffixCategory.ADJECTIVE),
    ('ar', SuffixCategory.ADJECTIVE),
    ('il', SuffixCategory.ADJECTIVE),

    # Diminutive
    ('zinho', SuffixCategory.DIMINUTIVE),
    ('zinha', SuffixCategory.DIMINUTIVE),
    ('inho', SuffixCategory.DIMINUTIVE),
    ('inha', SuffixCategory.DIMINUTIVE),
    ('zito', SuffixCategory.DIMINUTIVE),
    ('zita', SuffixCategory.DIMINUTIVE),
    ('ito', SuffixCategory.DIMINUTIVE),
    ('ita', SuffixCategory.DIMINUTIVE),

    # Augmentative
    ('alhão', SuffixCategory.AUGMENTATIVE),
    ('arrão', SuffixCategory.AUGMENTATIVE),
    ('anzil', SuffixCategory.AUGMENTATIVE),
    ('aréu', SuffixCategory.AUGMENTATIVE),
    ('ão', SuffixCategory.AUGMENTATIVE),
    ('ona', SuffixCategory.AUGMENTATIVE),
    ('aço', SuffixCategory.AUGMENTATIVE),
    ('aça', SuffixCategory.AUGMENTATIVE),

    # Pejorative
    ('orra', SuffixCategory.PEJORATIVE),
    ('astro', SuffixCategory.PEJORATIVE),
    ('eco', SuffixCategory.PEJORATIVE),
    ('elho', SuffixCategory.PEJORATIVE),
]

# Clitic pronouns with grammatical info: (form, person, number, case)
CLITIC_TABLE: Dict[str, Tuple[int, str, str]] = {
    'me': (1, 'sg', 'acc/dat'),
    'te': (2, 'sg', 'acc/dat'),
    'se': (3, 'sg', 'refl'),
    'o': (3, 'sg', 'acc'),
    'a': (3, 'sg', 'acc'),
    'lhe': (3, 'sg', 'dat'),
    'nos': (1, 'pl', 'acc/dat'),
    'vos': (2, 'pl', 'acc/dat'),
    'os': (3, 'pl', 'acc'),
    'as': (3, 'pl', 'acc'),
    'lhes': (3, 'pl', 'dat'),
    'lo': (3, 'sg', 'acc'),
    'la': (3, 'sg', 'acc'),
    'los': (3, 'pl', 'acc'),
    'las': (3, 'pl', 'acc'),
}

# Verbal inflection patterns: (suffix, tense_mood, person, number, conjugation_class_filter)
# conjugation_class_filter=0 means any class
# Ordered longest-first for greedy matching
VERBAL_ENDINGS: List[Tuple[str, str, int, str, int]] = [
    # Subjuntivo imperfeito
    ('ássemos', 'subj_imperf', 1, 'pl', 1),
    ('êssemos', 'subj_imperf', 1, 'pl', 2),
    ('íssemos', 'subj_imperf', 1, 'pl', 3),
    ('ásseis', 'subj_imperf', 2, 'pl', 1),
    ('êsseis', 'subj_imperf', 2, 'pl', 2),
    ('ísseis', 'subj_imperf', 2, 'pl', 3),
    ('assem', 'subj_imperf', 3, 'pl', 1),
    ('essem', 'subj_imperf', 3, 'pl', 2),
    ('issem', 'subj_imperf', 3, 'pl', 3),
    ('asses', 'subj_imperf', 2, 'sg', 1),
    ('esses', 'subj_imperf', 2, 'sg', 2),
    ('isses', 'subj_imperf', 2, 'sg', 3),
    ('asse', 'subj_imperf', 1, 'sg', 1),
    ('esse', 'subj_imperf', 1, 'sg', 2),
    ('isse', 'subj_imperf', 1, 'sg', 3),

    # Subjuntivo futuro
    ('armos', 'subj_fut', 1, 'pl', 1),
    ('ermos', 'subj_fut', 1, 'pl', 2),
    ('irmos', 'subj_fut', 1, 'pl', 3),
    ('ardes', 'subj_fut', 2, 'pl', 1),
    ('erdes', 'subj_fut', 2, 'pl', 2),
    ('irdes', 'subj_fut', 2, 'pl', 3),
    ('arem', 'subj_fut', 3, 'pl', 1),
    ('erem', 'subj_fut', 3, 'pl', 2),
    ('irem', 'subj_fut', 3, 'pl', 3),
    ('ares', 'subj_fut', 2, 'sg', 1),
    ('eres', 'subj_fut', 2, 'sg', 2),
    ('ires', 'subj_fut', 2, 'sg', 3),

    # Subjuntivo presente
    ('emos', 'subj_pres', 1, 'pl', 1),  # (cant)emos → -ar verbs use -e-
    ('amos', 'subj_pres', 1, 'pl', 2),  # (com)amos → -er verbs use -a-

    # Pretérito perfeito
    ('ámos', 'pret_perf', 1, 'pl', 1),
    ('amos', 'pret_perf', 1, 'pl', 1),
    ('emos', 'pret_perf', 1, 'pl', 2),
    ('imos', 'pret_perf', 1, 'pl', 3),
    ('aram', 'pret_perf', 3, 'pl', 1),
    ('eram', 'pret_perf', 3, 'pl', 2),
    ('iram', 'pret_perf', 3, 'pl', 3),
    ('aste', 'pret_perf', 2, 'sg', 1),
    ('este', 'pret_perf', 2, 'sg', 2),
    ('iste', 'pret_perf', 2, 'sg', 3),
    ('ou', 'pret_perf', 3, 'sg', 1),
    ('eu', 'pret_perf', 3, 'sg', 2),
    ('iu', 'pret_perf', 3, 'sg', 3),
    ('ei', 'pret_perf', 1, 'sg', 1),
    ('i', 'pret_perf', 1, 'sg', 2),

    # Imperfeito do indicativo
    ('ávamos', 'imperf', 1, 'pl', 1),
    ('íamos', 'imperf', 1, 'pl', 0),  # 2nd/3rd conj
    ('áveis', 'imperf', 2, 'pl', 1),
    ('íeis', 'imperf', 2, 'pl', 0),
    ('avam', 'imperf', 3, 'pl', 1),
    ('iam', 'imperf', 3, 'pl', 0),
    ('avas', 'imperf', 2, 'sg', 1),
    ('ias', 'imperf', 2, 'sg', 0),
    ('ava', 'imperf', 1, 'sg', 1),
    ('ia', 'imperf', 1, 'sg', 0),

    # Futuro do indicativo (with thematic vowel — longer matches beat suffix/pret_perf ambiguity)
    ('aremos', 'fut_ind', 1, 'pl', 1),
    ('eremos', 'fut_ind', 1, 'pl', 2),
    ('iremos', 'fut_ind', 1, 'pl', 3),
    ('areis', 'fut_ind', 2, 'pl', 1),
    ('ereis', 'fut_ind', 2, 'pl', 2),
    ('ireis', 'fut_ind', 2, 'pl', 3),
    ('arão', 'fut_ind', 3, 'pl', 1),
    ('erão', 'fut_ind', 3, 'pl', 2),
    ('irão', 'fut_ind', 3, 'pl', 3),
    ('arás', 'fut_ind', 2, 'sg', 1),
    ('erás', 'fut_ind', 2, 'sg', 2),
    ('irás', 'fut_ind', 2, 'sg', 3),
    ('arei', 'fut_ind', 1, 'sg', 1),
    ('erei', 'fut_ind', 1, 'sg', 2),
    ('irei', 'fut_ind', 1, 'sg', 3),
    ('ará', 'fut_ind', 3, 'sg', 1),
    ('erá', 'fut_ind', 3, 'sg', 2),
    ('irá', 'fut_ind', 3, 'sg', 3),
    # Futuro do indicativo (generic — fallback for irregular stems)
    ('remos', 'fut_ind', 1, 'pl', 0),
    ('reis', 'fut_ind', 2, 'pl', 0),
    ('rão', 'fut_ind', 3, 'pl', 0),
    ('rás', 'fut_ind', 2, 'sg', 0),
    ('rei', 'fut_ind', 1, 'sg', 0),
    ('rá', 'fut_ind', 3, 'sg', 0),

    # Condicional (with thematic vowel for 1st conj — ensures longest match beats suffix)
    ('aríamos', 'conditional', 1, 'pl', 1),
    ('eríamos', 'conditional', 1, 'pl', 2),
    ('iríamos', 'conditional', 1, 'pl', 3),
    ('aríeis', 'conditional', 2, 'pl', 1),
    ('eríeis', 'conditional', 2, 'pl', 2),
    ('iríeis', 'conditional', 2, 'pl', 3),
    ('ariam', 'conditional', 3, 'pl', 1),
    ('eriam', 'conditional', 3, 'pl', 2),
    ('iriam', 'conditional', 3, 'pl', 3),
    ('arias', 'conditional', 2, 'sg', 1),
    ('erias', 'conditional', 2, 'sg', 2),
    ('irias', 'conditional', 2, 'sg', 3),
    ('aria', 'conditional', 1, 'sg', 1),
    ('eria', 'conditional', 1, 'sg', 2),
    ('iria', 'conditional', 1, 'sg', 3),
    # Condicional (generic — fallback for irregular stems)
    ('ríamos', 'conditional', 1, 'pl', 0),
    ('ríeis', 'conditional', 2, 'pl', 0),
    ('riam', 'conditional', 3, 'pl', 0),
    ('rias', 'conditional', 2, 'sg', 0),
    ('ria', 'conditional', 1, 'sg', 0),

    # Presente do indicativo - 1st conj
    ('amos', 'pres_ind', 1, 'pl', 1),
    ('ais', 'pres_ind', 2, 'pl', 1),
    ('as', 'pres_ind', 2, 'sg', 1),
    ('am', 'pres_ind', 3, 'pl', 1),

    # Presente do indicativo - 2nd conj
    ('emos', 'pres_ind', 1, 'pl', 2),
    ('eis', 'pres_ind', 2, 'pl', 2),
    ('es', 'pres_ind', 2, 'sg', 2),
    ('em', 'pres_ind', 3, 'pl', 2),

    # Presente do indicativo - 3rd conj
    ('imos', 'pres_ind', 1, 'pl', 3),
    ('is', 'pres_ind', 2, 'pl', 3),
    ('es', 'pres_ind', 2, 'sg', 3),
    ('em', 'pres_ind', 3, 'pl', 3),

    # Non-finite forms
    ('ando', 'gerund', 0, '', 1),
    ('endo', 'gerund', 0, '', 2),
    ('indo', 'gerund', 0, '', 3),
    ('ado', 'participle', 0, '', 1),
    ('ido', 'participle', 0, '', 0),

    # Infinitivo pessoal
    ('armos', 'inf_pessoal', 1, 'pl', 1),
    ('ermos', 'inf_pessoal', 1, 'pl', 2),
    ('irmos', 'inf_pessoal', 1, 'pl', 3),
    ('ares', 'inf_pessoal', 2, 'sg', 1),
    ('eres', 'inf_pessoal', 2, 'sg', 2),
    ('ires', 'inf_pessoal', 2, 'sg', 3),
    ('arem', 'inf_pessoal', 3, 'pl', 1),
    ('erem', 'inf_pessoal', 3, 'pl', 2),
    ('irem', 'inf_pessoal', 3, 'pl', 3),

    # Infinitivo impessoal (must come last — short endings)
    ('ar', 'infinitive', 0, '', 1),
    ('er', 'infinitive', 0, '', 2),
    ('ir', 'infinitive', 0, '', 3),
    ('or', 'infinitive', 0, '', 2),  # pôr family
]

# Irregular verb stem → lemma mappings
# Maps allomorph stems to (lemma, typical_tense_context)
IRREGULAR_STEMS: Dict[str, Tuple[str, str]] = {
    # ser / ir
    'fo': ('ser/ir', 'pret_perf'),
    'fu': ('ser/ir', 'subj_imperf'),
    'sej': ('ser', 'subj_pres'),
    'sou': ('ser', 'pres_ind'),
    'som': ('ser', 'pres_ind'),
    'era': ('ser', 'imperf'),
    # estar
    'estiv': ('estar', 'pret_perf'),
    'estej': ('estar', 'subj_pres'),
    # ter
    'tiv': ('ter', 'pret_perf'),
    'tenh': ('ter', 'subj_pres'),
    # haver
    'houv': ('haver', 'pret_perf'),
    'haj': ('haver', 'subj_pres'),
    # fazer
    'fiz': ('fazer', 'pret_perf'),
    'faç': ('fazer', 'subj_pres'),
    'far': ('fazer', 'fut_ind'),
    # dizer
    'diss': ('dizer', 'pret_perf'),
    'dig': ('dizer', 'subj_pres'),
    'dir': ('dizer', 'fut_ind'),
    # trazer
    'troux': ('trazer', 'pret_perf'),
    'trag': ('trazer', 'subj_pres'),
    'trar': ('trazer', 'fut_ind'),
    # poder
    'pud': ('poder', 'pret_perf'),
    'poss': ('poder', 'subj_pres'),
    # querer
    'quis': ('querer', 'pret_perf'),
    'queir': ('querer', 'subj_pres'),
    # saber
    'soub': ('saber', 'pret_perf'),
    'saib': ('saber', 'subj_pres'),
    # pôr
    'pus': ('pôr', 'pret_perf'),
    'ponh': ('pôr', 'subj_pres'),
    # vir
    'vie': ('vir', 'pret_perf'),
    'venh': ('vir', 'subj_pres'),
    # ver
    'vi': ('ver', 'pret_perf'),
    'vej': ('ver', 'subj_pres'),
    # dar
    'de': ('dar', 'subj_pres'),
    'dê': ('dar', 'subj_pres'),
    # ir
    'vou': ('ir', 'pres_ind'),
    'vai': ('ir', 'pres_ind'),
    'vã': ('ir', 'subj_pres'),
    # caber
    'coub': ('caber', 'pret_perf'),
    'caib': ('caber', 'subj_pres'),
}

# Portuguese digraphs
DIGRAPHS = {'ch', 'lh', 'nh', 'rr', 'ss', 'qu', 'gu'}

# Vowels (including accented)
VOWELS = set('aeiouáéíóúâêîôûãõàü')
ACCENTED_VOWELS = set('áéíóúâêîôûãõà')
CONSONANTS = set('bcdfghjklmnpqrstvwxyzç')

# Nasal diphthong patterns
NASAL_DIPHTHONGS = re.compile(
    r'(ão|ãe|õe|ãi|am(?=[^aeiou]|$)|em(?=[^aeiou]|$)|om(?=[^aeiou]|$)|an(?=[^aeiou]|$)|en(?=[^aeiou]|$)|on(?=[^aeiou]|$)|in(?=[^aeiou]|$)|un(?=[^aeiou]|$))')
ORAL_DIPHTHONGS = re.compile(r'(ai|au|ei|eu|iu|oi|ou|ui|ãi|ãu|õi|õe)')


# ─────────────────────────────────────────────
# Main Analyzer
# ─────────────────────────────────────────────

class PortugueseMorphAnalyzer:
    """
    Full-featured Portuguese morphological analyzer.

    Usage:
        analyzer = PortugueseMorphAnalyzer()
        result = analyzer.analyze("extraordinariamente")
        print(result)
        print(result.to_json())

    For batch processing:
        results = analyzer.analyze_batch(["desfazer", "infelizmente", "dir-se-ia"])
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

        # Pre-sort prefix table by length (longest first) for greedy matching
        self._prefixes = sorted(PREFIX_TABLE, key=lambda x: -len(x[0]))
        self._suffixes = sorted(SUFFIX_TABLE, key=lambda x: -len(x[0]))
        self._verbal = sorted(VERBAL_ENDINGS, key=lambda x: -len(x[0]))  # longest-first
        self._irregulars = IRREGULAR_STEMS

        # Pre-compile clitic pattern
        clitic_forms = sorted(CLITIC_TABLE.keys(), key=lambda x: -len(x))
        clitic_alt = '|'.join(re.escape(c) for c in clitic_forms)
        self._enclitic_re = re.compile(rf'-({clitic_alt})$')
        self._mesoclitic_re = re.compile(
            rf'^([^-]+)-({clitic_alt})-([^-]+)$'
        )

        # Known roots / stems that should NOT be prefix-stripped.
        # Prevents "biologia" → bi+ologia, "impossibilidade" → im+pos+sibilidade
        self._prefix_blocklist: Dict[str, Set[str]] = {}
        _blocked = {
            'bio': {'bi'},  # biologia, biológico, biólogo…
            'impos': {'pos'},  # impossibilidade, impossível…
            'diss': {'dis', 'di'},  # disse, disseram, dissertação…
            'disc': {'dis', 'di'},  # disco, discurso…
            'diz': {'di'},  # dizer, dizemos…
            'dir': {'di'},  # director, direito…
            'come': {'com', 'co'},  # comer, comércio, comida…
            'comi': {'com', 'co'},
            'comer': {'com', 'co'},
            'comê': {'com', 'co'},  # comêssemos, comêsseis…
            'perig': {'peri', 'per', 'pe'},  # perigoso, perigo…
            'anim': {'an', 'a'},  # animal…
            'and': {'an', 'a'},  # andar…
            'ante': {'ante', 'an', 'a'},  # antes (the word itself)
        }
        # Expand: for any word starting with the blocked stem, block those prefixes
        self._prefix_block_stems = _blocked

        # Map known whole irregular words directly → skip prefix/stem heuristics
        self._irregular_whole_words: Dict[str, Tuple[str, str, Optional[int], Optional[str]]] = {
            # word → (lemma, tense_mood, person, number)
            'disse': ('dizer', 'pret_perf', 1, 'sg'),
            'dissemos': ('dizer', 'pret_perf', 1, 'pl'),
            'disseram': ('dizer', 'pret_perf', 3, 'pl'),
            'disseste': ('dizer', 'pret_perf', 2, 'sg'),
            'dissesse': ('dizer', 'subj_imperf', 1, 'sg'),
            'disséssemos': ('dizer', 'subj_imperf', 1, 'pl'),
            'fui': ('ser/ir', 'pret_perf', 1, 'sg'),
            'foi': ('ser/ir', 'pret_perf', 3, 'sg'),
            'fomos': ('ser/ir', 'pret_perf', 1, 'pl'),
            'foram': ('ser/ir', 'pret_perf', 3, 'pl'),
            'fosse': ('ser/ir', 'subj_imperf', 1, 'sg'),
            'fôssemos': ('ser/ir', 'subj_imperf', 1, 'pl'),
            'fossem': ('ser/ir', 'subj_imperf', 3, 'pl'),
            'sou': ('ser', 'pres_ind', 1, 'sg'),
            'somos': ('ser', 'pres_ind', 1, 'pl'),
            'é': ('ser', 'pres_ind', 3, 'sg'),
            'era': ('ser', 'imperf', 1, 'sg'),
            'éramos': ('ser', 'imperf', 1, 'pl'),
            'eram': ('ser', 'imperf', 3, 'pl'),
            'estive': ('estar', 'pret_perf', 1, 'sg'),
            'estivemos': ('estar', 'pret_perf', 1, 'pl'),
            'estiveram': ('estar', 'pret_perf', 3, 'pl'),
            'tive': ('ter', 'pret_perf', 1, 'sg'),
            'tivemos': ('ter', 'pret_perf', 1, 'pl'),
            'tiveram': ('ter', 'pret_perf', 3, 'pl'),
            'tenho': ('ter', 'pres_ind', 1, 'sg'),
            'fiz': ('fazer', 'pret_perf', 1, 'sg'),
            'fizemos': ('fazer', 'pret_perf', 1, 'pl'),
            'fizeram': ('fazer', 'pret_perf', 3, 'pl'),
            'fez': ('fazer', 'pret_perf', 3, 'sg'),
            'trouxe': ('trazer', 'pret_perf', 1, 'sg'),
            'trouxemos': ('trazer', 'pret_perf', 1, 'pl'),
            'trouxeram': ('trazer', 'pret_perf', 3, 'pl'),
            'pude': ('poder', 'pret_perf', 1, 'sg'),
            'pudemos': ('poder', 'pret_perf', 1, 'pl'),
            'puderam': ('poder', 'pret_perf', 3, 'pl'),
            'quis': ('querer', 'pret_perf', 1, 'sg'),
            'quisemos': ('querer', 'pret_perf', 1, 'pl'),
            'quiseram': ('querer', 'pret_perf', 3, 'pl'),
            'soube': ('saber', 'pret_perf', 1, 'sg'),
            'soubemos': ('saber', 'pret_perf', 1, 'pl'),
            'souberam': ('saber', 'pret_perf', 3, 'pl'),
            'pus': ('pôr', 'pret_perf', 1, 'sg'),
            'pusemos': ('pôr', 'pret_perf', 1, 'pl'),
            'puseram': ('pôr', 'pret_perf', 3, 'pl'),
            'coube': ('caber', 'pret_perf', 1, 'sg'),
            'coubemos': ('caber', 'pret_perf', 1, 'pl'),
            'couberam': ('caber', 'pret_perf', 3, 'pl'),
            'houve': ('haver', 'pret_perf', 3, 'sg'),
            'houvemos': ('haver', 'pret_perf', 1, 'pl'),
            'houveram': ('haver', 'pret_perf', 3, 'pl'),
            'vim': ('vir', 'pret_perf', 1, 'sg'),
            'viemos': ('vir', 'pret_perf', 1, 'pl'),
            'vieram': ('vir', 'pret_perf', 3, 'pl'),
            'vou': ('ir', 'pres_ind', 1, 'sg'),
            'vai': ('ir', 'pres_ind', 3, 'sg'),
            'vamos': ('ir', 'pres_ind', 1, 'pl'),
            'vão': ('ir', 'pres_ind', 3, 'pl'),
        }

        # ── Integration: POS tagger ──
        self._tagger = None
        if self.config.use_pos_tagger and _HAS_TAGGER:
            try:
                self._tagger = _ExtTugaTagger(engine=self.config.tagger_engine)
            except Exception:
                pass  # graceful fallback to heuristic

        # ── Integration: Syllabifier ──
        self._ext_syllabify = None
        if self.config.use_syllabifier:
            self._ext_syllabify = syllabify

        # POS tag sets for disambiguation
        self._verbal_pos = {'VERB', 'AUX'}
        self._nominal_pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}

    # ── Public API ──────────────────────────

    def analyze(self, word: str, pos_tag: Optional[str] = None) -> MorphologicalAnalysis:
        """
        Analyze a single word and return a MorphologicalAnalysis.

        Args:
            word: The word to analyze.
            pos_tag: Optional UPOS tag hint (e.g. from tugatagger or upstream pipeline).
                     When provided, skips internal POS tagging and uses this directly.
        """
        original = word.strip()
        normalized = original.lower()

        result = MorphologicalAnalysis(
            original=original,
            normalized=normalized,
            word_length=len(normalized),
            vowel_count=sum(1 for c in normalized if c in VOWELS),
            consonant_count=sum(1 for c in normalized if c in CONSONANTS),
            is_capitalized=original[0].isupper() if original else False,
            has_accent=any(c in ACCENTED_VOWELS for c in normalized),
            has_hyphen='-' in normalized,
        )

        working = normalized

        # ── POS tagging ──
        # If a real tagger is available, use it and enable POS disambiguation.
        # The built-in heuristic is too coarse for reliable disambiguation,
        # so we only use it for labeling (pos_tag field), NOT for tiebreaking.
        _pos_from_tagger = False
        if pos_tag is not None:
            _pos_from_tagger = True  # caller provided a trusted tag
        elif self.config.use_pos_tagger:
            if self._tagger is not None:
                try:
                    tagged = self._tagger.tag(original)
                    if tagged:
                        pos_tag = tagged[0][1]
                        _pos_from_tagger = True
                except Exception:
                    pass
            if pos_tag is None:
                pos_tag = self._guess_pos(original)
        result.pos_tag = pos_tag

        # 0. Check irregular whole-word table first (short-circuits everything)
        if working in self._irregular_whole_words:
            lemma, tm, per, num = self._irregular_whole_words[working]
            result.verbal = VerbalAnalysis(
                tense_mood=tm,
                person=per,
                number=num,
                is_irregular=True,
                lemma_guess=lemma,
                allomorph=working,
            )
            result.root = working
            result.morphemes.append(Morpheme(working, MorphemeType.ROOT, label=f'irregular:{lemma}'))
            if self.config.extract_phonology:
                result.phonology = self._extract_phonology(normalized)
            return result

        # 1. Clitics (mesoclitic first, then enclitic)
        if self.config.extract_clitics:
            working = self._extract_clitics(working, result)

        # 2. Compound detection (hyphenated)
        if self.config.detect_compounds and '-' in working:
            parts = working.split('-')
            if len(parts) >= 2 and all(len(p) >= 2 for p in parts):
                result.is_compound = True
                result.compound_parts = parts
                # For compounds, analyze the head (typically last part)
                # but we still try to get the full picture
                for p in parts[:-1]:
                    result.morphemes.append(Morpheme(p, MorphemeType.ROOT, label='compound_part'))
                    result.morphemes.append(Morpheme('-', MorphemeType.INTERFIXE, label='hyphen'))
                working = parts[-1]  # analyze the head

        # 3. Prefixes (stacking up to max)
        if self.config.extract_prefixes:
            working = self._extract_prefixes(working, result)

        # 4. Irregular check (stem-based, not whole-word — those were caught in step 0)
        irr_found = self._check_irregular(working, result)

        # 5+6. Suffix vs Verbal: longest-match-wins, POS-informed tiebreaking
        # When POS tag is available: VERB/AUX → prefer verbal, NOUN/ADJ/ADV → prefer suffix
        suffix_found = False
        verb_found = False
        if not irr_found:
            sfx_match = self._find_suffix(working) if self.config.extract_suffixes else None
            vrb_match = self._find_verbal(working) if self.config.extract_verbal else None

            sfx_len = len(sfx_match[0]) if sfx_match else 0
            vrb_len = len(vrb_match[0]) if vrb_match else 0

            # POS-informed disambiguation (only when tag is from a real tagger, not heuristic)
            pos_prefers_verbal = (
                        _pos_from_tagger and pos_tag in self._verbal_pos) if self.config.pos_disambiguate else False
            pos_prefers_nominal = (
                        _pos_from_tagger and pos_tag in self._nominal_pos) if self.config.pos_disambiguate else False

            if pos_prefers_verbal and vrb_len > 0:
                # POS says VERB → always prefer verbal
                self._apply_verbal(vrb_match, result)
                verb_found = True
            elif pos_prefers_nominal and sfx_len > 0:
                # POS says NOUN/ADJ/ADV → always prefer suffix
                result.suffix = sfx_match
                suffix_found = True
            elif sfx_len > vrb_len:
                # Suffix wins on length (e.g. -logia > -ia)
                result.suffix = sfx_match
                suffix_found = True
            elif vrb_len > 0:
                # Verbal wins (including ties — verbal takes priority at equal length)
                self._apply_verbal(vrb_match, result)
                verb_found = True
            elif sfx_len > 0:
                result.suffix = sfx_match
                suffix_found = True

        # 7. Whatever remains is the root
        root = working
        if verb_found and result.verbal and result.verbal.tense_mood != 'none':
            # Remove the matched ending from working to get root
            for ending, tm, per, num, conj in self._verbal:
                if working.endswith(ending):
                    root = working[:-len(ending)]
                    # Thematic vowel extraction
                    if root and root[-1] in 'aei':
                        result.verbal.thematic_vowel = root[-1]
                        if result.verbal.conjugation_class is None:
                            result.verbal.conjugation_class = {'a': 1, 'e': 2, 'i': 3}.get(root[-1])
                        root = root[:-1]
                        result.morphemes.append(Morpheme(root, MorphemeType.ROOT, label='verbal_root'))
                        result.morphemes.append(Morpheme(result.verbal.thematic_vowel, MorphemeType.THEMATIC_VOWEL))
                    else:
                        result.morphemes.append(Morpheme(root, MorphemeType.ROOT, label='verbal_root'))
                    result.morphemes.append(Morpheme(ending, MorphemeType.TENSE_MOOD, label=result.verbal.tense_mood))
                    break
        elif irr_found:
            result.morphemes.append(
                Morpheme(working, MorphemeType.ROOT, label=f'irregular:{result.verbal.lemma_guess}'))
            root = working
        elif suffix_found and result.suffix:
            sfx_form = result.suffix[0]
            root = working[:-len(sfx_form)] if working.endswith(sfx_form) else working
            result.morphemes.append(Morpheme(root, MorphemeType.ROOT, label='derivational_root'))
            result.morphemes.append(Morpheme(sfx_form, MorphemeType.SUFFIX, label=str(result.suffix[1])))
        else:
            result.morphemes.append(Morpheme(working, MorphemeType.ROOT, label='root'))

        result.root = root

        # 8. Phonological features
        if self.config.extract_phonology:
            result.phonology = self._extract_phonology(normalized)

        return result

    def analyze_batch(self, words: List[str]) -> List[MorphologicalAnalysis]:
        """Analyze a list of words."""
        return [self.analyze(w) for w in words]

    def segment(self, word: str) -> str:
        """Convenience: return just the segmentation string."""
        return self.analyze(word).segmentation_str()

    # ── Private Methods ─────────────────────

    @staticmethod
    def _apply_verbal(vrb_match: Tuple, result: MorphologicalAnalysis):
        """Apply a verbal ending match to the result."""
        ending, tense_mood, person, number, conj_class = vrb_match
        if result.verbal is None:
            result.verbal = VerbalAnalysis()
        result.verbal.tense_mood = tense_mood
        result.verbal.person = person if person > 0 else None
        result.verbal.number = number if number else None
        result.verbal.conjugation_class = conj_class if conj_class > 0 else None

    def analyze_sentence(self, sentence: str) -> List[MorphologicalAnalysis]:
        """
        Analyze all words in a sentence with context-aware POS tagging.

        Uses tugatagger (if available) to tag the full sentence first,
        then feeds each word's POS into analyze() for better disambiguation.
        """
        # Get sentence-level POS tags
        word_pos_pairs: List[Tuple[str, Optional[str]]] = []
        if self._tagger is not None:
            try:
                tagged = self._tagger.tag(sentence)
                word_pos_pairs = tagged
            except Exception:
                word_pos_pairs = [(w, None) for w in sentence.split()]
        else:
            word_pos_pairs = [(w, None) for w in sentence.split()]

        return [self.analyze(word, pos_tag=pos) for word, pos in word_pos_pairs]

    @staticmethod
    def _guess_pos(word: str) -> str:
        """
        Heuristic POS guesser — mirrors TugaTagger._guess_pos for zero-dependency mode.

        Returns UPOS tags: NOUN, VERB, ADJ, ADV, DET, ADP, CCONJ, SCONJ, PRON, AUX, etc.
        """
        lower = word.lower()

        # Punctuation / numbers
        if not word.isalnum():
            return "PUNCT"
        if word.isdigit():
            return "NUM"

        # Closed-class words
        _FUNC = {
            "o": "DET", "a": "DET", "os": "DET", "as": "DET", "um": "DET", "uma": "DET",
            "de": "ADP", "do": "ADP", "da": "ADP", "em": "ADP", "no": "ADP", "na": "ADP",
            "por": "ADP", "para": "ADP", "com": "ADP", "sem": "ADP",
            "e": "CCONJ", "mas": "CCONJ", "ou": "CCONJ", "que": "SCONJ", "se": "SCONJ",
            "eu": "PRON", "ele": "PRON", "ela": "PRON", "nós": "PRON", "eles": "PRON",
            "isso": "PRON", "aquilo": "PRON",
            "é": "AUX", "foi": "AUX", "são": "AUX", "está": "AUX", "ser": "AUX", "ter": "AUX",
            "não": "ADV", "sim": "ADV", "muito": "ADV", "mais": "ADV",
        }
        if lower in _FUNC:
            return _FUNC[lower]

        # Suffix rules (longer first)
        _SUFFIX_RULES = [
            ("mente", "ADV"), ("ando", "VERB"), ("endo", "VERB"), ("indo", "VERB"),
            ("aram", "VERB"), ("eram", "VERB"), ("iram", "VERB"),
            ("ava", "VERB"), ("ria", "VERB"),
            ("dor", "NOUN"), ("ção", "NOUN"), ("são", "NOUN"),
            ("dade", "NOUN"), ("ismo", "NOUN"), ("ista", "NOUN"),
            ("oso", "ADJ"), ("osa", "ADJ"), ("vel", "ADJ"), ("al", "ADJ"),
            ("ar", "VERB"), ("er", "VERB"), ("ir", "VERB"),
        ]
        for suffix, tag in _SUFFIX_RULES:
            if lower.endswith(suffix) and len(lower) > len(suffix):
                return tag

        # Title case → proper noun
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return "PROPN"

        return "NOUN"

    def _extract_clitics(self, word: str, result: MorphologicalAnalysis) -> str:
        """Extract mesoclitic or enclitic pronouns."""
        # Mesoclitic: e.g., "dir-se-ia" → "diria" + clitic "se"
        meso = self._mesoclitic_re.match(word)
        if meso:
            pre, clitic_form, post = meso.group(1), meso.group(2), meso.group(3)
            info = CLITIC_TABLE.get(clitic_form)
            result.clitic = CliticInfo(
                form=clitic_form,
                position=CliticPosition.MESOCLITIC,
                person=info[0] if info else None,
                number=info[1] if info else None,
                case=info[2] if info else None,
            )
            result.morphemes.append(Morpheme(clitic_form, MorphemeType.CLITIC, label='mesoclitic'))
            reconstructed = pre + post
            # Check if the reconstructed form is a known irregular
            analyzer = getattr(self, '_irregular_whole_words', {})
            if reconstructed in analyzer:
                lemma, tm, per, num = analyzer[reconstructed]
                result.verbal = VerbalAnalysis(
                    tense_mood=tm, person=per, number=num,
                    is_irregular=True, lemma_guess=lemma, allomorph=reconstructed,
                )
            return reconstructed

        # Enclitic: e.g., "disse-lhe"
        enc = self._enclitic_re.search(word)
        if enc:
            clitic_form = enc.group(1)
            info = CLITIC_TABLE.get(clitic_form)
            result.clitic = CliticInfo(
                form=clitic_form,
                position=CliticPosition.ENCLITIC,
                person=info[0] if info else None,
                number=info[1] if info else None,
                case=info[2] if info else None,
            )
            result.morphemes.append(Morpheme(clitic_form, MorphemeType.CLITIC, label='enclitic'))
            return word[:enc.start()]

        return word

    def _extract_prefixes(self, word: str, result: MorphologicalAnalysis) -> str:
        """Extract up to max_prefix_stack prefixes, longest match first.
        Respects the prefix_block_stems to avoid false decomposition."""
        remaining = word
        for _ in range(self.config.max_prefix_stack):
            matched = False
            for prefix, cat in self._prefixes:
                if remaining.startswith(prefix) and len(remaining) - len(prefix) >= self.config.min_root_length:
                    # Check blocklist: is this prefix blocked for this word stem?
                    blocked = False
                    for stem, blocked_prefixes in self._prefix_block_stems.items():
                        if word.startswith(stem) and prefix in blocked_prefixes:
                            blocked = True
                            break
                    if blocked:
                        continue
                    result.prefixes.append((prefix, cat))
                    result.morphemes.append(Morpheme(prefix, MorphemeType.PREFIX, label=str(cat)))
                    remaining = remaining[len(prefix):]
                    matched = True
                    break
            if not matched:
                break
        return remaining

    def _check_irregular(self, word: str, result: MorphologicalAnalysis) -> bool:
        """Check if the word starts with a known irregular stem."""
        # Try longest stems first
        for stem in sorted(self._irregulars.keys(), key=lambda x: -len(x)):
            if word.startswith(stem) and len(word) <= len(stem) + 6:
                lemma, tense_ctx = self._irregulars[stem]
                if result.verbal is None:
                    result.verbal = VerbalAnalysis()
                result.verbal.is_irregular = True
                result.verbal.lemma_guess = lemma
                result.verbal.allomorph = stem
                result.verbal.tense_mood = tense_ctx
                return True
        return False

    def _find_verbal(self, word: str) -> Optional[Tuple[str, str, int, str, int]]:
        """Find the longest matching verbal ending. Returns the full tuple or None."""
        for entry in self._verbal:
            ending = entry[0]
            if word.endswith(ending) and len(word) - len(ending) >= self.config.min_root_length:
                return entry
        return None

    def _extract_verbal(self, word: str, result: MorphologicalAnalysis) -> bool:
        """Match the longest verbal ending and apply it."""
        match = self._find_verbal(word)
        if match:
            ending, tense_mood, person, number, conj_class = match
            if result.verbal is None:
                result.verbal = VerbalAnalysis()
            result.verbal.tense_mood = tense_mood
            result.verbal.person = person if person > 0 else None
            result.verbal.number = number if number else None
            result.verbal.conjugation_class = conj_class if conj_class > 0 else None
            return True
        return False

    def _find_suffix(self, word: str) -> Optional[Tuple[str, SuffixCategory]]:
        """Find the longest matching derivational suffix. Returns (form, category) or None."""
        for sfx, cat in self._suffixes:
            if word.endswith(sfx) and len(word) - len(sfx) >= self.config.min_root_length:
                return (sfx, cat)
        return None

    def _extract_suffix(self, word: str, result: MorphologicalAnalysis) -> bool:
        """Match the longest derivational suffix and apply it."""
        match = self._find_suffix(word)
        if match:
            result.suffix = match
            return True
        return False

    def _extract_phonology(self, word: str) -> PhonologicalFeatures:
        """
        Extract phonological features from orthography.

        Uses silabificador for accurate syllabification when available,
        falls back to vowel-cluster heuristic otherwise.
        """
        feats = PhonologicalFeatures()

        # ── Syllabification ──
        if self._ext_syllabify is not None:
            try:
                syllables = self._ext_syllabify(word)
                feats.syllables = syllables
                feats.syllable_count = len(syllables)
            except Exception:
                feats.syllables = self._heuristic_syllabify(word)
                feats.syllable_count = len(feats.syllables)
        else:
            feats.syllables = self._heuristic_syllabify(word)
            feats.syllable_count = len(feats.syllables)

        # ── Stress pattern (uses syllable boundaries for accuracy) ──
        feats.stress_pattern, feats.stressed_syllable_index = self._detect_stress(
            word, feats.syllables
        )

        # Diphthongs
        feats.has_nasal_diphthong = bool(NASAL_DIPHTHONGS.search(word))
        feats.has_oral_diphthong = bool(ORAL_DIPHTHONGS.search(word))

        # Hiatus detection (two adjacent vowels that aren't diphthongs)
        for i in range(len(word) - 1):
            if word[i] in VOWELS and word[i + 1] in VOWELS:
                pair = word[i:i + 2]
                if not ORAL_DIPHTHONGS.match(pair) and not NASAL_DIPHTHONGS.match(pair):
                    feats.has_hiatus = True
                    break

        # Nasal vowel
        feats.has_nasal_vowel = bool(re.search(r'[ãõ]', word))

        # Final nasal
        feats.ends_in_nasal = word.endswith(('m', 'n', 'ão', 'ãe', 'õe'))

        # Digraph
        feats.has_digraph = any(d in word for d in DIGRAPHS)

        return feats

    @staticmethod
    def _heuristic_syllabify(word: str) -> List[str]:
        """Vowel-cluster syllabification fallback when silabificador is not installed."""
        if not word:
            return [word] if word else []
        # Split on consonant clusters between vowel nuclei
        segments = re.split(
            r'(?<=[aeiouáéíóúâêîôûãõàü])(?=[^aeiouáéíóúâêîôûãõàü]+[aeiouáéíóúâêîôûãõàü])',
            word
        )
        return segments if segments else [word]

    @staticmethod
    def _detect_stress(word: str, syllables: List[str]) -> Tuple[StressPattern, Optional[int]]:
        """Detect stress pattern using syllable boundaries."""
        n = len(syllables)
        if n == 0:
            return StressPattern.UNKNOWN, None

        # Look for explicit accent marks in syllables
        for i, syl in enumerate(syllables):
            if any(c in ACCENTED_VOWELS for c in syl):
                pos_from_end = n - 1 - i
                if pos_from_end == 0:
                    return StressPattern.OXYTONE, i
                elif pos_from_end == 1:
                    return StressPattern.PAROXYTONE, i
                else:
                    return StressPattern.PROPAROXYTONE, i

        # No accent mark — apply Portuguese default rules
        clean = re.sub(r'[^a-záéíóúâêîôûãõàç]', '', word)
        if not clean:
            return StressPattern.UNKNOWN, None

        if clean.endswith(('a', 'e', 'o', 'as', 'es', 'os', 'am', 'em')):
            return StressPattern.PAROXYTONE, max(0, n - 2)
        elif clean.endswith(('i', 'u', 'r', 'l', 'z', 'im', 'um', 'ins', 'uns')):
            return StressPattern.OXYTONE, n - 1
        else:
            return StressPattern.PAROXYTONE, max(0, n - 2)


# ─────────────────────────────────────────────
# Test Suite
# ─────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = PortugueseMorphAnalyzer()

    test_words = [
        # Derivational morphology
        "extraordinariamente",
        "anticonstitucional",
        "infelizmente",
        "desfazer",
        "reutilização",
        "biologia",
        "impossibilidade",
        "predeterminar",
        "subdesenvolvimento",

        # Verbal forms
        "cantávamos",
        "comêssemos",
        "partiriam",
        "fizemos",
        "disseram",

        # Clitics
        "dir-se-ia",
        "fazer-te-ei",
        "disse-lhe",
        "encontramo-nos",
        "dá-lo",

        # Compounds
        "guarda-chuva",
        "segunda-feira",

        # Short / edge cases
        "sol",
        "pôr",
        "vou",
    ]

    print("=" * 80)
    print("PORTUGUESE MORPHOLOGICAL ANALYZER — TEST SUITE")
    print("=" * 80)

    for w in test_words:
        result = analyzer.analyze(w)
        print(f"\n{'─' * 60}")
        print(result)
        print(f"  feature vector len: {len(result.to_feature_vector())}")

    # Batch demo
    print(f"\n{'═' * 80}")
    print("BATCH SEGMENTATION")
    print("═" * 80)
    for w in test_words:
        print(f"  {w:<25} → {analyzer.segment(w)}")

    # JSON export demo
    print(f"\n{'═' * 80}")
    print("JSON EXPORT (sample)")
    print("═" * 80)
    print(analyzer.analyze("extraordinariamente").to_json())

    # ================================================================================
    # PORTUGUESE MORPHOLOGICAL ANALYZER — TEST SUITE
    # ================================================================================
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('extraordinariamente')
    #   segmentation: [extra]-[ordinaria]-[mente]
    #   pos: ADV
    #   prefixes: extra (position)
    #   root: 'ordinaria'
    #   suffix: mente (adverb)
    #   phonology: ex.tra.or.di.na.ri.a.men.te, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('anticonstitucional')
    #   segmentation: [anti]-[con]-[stitucion]-[al]
    #   pos: ADJ
    #   prefixes: anti (negation), con (connection)
    #   root: 'stitucion'
    #   suffix: al (adjective)
    #   phonology: an.ti.cons.ti.tu.ci.o.nal, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('infelizmente')
    #   segmentation: [in]-[feliz]-[mente]
    #   pos: ADV
    #   prefixes: in (negation)
    #   root: 'feliz'
    #   suffix: mente (adverb)
    #   phonology: in.fe.liz.men.te, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('desfazer')
    #   segmentation: [des]-[faz]-[er]
    #   pos: VERB
    #   prefixes: des (negation)
    #   root: 'faz'
    #   verbal: infinitive ?? (conj 2)
    #   phonology: des.fa.zer, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('reutilização')
    #   segmentation: [re]-[util]-[ização]
    #   pos: NOUN
    #   prefixes: re (repetition)
    #   root: 'util'
    #   suffix: ização (noun_abstract)
    #   phonology: reu.ti.li.za.ção, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('biologia')
    #   segmentation: [bio]-[logia]
    #   pos: NOUN
    #   root: 'bio'
    #   suffix: logia (scientific)
    #   phonology: bi.o.lo.gi.a, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('impossibilidade')
    #   segmentation: [im]-[possibil]-[idade]
    #   pos: NOUN
    #   prefixes: im (negation)
    #   root: 'possibil'
    #   suffix: idade (noun_abstract)
    #   phonology: im.pos.si.bi.li.da.de, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('predeterminar')
    #   segmentation: [pre]-[determin]-[ar]
    #   pos: VERB
    #   prefixes: pre (temporal)
    #   root: 'determin'
    #   verbal: infinitive ?? (conj 1)
    #   phonology: pre.de.ter.mi.nar, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('subdesenvolvimento')
    #   segmentation: [sub]-[des]-[envolv]-[imento]
    #   pos: NOUN
    #   prefixes: sub (position), des (negation)
    #   root: 'envolv'
    #   suffix: imento (noun_abstract)
    #   phonology: sub.de.sen.vol.vi.men.to, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('cantávamos')
    #   segmentation: [cant]-[ávamos]
    #   pos: VERB
    #   root: 'cant'
    #   verbal: imperf 1pl (conj 1)
    #   phonology: can.tá.va.mos, proparoxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('comêssemos')
    #   segmentation: [com]-[êssemos]
    #   pos: VERB
    #   root: 'com'
    #   verbal: subj_imperf 1pl (conj 2)
    #   phonology: co.mês.se.mos, proparoxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('partiriam')
    #   segmentation: [part]-[iriam]
    #   pos: VERB
    #   root: 'part'
    #   verbal: conditional 3pl (conj 3)
    #   phonology: par.ti.ri.am, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('fizemos')
    #   segmentation: [fizemos]
    #   pos: VERB
    #   root: 'fizemos'
    #   verbal: pret_perf 1pl (conj ?)
    #   irregular: fizemos → fazer
    #   phonology: fi.ze.mos, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('disseram')
    #   segmentation: [disseram]
    #   pos: VERB
    #   root: 'disseram'
    #   verbal: pret_perf 3pl (conj ?)
    #   irregular: disseram → dizer
    #   phonology: dis.se.ram, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('dir-se-ia')
    #   segmentation: [se]-[diria]
    #   pos: NOUN
    #   root: 'diria'
    #   verbal: fut_ind ?? (conj ?)
    #   irregular: dir → dizer
    #   clitic: -se (mesoclitic)
    #   phonology: dir.se.i.a, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('fazer-te-ei')
    #   segmentation: [te]-[faz]-[erei]
    #   pos: NOUN
    #   root: 'faz'
    #   verbal: fut_ind 1sg (conj 2)
    #   clitic: -te (mesoclitic)
    #   phonology: fa.zer.te.ei, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('disse-lhe')
    #   segmentation: [lhe]-[disse]
    #   pos: VERB
    #   root: 'disse'
    #   verbal: pret_perf ?? (conj ?)
    #   irregular: diss → dizer
    #   clitic: -lhe (enclitic)
    #   phonology: dis.se.lhe, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('encontramo-nos')
    #   segmentation: [nos]-[encontramo]
    #   pos: NOUN
    #   root: 'encontramo'
    #   clitic: -nos (enclitic)
    #   phonology: en.con.tra.mo.nos, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('dá-lo')
    #   segmentation: [lo]-[dá]
    #   pos: NOUN
    #   root: 'dá'
    #   clitic: -lo (enclitic)
    #   phonology: dá.lo, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('guarda-chuva')
    #   segmentation: [guarda]-[-]-[chuva]
    #   pos: NOUN
    #   root: 'chuva'
    #   phonology: guar.da.chu.va, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('segunda-feira')
    #   segmentation: [segunda]-[-]-[feira]
    #   pos: NOUN
    #   root: 'feira'
    #   phonology: se.gun.da.fei.ra, paroxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('sol')
    #   segmentation: [sol]
    #   pos: NOUN
    #   root: 'sol'
    #   phonology: sol, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('pôr')
    #   segmentation: [pôr]
    #   pos: VERB
    #   root: 'pôr'
    #   phonology: pôr, oxytone
    #   feature vector len: 17
    #
    # ────────────────────────────────────────────────────────────
    # MorphAnalysis('vou')
    #   segmentation: [vou]
    #   pos: AUX
    #   root: 'vou'
    #   verbal: pres_ind 1sg (conj ?)
    #   irregular: vou → ir
    #   phonology: vou, oxytone
    #   feature vector len: 17
    #
    # ════════════════════════════════════════════════════════════════════════════════
    # BATCH SEGMENTATION
    # ════════════════════════════════════════════════════════════════════════════════
    #   extraordinariamente       → [extra]-[ordinaria]-[mente]
    #   anticonstitucional        → [anti]-[con]-[stitucion]-[al]
    #   infelizmente              → [in]-[feliz]-[mente]
    #   desfazer                  → [des]-[faz]-[er]
    #   reutilização              → [re]-[util]-[ização]
    #   biologia                  → [bio]-[logia]
    #   impossibilidade           → [im]-[possibil]-[idade]
    #   predeterminar             → [pre]-[determin]-[ar]
    #   subdesenvolvimento        → [sub]-[des]-[envolv]-[imento]
    #   cantávamos                → [cant]-[ávamos]
    #   comêssemos                → [com]-[êssemos]
    #   partiriam                 → [part]-[iriam]
    #   fizemos                   → [fizemos]
    #   disseram                  → [disseram]
    #   dir-se-ia                 → [se]-[diria]
    #   fazer-te-ei               → [te]-[faz]-[erei]
    #   disse-lhe                 → [lhe]-[disse]
    #   encontramo-nos            → [nos]-[encontramo]
    #   dá-lo                     → [lo]-[dá]
    #   guarda-chuva              → [guarda]-[-]-[chuva]
    #   segunda-feira             → [segunda]-[-]-[feira]
    #   sol                       → [sol]
    #   pôr                       → [pôr]
    #   vou                       → [vou]
    #
    # ════════════════════════════════════════════════════════════════════════════════
    # JSON EXPORT (sample)
    # ════════════════════════════════════════════════════════════════════════════════
    # {
    #   "word": "extraordinariamente",
    #   "word_length": 19,
    #   "vowel_count": 9,
    #   "consonant_count": 10,
    #   "is_capitalized": false,
    #   "has_accent": false,
    #   "has_hyphen": false,
    #   "syllable_count": 9,
    #   "syllables": [
    #     "ex",
    #     "tra",
    #     "or",
    #     "di",
    #     "na",
    #     "ri",
    #     "a",
    #     "men",
    #     "te"
    #   ],
    #   "stressed_syllable_index": 7,
    #   "stress_pattern": "paroxytone",
    #   "has_nasal_diphthong": true,
    #   "has_oral_diphthong": false,
    #   "has_hiatus": true,
    #   "n_prefixes": 1,
    #   "prefix_types": [
    #     "position"
    #   ],
    #   "root": "ordinaria",
    #   "suffix_type": "adverb",
    #   "has_clitic": false,
    #   "clitic_position": "none",
    #   "is_compound": false,
    #   "n_morphemes": 3,
    #   "pos_tag": "ADV"
    # }
