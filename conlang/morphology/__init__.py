"""Morphology: how words are built from morphemes.

Stage 3 of the toolkit. It models **inflection** (marking grammatical categories such as
number, case, tense, and person on a root) and basic **derivation** (forming new words,
sometimes of a new word class, with derivational affixes).

The pieces:

- ``features``   — grammatical categories and their values (Number, Case, Tense, …), the
  :class:`FeatureBundle` that picks one value per category, and :class:`WordClass`.
- ``affix``      — an :class:`Affix`: a phonological form plus the bundle it marks and
  where it attaches.
- ``paradigm``   — a :class:`Paradigm` that inflects a root for a bundle and lays out full
  tables, in either an agglutinative or a fusional style.
- ``generator``  — rolls a typologically plausible morphological system, generating affix
  forms with the Stage 1 word generator and optionally smoothing morpheme boundaries with
  a Stage 2 sound-change ruleset (sandhi).
"""

from conlang.morphology.features import (
    GrammaticalCategory,
    FeatureBundle,
    WordClass,
    Typology,
    CATEGORIES,
    WORD_CLASSES,
)
from conlang.morphology.affix import Affix, Position
from conlang.morphology.paradigm import Paradigm, DerivationRule
from conlang.morphology.generator import MorphologySystem, random_system

__all__ = [
    "GrammaticalCategory",
    "FeatureBundle",
    "WordClass",
    "Typology",
    "CATEGORIES",
    "WORD_CLASSES",
    "Affix",
    "Position",
    "Paradigm",
    "DerivationRule",
    "MorphologySystem",
    "random_system",
]
