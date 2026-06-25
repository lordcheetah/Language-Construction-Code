"""Grammatical categories, feature bundles, and word classes.

This is the morphological analogue of the phonological feature system in Stage 1. A
:class:`GrammaticalCategory` is a dimension a language may mark on a word — Number, Case,
Tense, Person, and so on — each with an ordered set of values and a *base* (the unmarked
value that the citation form carries, e.g. singular, nominative, present). Each category
also carries a ``commonness``: roughly how often the world's languages grammaticalize it,
used by the generator to decide which categories a rolled language marks.

A :class:`FeatureBundle` selects one value per category (``number=pl, case=acc``) and is
the key that a paradigm maps to an affix. A :class:`WordClass` lists the categories its
members can inflect for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Typology(Enum):
    """Morphological type — how many features one affix expresses.

    - ISOLATING: few categories marked. (A fuller model would express them with free
      grammatical words/particles rather than affixes; the current engine still affixes
      the few marked categories — true analytic particles are on the backlog.)
    - AGGLUTINATIVE: one affix per category value, stacked transparently.
    - FUSIONAL: a single affix expresses a whole bundle of categories at once.
    """

    ISOLATING = "isolating"
    AGGLUTINATIVE = "agglutinative"
    FUSIONAL = "fusional"


@dataclass(frozen=True)
class GrammaticalCategory:
    name: str
    values: tuple[str, ...]
    base: str          # the unmarked value carried by the citation form
    commonness: float  # approx. fraction of languages that mark this category

    def __post_init__(self) -> None:
        if self.base not in self.values:
            raise ValueError(f"base {self.base!r} not among values of {self.name!r}")

    @property
    def marked_values(self) -> tuple[str, ...]:
        """Values other than the base (the ones that typically take an overt affix)."""
        return tuple(v for v in self.values if v != self.base)


# --- Standard category inventory ----------------------------------------------------
# Values and commonness are deliberate approximations of cross-linguistic tendencies
# (WALS-flavoured), enough to drive plausible typology rather than to be authoritative.
CATEGORIES: dict[str, GrammaticalCategory] = {
    # ~half the world's languages lack obligatory affixal number marking on nouns, so
    # 0.70 rather than near-universal.
    "number": GrammaticalCategory("number", ("sg", "pl"), "sg", 0.70),
    "case": GrammaticalCategory(
        "case", ("nom", "acc", "gen", "dat", "loc"), "nom", 0.50
    ),
    "gender": GrammaticalCategory("gender", ("masc", "fem", "neut"), "masc", 0.30),
    "definiteness": GrammaticalCategory(
        "definiteness", ("indef", "def"), "indef", 0.30
    ),
    "tense": GrammaticalCategory("tense", ("pres", "past", "fut"), "pres", 0.70),
    "aspect": GrammaticalCategory(
        "aspect", ("imperfective", "perfective"), "imperfective", 0.60
    ),
    "mood": GrammaticalCategory(
        "mood", ("indicative", "subjunctive", "imperative"), "indicative", 0.45
    ),
    "person": GrammaticalCategory("person", ("1", "2", "3"), "3", 0.70),
    "polarity": GrammaticalCategory(
        "polarity", ("affirmative", "negative"), "affirmative", 0.20
    ),
    # Polypersonal agreement: the verb also cross-references its object (Basque, Bantu, …).
    # A minority feature, so low commonness.
    "object_person": GrammaticalCategory("object_person", ("1", "2", "3"), "3", 0.15),
    "object_number": GrammaticalCategory("object_number", ("sg", "pl"), "sg", 0.15),
}


@dataclass(frozen=True)
class WordClass:
    name: str
    category_names: tuple[str, ...]

    def categories(self) -> list[GrammaticalCategory]:
        return [CATEGORIES[n] for n in self.category_names]


# Categories each class *may* inflect for; a generated language marks a subset.
# The tuple order is the inner->outer affix template (the category listed first sits
# closest to the root). Verb order follows Bybee's relevance hierarchy — aspect is most
# relevant to the verb stem so it is innermost, agreement (person/number) is least so it
# is outermost — which is the cross-linguistically typical ordering. Noun order is
# Greenbergian: number inner, case outer.
WORD_CLASSES: dict[str, WordClass] = {
    "noun": WordClass("noun", ("number", "case", "gender", "definiteness")),
    # Object agreement is outermost (least relevant to the stem, Bybee), after subject agreement.
    "verb": WordClass("verb", ("aspect", "tense", "mood", "polarity", "person", "number",
                               "object_person", "object_number")),
    "adjective": WordClass("adjective", ("gender", "number", "case")),
}


@dataclass(frozen=True)
class FeatureBundle:
    """An immutable selection of one value per category (``number=pl, case=acc``)."""

    features: tuple[tuple[str, str], ...]  # always stored sorted by category name

    @classmethod
    def of(cls, **kwargs: str) -> "FeatureBundle":
        return cls(tuple(sorted(kwargs.items())))

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> "FeatureBundle":
        return cls(tuple(sorted(mapping.items())))

    def get(self, category: str) -> str | None:
        for cat, val in self.features:
            if cat == category:
                return val
        return None

    def items(self) -> tuple[tuple[str, str], ...]:
        return self.features

    def with_feature(self, category: str, value: str) -> "FeatureBundle":
        mapping = dict(self.features)
        mapping[category] = value
        return FeatureBundle.from_dict(mapping)

    def is_empty(self) -> bool:
        return not self.features

    def __str__(self) -> str:
        if not self.features:
            return "(base)"
        return ", ".join(f"{cat}={val}" for cat, val in self.features)
