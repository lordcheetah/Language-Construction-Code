"""Paradigms: inflecting a root for a feature bundle, and laying out full tables.

A :class:`Paradigm` ties a word class to the categories a language actually marks on it,
the affixes that realize them, and a morphological :class:`Typology`:

- **Agglutinative** — one affix per marked category value; an inflected form stacks the
  affixes for each category in order. Unmarked (base) values contribute a zero affix.
- **Fusional** — a single affix realizes the whole combination of category values at
  once; the paradigm maps each full :class:`FeatureBundle` to one affix.

A word class may have several **inflection classes** (declensions for nouns, conjugations
for verbs): each realizes the *same* marked categories with a *different* affix set, and
every lexeme belongs to one. The paradigm's own affix fields are inflection class ``"1"``
(the default); additional classes live in ``extra_classes`` keyed ``"2"``, ``"3"``, … and
:meth:`inflect` selects one per word. A simplification worth noting: classes differ only in
their *marked* cells (the citation form is shared). In a gender-marking language the lexicon
ties a noun's class to its gender (declensions track gender, as in Latin); otherwise the
class is assigned at random.

Optionally a Stage 2 :class:`~conlang.soundchange.ruleset.RuleSet` is applied after
affixation as **sandhi**, smoothing the morpheme boundaries. A :class:`StemAlternation`
adds the other kind of stem change — **allomorphy**: a bound/oblique stem, distinct from the
citation root, that an affix attaches to (final-stop voicing, umlaut, …).

:class:`DerivationRule` covers the other half of morphology: forming a new stem (often of
a different word class) by adding a derivational affix.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from conlang.phonology.features import Segment
from conlang.phonology.wordgen import Romanizer
from conlang.morphology.features import (
    FeatureBundle,
    GrammaticalCategory,
    Typology,
    WordClass,
)
from conlang.morphology.affix import Affix, Position


class SandhiLike(Protocol):
    """Anything that can rewrite a segment sequence — e.g. a Stage 2 ``RuleSet``."""

    def apply(self, segments: Sequence[Segment]) -> Sequence[Segment]: ...


DEFAULT_CLASS = "1"


@dataclass
class StemAlternation:
    """Stem allomorphy: the stem an affix attaches to differs from the citation root.

    Many languages have a *bound* (oblique) stem distinct from the free/citation form — the
    stem mutates at its edge once anything is suffixed (final-stop voicing, vowel raising/
    umlaut, …). ``change`` (a Stage-2 ``RuleSet``-like rewriter) maps the root to that bound
    stem; it fires only when the word is overtly inflected — i.e. some marked category takes
    a non-base value — so the citation form (all base values) is left untouched.

    ``trigger_category`` narrows when the bound stem appears: ``None`` means *any* overt
    inflection (a true two-stem / oblique-stem system), while a category name restricts it to
    that category's non-base values (e.g. number-triggered umlaut plurals, foot → feet).

    ``condition`` makes the alternation **affix-conditioned** — sensitive to the phonology of
    the suffix that follows the stem: ``"before_vowel"`` fires only before a vowel-initial
    suffix (the Finnish/Celtic lenition pattern; the stem stays strong before a consonant or
    word-finally), ``"before_consonant"`` only before a consonant-initial one, ``None`` is
    unconditioned. The trigger and the condition both have to hold.

    Simplification: the alternation is inflection-*class*-independent — one stem rule applies
    across all declensions/conjugations. Real two-stem systems are often class-bound.
    """

    change: SandhiLike            # root -> bound stem
    trigger_category: str | None = None
    condition: str | None = None  # None | "before_vowel" | "before_consonant"

    def stem(
        self,
        root: Sequence[Segment],
        full: FeatureBundle,
        marked: tuple,
        following: "Affix | None" = None,
    ) -> list[Segment]:
        if self._applies(full, marked, following):
            return list(self.change.apply(list(root)))
        return list(root)

    def _applies(self, full: FeatureBundle, marked: tuple, following) -> bool:
        cats = marked if self.trigger_category is None else [
            c for c in marked if c.name == self.trigger_category
        ]
        if not any(full.get(c.name) not in (None, c.base) for c in cats):
            return False
        if self.condition is None:
            return True
        vowel = following is not None and following.form and following.form[0].is_vowel
        if self.condition == "before_vowel":
            return bool(vowel)
        return following is not None and not vowel  # "before_consonant"


@dataclass
class InflectionClass:
    """One declension/conjugation: an affix set realizing the marked categories."""

    agglutinative_affixes: dict[tuple[str, str], Affix] = field(default_factory=dict)
    fusional_affixes: dict[FeatureBundle, Affix] = field(default_factory=dict)


@dataclass
class Paradigm:
    word_class: WordClass
    typology: Typology
    marked: tuple[GrammaticalCategory, ...]
    # The default inflection class ("1"): agglutinative (category, value) -> Affix,
    # fusional FeatureBundle -> Affix.
    agglutinative_affixes: dict[tuple[str, str], Affix] = field(default_factory=dict)
    fusional_affixes: dict[FeatureBundle, Affix] = field(default_factory=dict)
    # Additional inflection classes keyed "2", "3", … (same categories, other affixes).
    extra_classes: dict[str, InflectionClass] = field(default_factory=dict)
    romanizer: Romanizer = field(default_factory=Romanizer)
    sandhi: SandhiLike | None = None  # an optional RuleSet applied after affixation
    stem_alternation: StemAlternation | None = None  # optional bound-stem allomorphy

    def class_ids(self) -> list[str]:
        return [DEFAULT_CLASS, *sorted(self.extra_classes)]

    def _affixes(self, inflection_class: str | None):
        ic = self.extra_classes.get(inflection_class) if inflection_class else None
        if ic is not None:
            return ic.agglutinative_affixes, ic.fusional_affixes
        return self.agglutinative_affixes, self.fusional_affixes

    # --- Inflection ------------------------------------------------------------------
    def inflect(
        self,
        root: Sequence[Segment],
        bundle: FeatureBundle,
        inflection_class: str | None = None,
    ) -> list[Segment]:
        """Inflect *root* for *bundle* using *inflection_class* (default class if None).

        Missing marked categories default to their base value.
        """
        full = self._complete(bundle)
        agglutinative, fusional = self._affixes(inflection_class)
        if self.typology is Typology.FUSIONAL:
            affix = fusional.get(full)
            overt = affix if (affix is not None and not affix.is_zero) else None
            # Affix conditioning looks at what follows the stem's (right) edge: a suffix.
            following = overt if (overt and overt.position is Position.SUFFIX) else None
            stem = self._stem(root, full, following)
            form = overt.attach(stem) if overt else list(stem)
        else:  # agglutinative and isolating both stack affixes (isolating just has few)
            prefixes, suffixes = self._collect_affixes(full, agglutinative)
            following = suffixes[0] if suffixes else None  # the innermost (stem-adjacent) suffix
            stem = self._stem(root, full, following)
            form = self._attach_affixes(stem, prefixes, suffixes)
        return self._apply_sandhi(form)

    def _stem(
        self, root: Sequence[Segment], full: FeatureBundle, following: "Affix | None"
    ) -> list[Segment]:
        if self.stem_alternation is None:
            return list(root)
        return self.stem_alternation.stem(root, full, self.marked, following)

    def _collect_affixes(self, full: FeatureBundle, affixes: dict):
        """The (prefixes, suffixes) realizing *full*, inner-to-outer (stem-adjacent first)."""
        prefixes: list[Affix] = []
        suffixes: list[Affix] = []
        for cat in self.marked:
            value = full.get(cat.name)
            if value is None or value == cat.base:
                continue  # base value -> zero affix
            affix = affixes.get((cat.name, value))
            if affix is None or affix.is_zero:
                continue
            (prefixes if affix.position is Position.PREFIX else suffixes).append(affix)
        return prefixes, suffixes

    def _attach_affixes(
        self, stem: Sequence[Segment], prefixes: list, suffixes: list
    ) -> list[Segment]:
        out = list(stem)
        # Inner-to-outer: the first marked category sits closest to the root on each side.
        for affix in prefixes:
            out = [*affix.form, *out]
        for affix in suffixes:
            out = [*out, *affix.form]
        return out

    def _complete(self, bundle: FeatureBundle) -> FeatureBundle:
        """Return *bundle* restricted to marked categories, filling gaps with base.

        A value this category doesn't have (e.g. ``number="dual"`` asked of a sg/pl language)
        falls back to the base, so it yields the citation form rather than a silent mismatch.
        """
        mapping: dict[str, str] = {}
        for cat in self.marked:
            value = bundle.get(cat.name) or cat.base
            mapping[cat.name] = value if value in cat.values else cat.base
        return FeatureBundle.from_dict(mapping)

    def _apply_sandhi(self, form: list[Segment]) -> list[Segment]:
        if self.sandhi is None:
            return form
        return list(self.sandhi.apply(form))

    # --- Display ---------------------------------------------------------------------
    def romanize(self, segments: Sequence[Segment]) -> str:
        return self.romanizer.romanize([list(segments)])

    def enumerate_bundles(self) -> list[FeatureBundle]:
        """All combinations of the marked categories' values (the full paradigm)."""
        if not self.marked:
            return [FeatureBundle.of()]
        names = [c.name for c in self.marked]
        value_lists = [c.values for c in self.marked]
        return [
            FeatureBundle.from_dict(dict(zip(names, combo)))
            for combo in itertools.product(*value_lists)
        ]

    def table(
        self, root: Sequence[Segment], inflection_class: str | None = None
    ) -> list[tuple[FeatureBundle, list[Segment], str]]:
        """Full paradigm for *root* in one inflection class: (bundle, segments, roman)."""
        rows = []
        for bundle in self.enumerate_bundles():
            seg = self.inflect(root, bundle, inflection_class)
            rows.append((bundle, seg, self.romanize(seg)))
        return rows


@dataclass(frozen=True)
class DerivationRule:
    """A derivational affix that forms a new stem, optionally of a different word class."""

    affix: Affix
    from_class: str
    to_class: str
    gloss: str

    def apply(self, root: Sequence[Segment]) -> list[Segment]:
        return self.affix.attach(root)
