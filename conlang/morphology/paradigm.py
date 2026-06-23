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
:meth:`inflect` selects one per word. Simplifications worth noting: classes differ only in
their *marked* cells (the citation form is shared), and class membership is not yet tied to
gender (real declensions often correlate with it).

Optionally a Stage 2 :class:`~conlang.soundchange.ruleset.RuleSet` is applied after
affixation as **sandhi**, smoothing the morpheme boundaries.

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
            form = self._inflect_fusional(root, full, fusional)
        else:  # agglutinative and isolating both stack affixes (isolating just has few)
            form = self._inflect_agglutinative(root, full, agglutinative)
        return self._apply_sandhi(form)

    def _inflect_agglutinative(
        self, root: Sequence[Segment], full: FeatureBundle, affixes: dict
    ) -> list[Segment]:
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
        out = list(root)
        # Inner-to-outer: first marked category sits closest to the root on each side.
        for affix in prefixes:
            out = [*affix.form, *out]
        for affix in suffixes:
            out = [*out, *affix.form]
        return out

    def _inflect_fusional(
        self, root: Sequence[Segment], full: FeatureBundle, affixes: dict
    ) -> list[Segment]:
        affix = affixes.get(full)
        if affix is None or affix.is_zero:
            return list(root)
        return affix.attach(root)

    def _complete(self, bundle: FeatureBundle) -> FeatureBundle:
        """Return *bundle* restricted to marked categories, filling gaps with base."""
        mapping: dict[str, str] = {}
        for cat in self.marked:
            mapping[cat.name] = bundle.get(cat.name) or cat.base
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
