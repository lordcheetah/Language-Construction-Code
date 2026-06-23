"""Paradigms: inflecting a root for a feature bundle, and laying out full tables.

A :class:`Paradigm` ties a word class to the categories a language actually marks on it,
the affixes that realize them, and a morphological :class:`Typology`:

- **Agglutinative** — one affix per marked category value; an inflected form stacks the
  affixes for each category in order. Unmarked (base) values contribute a zero affix.
- **Fusional** — a single affix realizes the whole combination of category values at
  once; the paradigm maps each full :class:`FeatureBundle` to one affix.

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


@dataclass
class Paradigm:
    word_class: WordClass
    typology: Typology
    marked: tuple[GrammaticalCategory, ...]
    # Agglutinative: (category_name, value) -> Affix. Fusional: FeatureBundle -> Affix.
    agglutinative_affixes: dict[tuple[str, str], Affix] = field(default_factory=dict)
    fusional_affixes: dict[FeatureBundle, Affix] = field(default_factory=dict)
    romanizer: Romanizer = field(default_factory=Romanizer)
    sandhi: SandhiLike | None = None  # an optional RuleSet applied after affixation

    # --- Inflection ------------------------------------------------------------------
    def inflect(self, root: Sequence[Segment], bundle: FeatureBundle) -> list[Segment]:
        """Inflect *root* for *bundle*; missing marked categories default to their base."""
        full = self._complete(bundle)
        if self.typology is Typology.FUSIONAL:
            form = self._inflect_fusional(root, full)
        else:  # agglutinative and isolating both stack affixes (isolating just has few)
            form = self._inflect_agglutinative(root, full)
        return self._apply_sandhi(form)

    def _inflect_agglutinative(
        self, root: Sequence[Segment], full: FeatureBundle
    ) -> list[Segment]:
        prefixes: list[Affix] = []
        suffixes: list[Affix] = []
        for cat in self.marked:
            value = full.get(cat.name)
            if value is None or value == cat.base:
                continue  # base value -> zero affix
            affix = self.agglutinative_affixes.get((cat.name, value))
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
        self, root: Sequence[Segment], full: FeatureBundle
    ) -> list[Segment]:
        affix = self.fusional_affixes.get(full)
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

    def table(self, root: Sequence[Segment]) -> list[tuple[FeatureBundle, list[Segment], str]]:
        """Full paradigm for *root*: (bundle, inflected segments, romanization)."""
        rows = []
        for bundle in self.enumerate_bundles():
            seg = self.inflect(root, bundle)
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
