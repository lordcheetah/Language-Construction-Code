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

    ``classes`` makes the alternation **class-bound** — restricted to some declensions/
    conjugations rather than the whole word class. ``None`` means every class alternates (the
    default); a tuple of inflection-class ids (``("1", "3")``) means only those classes have
    the bound stem, the rest keep their root unchanged — a strong/weak split, as in Germanic
    strong vs. weak verbs or Latin's stem-varying declensions. ``classes`` is independent of
    ``trigger_category``/``condition`` (all three must hold), so how visible the split is still
    depends on them and on the shared-citation simplification noted on :class:`Paradigm`.
    """

    change: SandhiLike            # root -> bound stem
    trigger_category: str | None = None
    condition: str | None = None  # None | "before_vowel" | "before_consonant"
    classes: tuple[str, ...] | None = None  # inflection classes it applies to (None = all)

    def stem(
        self,
        root: Sequence[Segment],
        full: FeatureBundle,
        marked: tuple,
        following: "Affix | None" = None,
        inflection_class: str | None = None,
    ) -> list[Segment]:
        if self._applies(full, marked, following, inflection_class):
            return list(self.change.apply(list(root)))
        return list(root)

    def _applies(self, full: FeatureBundle, marked: tuple, following, inflection_class=None) -> bool:
        if self.classes is not None and (inflection_class or DEFAULT_CLASS) not in self.classes:
            return False  # this declension/conjugation keeps the unaltered root
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
        suppletive_stems: "tuple" = (),
    ) -> list[Segment]:
        """Inflect *root* for *bundle* using *inflection_class* (default class if None).

        Missing marked categories default to their base value. ``suppletive_stems`` (pairs of
        ``((category, value), form)``) overrides the whole word for a matching marked cell —
        true suppletion (go/went), returned verbatim with no affixation.
        """
        inflection_class = inflection_class or DEFAULT_CLASS  # normalize once: affix selection
        full = self._complete(bundle)                         # and class-binding agree on it
        if suppletive_stems:
            suppletive = self._suppletive_form(full, suppletive_stems)
            if suppletive is not None:
                return list(suppletive)  # full-form suppletion: the stored form *is* the word
        agglutinative, fusional = self._affixes(inflection_class)
        if self.typology is Typology.FUSIONAL:
            affix = fusional.get(full)
            overt = affix if (affix is not None and not affix.is_zero) else None
            # Affix conditioning looks at what follows the stem's (right) edge: a suffix.
            following = overt if (overt and overt.position is Position.SUFFIX) else None
            stem = self._stem(root, full, following, inflection_class)
            form = overt.attach(stem) if overt else list(stem)
        else:  # agglutinative and isolating both stack affixes (isolating just has few)
            prefixes, suffixes = self._collect_affixes(full, agglutinative)
            following = suffixes[0] if suffixes else None  # the innermost (stem-adjacent) suffix
            stem = self._stem(root, full, following, inflection_class)
            form = self._attach_affixes(stem, prefixes, suffixes)
        return self._apply_sandhi(form)

    def _stem(
        self,
        root: Sequence[Segment],
        full: FeatureBundle,
        following: "Affix | None",
        inflection_class: str | None = None,
    ) -> list[Segment]:
        if self.stem_alternation is None:
            return list(root)
        return self.stem_alternation.stem(
            root, full, self.marked, following, inflection_class
        )

    def _suppletive_form(self, full: FeatureBundle, suppletive_stems) -> "list | None":
        """A stored suppletive form for *full*, or None. Full-form suppletion: a single marked
        cell (a non-base value) maps to a complete irregular word that replaces root + affixes.
        Matches in marked-category order, so the result is deterministic if several could fire.

        Simplification: the stored form is invariant for every *other* category — a suppletive
        plural is the same across cases, a suppletive past across persons. That matches totally
        suppletive items (English went/people) but not stem-suppletion that still takes regular
        affixes (Latin fui/fuisti, Russian ljudi/ljudej), which is out of scope here.
        """
        table = dict(suppletive_stems)
        for cat in self.marked:
            value = full.get(cat.name)
            if value is not None and value != cat.base and (cat.name, value) in table:
                return list(table[(cat.name, value)])
        return None

    def suppletive_form(
        self, bundle: FeatureBundle, suppletive_stems, inflection_class: str | None = None
    ) -> "list | None":
        """The full-form suppletive word for *bundle* (go/went), or None if no cell matches.
        Exposes the same override :meth:`inflect` applies, for the isolating render path."""
        if not suppletive_stems:
            return None
        return self._suppletive_form(self._complete(bundle), suppletive_stems)

    def analytic_particles(
        self, bundle: FeatureBundle, inflection_class: str | None = None
    ) -> tuple[list, list]:
        """For an isolating language: the overt marked categories as free particle Affixes, to
        be rendered as separate words rather than bound to the stem. Returns ``(prefixes,
        suffixes)`` in inner-to-outer order (as :meth:`_collect_affixes` does); the stem itself
        is left bare. Reuses the same affix forms/glosses the bound paradigm would attach —
        isolating morphology realizes them as words, not suffixes (Chinese 了, Vietnamese đã).

        Idealization: *every* marked category becomes a particle, including any case/gender the
        language happens to mark. Real isolating languages grammaticalize TAM, number and
        definiteness analytically but rarely case and almost never gender; treating those as
        particles too is a simplification (the marking itself comes from the generator).
        """
        full = self._complete(bundle)
        agglutinative, _fusional = self._affixes(inflection_class)
        return self._collect_affixes(full, agglutinative)

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
