"""Phonotactics — the rules for how segments combine into syllables.

A :class:`SyllableTemplate` is written in the familiar conlanger notation where ``C`` is
a consonant slot, ``V`` a vowel slot, and parentheses mark a slot optional:

    (C)(C)V(C)      # 0-2 onset consonants, a vowel, an optional coda

:class:`Phonotactics` bundles an inventory with one or more weighted templates and emits
random syllables. Clusters are kept well-formed via the **Sonority Sequencing
Principle**: onset consonants rise in sonority toward the vowel and coda consonants fall
away from it (LCK, "Phonotactics"). Rather than rejection-sampling against the SSP, we
fill the slots and then *order* each cluster by sonority, which is cheap and always
valid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from conlang.phonology.features import Consonant, Vowel, Manner, sonority
from conlang.phonology.inventory import Inventory

# Manners that pattern as obstruents for the purpose of onset cluster restrictions.
_STOP_LIKE = (Manner.PLOSIVE, Manner.AFFRICATE)


@dataclass(frozen=True)
class _Slot:
    kind: str  # "C" or "V"
    optional: bool


@dataclass(frozen=True)
class SyllableTemplate:
    """A parsed syllable shape such as ``(C)(C)V(C)``."""

    notation: str
    slots: tuple[_Slot, ...]

    @classmethod
    def parse(cls, notation: str) -> "SyllableTemplate":
        slots: list[_Slot] = []
        i = 0
        text = notation.replace(" ", "")
        while i < len(text):
            ch = text[i]
            if ch == "(":
                inner = text[i + 1]
                if inner not in ("C", "V"):
                    raise ValueError(f"bad slot {inner!r} in template {notation!r}")
                if text[i + 2] != ")":
                    raise ValueError(f"unclosed optional slot in {notation!r}")
                slots.append(_Slot(inner, optional=True))
                i += 3
            elif ch in ("C", "V"):
                slots.append(_Slot(ch, optional=False))
                i += 1
            else:
                raise ValueError(f"unexpected char {ch!r} in template {notation!r}")
        if not any(s.kind == "V" for s in slots):
            raise ValueError(f"template {notation!r} has no vowel (nucleus)")
        return cls(notation, tuple(slots))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.notation


@dataclass
class Phonotactics:
    """An inventory plus weighted syllable templates; emits random syllables."""

    inventory: Inventory
    templates: list[SyllableTemplate] = field(default_factory=list)
    template_weights: list[float] = field(default_factory=list)
    # Minimum sonority *gap* required between adjacent consonants in a cluster. >=2
    # forbids equal/near-equal pairs such as /pt/ or /sʃ/ while permitting /pl/, /sn/.
    min_sonority_distance: int = 2
    # Base chance an optional slot is filled; cluster-extending slots are damped below
    # this so complex onsets/codas stay rarer than simple ones, as in natural languages.
    optional_probability: float = 0.5

    # --- Construction ----------------------------------------------------------------
    @classmethod
    def from_notation(
        cls,
        inventory: Inventory,
        templates: list[str] | str,
        weights: list[float] | None = None,
    ) -> "Phonotactics":
        if isinstance(templates, str):
            templates = [templates]
        parsed = [SyllableTemplate.parse(t) for t in templates]
        weights = weights or [1.0] * len(parsed)
        if len(weights) != len(parsed):
            raise ValueError("weights must match number of templates")
        return cls(inventory, parsed, weights)

    @classmethod
    def random(
        cls, inventory: Inventory, rng: random.Random | None = None,
        *, prefer_complex: bool = False,
    ) -> "Phonotactics":
        """Roll a plausible phonotactic profile (simple → complex syllable structure).

        ``prefer_complex`` skews toward the clustered, coda-heavy tiers (never pure CV) — used
        for a loanword donor, so borrowings read as foreign (more complex than a typical native
        profile) rather than occasionally simpler.
        """
        rng = rng or random.Random()
        # Complexity tiers, weighted toward the cross-linguistically common middle. Each
        # tier carries per-template weights so simpler shapes dominate within a profile
        # (e.g. CV is more frequent than CVC), rather than every shape being equally likely.
        profiles = [
            #  (templates,                      template_weights),          tier_weight
            ((["CV"], [1.0]), 0.20),
            ((["CV", "CVC"], [0.7, 0.3]), 0.30),
            ((["(C)V", "(C)V(C)"], [0.6, 0.4]), 0.30),
            ((["(C)(C)V(C)", "(C)V"], [0.4, 0.6]), 0.15),
            ((["(C)(C)V(C)(C)", "(C)(C)V(C)"], [0.3, 0.7]), 0.05),
        ]
        tier_weights = (
            [0.0, 0.10, 0.30, 0.35, 0.25] if prefer_complex else [p[1] for p in profiles]
        )
        (templates, weights) = rng.choices(
            [p[0] for p in profiles], weights=tier_weights, k=1
        )[0]
        return cls.from_notation(inventory, templates, weights)

    # --- Generation ------------------------------------------------------------------
    def random_syllable(self, rng: random.Random | None = None) -> list:
        """Return one syllable as a list of :class:`Segment`."""
        rng = rng or random.Random()
        template = rng.choices(self.templates, weights=self.template_weights, k=1)[0]

        onset: list[Consonant] = []
        nucleus: list[Vowel] = []
        coda: list[Consonant] = []
        seen_vowel = False

        for slot in template.slots:
            if slot.kind == "C":
                side = coda if seen_vowel else onset
                if slot.optional:
                    # Damp cluster-extending slots: each consonant already on this side
                    # halves the chance of adding another, so complex clusters are rare.
                    prob = self.optional_probability * (0.5 ** len(side))
                    if rng.random() >= prob:
                        continue
                cons = _weighted_choice(rng, self.inventory.consonants)
                if cons is not None:
                    side.append(cons)
            else:  # vowel / nucleus
                if slot.optional and rng.random() >= self.optional_probability:
                    continue
                vow = _weighted_choice(rng, self.inventory.vowels)
                if vow is not None:
                    nucleus.append(vow)
                seen_vowel = True

        # Order each cluster by the Sonority Sequencing Principle (onsets rise toward the
        # nucleus, codas fall away), then drop ill-formed members (OCP, minimum sonority
        # distance, stop+nasal onsets).
        onset = self._well_form(sorted(onset, key=sonority), rising=True)
        coda = self._well_form(sorted(coda, key=sonority, reverse=True), rising=False)
        return [*onset, *nucleus, *coda]

    def _well_form(self, cluster: list[Consonant], *, rising: bool) -> list[Consonant]:
        """Drop cluster members that violate well-formedness, keeping the first.

        Applied to an already sonority-ordered cluster, this enforces three constraints
        attested across natural languages:

        - **OCP**: no two identical adjacent segments (kills geminate onsets like /ss/).
        - **Minimum sonority distance**: adjacent segments must differ in sonority by at
          least ``min_sonority_distance`` (kills /pt/, /sʃ/ while allowing /pl/, /sn/).
        - **No stop/affricate + nasal onset**: /pm/, /kn/ etc. are cross-linguistically
          near-absent, unlike fricative + nasal (/sn/, /sm/), which are fine.
        """
        kept: list[Consonant] = []
        for seg in cluster:
            if not kept:
                kept.append(seg)
                continue
            prev = kept[-1]
            if seg.ipa == prev.ipa:  # OCP
                continue
            if abs(sonority(seg) - sonority(prev)) < self.min_sonority_distance:
                continue
            if rising and prev.manner in _STOP_LIKE and seg.manner is Manner.NASAL:
                continue
            kept.append(seg)
        return kept


# --- Helpers ------------------------------------------------------------------------
def _weighted_choice(rng, segments):
    """Pick one segment weighted by cross-linguistic frequency, or None if empty."""
    if not segments:
        return None
    weights = [max(s.frequency, 1e-9) for s in segments]
    return rng.choices(segments, weights=weights, k=1)[0]
