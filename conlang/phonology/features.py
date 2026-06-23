"""The IPA feature system and the :class:`Segment` model.

A *segment* (phoneme) is described by phonological features. Consonants are defined by
*place*, *manner*, and *voicing*; vowels by *height*, *backness*, and *roundedness*.
These features are not decoration — downstream stages reason over them: sound change
targets natural classes (e.g. "all voiceless plosives"), phonotactics ranks clusters by
sonority, and inventory generation enforces typological universals stated in features.

References: LCK ch. "Sounds"; the IPA chart; PHOIBLE/UPSID for the cross-linguistic
grounding used in :mod:`conlang.phonology.data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Place(Enum):
    """Place of articulation, ordered front-to-back of the vocal tract."""

    BILABIAL = "bilabial"
    LABIODENTAL = "labiodental"
    DENTAL = "dental"
    ALVEOLAR = "alveolar"
    POSTALVEOLAR = "postalveolar"
    RETROFLEX = "retroflex"
    PALATAL = "palatal"
    VELAR = "velar"
    UVULAR = "uvular"
    PHARYNGEAL = "pharyngeal"
    GLOTTAL = "glottal"


class Manner(Enum):
    """Manner of articulation, roughly ordered by decreasing constriction.

    The order also serves as the basis for the sonority hierarchy used by phonotactics
    (plosives least sonorous, approximants most), see :func:`sonority`.
    """

    PLOSIVE = "plosive"
    AFFRICATE = "affricate"
    FRICATIVE = "fricative"
    NASAL = "nasal"
    TRILL = "trill"
    TAP = "tap"
    LATERAL_FRICATIVE = "lateral fricative"
    LATERAL_APPROXIMANT = "lateral approximant"
    APPROXIMANT = "approximant"


class Voicing(Enum):
    VOICELESS = "voiceless"
    VOICED = "voiced"


class Height(Enum):
    """Vowel height (closeness of the tongue to the roof of the mouth)."""

    CLOSE = "close"
    NEAR_CLOSE = "near-close"
    CLOSE_MID = "close-mid"
    MID = "mid"
    OPEN_MID = "open-mid"
    NEAR_OPEN = "near-open"
    OPEN = "open"


class Backness(Enum):
    FRONT = "front"
    CENTRAL = "central"
    BACK = "back"


@dataclass(frozen=True)
class Segment:
    """Base class for a phoneme.

    ``ipa`` is the canonical IPA symbol and the identity of the segment (segments are
    hashable and compared by all fields, but ``ipa`` is unique within a chart).
    ``frequency`` is the approximate fraction of the world's languages that contain the
    segment (0..1), drawn from cross-linguistic surveys; it is the sampling weight used
    when rolling a random-but-plausible inventory.
    """

    ipa: str
    frequency: float = 0.0

    @property
    def is_consonant(self) -> bool:
        return isinstance(self, Consonant)

    @property
    def is_vowel(self) -> bool:
        return isinstance(self, Vowel)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.ipa


@dataclass(frozen=True)
class Consonant(Segment):
    place: Place = Place.GLOTTAL
    manner: Manner = Manner.PLOSIVE
    voicing: Voicing = Voicing.VOICELESS

    def describe(self) -> str:
        return f"{self.voicing.value} {self.place.value} {self.manner.value}"


@dataclass(frozen=True)
class Vowel(Segment):
    height: Height = Height.OPEN
    backness: Backness = Backness.CENTRAL
    rounded: bool = False
    long: bool = False

    def describe(self) -> str:
        roundness = "rounded" if self.rounded else "unrounded"
        length = "long " if self.long else ""
        return f"{length}{self.height.value} {self.backness.value} {roundness} vowel"


# --- Sonority -----------------------------------------------------------------------
# A single scale over both consonants and vowels. Higher = more sonorous. Phonotactics
# uses this to rank clusters: well-formed onsets rise in sonority toward the nucleus and
# codas fall away from it (the Sonority Sequencing Principle).

_MANNER_SONORITY: dict[Manner, int] = {
    Manner.PLOSIVE: 0,
    Manner.AFFRICATE: 1,
    Manner.FRICATIVE: 2,
    Manner.LATERAL_FRICATIVE: 2,
    Manner.NASAL: 3,
    Manner.TRILL: 4,
    Manner.TAP: 4,
    Manner.LATERAL_APPROXIMANT: 5,
    Manner.APPROXIMANT: 6,
}

# Vowels are the sonority peak — strictly above every consonant, including the
# approximants/glides at 12.
_VOWEL_SONORITY = 14


def sonority(segment: Segment) -> int:
    """Return a sonority rank for *segment* (higher = more sonorous, vowels highest).

    Consonant ranks come from doubling the manner rank and adding 1 for voiced
    obstruents, which keeps voicing as a tie-breaker *within* a manner's tier without
    ever leapfrogging the next manner. The resulting integer ladder is::

        plosive 0 / voiced 1   affricate 2 / voiced 3   fricative 4 / voiced 5
        nasal 6   trill·tap 8   lateral approximant 10   approximant 12   vowel 14

    (Preserve that no-leapfrog property if you edit ``_MANNER_SONORITY``.)
    """

    if isinstance(segment, Vowel):
        # Open vowels are marginally more sonorous than close ones; keep all vowels
        # above consonants regardless.
        return _VOWEL_SONORITY
    if isinstance(segment, Consonant):
        base = _MANNER_SONORITY[segment.manner]
        # Voiced obstruents are slightly more sonorous than their voiceless counterparts.
        if segment.voicing is Voicing.VOICED and segment.manner in (
            Manner.PLOSIVE,
            Manner.AFFRICATE,
            Manner.FRICATIVE,
            Manner.LATERAL_FRICATIVE,
        ):
            return base * 2 + 1
        return base * 2
    return 0
