"""IPA charts annotated with cross-linguistic frequencies.

Each segment carries a ``frequency`` — the approximate fraction of the world's languages
that contain it. The figures are rounded approximations of the segment frequencies in
large phonological surveys (PHOIBLE / UPSID) and are used as sampling weights when
rolling a random-but-plausible inventory. They are deliberately *approximate*: the goal
is plausible typology, not a citation-grade database.

The set below is curated, not exhaustive — it covers the segments that actually recur
across languages plus enough rarer ones to make generation interesting. Adding a segment
is as simple as appending a row here.
"""

from __future__ import annotations

from conlang.phonology.features import (
    Consonant,
    Vowel,
    Place,
    Manner,
    Voicing,
    Height,
    Backness,
)

# Shorthands for readability of the table below.
_VL = Voicing.VOICELESS
_VD = Voicing.VOICED


# --- Consonants ---------------------------------------------------------------------
# Ordered roughly by frequency within each manner for readability.
CONSONANTS: list[Consonant] = [
    # Plosives
    Consonant("p", 0.86, Place.BILABIAL, Manner.PLOSIVE, _VL),
    Consonant("t", 0.85, Place.ALVEOLAR, Manner.PLOSIVE, _VL),
    Consonant("k", 0.90, Place.VELAR, Manner.PLOSIVE, _VL),
    Consonant("b", 0.63, Place.BILABIAL, Manner.PLOSIVE, _VD),
    Consonant("d", 0.46, Place.ALVEOLAR, Manner.PLOSIVE, _VD),
    # /g/ is the voiced stop most often missing (the classic "missing /g/" gap),
    # so it sits *below* /d/ despite velars being common among voiceless stops.
    Consonant("g", 0.44, Place.VELAR, Manner.PLOSIVE, _VD),
    Consonant("q", 0.16, Place.UVULAR, Manner.PLOSIVE, _VL),
    Consonant("ɢ", 0.02, Place.UVULAR, Manner.PLOSIVE, _VD),
    Consonant("ʔ", 0.37, Place.GLOTTAL, Manner.PLOSIVE, _VL),
    Consonant("c", 0.14, Place.PALATAL, Manner.PLOSIVE, _VL),
    # Nasals
    Consonant("m", 0.96, Place.BILABIAL, Manner.NASAL, _VD),
    Consonant("n", 0.96, Place.ALVEOLAR, Manner.NASAL, _VD),
    Consonant("ŋ", 0.63, Place.VELAR, Manner.NASAL, _VD),
    Consonant("ɲ", 0.42, Place.PALATAL, Manner.NASAL, _VD),
    # Fricatives
    Consonant("s", 0.67, Place.ALVEOLAR, Manner.FRICATIVE, _VL),
    Consonant("h", 0.56, Place.GLOTTAL, Manner.FRICATIVE, _VL),
    Consonant("f", 0.44, Place.LABIODENTAL, Manner.FRICATIVE, _VL),
    Consonant("ʃ", 0.37, Place.POSTALVEOLAR, Manner.FRICATIVE, _VL),
    Consonant("x", 0.30, Place.VELAR, Manner.FRICATIVE, _VL),
    Consonant("z", 0.30, Place.ALVEOLAR, Manner.FRICATIVE, _VD),
    Consonant("v", 0.27, Place.LABIODENTAL, Manner.FRICATIVE, _VD),
    Consonant("ʒ", 0.15, Place.POSTALVEOLAR, Manner.FRICATIVE, _VD),
    Consonant("θ", 0.04, Place.DENTAL, Manner.FRICATIVE, _VL),
    Consonant("ð", 0.05, Place.DENTAL, Manner.FRICATIVE, _VD),
    Consonant("ɣ", 0.14, Place.VELAR, Manner.FRICATIVE, _VD),
    Consonant("χ", 0.06, Place.UVULAR, Manner.FRICATIVE, _VL),
    Consonant("ħ", 0.08, Place.PHARYNGEAL, Manner.FRICATIVE, _VL),
    # Affricates
    Consonant("t͡ʃ", 0.40, Place.POSTALVEOLAR, Manner.AFFRICATE, _VL),
    Consonant("d͡ʒ", 0.27, Place.POSTALVEOLAR, Manner.AFFRICATE, _VD),
    Consonant("t͡s", 0.21, Place.ALVEOLAR, Manner.AFFRICATE, _VL),
    # Lateral fricative
    Consonant("ɬ", 0.10, Place.ALVEOLAR, Manner.LATERAL_FRICATIVE, _VL),
    # Liquids — the tap is the more common rhotic; the trill is rarer than often assumed.
    Consonant("l", 0.68, Place.ALVEOLAR, Manner.LATERAL_APPROXIMANT, _VD),
    Consonant("ɾ", 0.27, Place.ALVEOLAR, Manner.TAP, _VD),
    Consonant("r", 0.22, Place.ALVEOLAR, Manner.TRILL, _VD),
    Consonant("ɽ", 0.06, Place.RETROFLEX, Manner.TAP, _VD),
    Consonant("ʁ", 0.05, Place.UVULAR, Manner.FRICATIVE, _VD),
    # Approximants / glides
    Consonant("j", 0.90, Place.PALATAL, Manner.APPROXIMANT, _VD),
    Consonant("w", 0.82, Place.BILABIAL, Manner.APPROXIMANT, _VD),
]


# --- Vowels -------------------------------------------------------------------------
VOWELS: list[Vowel] = [
    Vowel("i", 0.87, Height.CLOSE, Backness.FRONT, rounded=False),
    Vowel("a", 0.86, Height.OPEN, Backness.FRONT, rounded=False),
    Vowel("u", 0.82, Height.CLOSE, Backness.BACK, rounded=True),
    Vowel("e", 0.61, Height.CLOSE_MID, Backness.FRONT, rounded=False),
    Vowel("o", 0.60, Height.CLOSE_MID, Backness.BACK, rounded=True),
    Vowel("ɛ", 0.27, Height.OPEN_MID, Backness.FRONT, rounded=False),
    Vowel("ɔ", 0.27, Height.OPEN_MID, Backness.BACK, rounded=True),
    Vowel("ə", 0.30, Height.MID, Backness.CENTRAL, rounded=False),
    Vowel("ɨ", 0.16, Height.CLOSE, Backness.CENTRAL, rounded=False),
    Vowel("y", 0.05, Height.CLOSE, Backness.FRONT, rounded=True),
    Vowel("ø", 0.03, Height.CLOSE_MID, Backness.FRONT, rounded=True),
    Vowel("ɯ", 0.04, Height.CLOSE, Backness.BACK, rounded=False),
    Vowel("ɪ", 0.10, Height.NEAR_CLOSE, Backness.FRONT, rounded=False),
    Vowel("ʊ", 0.08, Height.NEAR_CLOSE, Backness.BACK, rounded=True),
    Vowel("æ", 0.08, Height.NEAR_OPEN, Backness.FRONT, rounded=False),
    Vowel("ɑ", 0.12, Height.OPEN, Backness.BACK, rounded=False),
]


# --- Lookups ------------------------------------------------------------------------
ALL_SEGMENTS = [*CONSONANTS, *VOWELS]
BY_IPA = {seg.ipa: seg for seg in ALL_SEGMENTS}


def consonant(ipa: str) -> Consonant:
    """Look up a consonant by IPA symbol (raises KeyError if unknown)."""
    seg = BY_IPA[ipa]
    assert isinstance(seg, Consonant), f"{ipa!r} is not a consonant"
    return seg


def vowel(ipa: str) -> Vowel:
    """Look up a vowel by IPA symbol (raises KeyError if unknown)."""
    seg = BY_IPA[ipa]
    assert isinstance(seg, Vowel), f"{ipa!r} is not a vowel"
    return seg
