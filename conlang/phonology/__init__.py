"""Phonology: the sounds of a language and the rules for combining them.

This is stage 1 of the toolkit and the foundation every later stage inherits:

- ``features``    — the IPA feature system and the :class:`Segment` model.
- ``data``        — IPA charts annotated with cross-linguistic frequencies.
- ``inventory``   — a language's phoneme inventory (random-plausible or guided).
- ``phonotactics``— syllable templates and onset/coda constraints.
- ``wordgen``     — a frequency-weighted word/root generator with romanization.
"""

from conlang.phonology.features import (
    Segment,
    Consonant,
    Vowel,
    Place,
    Manner,
    Voicing,
    Height,
    Backness,
)
from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics, SyllableTemplate
from conlang.phonology.wordgen import WordGenerator

__all__ = [
    "Segment",
    "Consonant",
    "Vowel",
    "Place",
    "Manner",
    "Voicing",
    "Height",
    "Backness",
    "Inventory",
    "Phonotactics",
    "SyllableTemplate",
    "WordGenerator",
]
