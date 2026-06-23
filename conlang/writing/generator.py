"""Roll a writing system for a language.

A script type is sampled (alphabets are the most common among the world's scripts, then
abugidas, then abjads and syllabaries), and a per-language :class:`Style` is chosen — a
slight slant, a stroke weight, and whether voicing is marked with a dot or a bar — so two
languages get visually distinct scripts even though both use the same featural logic. The
featural glyphs themselves are deterministic functions of the phonemes.
"""

from __future__ import annotations

import random

from conlang.phonology.features import Height
from conlang.phonology.inventory import Inventory
from conlang.writing.glyph import Style
from conlang.writing.featural import consonant_glyph, vowel_glyph, vowel_diacritic
from conlang.writing.system import WritingSystem, WritingSystemType, build_digit_glyphs

# Approximate prevalence of script types among the world's writing systems.
_TYPE_WEIGHTS: dict[WritingSystemType, float] = {
    WritingSystemType.ALPHABET: 0.45,
    WritingSystemType.ABUGIDA: 0.25,
    WritingSystemType.ABJAD: 0.15,
    WritingSystemType.SYLLABARY: 0.15,
}


def build_writing_system(
    inventory: Inventory,
    rng: random.Random | None = None,
    *,
    wtype: WritingSystemType | None = None,
) -> WritingSystem:
    rng = rng or random.Random()

    if wtype is None:
        types = list(_TYPE_WEIGHTS)
        wtype = rng.choices(types, weights=[_TYPE_WEIGHTS[t] for t in types], k=1)[0]

    style = Style(
        stroke_width=round(rng.uniform(5.0, 7.5), 1),
        slant=round(rng.uniform(-12.0, 12.0), 1),
        voiced_mark=rng.choice(["dot", "bar"]),
    )

    voiced_mark = style.voiced_mark
    consonants = {c.ipa: consonant_glyph(c, voiced_mark=voiced_mark) for c in inventory.consonants}
    vowels = {v.ipa: vowel_glyph(v) for v in inventory.vowels}
    diacritics = {v.ipa: vowel_diacritic(v) for v in inventory.vowels}

    # Real abugidas overwhelmingly use /a/ as the inherent vowel, so prefer an open vowel,
    # breaking ties by cross-linguistic frequency.
    inherent = None
    if wtype is WritingSystemType.ABUGIDA and inventory.vowels:
        inherent = max(
            inventory.vowels, key=lambda v: (v.height is Height.OPEN, v.frequency)
        ).ipa

    return WritingSystem(
        type=wtype,
        consonants=consonants,
        vowels=vowels,
        diacritics=diacritics,
        style=style,
        inherent_vowel=inherent,
        digit_glyphs=build_digit_glyphs(),  # bars-and-dots digits (pure geometry, no rng)
    )
