"""Writing systems: turning a language's sounds into glyphs.

Stage 6 of the toolkit and the last engine stage. It generates a *native script* for a
language as scalable vector graphics (SVG) — text output, no dependencies, deterministic,
and viewable in any browser.

The glyphs are **featural**, in the spirit of Hangul: a phoneme's articulatory features
drive its shape, so related sounds get visually related letters (all nasals share a motif,
voicing adds a consistent mark, vowel height/backness place a loop). This makes a rolled
script internally coherent and learnable rather than a set of arbitrary squiggles.

Four script types are supported, matching the cross-linguistic typology: ``ALPHABET``
(a glyph per consonant and vowel), ``ABJAD`` (consonants only), ``ABUGIDA`` (consonant
glyph plus a vowel diacritic), and ``SYLLABARY`` (a composed glyph per CV syllable).
"""

from conlang.writing.glyph import Glyph, Style, Line, Path, Circle
from conlang.writing.system import (
    WritingSystem, WritingSystemType, WritingDirection,
    maya_digit, build_digit_glyphs, build_punctuation,
)
from conlang.writing.generator import build_writing_system

__all__ = [
    "Glyph",
    "Style",
    "Line",
    "Path",
    "Circle",
    "WritingSystem",
    "WritingSystemType",
    "WritingDirection",
    "maya_digit",
    "build_digit_glyphs",
    "build_punctuation",
    "build_writing_system",
]
