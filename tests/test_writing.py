"""Tests for the writing-system stage.

These check that glyph generation is deterministic and featural (related sounds get
related glyphs), that the four script types break a word into the right units, and that
all SVG output is well-formed XML.
"""

import random
import xml.etree.ElementTree as ET

import pytest

from conlang.phonology import data
from conlang.phonology.inventory import Inventory
from conlang.writing.glyph import Glyph, Style, Line, Circle
from conlang.writing.featural import consonant_glyph, vowel_glyph, vowel_diacritic
from conlang.writing.system import (
    WritingSystem, WritingSystemType, maya_digit, build_digit_glyphs, _to_base_digits,
)
from conlang.writing.generator import build_writing_system


def segs(symbols: str):
    return [data.BY_IPA[s] for s in symbols.split()]


def is_well_formed_svg(svg: str) -> bool:
    root = ET.fromstring(svg)  # raises on malformed XML
    return root.tag.endswith("svg")


# --- Glyph / SVG --------------------------------------------------------------------
def test_glyph_svg_is_well_formed():
    g = Glyph((Line(10, 10, 90, 90), Circle(50, 50, 20)))
    assert is_well_formed_svg(g.svg())


def test_number_formatting_has_no_trailing_zeros():
    svg = Glyph((Line(10.0, 20.5, 30.0, 40.25),)).svg()
    assert "10 " in svg or 'x1="10"' in svg  # 10.00 -> "10"


# --- Featural determinism & coherence -----------------------------------------------
def test_glyph_generation_is_deterministic():
    assert consonant_glyph(data.consonant("p")) == consonant_glyph(data.consonant("p"))
    assert vowel_glyph(data.vowel("a")) == vowel_glyph(data.vowel("a"))


def test_distinct_phonemes_get_distinct_glyphs():
    assert consonant_glyph(data.consonant("p")) != consonant_glyph(data.consonant("k"))
    assert consonant_glyph(data.consonant("p")) != consonant_glyph(data.consonant("m"))
    assert vowel_glyph(data.vowel("i")) != vowel_glyph(data.vowel("u"))


def test_voicing_adds_a_mark():
    # /b/ is /p/ plus a voicing mark, so it has exactly one more stroke.
    p = consonant_glyph(data.consonant("p"), voiced_mark="dot")
    b = consonant_glyph(data.consonant("b"), voiced_mark="dot")
    assert len(b.strokes) == len(p.strokes) + 1


def test_same_manner_shares_body_strokes():
    # /p/ and /k/ are both plosives: their bodies match; only the place tick differs.
    p = consonant_glyph(data.consonant("p"))
    k = consonant_glyph(data.consonant("k"))
    # The first stroke is the manner body (a straight stem for plosives).
    assert p.strokes[0] == k.strokes[0]


def test_rounded_vowel_closes_the_loop():
    # /u/ is rounded (a closed circle); /i/ is unrounded (an open path).
    u = vowel_glyph(data.vowel("u"))
    i = vowel_glyph(data.vowel("i"))
    assert any(isinstance(s, Circle) for s in u.strokes)
    assert not any(isinstance(s, Circle) for s in i.strokes)


def test_distinct_vowels_get_distinct_diacritics():
    # The diacritic must encode height too, so vowels differing only in height differ.
    marks = {v.ipa: vowel_diacritic(v).svg() for v in data.VOWELS}
    assert len(set(marks.values())) == len(marks), "two vowels share a diacritic"
    # specifically, /i/, /e/, /a/ (front unrounded, three heights) must all differ
    assert len({vowel_diacritic(data.vowel(x)).svg() for x in ("i", "e", "a")}) == 3


# --- Writing system types -----------------------------------------------------------
def _system(wtype, inv_str="p t k b a i u"):
    inv = Inventory.from_ipa(inv_str)
    return build_writing_system(inv, random.Random(1), wtype=wtype)


def test_alphabet_writes_every_segment():
    ws = _system(WritingSystemType.ALPHABET)
    units = ws.render_segments(segs("p a t i"))
    assert [u[0] for u in units] == ["p", "a", "t", "i"]


def test_abjad_omits_vowels():
    ws = _system(WritingSystemType.ABJAD)
    units = ws.render_segments(segs("p a t i k"))
    assert [u[0] for u in units] == ["p", "t", "k"]


def test_abugida_leaves_inherent_vowel_bare_and_composes_others():
    ws = _system(WritingSystemType.ABUGIDA)
    inherent = ws.inherent_vowel
    other = next(v for v in ("a", "i", "u") if v != inherent)
    # consonant + inherent vowel -> bare consonant glyph
    bare = ws.render_segments(segs(f"p {inherent}"))
    assert bare[0][0] == "p" and bare[0][1] == ws.consonants["p"]
    # consonant + other vowel -> composed glyph with more strokes than the bare consonant
    composed = ws.render_segments(segs(f"p {other}"))
    assert composed[0][0] == "p" + other
    assert len(composed[0][1].strokes) > len(ws.consonants["p"].strokes)


def test_syllabary_composes_every_cv():
    ws = _system(WritingSystemType.SYLLABARY)
    units = ws.render_segments(segs("p a t i"))
    assert [u[0] for u in units] == ["pa", "ti"]


def test_coda_consonant_is_marked_distinct_from_bare_inherent():
    # In an abugida, a coda consonant (no following vowel) must not render identically to
    # the bare (inherent-vowel) form of the same consonant.
    ws = _system(WritingSystemType.ABUGIDA)
    inherent = ws.inherent_vowel
    bare = ws.render_segments(segs(f"p {inherent}"))[0][1]          # p + inherent -> bare
    coda = ws.render_segments(segs("a p"))[-1][1]                    # p as a coda
    assert coda != bare
    assert len(coda.strokes) > len(ws.consonants["p"].strokes)       # carries the virama


def test_style_color_applies_to_filled_marks():
    # A filled voicing dot must take the style colour, not fall back to black.
    g = consonant_glyph(data.consonant("b"))  # voiced -> has a filled mark
    svg = g.svg(Style(color="#ff0000"))
    assert 'color="#ff0000"' in svg and "#222222" not in svg


# --- SVG rendering ------------------------------------------------------------------
def test_word_and_chart_svg_well_formed_for_all_types():
    for wtype in WritingSystemType:
        ws = _system(wtype)
        assert is_well_formed_svg(ws.word_svg(segs("p a t i")))
        assert is_well_formed_svg(ws.chart_svg())


def test_chart_cell_count_matches_type():
    inv = "p t k b a i u"
    alpha = _system(WritingSystemType.ALPHABET, inv)
    assert len(alpha.chart_cells()) == 4 + 3  # consonants + vowels
    abjad = _system(WritingSystemType.ABJAD, inv)
    assert len(abjad.chart_cells()) == 4
    syll = _system(WritingSystemType.SYLLABARY, inv)
    assert len(syll.chart_cells()) == 4 * 3  # consonants x vowels


# --- Numeral glyphs -----------------------------------------------------------------
def test_base_decomposition():
    assert _to_base_digits(0, 10) == [0]
    assert _to_base_digits(42, 10) == [4, 2]
    assert _to_base_digits(100, 20) == [5, 0]   # 5*20 + 0
    assert _to_base_digits(7, 5) == [1, 2]        # 1*5 + 2


def test_maya_digits_encode_bars_and_dots():
    # value v -> (v // 5) bars (Lines) + (v % 5) dots (filled Circles); 0 is special.
    def counts(v):
        g = maya_digit(v)
        bars = sum(1 for s in g.strokes if isinstance(s, Line))
        dots = sum(1 for s in g.strokes if isinstance(s, Circle) and s.filled)
        return bars, dots
    assert counts(3) == (0, 3)
    assert counts(5) == (1, 0)
    assert counts(7) == (1, 2)
    assert counts(19) == (3, 4)
    assert maya_digit(0) != maya_digit(5)   # zero is a distinct shape
    # every digit 0..19 is a distinct glyph
    assert len({maya_digit(v).svg() for v in range(20)}) == 20


def test_number_svg_is_well_formed_and_uses_the_right_digit_count():
    ws = build_writing_system(Inventory.from_ipa("p t k a i u"), random.Random(1))
    assert len(ws.digit_glyphs) == 20
    for n, base, ndigits in ((0, 10, 1), (42, 10, 2), (100, 20, 2), (7, 5, 2)):
        svg = ws.number_svg(n, base)
        root = ET.fromstring(svg)
        # one outer <g> (a positioned digit cell) per base digit
        digit_cells = [c for c in root if c.tag.endswith("g")]
        assert len(digit_cells) == ndigits


def test_zero_in_a_nonfinal_position_renders_the_shell():
    ws = build_writing_system(Inventory.from_ipa("p t k a i u"), random.Random(1))
    root = ET.fromstring(ws.number_svg(100, 20))  # 100 base 20 -> digits [5, 0]
    cells = [c for c in root if c.tag.endswith("g")]
    assert len(cells) == 2
    second = ET.tostring(cells[1], encoding="unicode")  # the digit 0
    assert "path" in second.lower()  # the zero shell (a Path), not a blank cell


def test_number_svg_rejects_a_base_beyond_the_digit_glyphs():
    ws = build_writing_system(Inventory.from_ipa("p t k a i u"), random.Random(1))
    with pytest.raises(ValueError):
        ws.number_svg(5, base=36)  # only 0..19 digit glyphs exist


# --- Generator ----------------------------------------------------------------------
def test_build_writing_system_reproducible():
    inv = Inventory.from_ipa("p t k b a i u")
    a = build_writing_system(inv, random.Random(5))
    b = build_writing_system(inv, random.Random(5))
    assert a.type == b.type and a.style == b.style


def test_abugida_inherent_vowel_prefers_open():
    # Even though /i/ is slightly more frequent than /a/, the inherent vowel should be the
    # open vowel /a/, matching real abugidas.
    inv = Inventory.from_ipa("p t a i u")
    ws = build_writing_system(inv, random.Random(0), wtype=WritingSystemType.ABUGIDA)
    assert ws.inherent_vowel == "a"
