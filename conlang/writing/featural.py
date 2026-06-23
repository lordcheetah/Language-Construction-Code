"""Build a glyph from a segment's phonological features.

The mapping is *featural*, so the script encodes articulation visually:

- **Consonants**: the *manner* draws the central body (a plosive is a straight stem, a
  fricative a wave, a nasal a looped stem, a trill a sharp zigzag, an approximant a curve,
  a lateral a branched stem). The *place* sets the height of a short cross-tick on the
  body (front articulations high, back ones low). *Voicing* adds a consistent mark on top.
- **Vowels**: a loop whose horizontal position encodes *backness* and vertical position
  encodes *height*; a *rounded* vowel closes the loop, an unrounded one leaves it open.

Because the features drive the shapes, related sounds come out looking related — which is
what makes the script learnable.
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
from conlang.writing.glyph import Glyph, Line, Path, Circle, Stroke

# Place -> vertical height of the cross-tick (front high / small y, back low / large y).
_PLACE_Y: dict[Place, float] = {
    Place.BILABIAL: 22, Place.LABIODENTAL: 28, Place.DENTAL: 34, Place.ALVEOLAR: 40,
    Place.POSTALVEOLAR: 46, Place.RETROFLEX: 52, Place.PALATAL: 58, Place.VELAR: 66,
    Place.UVULAR: 74, Place.PHARYNGEAL: 82, Place.GLOTTAL: 90,
}

# Vowel loop centre: backness -> x, height -> y.
_BACKNESS_X: dict[Backness, float] = {Backness.FRONT: 36, Backness.CENTRAL: 50, Backness.BACK: 64}
_HEIGHT_Y: dict[Height, float] = {
    Height.CLOSE: 30, Height.NEAR_CLOSE: 37, Height.CLOSE_MID: 44, Height.MID: 52,
    Height.OPEN_MID: 60, Height.NEAR_OPEN: 68, Height.OPEN: 76,
}


# --- Consonant manner bodies --------------------------------------------------------
def _body_plosive() -> list[Stroke]:
    return [Line(50, 15, 50, 85)]


def _body_fricative() -> list[Stroke]:
    return [Path.polyline([(50, 15), (60, 29), (40, 43), (60, 57), (40, 71), (50, 85)])]


def _body_affricate() -> list[Stroke]:
    return [Line(50, 28, 50, 85), Path.polyline([(50, 15), (60, 20), (40, 24), (50, 28)])]


def _body_nasal() -> list[Stroke]:
    return [Line(50, 15, 50, 68), Circle(50, 76, 10)]


def _body_trill() -> list[Stroke]:
    return [Path.polyline([(50, 15), (63, 27), (37, 39), (63, 51), (37, 63), (63, 75), (50, 85)])]


def _body_tap() -> list[Stroke]:
    return [Line(50, 34, 50, 74)]


def _body_lateral_approx() -> list[Stroke]:
    return [Line(50, 15, 50, 85), Line(50, 50, 73, 37)]


def _body_lateral_fricative() -> list[Stroke]:
    return _body_fricative() + [Line(50, 50, 73, 37)]


def _body_approximant() -> list[Stroke]:
    return [Path.quad(42, 15, 72, 50, 42, 85)]


_MANNER_BODY = {
    Manner.PLOSIVE: _body_plosive,
    Manner.FRICATIVE: _body_fricative,
    Manner.AFFRICATE: _body_affricate,
    Manner.NASAL: _body_nasal,
    Manner.TRILL: _body_trill,
    Manner.TAP: _body_tap,
    Manner.LATERAL_APPROXIMANT: _body_lateral_approx,
    Manner.LATERAL_FRICATIVE: _body_lateral_fricative,
    Manner.APPROXIMANT: _body_approximant,
}


def consonant_glyph(c: Consonant, *, voiced_mark: str = "dot") -> Glyph:
    strokes: list[Stroke] = list(_MANNER_BODY[c.manner]())
    # Place cross-tick.
    y = _PLACE_Y[c.place]
    strokes.append(Line(40, y, 60, y))
    # Voicing mark on top.
    if c.voicing is Voicing.VOICED:
        if voiced_mark == "bar":
            strokes.append(Line(42, 9, 58, 9))
        else:
            strokes.append(Circle(50, 9, 4, filled=True))
    return Glyph(tuple(strokes))


def vowel_glyph(v: Vowel) -> Glyph:
    cx = _BACKNESS_X[v.backness]
    cy = _HEIGHT_Y[v.height]
    r = 15.0
    strokes: list[Stroke] = []
    if v.rounded:
        strokes.append(Circle(cx, cy, r))
    else:
        # Open 'C'-shaped loop (open toward the right).
        strokes.append(Path.quad(cx + r * 0.7, cy - r, cx - r, cy, cx + r * 0.7, cy + r))
    # A stem grounding the loop to the baseline so it reads as a standalone letter.
    strokes.append(Line(cx, cy + r, cx, 90))
    if v.long:
        strokes.append(Line(cx + 8, cy + r, cx + 8, 90))
    return Glyph(tuple(strokes))


def vowel_diacritic(v: Vowel) -> Glyph:
    """A small mark encoding a vowel, for abugida/syllabary composition (placed up top).

    It is the *full* vowel glyph scaled down and moved to the top of the cell, so the
    diacritic is as distinctive as the standalone vowel — height, backness, rounding (and
    length) all survive. A mark that encoded only backness/rounding would collapse many
    distinct vowels onto the same shape.
    """
    return vowel_glyph(v).scaled(0.30, dx=35.0, dy=0.0)
