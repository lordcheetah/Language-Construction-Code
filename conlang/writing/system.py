"""The writing system: a type, a glyph per unit, and SVG rendering of words and charts.

Four script types are modelled, matching the cross-linguistic typology:

- ``ALPHABET``  — a glyph for every consonant and every vowel.
- ``ABJAD``     — consonants only; vowels go unwritten.
- ``ABUGIDA``   — a consonant glyph carries an inherent vowel; other vowels add a
  diacritic, and the inherent vowel is left unmarked.
- ``SYLLABARY`` — every consonant-vowel sequence is one composed glyph.

Rendering produces SVG: :meth:`WritingSystem.word_svg` lays a word out left to right, and
:meth:`WritingSystem.chart_svg` draws the script's full inventory as a labelled grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from conlang.phonology.features import Segment
from conlang.writing.glyph import Glyph, Style, Line, slant_transform


class WritingSystemType(Enum):
    ALPHABET = "alphabet"
    ABJAD = "abjad"
    ABUGIDA = "abugida"
    SYLLABARY = "syllabary"


# A neutral vertical carrier used to display a vowel diacritic on its own in a chart.
_CARRIER = Glyph((Line(50, 25, 50, 85),))

# A "vowel-killer" (virama-like) mark: a consonant with no vowel of its own — a coda or a
# cluster member — carries it so it is not confused with the bare inherent-vowel form.
_VIRAMA = Glyph((Line(40, 92, 60, 84),))

_CELL_W = 110
_CELL_H = 135


@dataclass
class WritingSystem:
    type: WritingSystemType
    consonants: dict[str, Glyph]            # ipa -> consonant glyph
    vowels: dict[str, Glyph]                # ipa -> full vowel glyph
    diacritics: dict[str, Glyph]            # ipa -> vowel diacritic
    style: Style = field(default_factory=Style)
    inherent_vowel: str | None = None       # ipa of the abugida's inherent vowel

    # --- Rendering a word ------------------------------------------------------------
    def render_segments(self, segments: Sequence[Segment]) -> list[tuple[str, Glyph]]:
        """Break a segment sequence into (label, glyph) writing units for this script."""
        if self.type is WritingSystemType.ALPHABET:
            return [(s.ipa, self._glyph_for(s)) for s in segments if self._glyph_for(s)]
        if self.type is WritingSystemType.ABJAD:
            return [(s.ipa, self.consonants[s.ipa]) for s in segments
                    if s.is_consonant and s.ipa in self.consonants]
        compose_inherent = self.type is WritingSystemType.SYLLABARY
        return self._cv_units(segments, compose_inherent)

    def _glyph_for(self, s: Segment) -> Glyph | None:
        if s.is_consonant:
            return self.consonants.get(s.ipa)
        return self.vowels.get(s.ipa)

    def _cv_units(self, segments: Sequence[Segment], compose_inherent: bool):
        units: list[tuple[str, Glyph]] = []
        i, n = 0, len(segments)
        while i < n:
            s = segments[i]
            if s.is_consonant and s.ipa in self.consonants:
                base = self.consonants[s.ipa]
                nxt = segments[i + 1] if i + 1 < n else None
                if nxt is not None and nxt.is_vowel and nxt.ipa in self.diacritics:
                    if not compose_inherent and nxt.ipa == self.inherent_vowel:
                        units.append((s.ipa, base))  # inherent vowel: written bare
                    else:
                        units.append((s.ipa + nxt.ipa, base.overlay(self.diacritics[nxt.ipa])))
                    i += 2
                    continue
                # No following vowel: a coda or cluster consonant. Mark it vowelless so it
                # is distinct from the bare (inherent-vowel) form.
                units.append((s.ipa, base.overlay(_VIRAMA)))
                i += 1
            elif s.is_vowel and s.ipa in self.vowels:
                units.append((s.ipa, self.vowels[s.ipa]))
                i += 1
            else:
                i += 1  # unknown segment: skip
        return units

    def word_svg(self, segments: Sequence[Segment], size: int = 80) -> str:
        units = self.render_segments(segments)
        if not units:
            units = [("", Glyph())]
        cell = 100
        groups = []
        for i, (_, glyph) in enumerate(units):
            inner = glyph.to_svg_group(self.style, slant_transform(self.style.slant))
            groups.append(f'<g transform="translate({i * cell} 0)">{inner}</g>')
        width = len(units) * cell
        height = size
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{len(units) * size}" '
            f'height="{height}" viewBox="0 0 {width} 100">{"".join(groups)}</svg>'
        )

    # --- Rendering the chart ---------------------------------------------------------
    def chart_cells(self) -> list[tuple[str, Glyph]]:
        if self.type is WritingSystemType.ABJAD:
            return list(self.consonants.items())
        if self.type is WritingSystemType.ALPHABET:
            return [*self.consonants.items(), *self.vowels.items()]
        if self.type is WritingSystemType.ABUGIDA:
            cells = list(self.consonants.items())
            cells += [(ipa, _CARRIER.overlay(d)) for ipa, d in self.diacritics.items()]
            return cells
        # SYLLABARY: consonant x vowel grid.
        cells = []
        for c_ipa, c_glyph in self.consonants.items():
            for v_ipa, d in self.diacritics.items():
                cells.append((c_ipa + v_ipa, c_glyph.overlay(d)))
        return cells

    def chart_columns(self) -> int:
        if self.type is WritingSystemType.SYLLABARY and self.diacritics:
            return len(self.diacritics)
        return 6

    def chart_svg(self) -> str:
        cells = self.chart_cells()
        columns = max(1, self.chart_columns())
        rows = (len(cells) + columns - 1) // columns
        parts = []
        for idx, (label, glyph) in enumerate(cells):
            col, row = idx % columns, idx // columns
            x, y = col * _CELL_W, row * _CELL_H
            inner = glyph.to_svg_group(self.style, slant_transform(self.style.slant))
            parts.append(f'<g transform="translate({x + 5} {y + 5})">{inner}</g>')
            parts.append(
                f'<text x="{x + _CELL_W / 2:.0f}" y="{y + 122}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="20" fill="#222">{_esc(label)}</text>'
            )
        width = columns * _CELL_W
        height = rows * _CELL_H
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">{"".join(parts)}</svg>'
        )

    def summary(self) -> str:
        line = f"Writing system: {self.type.value}"
        if self.type is WritingSystemType.ABUGIDA and self.inherent_vowel:
            line += f" (inherent vowel /{self.inherent_vowel}/)"
        return (
            f"{line}\n"
            f"  {len(self.consonants)} consonant glyphs, {len(self.vowels)} vowel glyphs, "
            f"{len(self.chart_cells())} chart cells"
        )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
