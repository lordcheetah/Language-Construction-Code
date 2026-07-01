"""The writing system: a type, a glyph per unit, and SVG rendering of words and charts.

Four script types are modelled, matching the cross-linguistic typology:

- ``ALPHABET``  — a glyph for every consonant and every vowel.
- ``ABJAD``     — consonants only; vowels go unwritten.
- ``ABUGIDA``   — a consonant glyph carries an inherent vowel; other vowels add a
  diacritic, and the inherent vowel is left unmarked.
- ``SYLLABARY`` — every consonant-vowel sequence is one composed glyph.

Rendering produces SVG: :meth:`WritingSystem.word_svg` lays a word out in the script's
reading direction (left-to-right, right-to-left, or top-to-bottom — see
:class:`WritingDirection`), and :meth:`WritingSystem.chart_svg` draws the script's full
inventory as a labelled grid. The chart and the numeral rendering stay left-to-right
regardless of direction (a reference grid has no reading direction, and positional numerals
are conventionally left-to-right even in right-to-left scripts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from conlang.phonology.features import Segment
from conlang.writing.glyph import Glyph, Style, Line, Path, Circle, slant_transform, _n


class WritingSystemType(Enum):
    ALPHABET = "alphabet"
    ABJAD = "abjad"
    ABUGIDA = "abugida"
    SYLLABARY = "syllabary"


class WritingDirection(Enum):
    """The direction running text flows (the reference chart and numerals stay left-to-right)."""

    LTR = "left-to-right"
    RTL = "right-to-left"
    TTB = "top-to-bottom"


# A neutral vertical carrier used to display a vowel diacritic on its own in a chart.
_CARRIER = Glyph((Line(50, 25, 50, 85),))

# A "vowel-killer" (virama-like) mark: a consonant with no vowel of its own — a coda or a
# cluster member — carries it so it is not confused with the bare inherent-vowel form.
_VIRAMA = Glyph((Line(40, 92, 60, 84),))


def _stack_glyphs(glyphs: Sequence[Glyph]) -> Glyph:
    """Stack consonant glyphs into one conjunct, top to bottom (Brahmic-style).

    Each glyph is scaled down and offset vertically so a cluster reads as a single stacked
    ligature — the way Devanagari writes क्ष (k+ṣ) as one conjunct rather than two letters.
    """
    f = 0.9 / len(glyphs)
    dx = 50 * (1 - f)  # centre each scaled glyph horizontally
    out = Glyph()
    for k, g in enumerate(glyphs):
        out = out.overlay(g.scaled(f, dx=dx, dy=k * (100 * f) + 5))
    return out


def _join_runs(joinable: Sequence[bool]) -> list[tuple[int, int]]:
    """Index ranges (start, end) of each maximal run of ≥2 adjacent joinable cells."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(joinable)
    while i < n:
        if joinable[i]:
            j = i
            while j + 1 < n and joinable[j + 1]:
                j += 1
            if j > i:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs

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
    digit_glyphs: dict[int, Glyph] = field(default_factory=dict)  # digit value -> glyph
    punctuation: dict[str, Glyph] = field(default_factory=dict)   # "stop"/"pause"/"word" marks
    direction: WritingDirection = WritingDirection.LTR           # flow of running text
    cursive: bool = False        # glyphs in a word are joined by a baseline connecting stroke
    stack_clusters: bool = False  # (abugida) a consonant cluster stacks into one conjunct glyph

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
        # Conjunct stacking is only meaningful for an abugida — a syllabary already fuses CV.
        stack = self.stack_clusters and not compose_inherent
        units: list[tuple[str, Glyph]] = []
        i, n = 0, len(segments)
        while i < n:
            s = segments[i]
            if s.is_consonant and s.ipa in self.consonants:
                if stack:
                    # Gather a maximal adjacent-consonant run. (A real script forms conjuncts
                    # within a syllable/onset; with no syllabifier here we stack any run.)
                    run, j = [], i
                    while j < n and segments[j].is_consonant and segments[j].ipa in self.consonants:
                        run.append(segments[j])
                        j += 1
                    if len(run) >= 2:  # a cluster: stack it into one conjunct glyph
                        conjunct = _stack_glyphs([self.consonants[c.ipa] for c in run])
                        label = "".join(c.ipa for c in run)
                        nxt = segments[j] if j < n else None
                        if nxt is not None and nxt.is_vowel and nxt.ipa in self.diacritics:
                            if nxt.ipa == self.inherent_vowel:
                                units.append((label, conjunct))  # inherent vowel: bare
                            else:
                                units.append((label + nxt.ipa, conjunct.overlay(self.diacritics[nxt.ipa])))
                            i = j + 1
                        else:
                            units.append((label, conjunct.overlay(_VIRAMA)))
                            i = j
                        continue
                    # a lone consonant (not a cluster): fall through to per-consonant handling
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

    def _row_svg(
        self, glyphs: Sequence[Glyph], size: int, joinable: Sequence[bool] | None = None
    ) -> str:
        """Lay out a line of glyphs (one 100-unit cell each) in the script's direction.

        Left-to-right and right-to-left run horizontally (RTL places the first glyph at the
        right); top-to-bottom stacks the cells vertically. *joinable* marks which cells are
        letters that a cursive script strings together (defaults to all); word dividers and
        punctuation pass ``False`` so the connecting stroke breaks between words.
        """
        glyphs = list(glyphs) or [Glyph()]
        cell, n = 100, len(glyphs)
        vertical = self.direction is WritingDirection.TTB
        if joinable is None:
            joinable = [True] * n
        xs, cells = [], []
        for i, glyph in enumerate(glyphs):
            if vertical:
                x, y = 0, i * cell
            elif self.direction is WritingDirection.RTL:
                x, y = (n - 1 - i) * cell, 0
            else:
                x, y = i * cell, 0
            xs.append(x)
            inner = glyph.to_svg_group(self.style, slant_transform(self.style.slant))
            cells.append(f'<g transform="translate({x} {y})">{inner}</g>')
        # A cursive script joins adjacent letters with a connecting baseline stroke; one
        # stroke per run of joined cells, so it breaks at word gaps and punctuation. Each is a
        # bare <line> (not a positioned <g> cell) so it never counts as a glyph slot; drawn
        # before the cells so the glyphs sit on top.
        joins = []
        if self.cursive and not vertical:
            for a, b in _join_runs(joinable):
                left, right = min(xs[a], xs[b]), max(xs[a], xs[b])
                joins.append(
                    f'<line x1="{left + 15}" y1="78" x2="{right + 85}" y2="78" '
                    f'stroke="{self.style.color}" stroke-width="{_n(self.style.stroke_width)}" '
                    f'stroke-linecap="round"/>'
                )
        vb_w, vb_h = (cell, n * cell) if vertical else (n * cell, cell)
        px_w, px_h = (size, n * size) if vertical else (n * size, size)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{px_w}" '
            f'height="{px_h}" viewBox="0 0 {vb_w} {vb_h}">{"".join(joins)}{"".join(cells)}</svg>'
        )

    def word_svg(self, segments: Sequence[Segment], size: int = 80) -> str:
        return self._row_svg([g for _, g in self.render_segments(segments)], size)

    def sentence_svg(
        self, words: Sequence[Sequence[Segment]], terminator: str = "stop", size: int = 80
    ) -> str:
        """Render several words in a row, separated by the word divider and closed by a
        terminal mark. A missing divider/terminator just leaves a blank cell / no mark, so a
        script with no punctuation degrades to spaced (or run-on) writing."""
        divider = self.punctuation.get("word", Glyph())
        glyphs: list[Glyph] = []
        joinable: list[bool] = []  # dividers/punctuation are False so cursive joins break there
        for i, word in enumerate(words):
            if i > 0:
                glyphs.append(divider)
                joinable.append(False)
            for _, g in self.render_segments(word):
                glyphs.append(g)
                joinable.append(True)
        terminal = self.punctuation.get(terminator)
        if terminal is not None:
            glyphs.append(terminal)
            joinable.append(False)
        return self._row_svg(glyphs, size, joinable)

    # --- Numbers ---------------------------------------------------------------------
    def number_svg(self, n: int, base: int, size: int = 80) -> str:
        """Render a non-negative integer in positional notation with the digit glyphs.

        This is place-value notation and is independent of how the language's spelled-out
        numeral *words* group (those have their own multiplier/unit order).
        """
        if base > len(self.digit_glyphs):
            raise ValueError(
                f"base {base} exceeds the available digit glyphs ({len(self.digit_glyphs)})"
            )
        digits = _to_base_digits(n, base)
        cell = 100
        groups = []
        for i, d in enumerate(digits):
            glyph = self.digit_glyphs.get(d, Glyph())
            inner = glyph.to_svg_group(self.style, slant_transform(self.style.slant))
            groups.append(f'<g transform="translate({i * cell} 0)">{inner}</g>')
        width = len(digits) * cell
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{len(digits) * size}" '
            f'height="{size}" viewBox="0 0 {width} 100">{"".join(groups)}</svg>'
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
        traits = [self.direction.value]
        if self.cursive:
            traits.append("cursive (joined)")
        if self.stack_clusters and self.type is WritingSystemType.ABUGIDA:
            traits.append("stacked conjuncts")
        return (
            f"{line}\n"
            f"  {', '.join(traits)}\n"
            f"  {len(self.consonants)} consonant glyphs, {len(self.vowels)} vowel glyphs, "
            f"{len(self.chart_cells())} chart cells"
        )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --- Numeral glyphs (Maya-style bars and dots) --------------------------------------
def maya_digit(v: int) -> Glyph:
    """A digit glyph in the bars-and-dots style: each bar is five, each dot is one.

    Attested (Maya, and similar to Babylonian grouping), it scales cleanly to any base up
    to ~20 and stays visually coherent: e.g. 7 is one bar with two dots, 13 is two bars
    with three dots. (A stylization: true Maya stacks place-values vertically; here digits
    advance in a horizontal row like the scripts.)
    """
    if v == 0:  # a shell-like lens, as Maya wrote zero
        return Glyph((Path("M 28 55 Q 50 44 72 55 Q 50 66 28 55 Z"), Line(38, 55, 62, 55)))

    bars, dots = divmod(v, 5)
    strokes: list = []
    y = 86.0
    for _ in range(bars):
        strokes.append(Line(22, y, 78, y))
        y -= 13
    dot_y = (y - 13) if bars else 40
    for i in range(dots):
        x = 50 + (i - (dots - 1) / 2) * 16
        strokes.append(Circle(x, dot_y, 5, filled=True))
    return Glyph(tuple(strokes))


def build_digit_glyphs(max_digit: int = 19) -> dict[int, Glyph]:
    return {v: maya_digit(v) for v in range(max_digit + 1)}


# --- Punctuation marks (pure geometry, like the digits) -----------------------------
def build_punctuation() -> dict[str, Glyph]:
    """Three punctuation marks, in the spirit of the Brahmic daṇḍa and the interpunct:

    - ``stop``  — a full-height vertical stroke ending a sentence (the daṇḍa).
    - ``pause`` — a half-height stroke for a clause/phrase break.
    - ``word``  — a centred dot dividing words (an interpunct), for scripts that mark word
      boundaries rather than leaving them blank.
    """
    return {
        "stop": Glyph((Line(50, 14, 50, 90),)),
        "pause": Glyph((Line(50, 52, 50, 90),)),
        "word": Glyph((Circle(50, 52, 4, filled=True),)),
    }


def _to_base_digits(n: int, base: int) -> list[int]:
    if n < 0:
        raise ValueError("cannot render a negative number")
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(n % base)
        n //= base
    return digits[::-1]
