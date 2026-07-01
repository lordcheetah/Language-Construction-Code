"""Tests for the desktop-GUI capstone.

Only the *pure* pieces are exercised — the view model (display data) and the glyph draw-op
geometry — since both are Tkinter-free by design. The app widgets are thin I/O and need a
display, so they are left out of the headless suite. The key invariant checked here is the
same cross-front-end determinism the other front-ends guarantee: a seed fully determines
what the GUI shows.
"""

import inspect
import math

import pytest

from conlang.language import Language
from conlang.writing.glyph import Glyph, Line, Circle, Path
from conlang.gui import drawops, viewmodel
from conlang.gui.drawops import glyph_drawops, cell_positions, _parse_path, _QUAD_STEPS
from conlang.gui.viewmodel import build_view


# --- View model ---------------------------------------------------------------------
def test_build_view_is_deterministic():
    a, b = build_view(7), build_view(7)
    assert [r.roman for r in a.vocab] == [r.roman for r in b.vocab]
    assert [s.interlinear for s in a.sentences] == [s.interlinear for s in b.sentences]
    assert a.numbers == b.numbers
    assert a.overview == b.overview


def test_view_content_matches_the_language():
    view = build_view(3)
    lang = Language.generate(3)
    for row in view.vocab:
        entry = lang.lexicon.get(row.gloss)
        assert entry is not None and row.roman == entry.roman and row.ipa == entry.ipa
    for n, word in view.numbers:
        assert word == lang.numerals.number(n).roman
    assert view.script_type == lang.writing.type.value
    assert view.direction == lang.writing.direction.value


def test_random_view_regenerates_from_its_recorded_seed():
    # build_view(None) rolls a language and records its concrete seed; rebuilding from that
    # seed must reproduce the same content (the persistence story the GUI relies on).
    view = build_view(None)
    assert isinstance(view.seed, int)
    again = build_view(view.seed)
    assert [r.roman for r in again.vocab] == [r.roman for r in view.vocab]
    assert again.overview == view.overview


def test_view_exposes_renderable_writing_units():
    view = build_view(11)
    assert len(view.chart_cells()) > 0
    assert all(isinstance(g, Glyph) for _, g in view.chart_cells())
    assert all(isinstance(g, Glyph) for _, g in view.word_units())
    assert view.sample_gloss  # a real dictionary headword to render


def test_pure_modules_do_not_import_tkinter():
    # These must stay importable (and testable) on a headless machine with no display.
    for module in (drawops, viewmodel):
        assert "tkinter" not in inspect.getsource(module)


# --- Draw ops -----------------------------------------------------------------------
def test_line_glyph_becomes_one_line_op():
    ops = glyph_drawops(Glyph((Line(10, 20, 30, 40),)))
    assert ops == [("line", 10, 20, 30, 40)]


def test_slant_shears_x_by_tan_times_y():
    slant = 12.0
    ops = glyph_drawops(Glyph((Line(0, 0, 0, 100),)), slant)
    (_, x1, y1, x2, y2), = ops
    assert (x1, y1) == (0, 0)  # y=0 is unaffected
    assert x2 == pytest.approx(math.tan(math.radians(slant)) * 100)  # y=100 shifts by tan*y


def test_circle_becomes_its_bounding_oval():
    ops = glyph_drawops(Glyph((Circle(50, 50, 20, filled=True),)))
    assert ops == [("oval", 30, 30, 70, 70, True)]


def test_path_moveto_lineto_keeps_exact_points():
    polylines = _parse_path("M 0 0 L 100 100 L 0 100")
    assert polylines == [[(0.0, 0.0), (100.0, 100.0), (0.0, 100.0)]]


def test_quadratic_curve_is_sampled_and_ends_on_target():
    ops = glyph_drawops(Glyph((Path(Path.quad(0, 0, 50, 100, 100, 0).d),)))
    (kind, pts, filled), = ops
    assert kind == "poly" and filled is False
    assert pts[0] == (0.0, 0.0)                       # start preserved
    assert pts[-1] == pytest.approx((100.0, 0.0))     # ends on the Q target
    assert len(pts) == _QUAD_STEPS + 1                # start + sampled points


def test_closepath_returns_to_the_subpath_start():
    polylines = _parse_path("M 10 10 L 90 10 L 90 90 Z")
    assert polylines == [[(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 10.0)]]


def test_cell_positions_follow_reading_direction():
    assert cell_positions(3, "left-to-right") == [(0, 0), (1, 0), (2, 0)]
    assert cell_positions(3, "right-to-left") == [(2, 0), (1, 0), (0, 0)]
    assert cell_positions(3, "top-to-bottom") == [(0, 0), (0, 1), (0, 2)]
    assert cell_positions(0, "left-to-right") == [(0, 0)]  # never empty
