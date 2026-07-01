"""Turn a :class:`~conlang.writing.glyph.Glyph` into flat drawing operations.

This is the Tkinter-canvas parallel of the SVG renderer in ``conlang.writing``: the glyph
model is already vector primitives (a :class:`Line`, a :class:`Path`, a :class:`Circle`
on a 0..100 box), so a canvas can draw them directly — no SVG rasteriser, no dependency.

The functions here are deliberately **pure and Tkinter-free**: a glyph becomes a list of
``(kind, ...)`` tuples in the glyph's own 0..100 coordinate space (with the style slant
already baked in), and the caller scales/translates them onto a real canvas. Keeping the
geometry testable without a display mirrors the pure-logic/I-O split used elsewhere.
"""

from __future__ import annotations

import math
from typing import Sequence

from conlang.writing.glyph import Glyph, Line, Path, Circle

# Draw ops (all coordinates in the 0..100 glyph box):
#   ("line", x1, y1, x2, y2)
#   ("oval", x1, y1, x2, y2, filled)   -- bounding box of a circle
#   ("poly", [(x, y), ...], filled)    -- a polyline (curves are pre-sampled)
DrawOp = tuple

_QUAD_STEPS = 12  # segments a quadratic curve is sampled into


def glyph_drawops(glyph: Glyph, slant: float = 0.0) -> list[DrawOp]:
    """Flatten a glyph into draw ops, applying the style *slant* (skewX, in degrees)."""
    t = math.tan(math.radians(slant))

    def skew(x: float, y: float) -> tuple[float, float]:
        # SVG skewX about the box centre reduces to x += tan(a)*y (the centring cancels).
        return (x + t * y, y)

    ops: list[DrawOp] = []
    for s in glyph.strokes:
        if isinstance(s, Line):
            x1, y1 = skew(s.x1, s.y1)
            x2, y2 = skew(s.x2, s.y2)
            ops.append(("line", x1, y1, x2, y2))
        elif isinstance(s, Circle):
            cx, cy = skew(s.cx, s.cy)  # a slanted circle is drawn as an upright oval (slants are small)
            ops.append(("oval", cx - s.r, cy - s.r, cx + s.r, cy + s.r, s.filled))
        elif isinstance(s, Path):
            for pts in _parse_path(s.d):
                ops.append(("poly", [skew(x, y) for x, y in pts], False))
    return ops


def _parse_path(d: str) -> list[list[tuple[float, float]]]:
    """Parse an SVG path ``d`` (only the M/L/Q the glyph model emits) into polylines."""
    tokens = d.replace(",", " ").split()
    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    i = 0

    def num(k: int) -> float:
        return float(tokens[k])

    while i < len(tokens):
        cmd = tokens[i]
        if cmd == "M":
            if current:
                polylines.append(current)
            current = [(num(i + 1), num(i + 2))]
            i += 3
        elif cmd == "L":
            current.append((num(i + 1), num(i + 2)))
            i += 3
        elif cmd == "Q":
            cx, cy, x2, y2 = num(i + 1), num(i + 2), num(i + 3), num(i + 4)
            current.extend(_sample_quad(current[-1], (cx, cy), (x2, y2)))
            i += 5
        elif cmd in ("Z", "z"):  # closepath: return to the subpath's start
            if current:
                current.append(current[0])
            i += 1
        else:  # a bare coordinate pair or unknown command: skip defensively
            i += 1
    if current:
        polylines.append(current)
    return polylines


def _sample_quad(p0, p1, p2, steps: int = _QUAD_STEPS) -> list[tuple[float, float]]:
    """Sample a quadratic Bézier into ``steps`` line segments (skips the start point)."""
    pts = []
    for k in range(1, steps + 1):
        u = k / steps
        mu = 1 - u
        x = mu * mu * p0[0] + 2 * mu * u * p1[0] + u * u * p2[0]
        y = mu * mu * p0[1] + 2 * mu * u * p1[1] + u * u * p2[1]
        pts.append((x, y))
    return pts


def cell_positions(count: int, direction: str) -> list[tuple[int, int]]:
    """Grid offsets (in 100-unit cells) for *count* glyphs laid out in a reading direction.

    Mirrors ``WritingSystem._row_svg``: left-to-right and right-to-left run horizontally
    (RTL puts the first glyph on the right), top-to-bottom stacks the cells downward.
    """
    n = max(count, 1)
    if direction == "top-to-bottom":
        return [(0, i) for i in range(n)]
    if direction == "right-to-left":
        return [(n - 1 - i, 0) for i in range(n)]
    return [(i, 0) for i in range(n)]
