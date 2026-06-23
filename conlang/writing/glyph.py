"""The glyph model: vector strokes on a 0..100 box, rendered to SVG.

A :class:`Glyph` is a bag of strokes drawn in a 100x100 coordinate box (origin top-left,
the SVG convention). Strokes are deliberately few and simple — a straight :class:`Line`,
a :class:`Path` (for polylines and quadratic curves), and a :class:`Circle` — which is
enough to compose featural letters while keeping the SVG small and inspectable.

Per-language rendering options (stroke weight, slant, whether the voicing mark is a dot or
a bar) live in :class:`Style` and are applied at render time, so the same featural
geometry can be re-skinned per language without regenerating glyphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Style:
    stroke_width: float = 6.0
    slant: float = 0.0          # skewX degrees applied to the whole glyph
    voiced_mark: str = "dot"    # "dot" | "bar"
    color: str = "#222222"


# --- Strokes ------------------------------------------------------------------------
@dataclass(frozen=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_svg(self) -> str:
        return f'<line x1="{_n(self.x1)}" y1="{_n(self.y1)}" x2="{_n(self.x2)}" y2="{_n(self.y2)}"/>'


@dataclass(frozen=True)
class Path:
    """An arbitrary SVG path ``d`` string (used for polylines and quadratic curves)."""

    d: str

    def to_svg(self) -> str:
        return f'<path d="{self.d}"/>'

    @classmethod
    def polyline(cls, points: list[tuple[float, float]]) -> "Path":
        head = f"M {_n(points[0][0])} {_n(points[0][1])}"
        rest = " ".join(f"L {_n(x)} {_n(y)}" for x, y in points[1:])
        return cls(f"{head} {rest}".strip())

    @classmethod
    def quad(cls, x1, y1, cx, cy, x2, y2) -> "Path":
        return cls(f"M {_n(x1)} {_n(y1)} Q {_n(cx)} {_n(cy)} {_n(x2)} {_n(y2)}")


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    filled: bool = False

    def to_svg(self) -> str:
        fill = "currentColor" if self.filled else "none"
        return f'<circle cx="{_n(self.cx)}" cy="{_n(self.cy)}" r="{_n(self.r)}" fill="{fill}"/>'


Stroke = Line | Path | Circle


# --- Glyph --------------------------------------------------------------------------
@dataclass(frozen=True)
class Glyph:
    strokes: tuple[Stroke, ...] = field(default_factory=tuple)

    def add(self, *strokes: Stroke) -> "Glyph":
        return Glyph(self.strokes + tuple(strokes))

    def overlay(self, other: "Glyph") -> "Glyph":
        """Combine two glyphs (e.g. a consonant base plus a vowel diacritic)."""
        return Glyph(self.strokes + other.strokes)

    def scaled(self, factor: float, dx: float = 0.0, dy: float = 0.0) -> "Glyph":
        """Return a copy with every coordinate scaled and translated (for diacritics)."""
        return Glyph(tuple(_scale_stroke(s, factor, dx, dy) for s in self.strokes))

    def to_svg_group(self, style: Style, transform: str = "") -> str:
        inner = "".join(s.to_svg() for s in self.strokes)
        # `color` is set so filled strokes (which use fill="currentColor") match the
        # stroke colour instead of falling back to inherited black.
        attrs = (
            f'color="{style.color}" fill="none" stroke="{style.color}" '
            f'stroke-width="{_n(style.stroke_width)}" '
            f'stroke-linecap="round" stroke-linejoin="round"'
        )
        t = f' transform="{transform}"' if transform else ""
        return f"<g {attrs}{t}>{inner}</g>"

    def svg(self, style: Style | None = None, size: int = 80) -> str:
        """A complete standalone SVG document for this glyph."""
        style = style or Style()
        transform = slant_transform(style.slant)
        body = self.to_svg_group(style, transform)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 100 100">{body}</svg>'
        )


# --- Helpers ------------------------------------------------------------------------
def _n(value: float) -> str:
    """Format a number compactly (no trailing zeros, no scientific notation)."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def slant_transform(slant: float) -> str:
    """An SVG transform that skews a glyph by *slant* degrees about its centre."""
    if not slant:
        return ""
    return f"translate(50 0) skewX({_n(slant)}) translate(-50 0)"


def _scale_stroke(stroke: Stroke, f: float, dx: float, dy: float) -> Stroke:
    def sx(x: float) -> float:
        return x * f + dx

    def sy(y: float) -> float:
        return y * f + dy

    if isinstance(stroke, Line):
        return Line(sx(stroke.x1), sy(stroke.y1), sx(stroke.x2), sy(stroke.y2))
    if isinstance(stroke, Circle):
        return Circle(sx(stroke.cx), sy(stroke.cy), stroke.r * f, stroke.filled)
    return _scaled_path(stroke, f, dx, dy)


def _scaled_path(path: Path, f: float, dx: float, dy: float) -> Path:
    # Scale numeric tokens in the d-string in pairs following a command letter.
    tokens = path.d.replace(",", " ").split()
    out: list[str] = []
    coord_index = 0
    for tok in tokens:
        if tok.isalpha():
            out.append(tok)
            coord_index = 0
        else:
            val = float(tok)
            val = val * f + (dx if coord_index % 2 == 0 else dy)
            out.append(_n(val))
            coord_index += 1
    return Path(" ".join(out))
