"""Desktop GUI capstone for the conlang toolkit (Tkinter, standard-library only).

The pure, testable pieces — :mod:`conlang.gui.viewmodel` (display data) and
:mod:`conlang.gui.drawops` (glyph geometry) — import no GUI toolkit. Only :func:`run`
touches Tkinter, and it is imported lazily so headless environments can still import this
package (and its pure submodules) without a display.
"""

from __future__ import annotations


def run(seed: int | None = None, *, sandhi: bool = False) -> int:
    """Open the desktop app (lazy Tkinter import so the package imports without a display)."""
    from conlang.gui.app import run as _run

    return _run(seed, sandhi=sandhi)
