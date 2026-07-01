"""A Tkinter desktop front-end over the whole conlang engine.

Tkinter is part of the Python standard library, so the GUI keeps the project's pure-Python,
zero-dependency, offline promise — no Qt, no browser, no packaging surprises. The window
regenerates a language from a seed and shows it across tabs: an overview, the dictionary
(with a Speak button that drives the formant synthesiser), glossed sentences, the numerals,
and the native script drawn straight onto a canvas from the glyph stroke primitives.

All display data comes from :mod:`conlang.gui.viewmodel` and all glyph geometry from
:mod:`conlang.gui.drawops`; this module is just widgets and event wiring.
"""

from __future__ import annotations

import atexit
import os
import random
import sys
import tempfile
import tkinter as tk
from tkinter import ttk

from conlang.gui.drawops import glyph_drawops, cell_positions
from conlang.gui.viewmodel import build_view, LanguageView

_CELL = 56          # on-screen size of one glyph cell, in pixels
_CHART_COLS = 8     # glyphs per row in the inventory chart


class ConlangApp:
    def __init__(self, root: tk.Tk, seed: int | None = None, sandhi: bool = False):
        self.root = root
        self.sandhi = sandhi
        self.view: LanguageView | None = None
        self._render_job: str | None = None  # pending debounced canvas redraw
        self._last_wav: str | None = None     # temp clip from the previous Speak
        root.title("Conlang Studio")
        root.geometry("860x640")
        self._build_toolbar()
        self._build_tabs()
        self.generate(seed)

    # --- Layout ----------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Seed:").pack(side=tk.LEFT)
        self.seed_var = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.seed_var, width=22)
        entry.pack(side=tk.LEFT, padx=(4, 8))
        entry.bind("<Return>", lambda _e: self._on_generate())
        ttk.Button(bar, text="Generate", command=self._on_generate).pack(side=tk.LEFT)
        ttk.Button(bar, text="Random", command=lambda: self.generate(None)).pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(bar, text="")
        self.status.pack(side=tk.RIGHT)

    def _build_tabs(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.overview_text = self._text_tab(nb, "Overview")
        self._build_dictionary_tab(nb)
        self.sentences_text = self._text_tab(nb, "Sentences")
        self._build_numbers_tab(nb)
        self._build_script_tab(nb)

    def _text_tab(self, nb: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        text = tk.Text(frame, wrap="word", font=("Consolas", 11), padx=8, pady=8,
                       borderwidth=0, background="#fbfbfb")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set, state=tk.DISABLED)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return text

    def _build_dictionary_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Dictionary")
        cols = ("gloss", "roman", "ipa", "say")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=14)
        for c, w in zip(cols, (140, 140, 150, 160)):
            tree.heading(c, text={"say": "say (TTS)"}.get(c, c))
            tree.column(c, width=w, anchor=tk.W)
        scroll = ttk.Scrollbar(frame, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.dict_tree = tree
        bar = ttk.Frame(frame, padding=(0, 6))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.speak_btn = ttk.Button(bar, text="🔊  Speak selected", command=self._speak_selected)
        self.speak_btn.pack(side=tk.LEFT)
        if _audio_supported():
            tree.bind("<Double-1>", lambda _e: self._speak_selected())
        else:
            self.speak_btn.state(["disabled"])
            ttk.Label(bar, text="(audio playback is Windows-only)").pack(side=tk.LEFT, padx=8)

    def _build_numbers_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Numbers")
        tree = ttk.Treeview(frame, columns=("n", "word"), show="headings", height=12)
        tree.heading("n", text="value")
        tree.heading("word", text="word")
        tree.column("n", width=80, anchor=tk.E)
        tree.column("word", width=260, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        self.numbers_tree = tree

    def _build_script_tab(self, nb: ttk.Notebook) -> None:
        frame = ttk.Frame(nb)
        nb.add(frame, text="Script")
        self.script_header = ttk.Label(frame, padding=(8, 6))
        self.script_header.pack(side=tk.TOP, anchor=tk.W)
        canvas = tk.Canvas(frame, background="white", highlightthickness=0)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = canvas
        # Resize fires in bursts during a drag; debounce so we redraw at most ~15x/s.
        canvas.bind("<Configure>", lambda _e: self._schedule_render())

    # --- Actions ---------------------------------------------------------------------
    def _on_generate(self) -> None:
        raw = self.seed_var.get().strip()
        if not raw:
            self.generate(None)
            return
        try:
            self.generate(int(raw))
        except ValueError:
            self.status.configure(text="seed must be a whole number")

    def generate(self, seed: int | None) -> None:
        self.view = build_view(seed, sandhi=self.sandhi)
        self.seed_var.set(str(self.view.seed))
        self.status.configure(text=f"{self.view.script_type} · {self.view.direction}")
        self._refresh()

    def _refresh(self) -> None:
        view = self.view
        self._set_text(self.overview_text, view.overview)
        self._set_text(self.sentences_text, "\n\n".join(
            f"“{b.english}”\n{b.interlinear}" for b in view.sentences))
        self.dict_tree.delete(*self.dict_tree.get_children())
        for row in view.vocab:
            self.dict_tree.insert("", tk.END, values=(row.gloss, row.roman, row.ipa, row.say))
        if view.vocab:
            first = self.dict_tree.get_children()[0]
            self.dict_tree.selection_set(first)
        self.numbers_tree.delete(*self.numbers_tree.get_children())
        for n, word in view.numbers:
            self.numbers_tree.insert("", tk.END, values=(n, word))
        self.script_header.configure(
            text=f"{view.script_type} — {view.direction}   (sample word: “{view.sample_gloss}”)")
        self._render_script()

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state=tk.DISABLED)

    def _speak_selected(self) -> None:
        if self.view is None or not _audio_supported():
            return
        sel = self.dict_tree.selection()
        gloss = self.dict_tree.item(sel[0])["values"][0] if sel else None
        entry = self.view.language.lexicon.get(str(gloss)) if gloss else None
        if entry is None:
            return
        # The previous clip's async playback has finished by now; reclaim its temp file.
        if self._last_wav is not None:
            _remove_temp(self._last_wav)
            self._last_wav = None
        try:
            path = _synthesize_to_wav(self.view.language, entry, self.view.seed)
        except Exception as exc:  # synthesis is best-effort; never crash the UI
            self.status.configure(text=f"speech failed: {exc}")
            return
        self._last_wav = path
        _play_wav(path)
        self.status.configure(text=f"spoke “{entry.gloss}” ({entry.roman})")

    # --- Script rendering ------------------------------------------------------------
    def _schedule_render(self) -> None:
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(60, self._render_script)

    def _render_script(self) -> None:
        self._render_job = None
        if self.view is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        style = self.view.language.writing.style
        pad = 16
        x0 = pad
        y = pad

        y = self._canvas_label(x0, y, "Inventory")
        cells = self.view.chart_cells()
        for idx, (label, glyph) in enumerate(cells):
            col, rowi = idx % _CHART_COLS, idx // _CHART_COLS
            cx = x0 + col * (_CELL + 22)
            cy = y + rowi * (_CELL + 22)
            self._draw_glyph(glyph, cx, cy, _CELL, style)
            canvas.create_text(cx + _CELL / 2, cy + _CELL + 8, text=label,
                               font=("Consolas", 9), fill="#444")
        rows = (len(cells) + _CHART_COLS - 1) // _CHART_COLS
        y += rows * (_CELL + 22) + pad

        y = self._canvas_label(x0, y, f"Word: “{self.view.sample_gloss}”")
        y = self._draw_row(self.view.word_units(), x0, y, style) + pad

        y = self._canvas_label(x0, y, "Sentence")
        for units in self.view.sentence_words_units():
            y = self._draw_row(units, x0, y, style)
        # bbox("all") covers the true drawn extent (both axes), so wide rows stay scrollable.
        bbox = canvas.bbox("all")
        canvas.configure(scrollregion=bbox or (0, 0, canvas.winfo_width(), y + pad))

    def _canvas_label(self, x: int, y: int, text: str) -> int:
        self.canvas.create_text(x, y, text=text, anchor=tk.NW, font=("Segoe UI", 10, "bold"),
                                 fill="#333")
        return y + 22

    def _draw_row(self, units, x0: int, y: int, style) -> int:
        """Draw one row of glyph units honouring the script's reading direction; returns the
        y below the row. Cursive scripts get a connecting baseline stroke per row."""
        glyphs = [g for _, g in units] or []
        if not glyphs:
            return y + _CELL + 8
        positions = cell_positions(len(glyphs), self.view.direction)
        gap = _CELL + 10
        if self.view.language.writing.cursive and self.view.direction != "top-to-bottom" and len(glyphs) > 1:
            self.canvas.create_line(x0 + 8, y + _CELL * 0.78, x0 + (len(glyphs) - 1) * gap + _CELL - 8,
                                    y + _CELL * 0.78, fill=style.color,
                                    width=max(1.0, style.stroke_width * _CELL / 100))
        max_row = 0
        for (col, rowi), glyph in zip(positions, glyphs):
            self._draw_glyph(glyph, x0 + col * gap, y + rowi * gap, _CELL, style)
            max_row = max(max_row, rowi)
        return y + (max_row + 1) * gap + 8

    def _draw_glyph(self, glyph, ox: float, oy: float, size: int, style) -> None:
        scale = size / 100.0
        width = max(1.0, style.stroke_width * scale)
        color = style.color
        for op in glyph_drawops(glyph, style.slant):
            if op[0] == "line":
                _, x1, y1, x2, y2 = op
                self.canvas.create_line(ox + x1 * scale, oy + y1 * scale,
                                        ox + x2 * scale, oy + y2 * scale,
                                        fill=color, width=width, capstyle=tk.ROUND)
            elif op[0] == "oval":
                _, x1, y1, x2, y2, filled = op
                self.canvas.create_oval(ox + x1 * scale, oy + y1 * scale,
                                        ox + x2 * scale, oy + y2 * scale,
                                        outline=color, width=width,
                                        fill=color if filled else "")
            elif op[0] == "poly":
                _, pts, _filled = op
                if len(pts) >= 2:
                    flat = [c for x, y in pts for c in (ox + x * scale, oy + y * scale)]
                    self.canvas.create_line(*flat, fill=color, width=width,
                                            capstyle=tk.ROUND, joinstyle=tk.ROUND)


# --- Audio (optional, best-effort) ---------------------------------------------------
_TEMP_WAVS: set[str] = set()  # temp clips to unlink at exit (in case one is still tracked)


def _audio_supported() -> bool:
    return sys.platform.startswith("win")


def _remove_temp(path: str) -> None:
    _TEMP_WAVS.discard(path)
    try:
        os.remove(path)
    except OSError:  # already gone, or still locked by an async play — leave for atexit
        pass


@atexit.register
def _cleanup_temp_wavs() -> None:
    for path in list(_TEMP_WAVS):
        _remove_temp(path)


def _synthesize_to_wav(language, entry, seed: int) -> str:
    from conlang.speech.synth import Synthesizer, Voice

    synth = Synthesizer(Voice(), random.Random(seed))
    samples = synth.synthesize(list(entry.form))
    fd, path = tempfile.mkstemp(prefix="conlang-", suffix=".wav")
    os.close(fd)
    synth.write_wav(path, samples)
    _TEMP_WAVS.add(path)
    return path


def _play_wav(path: str) -> None:
    if not _audio_supported():
        return
    import winsound

    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


def run(seed: int | None = None, *, sandhi: bool = False) -> int:
    """Open the desktop app. Returns 0 on a clean close; 1 if Tkinter is unavailable."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display / no Tk build
        print(f"gui: cannot open a window ({exc}). Tkinter needs a display.", file=sys.stderr)
        return 1
    ConlangApp(root, seed=seed, sandhi=sandhi)
    root.mainloop()
    return 0
