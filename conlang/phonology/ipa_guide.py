"""Plain-English pronunciation hints for IPA symbols.

The generator speaks in IPA — ``/ˈpataka/`` — but most people don't read it, so every
symbol the engine can produce gets a one-line hint anchored to an English (or familiar
foreign) example word. The hints are deliberately informal "say it like this" cues, not
phonetics-class definitions: the goal is a learner who can *approximate* the sound, not a
narrow transcription.

Two entry points:

- :func:`describe` — the hint for one symbol or :class:`Segment`.
- :func:`pronunciation_key` — a formatted, aligned key for a collection of segments (e.g.
  a language's inventory), so it can be printed under the phoneme chart.

Any symbol missing from the curated table falls back to a description assembled from its
articulatory features, so the guide never crashes on a segment that lacks a hand-written
cue — it just gives a drier description.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

from conlang.phonology.features import Consonant, Segment, Vowel


def _display_width(text: str) -> int:
    """Visual width: ignore combining marks so an affricate tie-bar (t͡ʃ) doesn't inflate it."""
    return sum(1 for ch in text if not unicodedata.combining(ch))


# IPA symbol -> an informal "pronounce it like…" hint. Covers every segment in
# ``phonology.data``; anchored to English examples where one exists, else to a well-known
# foreign word plus an articulatory nudge.
PRONUNCIATION: dict[str, str] = {
    # --- Plosives ---
    "p": "p as in 'spin'",
    "t": "t as in 'stop'",
    "k": "k as in 'skip'",
    "b": "b as in 'bat'",
    "d": "d as in 'dog'",
    "g": "g as in 'go'",
    "q": "like 'k' but further back in the throat (Arabic 'q' in Qatar)",
    "ɢ": "a voiced 'q' — 'g' made deep in the throat (uvular)",
    "ʔ": "the catch in the middle of 'uh-oh' (glottal stop)",
    "c": "roughly the 'cu' in 'cute', tongue on the hard palate",
    # --- Nasals ---
    "m": "m as in 'man'",
    "n": "n as in 'net'",
    "ŋ": "ng as in 'sing'",
    "ɲ": "ny as in 'canyon' (Spanish ñ in 'niño')",
    # --- Fricatives ---
    "s": "s as in 'see'",
    "h": "h as in 'hat'",
    "f": "f as in 'fan'",
    "ʃ": "sh as in 'ship'",
    "x": "ch as in Scottish 'loch' / German 'Bach'",
    "z": "z as in 'zoo'",
    "v": "v as in 'van'",
    "ʒ": "s as in 'measure' (French j in 'jour')",
    "θ": "th as in 'thin' (no buzz)",
    "ð": "th as in 'this' (with buzz)",
    "ɣ": "a soft buzzing 'g' (Greek γ in 'gamma'; Spanish g between vowels in 'agua')",
    "χ": "a raspy 'ch', further back than 'loch' (uvular)",
    "ħ": "a tight, breathy 'h' from the throat (Arabic ḥ)",
    "ʁ": "French r as in 'rouge' (gargled in the throat)",
    # --- Affricates (tie-barred) ---
    "t͡ʃ": "ch as in 'church'",
    "d͡ʒ": "j as in 'jump'",
    "t͡s": "ts as in 'cats'",
    # --- Lateral fricative ---
    "ɬ": "Welsh 'll' in 'Llan' — an 's'-like L blown around the tongue",
    # --- Liquids & glides ---
    "l": "l as in 'let'",
    "ɾ": "the quick 'tt' in American 'butter' (Spanish r in 'pero')",
    "r": "a rolled/trilled r (Spanish rr in 'perro')",
    "ɽ": "an r with the tongue curled back (retroflex flap)",
    "j": "y as in 'yes'",
    "w": "w as in 'we'",
    # --- Vowels ---
    "i": "ee as in 'see'",
    "a": "a as in 'father' (short), Spanish 'a'",
    "u": "oo as in 'boot'",
    "e": "ay as in 'say' (cut short, no glide), Spanish 'e'",
    "o": "o as in 'go' (cut short, no glide), Spanish 'o'",
    "ɛ": "e as in 'bed'",
    "ɔ": "aw as in 'law' / 'caught' (rounded; British 'thought')",
    "ə": "the unstressed a in 'about' (the 'schwa')",
    "ɨ": "a central 'ih', Russian ы",
    "y": "German ü — say 'ee' with tightly rounded lips",
    "ø": "German ö — say 'ay' with rounded lips",
    "ɯ": "say 'oo' but with lips spread, not rounded (Japanese u)",
    "ɪ": "i as in 'sit'",
    "ʊ": "oo as in 'foot'",
    "æ": "a as in 'cat'",
    "ɑ": "the broad o in American 'hot'",
}


def _auto_describe(segment: Segment) -> str:
    """A feature-based fallback hint for a symbol with no hand-written cue."""
    if isinstance(segment, Consonant):
        return f"{segment.voicing.value} {segment.place.value} {segment.manner.value}"
    if isinstance(segment, Vowel):
        rounding = "rounded" if segment.rounded else "unrounded"
        return f"{segment.height.value} {segment.backness.value} {rounding} vowel"
    return "an unfamiliar sound"  # pragma: no cover - defensive


def _hint_for(segment: Segment) -> str:
    """The curated hint for *segment*, or a feature-based description if none is on file."""
    curated = PRONUNCIATION.get(segment.ipa)
    return curated if curated is not None else _auto_describe(segment)


def describe(symbol: "str | Segment") -> str:
    """A plain-English pronunciation hint for an IPA symbol or :class:`Segment`.

    Accepts either the IPA string (``"ʃ"``) or a segment object. A curated hint is used
    when available; otherwise a description is assembled from the segment's features (only
    possible when a :class:`Segment` is passed — a bare unknown string yields a generic
    note, since its features aren't known here)."""
    if isinstance(symbol, Segment):
        return _hint_for(symbol)
    curated = PRONUNCIATION.get(symbol)
    return curated if curated is not None else f"{symbol}: (no pronunciation hint on file)"


def pronunciation_key(segments: Iterable[Segment], *, indent: str = "  ") -> str:
    """A formatted, aligned pronunciation key for *segments*, one symbol per line.

    Symbols are listed in the order given (so a caller can pass consonants-then-vowels),
    de-duplicated, with the IPA symbol shown in slashes and its hint aligned in a column."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for seg in segments:
        if seg.ipa in seen:
            continue
        seen.add(seg.ipa)
        rows.append((f"/{seg.ipa}/", _hint_for(seg)))
    if not rows:
        return ""
    width = max(_display_width(sym) for sym, _ in rows)
    return "\n".join(
        f"{indent}{sym}{' ' * (width - _display_width(sym))}  {hint}" for sym, hint in rows
    )
