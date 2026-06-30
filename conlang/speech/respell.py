"""Respell a conlang word so a generic (English) text-to-speech engine can approximate it.

The built-in synthesizer is one way to *hear* a word; this is the other — turn the word's
phonemes into an ASCII, English-orthography-ish spelling that a stock TTS voice reads back
close to the intended sounds (``/ʃ/`` → "sh", ``/a/`` → "ah", ``/y/`` → "ew"). So you can paste
a conlang word into any system/web TTS and get a fair approximation, without bundling a speech
engine.

It is deliberately approximate: an English TTS cannot make every sound, so exotic phonemes are
mapped to their nearest *writable* English cue (a velar fricative ``/x/`` → "kh", a glottal stop
``/ʔ/`` is dropped). The frequent phonemes are read back well; the weakest cues are the rare
ones with no faithful English grapheme — ``/ʒ/`` ("zh", which some voices read as plain /z/),
``/ø/`` ("er", r-coloured on a rhotic voice) and ``/y/`` ("ew") — they degrade gracefully to a
near neighbour. Output is plain ASCII so it survives any input box. This is distinct from the
language's own romanization, whose letters follow the conlang's logic rather than English
reading conventions.
"""

from __future__ import annotations

from typing import Sequence

from conlang.phonology.features import Segment, Vowel


# IPA symbol -> an English-TTS-friendly ASCII respelling. Covers every segment in
# ``phonology.data``. The common vowels/consonants are reliable; the rarer rows are the closest
# cue a stock English voice can read (it simply cannot produce e.g. a true uvular fricative).
_RESPELL: dict[str, str] = {
    # --- Plosives ---
    "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "g",
    "q": "k",      # uvular stop → k (further back, but English has no uvular)
    "ɢ": "g",
    "ʔ": "",       # glottal stop: dropped so the word still flows (TTS has no glottal stop)
    "c": "ky",     # palatal stop ≈ the "cu" in "cute"
    # --- Nasals ---
    "m": "m", "n": "n", "ŋ": "ng", "ɲ": "ny",
    # --- Fricatives ---
    "s": "s", "h": "h", "f": "f", "ʃ": "sh", "z": "z", "v": "v",
    "x": "kh", "χ": "kh",   # velar / uvular fricative → "kh" (read as k)
    "ɣ": "g",              # voiced velar fricative → g
    "ʒ": "zh",            # as in "measure"
    "θ": "th", "ð": "th",  # English th (the voicing falls out from context)
    "ħ": "h",             # pharyngeal → h
    "ʁ": "r",             # uvular r → English r
    # --- Affricates ---
    "t͡ʃ": "ch", "d͡ʒ": "j", "t͡s": "ts",
    # --- Lateral fricative ---
    "ɬ": "hl",            # Welsh "ll" ≈ a breathy "hl"
    # --- Liquids & glides ---
    "l": "l", "ɾ": "r", "r": "r", "ɽ": "r", "j": "y", "w": "w",
    # --- Vowels ---
    "i": "ee", "ɪ": "ih", "e": "ay", "ɛ": "eh", "æ": "a", "a": "ah", "ɑ": "ah",
    "ə": "uh", "ɨ": "ih", "u": "oo", "ʊ": "oo", "o": "oh", "ɔ": "aw",
    "y": "ew",            # front rounded ≈ the "ew" in "few"
    "ø": "er",            # front rounded mid ≈ non-rhotic "er"
    "ɯ": "oo",            # unrounded back ≈ oo
}


def _auto_respell(segment: Segment) -> str:
    """A safe fallback for a segment with no curated respelling (keeps output pronounceable)."""
    return "uh" if isinstance(segment, Vowel) else "h"


def respell(segments: Sequence[Segment]) -> str:
    """A TTS-friendly ASCII respelling of *segments* (one phone sequence = one word)."""
    out = []
    for seg in segments:
        cue = _RESPELL.get(seg.ipa)
        out.append(cue if cue is not None else _auto_respell(seg))
    return "".join(out)
