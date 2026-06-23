"""The formant synthesizer: turn an acoustic plan into audio and write a WAV.

A classic source/filter model. Voiced sounds excite a glottal pulse train (a buzz at the
fundamental F0) and pass it through a cascade of two-pole formant resonators; voiceless
sounds use white noise shaped by a single band; plosive closures are silence (or a faint
voiced bar). Parts are enveloped to avoid clicks and concatenated; the whole utterance is
normalized. Output is 16-bit mono PCM via the standard-library ``wave`` module.
"""

from __future__ import annotations

import io
import math
import random
import wave
from dataclasses import dataclass
from typing import Sequence

from conlang.phonology.features import Segment
from conlang.speech.phones import Part, plan_for


@dataclass(frozen=True)
class Voice:
    f0: float = 120.0          # fundamental frequency (pitch), Hz
    sample_rate: int = 16000
    rate: float = 1.0          # speech speed multiplier (>1 = faster)


class Synthesizer:
    def __init__(self, voice: Voice | None = None, rng: random.Random | None = None) -> None:
        self.voice = voice or Voice()
        self.rng = rng or random.Random()
        # Snapshot the RNG so every synthesize() call on this instance is reproducible
        # (determinism is per-call, not just per-fresh-instance).
        self._rng0 = self.rng.getstate()

    # --- Public API ------------------------------------------------------------------
    def synthesize(self, segments: Sequence[Segment]) -> list[float]:
        self.rng.setstate(self._rng0)
        sr = self.voice.sample_rate
        period = max(1, int(sr / self.voice.f0))
        overlap = max(1, int(0.008 * sr))  # ~8 ms crossfade between adjacent parts

        rendered: list[list[float]] = []
        phase = 0  # glottal phase carried across parts so the buzz stays continuous
        for seg in segments:
            for part in plan_for(seg):
                samples, phase = self._render(part, phase, period)
                rendered.append(samples)

        out = _crossfade_concat(rendered, overlap)
        return _normalize(_envelope(out, sr))

    def synthesize_word(self, word) -> list[float]:
        segments = [s for syl in word.syllables for s in syl]
        return self.synthesize(segments)

    def to_wav_bytes(self, samples: Sequence[float]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.voice.sample_rate)
            w.writeframes(_to_pcm16(samples))
        return buf.getvalue()

    def write_wav(self, path: str, samples: Sequence[float]) -> None:
        with open(path, "wb") as fh:
            fh.write(self.to_wav_bytes(samples))

    # --- Rendering one part ----------------------------------------------------------
    def _render(self, part: Part, phase: int, period: int) -> tuple[list[float], int]:
        sr = self.voice.sample_rate
        n = max(1, int(part.duration * sr / self.voice.rate))
        next_phase = (phase + n) % period

        if part.kind == "silence":
            return [0.0] * n, next_phase

        if part.kind == "voiced":
            src = self._glottal_source(n, phase, period)
            out = _cascade(src, part.formants, sr)
        else:  # "noise"
            out = [self.rng.uniform(-1.0, 1.0) for _ in range(n)]
            if part.formants:
                center, bw = part.formants[0]
                out = _resonator(out, center, bw, sr)
            if part.voiced_noise:  # voiced fricative: add a glottal buzz underneath
                buzz = _cascade(self._glottal_source(n, phase, period), ((250, 90),), sr)
                out = [o + 0.5 * b for o, b in zip(out, buzz)]

        out = [s * part.amp for s in out]
        if part.modulate:
            out = self._amplitude_modulate(out, part.modulate, sr)
        return out, next_phase

    def _glottal_source(self, n: int, phase: int, period: int) -> list[float]:
        # Pulse train at the fundamental, continued from `phase` for cross-part continuity.
        src = [0.0] * n
        for i in range(n):
            if (i + phase) % period == 0:
                src[i] = 1.0
        # Shape each impulse into a decaying pulse (spectral tilt, less aliasing buzz) and
        # remove the resulting DC offset before it reaches the formant filters.
        decay = 0.9
        y = 0.0
        for i in range(n):
            y = src[i] + decay * y
            src[i] = y
        mean = sum(src) / n
        return [s - mean for s in src]

    def _amplitude_modulate(self, samples: list[float], hz: float, sr: int) -> list[float]:
        return [
            s * (0.55 + 0.45 * math.sin(2 * math.pi * hz * i / sr))
            for i, s in enumerate(samples)
        ]


# --- DSP helpers --------------------------------------------------------------------
def _resonator(x: Sequence[float], freq: float, bw: float, sr: int) -> list[float]:
    """A two-pole resonator (formant filter) at *freq* with bandwidth *bw*."""
    r = math.exp(-math.pi * bw / sr)
    c = 2.0 * r * math.cos(2.0 * math.pi * freq / sr)
    # Normalize so the resonance peaks near unity regardless of bandwidth, otherwise a
    # narrow formant would dominate the others (relative formant amplitudes matter).
    g = (1.0 - r) * math.sqrt(1.0 - r * r)
    y = [0.0] * len(x)
    y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yn = g * xi + c * y1 - r * r * y2
        y[i] = yn
        y2, y1 = y1, yn
    return y


def _cascade(x: Sequence[float], formants: Sequence[tuple], sr: int) -> list[float]:
    out = list(x)
    for freq, bw in formants:
        out = _resonator(out, freq, bw, sr)
    return out


def _crossfade_concat(parts: list[list[float]], overlap: int) -> list[float]:
    """Join rendered parts with a short linear crossfade so boundaries don't click or notch."""
    out: list[float] = []
    for seg in parts:
        if not seg:
            continue
        if not out:
            out = list(seg)
            continue
        ov = min(overlap, len(out), len(seg))
        for i in range(ov):
            g = (i + 1) / (ov + 1)
            out[len(out) - ov + i] = out[len(out) - ov + i] * (1 - g) + seg[i] * g
        out.extend(seg[ov:])
    return out


def _envelope(samples: list[float], sr: int, ramp_s: float = 0.006) -> list[float]:
    """Apply a short linear fade in/out so concatenated parts don't click."""
    k = min(len(samples) // 2, max(1, int(ramp_s * sr)))
    for i in range(k):
        g = i / k
        samples[i] *= g
        samples[-1 - i] *= g
    return samples


def _normalize(samples: list[float], peak: float = 0.9) -> list[float]:
    hi = max((abs(s) for s in samples), default=0.0)
    if hi == 0.0:
        return samples
    scale = peak / hi
    return [s * scale for s in samples]


def _to_pcm16(samples: Sequence[float]) -> bytes:
    out = bytearray()
    for s in samples:
        v = int(max(-1.0, min(1.0, s)) * 32767)
        out += int(v).to_bytes(2, "little", signed=True)
    return bytes(out)
