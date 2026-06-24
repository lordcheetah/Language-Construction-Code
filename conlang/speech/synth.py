"""The formant synthesizer: build a formant track, then synthesize it in one pass.

A source/filter model with *formant transitions*. The phones' formant anchors are laid on
a timeline and linearly interpolated, so a vowel beside a consonant glides toward that
consonant's locus and back — the transition that cues place of articulation. The whole
utterance is then synthesized in a single pass: a continuous glottal pulse train (for
voiced stretches) runs through three time-varying cascade formant resonators whose
frequencies follow the track, while noise stretches (fricatives, bursts) are shaped by
their own band. Output is 16-bit mono PCM WAV via the standard-library ``wave`` module.
"""

from __future__ import annotations

import io
import math
import random
import wave
from dataclasses import dataclass
from typing import Sequence

from conlang.phonology.features import Segment
from conlang.speech.phones import Phone, plan_phone, apply_velar_pinch

# Formant bandwidths (Hz) for the three resonators; fixed, the frequencies vary over time.
_BW = (80.0, 90.0, 150.0)
_TRANSITION_S = 0.045  # how long a formant glide into/out of a steady phone lasts


@dataclass(frozen=True)
class Voice:
    f0: float = 120.0          # fundamental frequency (pitch), Hz
    sample_rate: int = 16000
    rate: float = 1.0          # speech speed multiplier (>1 = faster)


class Synthesizer:
    def __init__(self, voice: Voice | None = None, rng: random.Random | None = None) -> None:
        self.voice = voice or Voice()
        self.rng = rng or random.Random()
        self._rng0 = self.rng.getstate()  # reseed each synthesize() for per-call determinism

    # --- Public API ------------------------------------------------------------------
    def synthesize(self, segments: Sequence[Segment]) -> list[float]:
        self.rng.setstate(self._rng0)
        sr = self.voice.sample_rate
        phones = apply_velar_pinch(segments, [plan_phone(s) for s in segments])
        segs = [(max(1, int(s.duration * sr / self.voice.rate)), s)
                for ph in phones for s in ph.sources]
        total = sum(n for n, _ in segs)
        if total == 0:
            return []

        f1s, f2s, f3s = self._formant_tracks(phones, total)
        glottal = self._glottal_source(total)
        out = self._render(segs, total, f1s, f2s, f3s, glottal)
        return _normalize(_envelope(out, sr))

    def synthesize_word(self, word) -> list[float]:
        return self.synthesize([s for syl in word.syllables for s in syl])

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

    # --- Formant track ---------------------------------------------------------------
    def _formant_tracks(self, phones: list[Phone], total: int):
        """Interpolated F1/F2/F3 over every sample, from each phone's anchors."""
        sr = self.voice.sample_rate
        anchors: list[tuple[int, tuple]] = []  # (sample index, (f1, f2, f3))
        t = 0
        for ph in phones:
            dur = sum(max(1, int(s.duration * sr / self.voice.rate)) for s in ph.sources)
            trans = min(int(_TRANSITION_S * sr), dur // 2)
            if ph.voiced_resonant:  # a steady plateau, with glides only at the edges
                anchors.append((t + trans, ph.formants))
                anchors.append((t + dur - trans, ph.formants))
            else:  # obstruent: aim the neighbours' transitions at the locus
                anchors.append((t, ph.formants))
                anchors.append((t + dur, ph.formants))
            t += dur
        # Hold the first/last anchor out to the utterance edges. A word-initial obstruent's
        # locus thus sits flat from t=0 (no pre-locus glide), which is fine — there is no
        # preceding sound to transition from.
        anchors = [(0, anchors[0][1]), *anchors, (total, anchors[-1][1])]

        f1 = [0.0] * total
        f2 = [0.0] * total
        f3 = [0.0] * total
        for (i0, a), (i1, b) in zip(anchors, anchors[1:]):
            i0, i1 = max(0, i0), min(total, i1)
            if i1 <= i0:
                continue
            span = i1 - i0
            for i in range(i0, i1):
                u = (i - i0) / span
                f1[i] = a[0] + (b[0] - a[0]) * u
                f2[i] = a[1] + (b[1] - a[1]) * u
                f3[i] = a[2] + (b[2] - a[2]) * u
        return f1, f2, f3

    def _glottal_source(self, total: int) -> list[float]:
        period = max(1, int(self.voice.sample_rate / self.voice.f0))
        src = [0.0] * total
        for i in range(0, total, period):
            src[i] = 1.0
        # Shape each impulse into a decaying pulse (spectral tilt) and remove the DC offset.
        y = 0.0
        for i in range(total):
            y = src[i] + 0.9 * y
            src[i] = y
        mean = sum(src) / total
        return [s - mean for s in src]

    # --- Single-pass render ----------------------------------------------------------
    def _render(self, segs, total, f1s, f2s, f3s, glottal) -> list[float]:
        sr = self.voice.sample_rate
        two_pi = 2.0 * math.pi
        # Cascade resonators: constant radii/gains, frequencies vary per sample.
        r = tuple(math.exp(-math.pi * bw / sr) for bw in _BW)
        g = tuple((1.0 - ri) * math.sqrt(1.0 - ri * ri) for ri in r)
        s1a = s1b = s2a = s2b = s3a = s3b = 0.0  # cascade states
        na = nb = 0.0                            # noise-band state

        out = [0.0] * total
        i = 0
        for n, src in segs:
            if src.kind == "silence":
                # A voiceless closure resets the resonators, so the following vowel starts
                # from a clean state rather than the frozen pre-closure one (avoids a click).
                s1a = s1b = s2a = s2b = s3a = s3b = 0.0
            if src.kind == "noise" and src.noise_band:
                nf, nbw = src.noise_band
                rn = math.exp(-math.pi * nbw / sr)
                cn = 2.0 * rn * math.cos(two_pi * nf / sr)
                gn = 1.0 - rn
            for _ in range(n):
                if src.kind == "voiced":
                    x = glottal[i]
                elif src.kind == "noise":
                    x = self.rng.uniform(-1.0, 1.0)
                    if src.voiced_noise:
                        x = x * 0.7 + 0.6 * glottal[i]
                else:
                    x = 0.0
                if src.modulate:
                    x *= 0.55 + 0.45 * math.sin(two_pi * src.modulate * i / sr)

                if src.kind == "voiced":
                    c1 = 2.0 * r[0] * math.cos(two_pi * f1s[i] / sr)
                    y1 = g[0] * x + c1 * s1a - r[0] * r[0] * s1b
                    s1b, s1a = s1a, y1
                    c2 = 2.0 * r[1] * math.cos(two_pi * f2s[i] / sr)
                    y2 = g[1] * y1 + c2 * s2a - r[1] * r[1] * s2b
                    s2b, s2a = s2a, y2
                    c3 = 2.0 * r[2] * math.cos(two_pi * f3s[i] / sr)
                    y3 = g[2] * y2 + c3 * s3a - r[2] * r[2] * s3b
                    s3b, s3a = s3a, y3
                    val = y3
                elif src.kind == "noise" and src.noise_band:
                    y = gn * x + cn * na - rn * rn * nb
                    nb, na = na, y
                    val = y
                else:
                    val = x
                out[i] = val * src.amp
                i += 1
        return out


# --- DSP helpers --------------------------------------------------------------------
def _envelope(samples: list[float], sr: int, ramp_s: float = 0.006) -> list[float]:
    """Apply a short linear fade in/out so the utterance edges don't click."""
    k = min(len(samples) // 2, max(1, int(ramp_s * sr)))
    for i in range(k):
        gain = i / k
        samples[i] *= gain
        samples[-1 - i] *= gain
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
