"""Map a phoneme's features to an acoustic plan.

Each segment becomes a short list of :class:`Part` chunks the synthesizer can render. The
mapping is articulatory: a vowel is a voiced buzz shaped by formants computed from its
height and backness; a fricative is noise shaped to a place-dependent band; a plosive is a
closure (silent, or a faint voiced bar) followed by a noise burst; nasals, liquids, and
glides are low-amplitude voiced resonances.
"""

from __future__ import annotations

from dataclasses import dataclass

from conlang.phonology.features import (
    Segment,
    Consonant,
    Vowel,
    Place,
    Manner,
    Voicing,
    Height,
    Backness,
)


@dataclass(frozen=True)
class Part:
    """One acoustic chunk: a voiced resonance, shaped noise, or silence."""

    kind: str                       # "voiced" | "noise" | "silence"
    duration: float                 # seconds (before the voice's rate scaling)
    formants: tuple = ()            # ((freq_hz, bandwidth_hz), ...) — resonator bands
    amp: float = 1.0
    voiced_noise: bool = False      # a voiced fricative: noise plus a glottal buzz
    modulate: float = 0.0           # amplitude-modulation rate in Hz (for trills)


# --- Vowel formants from features ---------------------------------------------------
_HEIGHT_F1 = {
    Height.CLOSE: 300, Height.NEAR_CLOSE: 350, Height.CLOSE_MID: 400, Height.MID: 500,
    Height.OPEN_MID: 600, Height.NEAR_OPEN: 680, Height.OPEN: 750,
}
_BACKNESS_F2 = {Backness.FRONT: 2150, Backness.CENTRAL: 1550, Backness.BACK: 900}


def vowel_formants(v: Vowel) -> tuple:
    f1 = _HEIGHT_F1[v.height]
    f2 = _BACKNESS_F2[v.backness]
    f3 = 2600 if v.backness is Backness.FRONT else 2500
    if v.rounded:  # rounding lowers the upper formants
        f2 -= 170
        f3 -= 180
    return ((f1, 60), (f2, 90), (f3, 160))


# --- Consonant place tables ---------------------------------------------------------
_PLACE_BURST = {
    Place.BILABIAL: 900, Place.LABIODENTAL: 1200, Place.DENTAL: 2600, Place.ALVEOLAR: 3200,
    Place.POSTALVEOLAR: 2500, Place.RETROFLEX: 2200, Place.PALATAL: 2800, Place.VELAR: 1800,
    Place.UVULAR: 1400, Place.PHARYNGEAL: 1200, Place.GLOTTAL: 1000,
}
_PLACE_FRICATIVE = {
    Place.BILABIAL: (1500, 1200), Place.LABIODENTAL: (4500, 2500), Place.DENTAL: (6500, 2500),
    Place.ALVEOLAR: (6000, 1800), Place.POSTALVEOLAR: (3500, 1500), Place.RETROFLEX: (2800, 1500),
    Place.PALATAL: (3000, 1800), Place.VELAR: (1600, 1200), Place.UVULAR: (1200, 1000),
    Place.PHARYNGEAL: (1000, 900), Place.GLOTTAL: (1500, 3000),
}
_NASAL_F2 = {
    Place.BILABIAL: 1000, Place.LABIODENTAL: 1100, Place.DENTAL: 1500, Place.ALVEOLAR: 1500,
    Place.POSTALVEOLAR: 1700, Place.RETROFLEX: 1600, Place.PALATAL: 1900, Place.VELAR: 2100,
    Place.UVULAR: 1300, Place.PHARYNGEAL: 1100, Place.GLOTTAL: 1200,
}

_LATERAL_FORMANTS = ((350, 80), (1100, 90), (2600, 160))
_RHOTIC_FORMANTS = ((350, 90), (1300, 120), (2400, 160))


def _is_voiced(c: Consonant) -> bool:
    return c.voicing is Voicing.VOICED


# --- The plan -----------------------------------------------------------------------
def plan_for(seg: Segment) -> list[Part]:
    if isinstance(seg, Vowel):
        dur = 0.25 if seg.long else 0.15
        return [Part("voiced", dur, vowel_formants(seg), amp=1.0)]

    assert isinstance(seg, Consonant)
    m = seg.manner
    voiced = _is_voiced(seg)

    if m in (Manner.PLOSIVE, Manner.AFFRICATE):
        parts: list[Part] = []
        if voiced:  # a voiced closure has a low "voice bar" rather than silence
            parts.append(Part("voiced", 0.05, ((180, 80),), amp=0.35))
        else:
            parts.append(Part("silence", 0.055))
        burst_center = _PLACE_BURST[seg.place]
        parts.append(Part("noise", 0.012, ((burst_center, 1500),), amp=0.7))
        if m is Manner.AFFRICATE:  # release into frication
            center, bw = _PLACE_FRICATIVE[seg.place]
            parts.append(Part("noise", 0.07, ((center, bw),), amp=0.5, voiced_noise=voiced))
        return parts

    if m in (Manner.FRICATIVE, Manner.LATERAL_FRICATIVE):
        center, bw = _PLACE_FRICATIVE[seg.place]
        return [Part("noise", 0.13, ((center, bw),), amp=0.5, voiced_noise=voiced)]

    if m is Manner.NASAL:
        f2 = _NASAL_F2[seg.place]
        return [Part("voiced", 0.09, ((250, 80), (f2, 150), (2500, 200)), amp=0.6)]

    if m is Manner.TRILL:
        return [Part("voiced", 0.11, _RHOTIC_FORMANTS, amp=0.6, modulate=28.0)]

    if m is Manner.TAP:
        return [Part("voiced", 0.03, _RHOTIC_FORMANTS, amp=0.55)]

    if m is Manner.LATERAL_APPROXIMANT:
        return [Part("voiced", 0.08, _LATERAL_FORMANTS, amp=0.6)]

    # Plain approximants / glides: a short vowel-like resonance (/j/ like i, /w/ like u).
    if seg.place is Place.PALATAL:
        formants = ((300, 70), (2150, 100), (2600, 160))   # j ~ i
    else:
        formants = ((300, 70), (730, 100), (2400, 160))    # w ~ u
    return [Part("voiced", 0.07, formants, amp=0.65)]
