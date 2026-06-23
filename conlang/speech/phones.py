"""Map a phoneme to an acoustic plan: a formant target plus timed source segments.

Each segment becomes a :class:`Phone` with a single formant *anchor* — the (F1, F2, F3)
the formant track aims at — and a list of :class:`Source` segments describing how it is
excited over time (a voiced buzz, shaped noise, or silence). The synthesizer interpolates
the formant anchors across the whole word, so a vowel next to a consonant glides toward
that consonant's anchor (its *locus*) and back. Those formant transitions, especially in
F2, are the main cue for a consonant's place of articulation — the thing the previous
butt-jointed model lacked.

For voiced "resonant" phones (vowels, nasals, liquids, glides) the anchor is the steady
formant the phone holds; for obstruents (stops, fricatives, affricates) it is a locus that
only shapes the neighbours' transitions, since the obstruent itself is silence or noise.
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
class Source:
    """One excitation segment of a phone."""

    duration: float                 # seconds (before the voice's rate scaling)
    kind: str                       # "voiced" | "noise" | "silence"
    amp: float = 1.0
    noise_band: tuple[float, float] | None = None  # (center_hz, bw_hz) for shaped noise
    voiced_noise: bool = False       # voiced fricative: noise plus a glottal buzz
    modulate: float = 0.0            # amplitude-modulation rate in Hz (for trills)


@dataclass(frozen=True)
class Phone:
    formants: tuple[float, float, float]  # (F1, F2, F3) anchor frequencies in Hz
    sources: tuple[Source, ...]
    voiced_resonant: bool  # True for vowels/sonorants (hold a steady formant)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.sources)


# --- Vowel formants from features ---------------------------------------------------
_HEIGHT_F1 = {
    Height.CLOSE: 300, Height.NEAR_CLOSE: 350, Height.CLOSE_MID: 400, Height.MID: 500,
    Height.OPEN_MID: 600, Height.NEAR_OPEN: 680, Height.OPEN: 750,
}
_BACKNESS_F2 = {Backness.FRONT: 2150, Backness.CENTRAL: 1550, Backness.BACK: 900}


def vowel_formants(v: Vowel) -> tuple:
    """(F1, F2, F3) for a vowel: height sets F1, backness sets F2, rounding lowers both upper."""
    f1 = _HEIGHT_F1[v.height]
    f2 = _BACKNESS_F2[v.backness]
    f3 = 2600 if v.backness is Backness.FRONT else 2500
    if v.rounded:
        f2 -= 170
        f3 -= 180
    return (f1, f2, f3)


# --- Consonant formant loci ---------------------------------------------------------
# F2 locus by place — the key place-of-articulation cue carried by the transition.
_PLACE_F2 = {
    Place.BILABIAL: 800, Place.LABIODENTAL: 1000, Place.DENTAL: 1700, Place.ALVEOLAR: 1700,
    Place.POSTALVEOLAR: 2000, Place.RETROFLEX: 1600, Place.PALATAL: 2300, Place.VELAR: 2000,
    Place.UVULAR: 1400, Place.PHARYNGEAL: 1200, Place.GLOTTAL: 1500,
}
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


def consonant_formants(c: Consonant) -> tuple:
    """(F1, F2, F3) locus/steady formants — the target the neighbouring transitions aim at."""
    f2 = _PLACE_F2[c.place]
    m = c.manner
    if m is Manner.NASAL:
        return (250, f2, 2500)
    if m is Manner.LATERAL_APPROXIMANT:
        return (350, 1100, 2600)
    if m in (Manner.TRILL, Manner.TAP):
        return (350, 1300, 1600)  # lowered F3 is the rhotic cue
    if m is Manner.APPROXIMANT:
        if c.place is Place.PALATAL:
            return (300, 2200, 2900)  # /j/ ~ i
        return (300, 800, 2200)       # /w/ ~ u
    # Obstruents: low F1, place-based F2 locus. The velar locus is fixed (~2000); real
    # velars vary with the adjacent vowel (the "velar pinch") — a deliberate simplification.
    return (300, f2, 2500)


# --- The plan -----------------------------------------------------------------------
def plan_phone(seg: Segment) -> Phone:
    if isinstance(seg, Vowel):
        dur = 0.25 if seg.long else 0.15
        return Phone(vowel_formants(seg), (Source(dur, "voiced", 1.0),), voiced_resonant=True)

    assert isinstance(seg, Consonant)
    formants = consonant_formants(seg)
    m = seg.manner
    voiced = seg.voicing is Voicing.VOICED

    if m in (Manner.PLOSIVE, Manner.AFFRICATE):
        sources: list[Source] = []
        if voiced:
            sources.append(Source(0.05, "voiced", 0.3))  # a low voice bar during closure
        else:
            sources.append(Source(0.055, "silence"))
        sources.append(Source(0.012, "noise", amp=0.7, noise_band=(_PLACE_BURST[seg.place], 1500)))
        if m is Manner.AFFRICATE:
            band = _PLACE_FRICATIVE[seg.place]
            sources.append(Source(0.07, "noise", amp=0.5, noise_band=band, voiced_noise=voiced))
        return Phone(formants, tuple(sources), voiced_resonant=False)

    if m in (Manner.FRICATIVE, Manner.LATERAL_FRICATIVE):
        band = _PLACE_FRICATIVE[seg.place]
        return Phone(
            formants, (Source(0.13, "noise", amp=0.5, noise_band=band, voiced_noise=voiced),),
            voiced_resonant=False,
        )

    if m is Manner.NASAL:
        return Phone(formants, (Source(0.09, "voiced", 0.65),), voiced_resonant=True)
    if m is Manner.TRILL:
        return Phone(formants, (Source(0.11, "voiced", 0.65, modulate=28.0),), voiced_resonant=True)
    if m is Manner.TAP:
        return Phone(formants, (Source(0.03, "voiced", 0.6),), voiced_resonant=True)
    if m is Manner.LATERAL_APPROXIMANT:
        return Phone(formants, (Source(0.08, "voiced", 0.65),), voiced_resonant=True)
    # plain approximant / glide
    return Phone(formants, (Source(0.07, "voiced", 0.7),), voiced_resonant=True)
