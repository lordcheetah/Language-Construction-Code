"""Word/root generation and romanization.

A :class:`WordGenerator` strings syllables from a :class:`Phonotactics` into words. Word
*length* (in syllables) follows a weighted distribution that favours the short 1-3
syllable roots typical of natural lexicons. Each emitted :class:`Word` carries both its
IPA form and a romanization — a reader-friendly Latin spelling produced by a
:class:`Romanizer`, which is fully customizable for the guided workflow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from conlang.phonology.features import Segment
from conlang.phonology.phonotactics import Phonotactics

# Default IPA → Latin spelling for symbols that aren't already comfortable ASCII.
# Guided users can override any of these via Romanizer(overrides=...).
_DEFAULT_ROMANIZATION: dict[str, str] = {
    "ʃ": "sh", "ʒ": "zh", "t͡ʃ": "ch", "d͡ʒ": "j", "t͡s": "ts",
    "ŋ": "ng", "ɲ": "ny", "θ": "th", "ð": "dh", "x": "kh", "ɣ": "gh", "χ": "qh",
    "ɬ": "lh",
    "ʔ": "'", "ħ": "h", "ʁ": "r", "ɾ": "r", "ɽ": "r", "q": "q", "c": "ky",
    "ɛ": "e", "ɔ": "o", "ə": "e", "ɨ": "y", "y": "ü", "ø": "ö",
    "ɯ": "u", "ɪ": "i", "ʊ": "u", "æ": "ae", "ɑ": "a",
}

# Word-length distribution: index 0 unused; weight[i] is the relative chance of i
# syllables. Peaks at 2, the modal root length across natural lexicons.
_LENGTH_WEIGHTS = [0.0, 0.30, 0.40, 0.22, 0.08]


@dataclass
class Romanizer:
    """Maps IPA segments to a readable Latin spelling."""

    overrides: dict[str, str] = field(default_factory=dict)

    def spell(self, segment: Segment) -> str:
        ipa = segment.ipa
        if ipa in self.overrides:
            return self.overrides[ipa]
        return _DEFAULT_ROMANIZATION.get(ipa, ipa)

    def romanize(self, syllables: list[list[Segment]]) -> str:
        return "".join(self.spell(seg) for syl in syllables for seg in syl)


@dataclass(frozen=True)
class Word:
    """A generated word: its syllable structure plus IPA and romanized forms."""

    syllables: tuple[tuple[Segment, ...], ...]
    roman: str

    @property
    def ipa(self) -> str:
        return "".join(seg.ipa for syl in self.syllables for seg in syl)

    @property
    def syllabified_ipa(self) -> str:
        return ".".join("".join(seg.ipa for seg in syl) for syl in self.syllables)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.roman} /{self.ipa}/"


@dataclass
class WordGenerator:
    """Generates words from a phonotactic profile."""

    phonotactics: Phonotactics
    romanizer: Romanizer = field(default_factory=Romanizer)

    def word(
        self,
        rng: random.Random | None = None,
        *,
        min_syllables: int = 1,
        max_syllables: int = 4,
    ) -> Word:
        rng = rng or random.Random()
        n = self._roll_length(rng, min_syllables, max_syllables)
        syllables: list[list[Segment]] = []
        for _ in range(n):
            syl = self.phonotactics.random_syllable(rng)
            if syl:  # guard against an all-optional template rolling empty
                syllables.append(syl)
        if not syllables:  # extremely unlikely; ensure at least one syllable
            syllables.append(self.phonotactics.random_syllable(rng))
        roman = self.romanizer.romanize(syllables)
        return Word(tuple(tuple(s) for s in syllables), roman)

    def lexicon(
        self, n: int, rng: random.Random | None = None, **kwargs
    ) -> list[Word]:
        """Generate *n* distinct words (by IPA form), giving up gracefully if the
        phonotactics are too small to yield that many."""
        rng = rng or random.Random()
        seen: set[str] = set()
        words: list[Word] = []
        attempts = 0
        max_attempts = n * 50
        while len(words) < n and attempts < max_attempts:
            attempts += 1
            w = self.word(rng, **kwargs)
            if w.ipa not in seen:
                seen.add(w.ipa)
                words.append(w)
        return words

    def _roll_length(self, rng, lo: int, hi: int) -> int:
        lo = max(1, lo)
        hi = max(lo, hi)
        candidates = list(range(lo, hi + 1))
        weights = [
            _LENGTH_WEIGHTS[i] if i < len(_LENGTH_WEIGHTS) else 0.02
            for i in candidates
        ]
        if sum(weights) == 0:
            weights = [1.0] * len(candidates)
        return rng.choices(candidates, weights=weights, k=1)[0]
