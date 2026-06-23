"""A language's phoneme inventory — the set of sounds it actually uses.

An :class:`Inventory` can be built two ways:

- **Guided**: hand it explicit segments (``Inventory.from_ipa("p t k a i u")``), or start
  empty and ``add``/``remove``.
- **Random-but-plausible**: ``Inventory.random(rng)`` samples segments weighted by their
  cross-linguistic frequency, then enforces a handful of typological universals so the
  result looks like a real language rather than a random bag of IPA symbols.

The universals enforced are intentionally a small, well-attested set (LCK, "Sounds"):

- /m/ and /n/ are effectively universal — every inventory gets them.
- A *voiced* obstruent at a given place/manner implies the *voiceless* one (so /g/
  without /k/ is repaired by adding /k/).
- Vowels are not sampled independently; they are chosen from canonical, well-dispersed
  systems by size (3 → i a u, 5 → i e a o u, …), because real vowel systems disperse
  evenly through the vowel space rather than clustering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from conlang.phonology import data
from conlang.phonology.features import (
    Consonant,
    Vowel,
    Segment,
    Manner,
    Voicing,
)

# Canonical vowel systems keyed by size, each well-dispersed through the vowel space.
# These are the cross-linguistically dominant systems; ``random`` picks one by size.
_VOWEL_SYSTEMS: dict[int, list[str]] = {
    3: ["i", "a", "u"],
    4: ["i", "e", "a", "u"],
    5: ["i", "e", "a", "o", "u"],
    6: ["i", "e", "a", "o", "u", "ə"],
    7: ["i", "e", "ɛ", "a", "ɔ", "o", "u"],
    8: ["i", "e", "ɛ", "a", "ɔ", "o", "u", "ə"],
    9: ["i", "y", "e", "ø", "a", "o", "u", "ɛ", "ɔ"],
}

# Obstruent manners for which voicing is contrastive and the implicational universal
# "voiced implies voiceless" is enforced.
_OBSTRUENT_MANNERS = (Manner.PLOSIVE, Manner.AFFRICATE, Manner.FRICATIVE)


@dataclass
class Inventory:
    """The consonants and vowels of a single language."""

    consonants: list[Consonant] = field(default_factory=list)
    vowels: list[Vowel] = field(default_factory=list)

    # --- Construction ----------------------------------------------------------------
    @classmethod
    def from_ipa(cls, symbols: str | list[str]) -> "Inventory":
        """Build from IPA symbols (a space-separated string or a list).

        Unknown symbols raise ``KeyError`` so typos surface immediately.
        """
        if isinstance(symbols, str):
            symbols = symbols.split()
        inv = cls()
        for sym in symbols:
            inv.add(data.BY_IPA[sym])
        return inv

    @classmethod
    def random(
        cls,
        rng: random.Random | None = None,
        *,
        consonant_target: int | None = None,
        vowel_target: int | None = None,
    ) -> "Inventory":
        """Roll a random-but-plausible inventory.

        ``consonant_target`` / ``vowel_target`` pin the sizes; when omitted they are
        themselves sampled from typologically realistic ranges. Pass a seeded
        ``random.Random`` for reproducible output.
        """
        rng = rng or random.Random()
        inv = cls()
        inv._fill_consonants(rng, consonant_target)
        inv._fill_vowels(rng, vowel_target)
        return inv

    # --- Mutation --------------------------------------------------------------------
    def add(self, segment: Segment) -> None:
        if isinstance(segment, Consonant):
            if segment not in self.consonants:
                self.consonants.append(segment)
        elif isinstance(segment, Vowel):
            if segment not in self.vowels:
                self.vowels.append(segment)
        else:  # pragma: no cover - defensive
            raise TypeError(f"not a consonant or vowel: {segment!r}")

    def remove(self, segment: Segment) -> None:
        if segment in self.consonants:
            self.consonants.remove(segment)
        elif segment in self.vowels:
            self.vowels.remove(segment)

    # --- Queries ---------------------------------------------------------------------
    @property
    def segments(self) -> list[Segment]:
        return [*self.consonants, *self.vowels]

    @property
    def size(self) -> int:
        return len(self.consonants) + len(self.vowels)

    def has(self, ipa: str) -> bool:
        return any(s.ipa == ipa for s in self.segments)

    # --- Random fill helpers ---------------------------------------------------------
    def _fill_consonants(self, rng: random.Random, target: int | None) -> None:
        # Most languages fall in ~15-25 consonants; sample a target if not pinned.
        if target is None:
            target = max(6, min(len(data.CONSONANTS), round(rng.gauss(21, 5))))

        pool = list(data.CONSONANTS)
        chosen: list[Consonant] = []

        # Seed with the near-universal nasals so every language has /m n/.
        for ipa in ("m", "n"):
            seg = data.consonant(ipa)
            chosen.append(seg)
            pool.remove(seg)

        # Weighted sampling without replacement for the remainder.
        while len(chosen) < target and pool:
            seg = _weighted_pop(rng, pool, key=lambda s: s.frequency)
            chosen.append(seg)

        self.consonants = chosen
        self._enforce_voicing_implication()
        self._enforce_velar_nasal_support()

    def _fill_vowels(self, rng: random.Random, target: int | None) -> None:
        if target is None:
            # 5-vowel systems are the global mode; cluster around it.
            target = max(3, min(9, round(rng.gauss(5.2, 1.3))))
        target = max(3, min(9, target))
        system = _VOWEL_SYSTEMS[target]
        self.vowels = [data.vowel(ipa) for ipa in system]

    def _enforce_voicing_implication(self) -> None:
        """Add the voiceless counterpart of any voiced obstruent that lacks one.

        Encodes the implicational universal that a voiced obstruent series presupposes
        the corresponding voiceless one.
        """
        present = {(c.place, c.manner, c.voicing) for c in self.consonants}
        for c in list(self.consonants):
            if c.voicing is Voicing.VOICED and c.manner in _OBSTRUENT_MANNERS:
                key = (c.place, c.manner, Voicing.VOICELESS)
                if key not in present:
                    counterpart = _find_segment(c.place, c.manner, Voicing.VOICELESS)
                    if counterpart is not None:
                        self.consonants.append(counterpart)
                        present.add(key)

    def _enforce_velar_nasal_support(self) -> None:
        """Ensure /ŋ/ is backed by a velar stop.

        A velar nasal in a language that has no velar stop (/k/ or /g/) is very rare;
        if /ŋ/ was rolled without one, add /k/ (the most common consonant overall).
        """
        if self.has("ŋ") and not (self.has("k") or self.has("g")):
            self.consonants.append(data.consonant("k"))

    # --- Display ---------------------------------------------------------------------
    def summary(self) -> str:
        cons = " ".join(c.ipa for c in self.consonants)
        vows = " ".join(v.ipa for v in self.vowels)
        return (
            f"Inventory ({self.size} phonemes: "
            f"{len(self.consonants)}C / {len(self.vowels)}V)\n"
            f"  Consonants: {cons}\n"
            f"  Vowels:     {vows}"
        )


# --- Module helpers -----------------------------------------------------------------
def _weighted_pop(rng, pool, key):
    """Pop one element from *pool* with probability proportional to ``key(element)``.

    Falls back to a uniform choice if all weights are zero. Mutates *pool*.
    """
    weights = [max(key(s), 1e-9) for s in pool]
    chosen = rng.choices(pool, weights=weights, k=1)[0]
    pool.remove(chosen)
    return chosen


def _find_segment(place, manner, voicing) -> Consonant | None:
    for c in data.CONSONANTS:
        if c.place is place and c.manner is manner and c.voicing is voicing:
            return c
    return None
