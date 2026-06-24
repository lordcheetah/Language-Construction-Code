"""Numeral systems: count past the handful of atomic number words.

A language doesn't list a separate word for every number — it has a small set of atoms and
a **base**, and builds the rest compositionally (decimal "twenty-four" = two-ten-four;
vigesimal systems group by twenty; and so on). This module rolls a base, reuses the
lexicon's existing small number words (one…five) as atoms, generates the missing atoms and
the base/“hundred” words, and composes the word for any number up to base³ − 1.

A few cross-linguistic switches are rolled per language and biased by word order (a
head-final language tends to put the multiplier and units first): whether a multiplier
precedes the base ("two-ten" vs "ten-two"), whether units precede tens, and whether one ×
base is said bare ("ten") or as "one-ten". Some languages (with base ≥ 10) also have **irregular teens** — the
first few numbers above the base (``base+1 .. base+3``) take their own suppletive root rather
than the regular composition (English *eleven*, *twelve*), and those roots are reused
compositionally (so "one-hundred eleven", not "one-hundred ten-one").

Remaining simplification: no suppletive decades and no sub-bases (French *quatre-vingts*).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from conlang.phonology.wordgen import WordGenerator, Romanizer

# Lexicon glosses for the atomic numbers, by value.
_DIGIT_GLOSS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

# Plausible bases and their rough prevalence (decimal dominates; vigesimal notable).
_BASE_WEIGHTS = {10: 0.72, 20: 0.15, 5: 0.10, 12: 0.03}


@dataclass(frozen=True)
class Numeral:
    value: int
    roman: str
    ipa: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.value} = {self.roman} /{self.ipa}/"


@dataclass
class NumeralSystem:
    base: int
    atoms: dict[int, Numeral]   # 1 .. base-1
    base_word: Numeral          # value == base
    square_word: Numeral        # value == base**2
    multiplier_before_base: bool
    units_before_tens: bool
    bare_base_for_one: bool
    irregular: dict[int, Numeral] = field(default_factory=dict)  # suppletive teens, by value

    @property
    def max_value(self) -> int:
        return self.base ** 3 - 1

    def number(self, n: int) -> Numeral:
        """The numeral word for *n* (1 .. base**3 - 1)."""
        if not 1 <= n <= self.max_value:
            raise ValueError(f"{n} is out of range for this base-{self.base} system (1..{self.max_value})")
        parts = self._components(n)
        return Numeral(n, " ".join(p.roman for p in parts), " ".join(p.ipa for p in parts))

    def _components(self, n: int) -> list[Numeral]:
        if n in self.irregular:  # a suppletive teen, used directly and within larger numbers
            return [self.irregular[n]]
        if n < self.base:
            return [self.atoms[n]]
        if n < self.base ** 2:
            tens, units = divmod(n, self.base)
            group = self._group(tens, self.base_word)
            if not units:
                return group
            unit = [self.atoms[units]]
            return unit + group if self.units_before_tens else group + unit
        hundreds, rest = divmod(n, self.base ** 2)
        group = self._group(hundreds, self.square_word)
        return group + (self._components(rest) if rest else [])

    def _group(self, multiplier: int, base_word: Numeral) -> list[Numeral]:
        """A multiplier times a base word, e.g. 2 x ten = 'two ten' (order rolled)."""
        if multiplier == 1 and self.bare_base_for_one:
            return [base_word]
        mult = self.atoms[multiplier]
        return [mult, base_word] if self.multiplier_before_base else [base_word, mult]

    def counting(self, upto: int) -> list[Numeral]:
        return [self.number(n) for n in range(1, min(upto, self.max_value) + 1)]


def build_numerals(
    lexicon,
    phonotactics,
    rng: random.Random | None = None,
    *,
    romanizer: Romanizer | None = None,
    base: int | None = None,
    head_final: bool = False,
) -> NumeralSystem:
    rng = rng or random.Random()
    romanizer = romanizer or Romanizer()
    gen = WordGenerator(phonotactics, romanizer)
    used: set[str] = set()

    def word_for(value: int) -> Numeral:
        # Generated number words are de-duplicated against each other (and reused lexicon
        # numbers) but may coincide with an unrelated lexicon word — harmless homophony.
        gloss = _DIGIT_GLOSS.get(value)
        entry = lexicon.get(gloss) if gloss else None
        if entry is not None:  # reuse the lexicon's existing small-number word
            used.add(entry.ipa)
            return Numeral(value, entry.roman, entry.ipa)
        word = gen.word(rng, min_syllables=1, max_syllables=2)
        for _ in range(40):
            if word.ipa not in used:
                break
            word = gen.word(rng, min_syllables=1, max_syllables=2)
        used.add(word.ipa)
        return Numeral(value, word.roman, word.ipa)

    if base is None:
        bases = list(_BASE_WEIGHTS)
        base = rng.choices(bases, weights=[_BASE_WEIGHTS[b] for b in bases], k=1)[0]

    atoms = {v: word_for(v) for v in range(1, base)}
    base_word = word_for(base)
    square_word = word_for(base ** 2)

    # Some languages give the first few numbers above the base (base+1 .. base+3) their own
    # suppletive root (English eleven/twelve) instead of the regular "base + unit"
    # composition. Restricted to base >= 10, where opaque "teen" roots are the attested
    # pattern; for a small base (5) regular composition reads more plausibly.
    irregular: dict[int, Numeral] = {}
    if base >= 10 and rng.random() < 0.30:
        for v in range(base + 1, base + 1 + rng.randint(1, 3)):
            irregular[v] = word_for(v)  # base+3 < base**2 for every rolled base

    # A head-final language leans toward modifier-first numerals (multiplier and units
    # before the base), a head-initial one toward base/tens-first.
    return NumeralSystem(
        base=base,
        atoms=atoms,
        base_word=base_word,
        square_word=square_word,
        multiplier_before_base=rng.random() < (0.85 if head_final else 0.55),
        units_before_tens=rng.random() < (0.35 if head_final else 0.10),
        bare_base_for_one=rng.random() < 0.5,
        irregular=irregular,
    )
