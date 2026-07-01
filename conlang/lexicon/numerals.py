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

A minority of base-10/12 languages also form their **decades** irregularly, in one of two ways
(mutually exclusive; most compose regularly):

- **Suppletive decades** — the tens (``2·base``, ``3·base``, …) take their own opaque roots
  rather than "multiplier + base" (loosely, English *twenty/thirty* vs. *two-ten/three-ten*).
- A **vigesimal sub-base** (base 10 only) — decades are grouped in twenties (French
  *quatre-vingts* 80 = four-twenty, *quatre-vingt-dix* 90 = four-twenty-ten), an odd decade
  adding a leftover ten. This scores *every* even decade (40 = two-score, 60 = three-score), so
  it is the fully-vigesimal Welsh/Breton/Danish "scores" system rather than French-exact (French
  keeps *quarante*/*soixante* decimal and only goes vigesimal at 80).
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
    decades: dict[int, Numeral] = field(default_factory=dict)    # suppletive tens (2·base…), by value
    score_word: "Numeral | None" = None  # a "twenty" (2·base) root -> a vigesimal sub-base for decades

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
            group = self._tens_group(tens)
            if not units:
                return group
            unit = [self.atoms[units]]
            return unit + group if self.units_before_tens else group + unit
        hundreds, rest = divmod(n, self.base ** 2)
        group = self._group(hundreds, self.square_word)
        return group + (self._components(rest) if rest else [])

    def _tens_group(self, tens: int) -> list[Numeral]:
        """The words for ``tens · base`` (a decade): a vigesimal score grouping, a suppletive
        decade root, or the regular multiplier × base — whichever this language uses."""
        if self.score_word is not None:              # vigesimal sub-base (four-twenty…)
            return self._score_group(tens)
        decade = self.decades.get(tens * self.base)  # a suppletive decade root
        if decade is not None:
            return [decade]
        return self._group(tens, self.base_word)     # regular: multiplier × base

    def _score_group(self, tens: int) -> list[Numeral]:
        """A decade under a vigesimal sub-base: ``tens·base`` counted in twenties, plus a
        leftover ten for an odd decade (French quatre-vingt-dix, 90 = four-twenty-ten)."""
        twenties, rem = divmod(tens * self.base, 2 * self.base)
        parts = self._group(twenties, self.score_word) if twenties else []
        if rem:  # a leftover ten (odd decade): "… ten"
            parts = parts + [self.base_word]
        return parts

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
    multiplier_before_base = rng.random() < (0.85 if head_final else 0.55)
    units_before_tens = rng.random() < (0.35 if head_final else 0.10)
    bare_base_for_one = rng.random() < 0.5

    # Irregular decade formation (base 10/12 only): either suppletive decade roots, or — base 10
    # only — a vigesimal sub-base that groups the decades in twenties (French quatre-vingts).
    # Mutually exclusive and a minority; rolled last so a regular result stays byte-identical.
    decades: dict[int, Numeral] = {}
    score_word: Numeral | None = None
    if base in (10, 12):
        roll = rng.random()
        if base == 10 and roll < 0.15:
            score_word = word_for(2 * base)  # a distinct "twenty" (score) root
        elif roll < 0.35:
            for tens in range(2, base):
                decades[tens * base] = word_for(tens * base)

    return NumeralSystem(
        base=base,
        atoms=atoms,
        base_word=base_word,
        square_word=square_word,
        multiplier_before_base=multiplier_before_base,
        units_before_tens=units_before_tens,
        bare_base_for_one=bare_base_for_one,
        irregular=irregular,
        decades=decades,
        score_word=score_word,
    )
