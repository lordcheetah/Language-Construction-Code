"""Lexicon: the vocabulary of a language, organized by meaning.

Stage 5 of the toolkit, after Mark Rosenfelder's *Conlanger's Lexipedia*. Where a
dictionary is *semasiological* (form -> meaning), this stage is *onomasiological*: it
starts from a curated inventory of concepts grouped into semantic fields and gives each a
word. The result feels like a real lexicon rather than a pile of unrelated roots because
it models three ways languages avoid coining everything from scratch:

- **Colexification** — concepts the world's languages commonly express with one word
  (tree/wood, sun/day, the "grue" green/blue) are merged with realistic probability.
- **Derivation** — some words are built from others via the Stage 3 derivational affixes
  (an agent noun from a verb, an adjective from a noun).
- **Compounding** — some words are two roots joined (water + fall -> waterfall).

Root length follows Zipf's law of abbreviation: the most basic, frequent concepts get the
shortest words.
"""

from conlang.lexicon.concepts import (
    Concept,
    CONCEPTS,
    FIELDS,
    COLEXIFICATION,
    DERIVATIONS,
    COMPOUNDS,
    BY_GLOSS,
)
from conlang.lexicon.lexicon import LexicalEntry, Lexicon, Etymology
from conlang.lexicon.generator import build_lexicon
from conlang.lexicon.numerals import NumeralSystem, Numeral, build_numerals

__all__ = [
    "Concept",
    "CONCEPTS",
    "FIELDS",
    "COLEXIFICATION",
    "DERIVATIONS",
    "COMPOUNDS",
    "BY_GLOSS",
    "LexicalEntry",
    "Lexicon",
    "Etymology",
    "build_lexicon",
    "NumeralSystem",
    "Numeral",
    "build_numerals",
]
