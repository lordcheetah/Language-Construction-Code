"""Syntax: how words combine into phrases and sentences.

Stage 4 of the toolkit, and the point where the earlier stages pay off together: a clause
is built from generated lexemes, its nouns and verb are inflected by the Stage 3
morphology (case by alignment, agreement on the verb), and the constituents are ordered
by the language's word-order parameters into an actual sentence.

The pieces:

- ``parameters`` — :class:`SyntaxParameters`: basic constituent order (SOV/SVO/…),
  head-directionality and its harmonic correlates (adposition, genitive, relative, and
  adjective placement), and morphosyntactic alignment (nominative-accusative vs
  ergative-absolutive).
- ``structure``  — a minimal clause model: :class:`Lexeme`, :class:`NounPhrase`,
  :class:`Clause`.
- ``linearizer`` — assigns case, applies agreement, and orders a clause into a glossed
  surface sentence.
- ``generator``  — rolls a typologically plausible set of parameters.
"""

from conlang.syntax.parameters import (
    SyntaxParameters,
    WordOrder,
    Side,
    Adposition,
    Alignment,
    Negation,
    PolarQuestion,
    derive_correlates,
)
from conlang.syntax.structure import Lexeme, NounPhrase, Clause, Role
from conlang.syntax.linearizer import Linearizer, GlossedWord, Sentence
from conlang.syntax.generator import random_syntax

__all__ = [
    "SyntaxParameters",
    "WordOrder",
    "Side",
    "Adposition",
    "Alignment",
    "Negation",
    "PolarQuestion",
    "derive_correlates",
    "Lexeme",
    "NounPhrase",
    "Clause",
    "Role",
    "Linearizer",
    "GlossedWord",
    "Sentence",
    "random_syntax",
]
