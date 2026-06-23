"""Flashcards and the deck built from a language.

Each dictionary word becomes two cards: a **recognition** card (shown the conlang word,
recall the meaning) and a **production** card (shown the meaning, recall the conlang word).
The two are scheduled independently because recognition is usually the easier direction.
The deck is ordered by *basicness* so the most frequent, useful words are taught first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    RECOGNITION = "recognition"  # conlang word -> meaning
    PRODUCTION = "production"     # meaning -> conlang word


@dataclass(frozen=True)
class Card:
    gloss: str       # the English meaning
    roman: str       # the conlang word (romanized)
    ipa: str         # its pronunciation
    pos: str
    field: str
    direction: Direction

    @property
    def id(self) -> str:
        return f"{self.gloss}:{self.direction.value}"

    @property
    def prompt(self) -> str:
        return self.roman if self.direction is Direction.RECOGNITION else self.gloss

    @property
    def answer(self) -> str:
        return self.gloss if self.direction is Direction.RECOGNITION else self.roman


def build_deck(language) -> list[Card]:
    """All cards for a language's lexicon, most basic words first.

    All recognition cards come before all production cards (each block basic-first), so a
    word's harder production card is introduced well after its recognition card rather than
    immediately afterward — recognition is learned first, and production isn't trivially
    primed by having just seen the answer.
    """
    # Grammatical particles (the negator, the question marker) aren't taught as vocabulary.
    entries = sorted(
        (e for e in language.lexicon.entries.values() if e.concept.pos != "particle"),
        key=lambda e: -e.concept.basicness,
    )

    def card(entry, direction: Direction) -> Card:
        return Card(
            gloss=entry.gloss, roman=entry.roman, ipa=entry.ipa,
            pos=entry.concept.pos, field=entry.concept.field, direction=direction,
        )

    return (
        [card(e, Direction.RECOGNITION) for e in entries]
        + [card(e, Direction.PRODUCTION) for e in entries]
    )
