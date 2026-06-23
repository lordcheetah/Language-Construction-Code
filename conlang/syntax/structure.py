"""A minimal clause model: lexemes, noun phrases, and clauses.

Deliberately small — just enough structure for the linearizer to order and inflect a
sentence. A :class:`NounPhrase` is a head noun with optional adjective, possessor, and
grammatical features (number, definiteness). A :class:`Clause` is a subject, a verb, and
an optional object plus the verb's tense; whether it is transitive is simply whether it
has an object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from conlang.phonology.features import Segment


class Role(Enum):
    """Grammatical relation of a noun phrase within its clause."""

    SUBJECT = "subject"
    OBJECT = "object"


@dataclass(frozen=True)
class Lexeme:
    """A dictionary word: a phonological root, its word class, and an English gloss."""

    root: tuple[Segment, ...]
    word_class: str
    gloss: str
    inflection_class: str | None = None  # which declension/conjugation this word takes

    @property
    def ipa(self) -> str:
        return "".join(s.ipa for s in self.root)


@dataclass
class NounPhrase:
    head: Lexeme
    adjective: Lexeme | None = None
    number: str = "sg"
    definiteness: str | None = None  # "def" / "indef" / None (unmarked)
    genitive: "NounPhrase | None" = None  # a possessor noun phrase

    @property
    def gloss(self) -> str:
        parts = []
        if self.definiteness == "def":
            parts.append("the")
        elif self.definiteness == "indef":
            parts.append("a")
        if self.adjective is not None:
            parts.append(self.adjective.gloss)
        parts.append(self.head.gloss)
        if self.number == "pl":
            parts[-1] = parts[-1] + "s"
        if self.genitive is not None:
            parts.append(f"of-{self.genitive.head.gloss}")
        return " ".join(parts)


@dataclass
class AdpositionalPhrase:
    """An adposition plus its noun phrase (a preposition or postposition + NP)."""

    adposition: Lexeme
    np: NounPhrase
    relation: str = ""  # a short gloss for the adposition, e.g. "near" / "LOC"


@dataclass
class Clause:
    subject: NounPhrase
    verb: Lexeme
    object: NounPhrase | None = None
    tense: str = "pres"
    obliques: list[AdpositionalPhrase] = field(default_factory=list)
    negated: bool = False
    # Sentence type: "declarative", "interrogative" (yes/no question), or "imperative".
    mood: str = "declarative"

    @property
    def is_transitive(self) -> bool:
        return self.object is not None

    @property
    def is_imperative(self) -> bool:
        return self.mood == "imperative"
