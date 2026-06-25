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
    gender: str | None = None            # a noun's lexical gender (for agreement), if any

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
    relative: "RelativeClause | None" = None  # a modifying relative clause
    person: str | None = None  # "1"/"2"/"3" for a pronoun; None = a 3rd-person full NP

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
        if self.relative is not None:
            parts.append(f"[that {self.relative.english()}]")
        return " ".join(parts)


@dataclass
class Coordination:
    """Two or more coordinated elements joined by a coordinator ("and" / "or").

    The conjuncts are either all :class:`NounPhrase` (a coordinated argument, e.g. "the dog
    and the cat") or all :class:`Clause` (a compound sentence, e.g. "the dog sleeps and the
    cat runs"). The coordinator is placed medially — between each pair of conjuncts — which
    is the dominant cross-linguistic pattern; ``coordinator`` names its particle ("and" for
    conjunction, "or" for disjunction).
    """

    conjuncts: list  # list[NounPhrase] | list[Clause]
    coordinator: str = "and"

    def __post_init__(self) -> None:
        if len(self.conjuncts) < 2:
            raise ValueError("a coordination needs at least two conjuncts")

    @property
    def number(self) -> str:
        # A conjoined subject is semantically plural and controls plural agreement; a
        # disjoined ("or") one keeps its first conjunct's number (a simplification of the
        # cross-linguistically variable agreement with disjoined subjects).
        if self.coordinator == "and":
            return "pl"
        return getattr(self.conjuncts[0], "number", "sg")

    @property
    def gloss(self) -> str:
        return f" {self.coordinator} ".join(c.gloss for c in self.conjuncts)


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
    object: NounPhrase | None = None        # the direct object / theme
    indirect_object: NounPhrase | None = None  # the recipient of a ditransitive (give X to Y)
    tense: str = "pres"
    obliques: list[AdpositionalPhrase] = field(default_factory=list)
    negated: bool = False
    # Sentence type: "declarative", "interrogative" (yes/no question), or "imperative".
    mood: str = "declarative"
    # If set, this is a content (wh-) question: that role's NP is an interrogative pronoun.
    # Only core arguments (subject/object) can be questioned — no oblique/adjunct/possessor
    # wh, no multiple wh — and fronting moves only the wh-word (no auxiliary inversion / V2).
    questioned: "Role | None" = None

    @property
    def is_transitive(self) -> bool:
        return self.object is not None

    @property
    def is_ditransitive(self) -> bool:
        return self.object is not None and self.indirect_object is not None

    @property
    def is_imperative(self) -> bool:
        return self.mood == "imperative"


@dataclass
class RelativeClause:
    """A clause modifying a noun, where the head fills *role* in the clause (the gap).

    The head noun is stored as that role of ``clause`` (for agreement) but is omitted from
    the surface — e.g. "the dog [that __ sees the bird]" gaps the embedded subject. By
    convention the gapped role's noun phrase should be the matrix head (its number drives
    the embedded verb's agreement); its other features there are unused, since it is not
    rendered.
    """

    clause: Clause
    role: Role  # the head's role in the embedded clause: SUBJECT or OBJECT

    def english(self) -> str:
        verb = self.clause.verb.gloss
        if self.role is Role.SUBJECT:  # head is the subject: "that sees the bird"
            tail = self.clause.object.gloss if self.clause.object is not None else ""
            return f"{verb} {tail}".strip()
        # head is the object: "that the woman sees"
        return f"{self.clause.subject.gloss} {verb}".strip()
