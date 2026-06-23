"""Affixes: bound morphemes that attach to a stem to mark grammatical features.

Stage 3 handles the two most common, concatenative positions — **prefix** and **suffix**.
A zero affix (empty form) is how unmarked values surface, so a citation form is just the
root with zero affixes. Non-concatenative morphology (infixes, ablaut, reduplication,
templatic root-and-pattern) is deferred to the backlog.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from conlang.phonology.features import Segment
from conlang.morphology.features import FeatureBundle


class Position(Enum):
    PREFIX = "prefix"
    SUFFIX = "suffix"


@dataclass(frozen=True)
class Affix:
    """A bound morpheme: a phonological form, where it attaches, and what it marks."""

    form: tuple[Segment, ...]
    position: Position
    marks: FeatureBundle
    gloss: str = ""  # optional human label, e.g. "PL" or "AGENT"

    @property
    def is_zero(self) -> bool:
        return len(self.form) == 0

    def attach(self, stem: Sequence[Segment]) -> list[Segment]:
        if self.position is Position.PREFIX:
            return [*self.form, *stem]
        return [*stem, *self.form]

    @property
    def ipa(self) -> str:
        return "".join(s.ipa for s in self.form)

    def __str__(self) -> str:  # pragma: no cover - trivial
        shape = self.ipa or "∅"
        return f"{shape} ({self.position.value}: {self.marks})"
