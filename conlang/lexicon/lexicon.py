"""Lexical entries and the lexicon container.

A :class:`LexicalEntry` pairs a concept with its word form and records the word's
*etymology* — whether it is a fresh root, a colexification of another concept, a
derivation, or a compound. The :class:`Lexicon` holds the entries and offers lookup by
concept or semantic field plus a printable glossary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from conlang.phonology.features import Segment
from conlang.lexicon.concepts import Concept, FIELDS


class Etymology(Enum):
    ROOT = "root"             # an independently coined root
    COLEXIFIED = "colexified"  # shares another concept's word
    DERIVED = "derived"        # built from another word with a derivational affix
    COMPOUND = "compound"      # built from two roots joined


@dataclass(frozen=True)
class LexicalEntry:
    concept: Concept
    form: tuple[Segment, ...]
    roman: str
    etymology: Etymology = Etymology.ROOT
    note: str = ""  # e.g. "= tree", "from hunt (AGENT)", "water+fall"

    @property
    def ipa(self) -> str:
        return "".join(s.ipa for s in self.form)

    @property
    def gloss(self) -> str:
        return self.concept.gloss

    def __str__(self) -> str:  # pragma: no cover - trivial
        tail = f"  [{self.etymology.value}: {self.note}]" if self.note else ""
        return f"{self.gloss}: {self.roman} /{self.ipa}/{tail}"


@dataclass
class Lexicon:
    entries: dict[str, LexicalEntry] = field(default_factory=dict)

    def get(self, gloss: str) -> LexicalEntry | None:
        return self.entries.get(gloss)

    def __len__(self) -> int:
        return len(self.entries)

    def by_field(self) -> dict[str, list[LexicalEntry]]:
        """Entries grouped by semantic field, in the canonical field order."""
        grouped: dict[str, list[LexicalEntry]] = {f: [] for f in FIELDS}
        for entry in self.entries.values():
            grouped.setdefault(entry.concept.field, []).append(entry)
        # Within a field, most basic concepts first.
        for entries in grouped.values():
            entries.sort(key=lambda e: -e.concept.basicness)
        return {f: e for f, e in grouped.items() if e}

    def of_etymology(self, etymology: Etymology) -> list[LexicalEntry]:
        return [e for e in self.entries.values() if e.etymology is etymology]

    def glossary(self) -> str:
        lines = []
        for fld, entries in self.by_field().items():
            lines.append(f"{fld}:")
            width = max((len(e.gloss) for e in entries), default=0)
            for e in entries:
                tail = f"   [{e.note}]" if e.note else ""
                lines.append(f"  {e.gloss:<{width}}  {e.roman} /{e.ipa}/{tail}")
        return "\n".join(lines)
