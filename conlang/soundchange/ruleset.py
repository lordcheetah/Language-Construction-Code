"""An ordered set of sound changes, plus derivation over words and lexicons.

A ruleset is written as a small text block: comment lines start with ``#``, category
definitions use ``=``, and rules use ``>``. Categories must be defined before the rules
that reference them, and rules apply top-to-bottom — so the *order* encodes the relative
chronology of the changes (feeding and bleeding), exactly as in historical linguistics::

    # West-coast lenition (intervocalic voicing, then final devoicing)
    K = p t k
    K > [+voiced] / V_V
    [voiced obstruent] > [-voiced] / _#

``V`` (all vowels) and ``C`` (all consonants) are predefined. Comments occupy their own
line; an inline ``#`` is never a comment because ``#`` is the word-boundary symbol.
Applying a ruleset to a generated lexicon yields a daughter language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from conlang.phonology import data
from conlang.phonology.features import Segment
from conlang.phonology.wordgen import Word, Romanizer
from conlang.soundchange.matcher import CategoryMatcher
from conlang.soundchange.rule import SoundChange


def _default_categories() -> dict[str, CategoryMatcher]:
    return {
        "V": CategoryMatcher("V", frozenset(v.ipa for v in data.VOWELS)),
        "C": CategoryMatcher("C", frozenset(c.ipa for c in data.CONSONANTS)),
    }


@dataclass(frozen=True)
class Derivation:
    """The history of one word as a ruleset is applied, stage by stage."""

    original: tuple[Segment, ...]
    stages: tuple[tuple[str, tuple[Segment, ...]], ...]  # (rule_source, result_after_rule)

    @property
    def final(self) -> tuple[Segment, ...]:
        return self.stages[-1][1] if self.stages else self.original

    @staticmethod
    def _ipa(segments: Sequence[Segment]) -> str:
        return "".join(s.ipa for s in segments)

    @property
    def original_ipa(self) -> str:
        return self._ipa(self.original)

    @property
    def final_ipa(self) -> str:
        return self._ipa(self.final)

    @property
    def changed(self) -> bool:
        return self.original_ipa != self.final_ipa

    def trace(self) -> str:
        """A human-readable derivation showing only the rules that altered the word."""
        lines = [f"  {self.original_ipa}"]
        current = self.original_ipa
        for source, result in self.stages:
            result_ipa = self._ipa(result)
            if result_ipa != current:
                lines.append(f"  > {result_ipa:<16} ({source})")
                current = result_ipa
        return "\n".join(lines)


@dataclass(frozen=True)
class EvolvedWord:
    """A lexicon entry carried through a ruleset, keeping a link to its ancestor."""

    original: Word
    segments: tuple[Segment, ...]
    roman: str

    @property
    def ipa(self) -> str:
        return "".join(s.ipa for s in self.segments)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.original.roman} /{self.original.ipa}/ -> {self.roman} /{self.ipa}/"


@dataclass
class RuleSet:
    rules: list[SoundChange] = field(default_factory=list)
    categories: dict[str, CategoryMatcher] = field(default_factory=_default_categories)

    @classmethod
    def parse(
        cls, text: str, base_categories: dict[str, CategoryMatcher] | None = None
    ) -> "RuleSet":
        categories = dict(base_categories) if base_categories is not None else _default_categories()
        rules: list[SoundChange] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            # Comments are whole-line only (start with '#'); an inline '#' is never a
            # comment because '#' is the word-boundary symbol in environments (e.g. _#).
            if not line or line.startswith("#"):
                continue
            try:
                if ">" in line:
                    rules.append(SoundChange.parse(line, categories))
                elif "=" in line:
                    name, _, members = line.partition("=")
                    categories[name.strip()] = _parse_category(name.strip(), members)
                else:
                    raise ValueError(f"line is neither a category (=) nor a rule (>): {line!r}")
            except ValueError as exc:
                raise ValueError(f"sound-change ruleset, line {lineno}: {exc}") from exc
        return cls(rules, categories)

    @classmethod
    def from_rules(cls, rules: list[str]) -> "RuleSet":
        """Convenience: build from a list of rule strings using the default categories."""
        cats = _default_categories()
        return cls([SoundChange.parse(r, cats) for r in rules], cats)

    # --- Application -----------------------------------------------------------------
    def apply(self, segments: Sequence[Segment]) -> list[Segment]:
        current = list(segments)
        for rule in self.rules:
            current = rule.apply(current)
        return current

    def derive(self, segments: Sequence[Segment]) -> Derivation:
        original = tuple(segments)
        current: list[Segment] = list(segments)
        stages: list[tuple[str, tuple[Segment, ...]]] = []
        for rule in self.rules:
            current = rule.apply(current)
            stages.append((rule.source, tuple(current)))
        return Derivation(original, tuple(stages))

    def evolve_word(self, word: Word, romanizer: Romanizer | None = None) -> EvolvedWord:
        romanizer = romanizer or Romanizer()
        segments = [seg for syl in word.syllables for seg in syl]
        evolved = tuple(self.apply(segments))
        roman = romanizer.romanize([list(evolved)])
        return EvolvedWord(word, evolved, roman)

    def evolve_lexicon(
        self, words: Sequence[Word], romanizer: Romanizer | None = None
    ) -> list[EvolvedWord]:
        romanizer = romanizer or Romanizer()
        return [self.evolve_word(w, romanizer) for w in words]


def _parse_category(name: str, members: str) -> CategoryMatcher:
    symbols = members.replace(",", " ").split()
    if not symbols:
        raise ValueError(f"category {name!r} has no members")
    for sym in symbols:
        if sym not in data.BY_IPA:
            raise ValueError(f"category {name!r} references unknown symbol {sym!r}")
    return CategoryMatcher(name, frozenset(symbols))
