"""A single sound change: parse ``target > replacement / environment`` and apply it.

The environment uses ``_`` to mark the target's position and may reference boundaries
(``#``), categories, literals, and feature classes. Either side of ``_`` may be empty
(unconstrained). Tokens may be space-separated (required for multi-character categories
or multi-codepoint IPA such as ``t͡ʃ``) or written compactly for single-character tokens
(``V_V``). Examples::

    p > b / V_V
    [voiceless plosive] > [+voiced] / V_V
    [voiced obstruent] > [-voiced] / _#
    h > 0 / V_V                # 0 (also ∅) deletes
    n > m / _ [bilabial]

A rule applies **simultaneously** across the word: every site is matched against the
original form, so a change does not feed itself within a single pass (ordering between
*different* rules is handled by :class:`~conlang.soundchange.ruleset.RuleSet`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from conlang.phonology import data
from conlang.phonology.features import Segment
from conlang.soundchange.matcher import (
    BOUNDARY,
    Element,
    Matcher,
    LiteralMatcher,
    CategoryMatcher,
    BoundaryMatcher,
    FeatureClass,
    apply_delta,
)

_DELETE_TOKENS = {"", "0", "∅", "Ø", "-"}
# Bare feature words allowed in a replacement, normalized to deltas.
_BARE_TO_DELTA = {
    "voiced": "+voiced",
    "voiceless": "-voiced",
    "long": "+long",
    "short": "-long",
    "rounded": "+rounded",
    "unrounded": "-rounded",
}


# --- Replacements -------------------------------------------------------------------
@dataclass(frozen=True)
class _Literal:
    segment: Segment

    def apply(self, matched: Segment) -> Segment | None:
        return self.segment


@dataclass(frozen=True)
class _Delete:
    def apply(self, matched: Segment) -> Segment | None:
        return None


@dataclass(frozen=True)
class _Delta:
    deltas: tuple[str, ...]

    def apply(self, matched: Segment) -> Segment | None:
        seg: Segment | None = matched
        for d in self.deltas:
            if seg is None:
                return None
            seg = apply_delta(seg, d)
        return seg


# --- Rule ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SoundChange:
    source: str
    target: Matcher
    replacement: object  # _Literal | _Delete | _Delta
    left: tuple[Matcher, ...] = field(default_factory=tuple)
    right: tuple[Matcher, ...] = field(default_factory=tuple)
    deletes: bool = False  # True for a deletion replacement (affects output length)

    @classmethod
    def parse(cls, text: str, categories: dict[str, CategoryMatcher] | None = None) -> "SoundChange":
        categories = categories or {}
        if ">" not in text:
            raise ValueError(f"rule missing '>': {text!r}")
        rule_part, _, env_part = text.partition("/")
        target_str, _, repl_str = rule_part.partition(">")

        target = _parse_token(target_str.strip(), categories, allow_boundary=False)
        replacement = _parse_replacement(repl_str.strip())
        left, right = _parse_environment(env_part, categories)
        return cls(
            source=text.strip(),
            target=target,
            replacement=replacement,
            left=tuple(left),
            right=tuple(right),
            deletes=isinstance(replacement, _Delete),
        )

    def apply(self, segments: Sequence[Segment]) -> list[Segment]:
        """Apply this change to a word (a sequence of segments)."""
        aug: list[Element] = [BOUNDARY, *segments, BOUNDARY]
        # Decide every site against the original form (simultaneous application).
        decisions: dict[int, object] = {}
        for i in range(1, len(aug) - 1):
            if not self.target.matches(aug[i]):
                continue
            if not _left_matches(aug, i, self.left):
                continue
            if not _right_matches(aug, i, self.right):
                continue
            decisions[i] = self.replacement

        out: list[Segment] = []
        for i in range(1, len(aug) - 1):
            seg = aug[i]
            assert isinstance(seg, Segment)
            repl = decisions.get(i)
            if repl is None:
                out.append(seg)
                continue
            result = repl.apply(seg)
            if result is None and not isinstance(repl, _Delete):
                # A feature delta with no attested result: leave the segment unchanged.
                out.append(seg)
            elif result is not None:
                out.append(result)
            # _Delete (result is None) appends nothing.
        return out


# --- Parsing helpers ----------------------------------------------------------------
def _parse_token(token: str, categories, *, allow_boundary: bool = True) -> Matcher:
    if token == "#":
        if not allow_boundary:
            raise ValueError("'#' boundary is only valid in an environment")
        return BoundaryMatcher()
    if token.startswith("[") and token.endswith("]"):
        return FeatureClass.parse(token[1:-1])
    if token in categories:
        return categories[token]
    if token in data.BY_IPA:
        return LiteralMatcher(data.BY_IPA[token])
    raise ValueError(f"unknown token {token!r} (not a category, IPA symbol, or [feature class])")


def _parse_replacement(token: str):
    if token in _DELETE_TOKENS:
        return _Delete()
    if token.startswith("[") and token.endswith("]"):
        words = token[1:-1].replace(",", " ").split()
        deltas: list[str] = []
        for w in words:
            if w in _BARE_TO_DELTA:
                deltas.append(_BARE_TO_DELTA[w])
            elif w[:1] in "+-":
                deltas.append(w)
            else:
                raise ValueError(f"replacement feature {w!r} must be a delta (+/-) or a known feature")
        if not deltas:
            raise ValueError(f"empty replacement []")
        return _Delta(tuple(deltas))
    if token in data.BY_IPA:
        return _Literal(data.BY_IPA[token])
    raise ValueError(f"unknown replacement {token!r}")


def _parse_environment(env_part: str, categories) -> tuple[list[Matcher], list[Matcher]]:
    env = env_part.strip()
    if not env:
        return [], []
    if "_" not in env:
        raise ValueError(f"environment missing '_': {env_part!r}")
    left_str, _, right_str = env.partition("_")
    left = [_parse_token(t, categories) for t in _tokenize(left_str, categories)]
    right = [_parse_token(t, categories) for t in _tokenize(right_str, categories)]
    return left, right


def _tokenize(side: str, categories: dict) -> list[str]:
    side = side.strip()
    if not side:
        return []
    if any(ch.isspace() for ch in side):
        return side.split()
    # Compact mode: greedily match the longest known token at each position so that
    # multi-codepoint IPA (e.g. the affricate /t͡ʃ/, which carries a tie bar) and
    # multi-character category names stay intact instead of being split per codepoint.
    known = sorted({*categories, *data.BY_IPA, "#"}, key=len, reverse=True)
    tokens: list[str] = []
    i = 0
    while i < len(side):
        if side[i] == "[":
            j = side.index("]", i)
            tokens.append(side[i : j + 1])
            i = j + 1
            continue
        for tok in known:
            if tok and side.startswith(tok, i):
                tokens.append(tok)
                i += len(tok)
                break
        else:
            # Unknown character: emit it so _parse_token raises a clear error.
            tokens.append(side[i])
            i += 1
    return tokens


def _left_matches(aug: list[Element], i: int, left: tuple[Matcher, ...]) -> bool:
    if not left:
        return True
    start = i - len(left)
    if start < 0:
        return False
    return all(m.matches(aug[start + k]) for k, m in enumerate(left))


def _right_matches(aug: list[Element], i: int, right: tuple[Matcher, ...]) -> bool:
    if not right:
        return True
    end = i + 1 + len(right)
    if end > len(aug):
        return False
    return all(m.matches(aug[i + 1 + k]) for k, m in enumerate(right))
