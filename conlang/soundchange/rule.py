"""A single sound change: parse ``target > replacement / environment`` and apply it.

The environment uses ``_`` to mark the target's position and may reference boundaries
(``#``), categories, literals, and feature classes. Either side of ``_`` may be empty
(unconstrained). Tokens may be space-separated (required for multi-character categories or
multi-codepoint IPA such as ``t͡ʃ``) or written compactly for single-character tokens
(``V_V``). Examples::

    p > b / V_V                               # context-sensitive voicing
    [voiceless plosive] > [+voiced] / V_V     # feature-class transform
    h > 0 / V_V                               # 0 (also ∅) deletes
    0 > ə / C_C                               # 0 on the left inserts (epenthesis)
    [plosive] > 0 / _(C)#                     # (X) is an optional environment element
    a > e / _C*i                              # X* / X+ = zero-or-more / one-or-more (a wildcard)
    [nasal] > [αplace] / _[αplace plosive]    # place assimilation (α-feature agreement)

Multi-segment rules act on a *window* of adjacent segments and build their output from
positional backreferences (``1`` = the first matched segment, ``2`` = the second, …) and
literals. This covers metathesis, gemination, and cluster splits or deletions::

    [stop] [liquid] > 2 1 / _                 # metathesis (swap the two segments)
    [voiceless plosive] > 1 1 / V_V           # gemination (double a segment)
    s k > k                                   # cluster reduction (two segments -> one)
    [plosive] > ʔ 1 / #_                      # split: prefix a glottal stop

A rule applies **simultaneously** across the word: every site is matched against the
original form, so a change does not feed itself within a single pass (ordering between
*different* rules is handled by :class:`~conlang.soundchange.ruleset.RuleSet`). Multi-segment
windows are applied left-to-right and never overlap (so under an even-width window an odd
trailing segment is left intact). Window targets are *adjacent* only — long-distance
metathesis is out of scope — and their output is backreferences and literals, not feature
changes (use a single-segment rule for feature transforms or α-feature agreement).
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
    set_feature,
    DIMENSIONS,
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


# --- Environment slots --------------------------------------------------------------
@dataclass(frozen=True)
class _Slot:
    matcher: Matcher
    # How many elements this slot consumes: "one" (exactly one), "optional" (0 or 1, from
    # ``(X)``), "star" (0 or more, from ``X*``), or "plus" (1 or more, from ``X+``). Capturing
    # an α-feature inside a repeated slot is undefined (last match wins); if such a slot
    # matches zero elements the variable is left unbound and the rule leaves the target as-is.
    quant: str = "one"


# --- Replacements -------------------------------------------------------------------
@dataclass(frozen=True)
class _Literal:
    segment: Segment

    def apply(self, matched, bindings) -> Segment | None:
        return self.segment


@dataclass(frozen=True)
class _Delete:
    def apply(self, matched, bindings) -> Segment | None:
        return None


@dataclass(frozen=True)
class _FeatureChange:
    """A sequence of feature operations: deltas (``+voiced``) and agreement assignments."""

    ops: tuple  # tuple of ("delta", str) or ("assign", (variable, attribute))

    def apply(self, matched: Segment, bindings: dict) -> Segment | None:
        seg: Segment | None = matched
        for kind, payload in self.ops:
            if seg is None:
                return None
            if kind == "delta":
                seg = apply_delta(seg, payload)
            else:  # assign a captured feature (agreement)
                var, attr = payload
                if var not in bindings:
                    return None
                seg = set_feature(seg, attr, bindings[var])
        return seg


# --- Rule ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SoundChange:
    source: str
    target: Matcher | None  # None for an insertion (epenthesis)
    replacement: object     # _Literal | _Delete | _FeatureChange
    left: tuple = field(default_factory=tuple)   # tuple of _Slot
    right: tuple = field(default_factory=tuple)  # tuple of _Slot
    deletes: bool = False
    inserts: bool = False
    target_seq: tuple = field(default_factory=tuple)        # multi-segment: tuple of Matcher
    replacement_seq: tuple = field(default_factory=tuple)   # tuple of ("ref", k) | ("lit", Segment)
    sequence: bool = False

    @classmethod
    def parse(cls, text: str, categories: dict[str, CategoryMatcher] | None = None) -> "SoundChange":
        categories = categories or {}
        if ">" not in text:
            raise ValueError(f"rule missing '>': {text!r}")
        rule_part, _, env_part = text.partition("/")
        target_str, _, repl_str = rule_part.partition(">")
        target_str, repl_str = target_str.strip(), repl_str.strip()

        left, right = _parse_environment(env_part, categories)

        if target_str in _DELETE_TOKENS:  # insertion (epenthesis)
            replacement = _parse_replacement(repl_str)
            if not isinstance(replacement, _Literal):
                raise ValueError("an insertion (0 > X) must insert a literal segment")
            if not left and not right:
                raise ValueError("an insertion (0 > X) needs an environment, e.g. 0 > ə / C_C")
            return cls(source=text.strip(), target=None, replacement=replacement,
                       left=tuple(left), right=tuple(right), inserts=True)

        target_tokens = _tokenize(target_str, categories)
        repl_tokens = [] if repl_str in _DELETE_TOKENS else _tokenize(repl_str, categories)
        # A window rule: more than one target segment, or an output that reorders/multiplies
        # segments (multiple output tokens or a positional backreference).
        if len(target_tokens) > 1 or len(repl_tokens) > 1 or any(_is_backref(t) for t in repl_tokens):
            target_seq = tuple(_parse_token(t, categories, allow_boundary=False) for t in target_tokens)
            replacement_seq = _parse_replacement_seq(repl_str, repl_tokens, len(target_seq))
            # Invariant: replacement is None iff sequence; apply() dispatches on .sequence.
            return cls(source=text.strip(), target=None, replacement=None,
                       left=tuple(left), right=tuple(right), sequence=True,
                       target_seq=target_seq, replacement_seq=replacement_seq,
                       deletes=not replacement_seq)

        # Single-segment rewrite (the common case).
        replacement = _parse_replacement(repl_str)
        target = _parse_token(target_str, categories, allow_boundary=False)
        # Every agreement variable used in the replacement must be captured in the context.
        _check_bound_variables(replacement, target, left, right)
        return cls(
            source=text.strip(),
            target=target,
            replacement=replacement,
            left=tuple(left),
            right=tuple(right),
            deletes=isinstance(replacement, _Delete),
        )

    def apply(self, segments: Sequence[Segment]) -> list[Segment]:
        aug: list[Element] = [BOUNDARY, *segments, BOUNDARY]
        if self.inserts:
            return self._apply_insertion(aug)
        if self.sequence:
            return self._apply_sequence(aug)
        return self._apply_rewrite(aug)

    def _apply_sequence(self, aug: list[Element]) -> list[Segment]:
        """Apply a multi-segment window rule, left-to-right, non-overlapping.

        Windows are matched against the original form; once a window is taken the scan
        resumes past it, so a rule never reapplies inside its own output within one pass.
        """
        last_real = len(aug) - 2  # aug[1..last_real] are real segments
        width = len(self.target_seq)
        taken: dict[int, list[Segment]] = {}
        i = 1
        while i <= last_real - width + 1:
            window = aug[i:i + width]
            if all(isinstance(window[k], Segment) and self.target_seq[k].matches(window[k])
                   for k in range(width)):
                right = _match_side(aug[i + width:], self.right)
                left = _match_side(list(reversed(aug[:i])), tuple(reversed(self.left)))
                if right is not None and left is not None:
                    taken[i] = window  # type: ignore[assignment]
                    i += width
                    continue
            i += 1

        out: list[Segment] = []
        i = 1
        while i <= last_real:
            if i in taken:
                window = taken[i]
                for kind, val in self.replacement_seq:
                    out.append(window[val - 1] if kind == "ref" else val)
                i += width
            else:
                out.append(aug[i])  # type: ignore[arg-type]
                i += 1
        return out

    def _apply_rewrite(self, aug: list[Element]) -> list[Segment]:
        # Decide every site against the original form (simultaneous application).
        decisions: dict[int, dict] = {}
        for i in range(1, len(aug) - 1):
            if not self.target.matches(aug[i]):
                continue
            bound = self._match_environment(aug, i + 1, i)
            if bound is None:
                continue
            bound.update(_bindings(self.target, aug[i]))
            decisions[i] = bound

        out: list[Segment] = []
        for i in range(1, len(aug) - 1):
            seg = aug[i]
            assert isinstance(seg, Segment)
            if i not in decisions:
                out.append(seg)
                continue
            result = self.replacement.apply(seg, decisions[i])
            if result is None and not isinstance(self.replacement, _Delete):
                out.append(seg)  # a transform with no attested result: leave it unchanged
            elif result is not None:
                out.append(result)
        return out

    def _apply_insertion(self, aug: list[Element]) -> list[Segment]:
        n = len(aug) - 2
        inserts: dict[int, dict] = {}
        for g in range(1, len(aug)):  # gap between aug[g-1] and aug[g]
            bound = self._match_environment(aug, g, g)
            if bound is not None:
                inserts[g] = bound

        out: list[Segment] = []
        for g in range(1, len(aug)):
            if g in inserts:
                inserted = self.replacement.apply(None, inserts[g])
                if inserted is not None:
                    out.append(inserted)
            if g <= n:  # aug[g] is a real segment (not the closing boundary)
                out.append(aug[g])  # type: ignore[arg-type]
        return out

    def _match_environment(self, aug: list[Element], right_start: int, left_end: int) -> dict | None:
        """Match both sides; return merged bindings, or None if either side fails.

        ``right_start`` is the first index of the right context; ``left_end`` is one past
        the last index of the left context (the target index for a rewrite, or the gap
        index for an insertion).

        Binding precedence (only matters if one variable is captured in more than one
        place): right context wins over left, and the target wins over both. Capturing the
        same variable on both sides is unusual; for the common single-capture rule this is
        moot.
        """
        right = _match_side(aug[right_start:], self.right)
        if right is None:
            return None
        left = _match_side(list(reversed(aug[:left_end])), tuple(reversed(self.left)))
        if left is None:
            return None
        return {**left, **right}


# --- Parsing helpers ----------------------------------------------------------------
def _check_bound_variables(replacement, target, left, right) -> None:
    """Raise if the replacement copies an agreement variable that nothing in the context binds."""
    if not isinstance(replacement, _FeatureChange):
        return
    used = {payload[0] for kind, payload in replacement.ops if kind == "assign"}
    if not used:
        return
    matchers = [s.matcher for s in (*left, *right)]
    if target is not None:
        matchers.append(target)
    bound = {var for m in matchers for var, _attr in getattr(m, "captures", ())}
    missing = used - bound
    if missing:
        raise ValueError(
            f"replacement uses unbound agreement variable(s) {sorted(missing)}; "
            "capture them in the environment, e.g. _[αplace plosive]"
        )


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
        ops = []
        for w in token[1:-1].replace(",", " ").split():
            key = w.lower()
            if key and key[0] in "αβγ@" and key[1:] in DIMENSIONS:
                var = "α" if key[0] == "@" else key[0]
                ops.append(("assign", (var, DIMENSIONS[key[1:]])))
            elif w in _BARE_TO_DELTA:
                ops.append(("delta", _BARE_TO_DELTA[w]))
            elif w[:1] in "+-":
                ops.append(("delta", w))
            else:
                raise ValueError(f"replacement feature {w!r} must be a delta (+/-), a known feature, or an αfeature")
        if not ops:
            raise ValueError("empty replacement []")
        return _FeatureChange(tuple(ops))
    if token in data.BY_IPA:
        return _Literal(data.BY_IPA[token])
    raise ValueError(f"unknown replacement {token!r}")


def _is_backref(token: str) -> bool:
    # A bare digit 1-9 is a positional backreference — unless it is itself an IPA symbol
    # (no digit-bearing symbols are in the inventory today, but tone numerals could appear).
    return len(token) == 1 and token in "123456789" and token not in data.BY_IPA


def _parse_replacement_seq(repl_str: str, tokens: list[str], n_target: int) -> tuple:
    """Parse a window rule's output: positional backreferences and/or literal segments.

    Returns a tuple of ``("ref", k)`` (copy the k-th matched segment) or ``("lit", seg)``.
    An empty tuple means the whole matched window is deleted.
    """
    if repl_str in _DELETE_TOKENS:
        return ()
    specs: list[tuple] = []
    for tok in tokens:
        if _is_backref(tok):
            k = int(tok)
            if not 1 <= k <= n_target:
                raise ValueError(
                    f"backreference {k} out of range; the target has {n_target} segment(s)"
                )
            specs.append(("ref", k))
        elif tok in data.BY_IPA:
            specs.append(("lit", data.BY_IPA[tok]))
        elif tok.startswith("[") and tok.endswith("]"):
            raise ValueError(
                "feature changes (e.g. [+voiced], [αplace]) are not supported in a "
                "multi-segment replacement; use a single-segment rule for those"
            )
        else:
            raise ValueError(
                f"sequence replacement token {tok!r} must be a backreference "
                f"(1..{n_target}) or an IPA segment"
            )
    if not specs:
        raise ValueError("empty sequence replacement")
    return tuple(specs)


def _parse_environment(env_part: str, categories) -> tuple[list[_Slot], list[_Slot]]:
    env = env_part.strip()
    if not env:
        return [], []
    if "_" not in env:
        raise ValueError(f"environment missing '_': {env_part!r}")
    left_str, _, right_str = env.partition("_")
    left = [_parse_slot(t, categories) for t in _fold_quantifiers(_tokenize(left_str, categories))]
    right = [_parse_slot(t, categories) for t in _fold_quantifiers(_tokenize(right_str, categories))]
    return left, right


def _fold_quantifiers(tokens: list[str]) -> list[str]:
    """Attach a bare ``*``/``+`` quantifier onto the preceding token (the tokenizer emits it
    as its own character), so ``C*`` arrives here as ``["C", "*"]`` and leaves as ``["C*"]``."""
    out: list[str] = []
    for t in tokens:
        if t in ("*", "+") and out and out[-1] not in ("*", "+"):
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return out


def _parse_slot(token: str, categories) -> _Slot:
    if token.startswith("(") and token.endswith(")"):
        return _Slot(_parse_token(token[1:-1].strip(), categories), quant="optional")
    if token and token[-1] in "*+":
        mark = token[-1]
        base = token[:-1]
        # The quantifier must follow exactly one class/segment — not nothing (`*i`), another
        # quantifier (`C**`), a group (`(C)*`), or a boundary (`#*`, which is meaningless).
        if not base or base[-1] in "*+" or base.startswith("(") or base == "#":
            raise ValueError(
                f"a {mark!r} quantifier must follow a single class or segment, got {token!r}"
            )
        quant = "star" if mark == "*" else "plus"
        return _Slot(_parse_token(base, categories), quant=quant)
    return _Slot(_parse_token(token, categories), quant="one")


def _tokenize(side: str, categories: dict) -> list[str]:
    """Split an environment side into tokens.

    Bracketed feature classes ``[...]`` and parenthesised optional groups ``(...)`` are
    atomic even when they contain spaces (e.g. ``[αplace plosive]``). Outside groups, spaces
    separate tokens; a run with no spaces is split by longest-match against known tokens, so
    compact forms (``VC``) and multi-codepoint IPA (``t͡ʃ``) both work.
    """
    side = side.strip()
    if not side:
        return []
    known = sorted({*categories, *data.BY_IPA, "#"}, key=len, reverse=True)
    tokens: list[str] = []
    i = 0
    while i < len(side):
        ch = side[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "([":
            close = ")" if ch == "(" else "]"
            j = side.find(close, i)
            if j == -1:
                raise ValueError(f"unbalanced {ch!r} in environment: {side!r}")
            tokens.append(side[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < len(side) and not side[j].isspace() and side[j] not in "([":
            j += 1
        tokens.extend(_split_run(side[i:j], known))
        i = j
    return tokens


def _split_run(run: str, known: list[str]) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(run):
        for tok in known:
            if tok and run.startswith(tok, i):
                tokens.append(tok)
                i += len(tok)
                break
        else:
            tokens.append(run[i])  # unknown char: let _parse_token raise a clear error
            i += 1
    return tokens


def _bindings(matcher, element) -> dict:
    fn = getattr(matcher, "bindings", None)
    return fn(element) if fn is not None else {}


def _match_side(elements: list, slots: tuple) -> dict | None:
    """Match *slots* against a prefix of *elements*; return bindings or None.

    A required slot consumes one element; ``(X)`` consumes zero or one; ``X*``/``X+`` consume
    a run of zero-or-more / one-or-more (greedily, with backtracking). Elements beyond the
    matched slots are ignored. Returns the bindings captured by any feature-class slots along
    the successful path (for a repeated slot, the last consumed element's bindings win).
    """
    return _seq(elements, slots, 0, 0)


def _seq(elements: list, slots: tuple, ei: int, si: int) -> dict | None:
    if si == len(slots):
        return {}
    slot = slots[si]
    if slot.quant in ("star", "plus"):
        # Greedily consume the run of matches, then backtrack from longest to shortest.
        binds = []
        j = ei
        while j < len(elements) and slot.matcher.matches(elements[j]):
            binds.append(_bindings(slot.matcher, elements[j]))
            j += 1
        min_k = 1 if slot.quant == "plus" else 0
        for k in range(len(binds), min_k - 1, -1):
            rest = _seq(elements, slots, ei + k, si + 1)
            if rest is not None:
                merged: dict = {}
                for b in binds[:k]:
                    merged.update(b)
                return {**merged, **rest}
        return None
    if ei < len(elements) and slot.matcher.matches(elements[ei]):
        rest = _seq(elements, slots, ei + 1, si + 1)
        if rest is not None:
            return {**_bindings(slot.matcher, elements[ei]), **rest}
    if slot.quant == "optional":
        rest = _seq(elements, slots, ei, si + 1)
        if rest is not None:
            return rest
    return None
