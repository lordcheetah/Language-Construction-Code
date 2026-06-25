"""Matchers and the feature machinery sound-change rules are built from.

Three kinds of thing can occupy a position in a rule's target or environment:

- a **literal** segment (an IPA symbol),
- a **category** — a named set of segments such as ``V`` or ``K`` (defined in a ruleset),
- a **feature class** — a natural class written ``[voiceless plosive]`` that matches any
  segment with those features.

A word boundary ``#`` is represented by the :data:`BOUNDARY` sentinel so that
environments like ``_#`` (word-finally) match uniformly alongside segment matchers.

This module also owns the **reverse feature lookup**: applying a delta such as
``[+voiced]`` to a matched segment means recomputing its features and finding the
attested IPA segment that has them (so output is always a real, romanizable symbol).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Union

from conlang.phonology import data
from conlang.phonology.features import (
    Segment,
    Consonant,
    Vowel,
    Place,
    Manner,
    Voicing,
    Height,
    Backness,
)


class _Boundary:
    """Sentinel for a word edge (``#``)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "#"


BOUNDARY = _Boundary()

# An element of a word as seen by the matcher: a real segment or a word boundary.
Element = Union[Segment, _Boundary]


# --- Reverse feature lookup ---------------------------------------------------------
# Indices from a feature tuple to the attested segment that has those features, so a
# feature delta resolves to a real IPA symbol rather than a fabricated one.
_CONSONANT_BY_FEATURES: dict[tuple, Consonant] = {
    (c.place, c.manner, c.voicing): c for c in data.CONSONANTS
}
_VOWEL_BY_FEATURES: dict[tuple, Vowel] = {
    (v.height, v.backness, v.rounded, v.long): v for v in data.VOWELS
}


def _vowel_with_length(seg: Vowel, long: bool) -> Vowel | None:
    """Return *seg* with the given length.

    Long vowels are systematically the short symbol plus the length mark ``ː``, so they
    are constructed directly rather than required to exist in the chart.
    """
    if seg.long == long:
        return seg
    if long:
        return replace(seg, long=True, ipa=seg.ipa.rstrip("ː") + "ː")
    return replace(seg, long=False, ipa=seg.ipa.rstrip("ː"))


# Feature deltas understood in a replacement, e.g. ``[+voiced]``. Each maps to a function
# (segment -> modified segment or None if the result is not an attested segment).
def _set_voicing(seg: Segment, voiced: bool) -> Segment | None:
    if not isinstance(seg, Consonant):
        return None
    target = Voicing.VOICED if voiced else Voicing.VOICELESS
    return _CONSONANT_BY_FEATURES.get((seg.place, seg.manner, target))


def _set_rounded(seg: Segment, rounded: bool) -> Segment | None:
    if not isinstance(seg, Vowel):
        return None
    return _VOWEL_BY_FEATURES.get((seg.height, seg.backness, rounded, seg.long))


_FEATURE_DELTAS = {
    "+voiced": lambda s: _set_voicing(s, True),
    "-voiced": lambda s: _set_voicing(s, False),
    "+voice": lambda s: _set_voicing(s, True),
    "-voice": lambda s: _set_voicing(s, False),
    "+long": lambda s: _vowel_with_length(s, True) if isinstance(s, Vowel) else None,
    "-long": lambda s: _vowel_with_length(s, False) if isinstance(s, Vowel) else None,
    "+round": lambda s: _set_rounded(s, True),
    "-round": lambda s: _set_rounded(s, False),
    "+rounded": lambda s: _set_rounded(s, True),
    "-rounded": lambda s: _set_rounded(s, False),
}


def apply_delta(seg: Segment, delta: str) -> Segment | None:
    """Apply a feature delta like ``+voiced`` to *seg*; None if it has no valid result."""
    fn = _FEATURE_DELTAS.get(delta)
    if fn is None:
        raise ValueError(f"unknown feature delta: [{delta}]")
    return fn(seg)


def set_feature(seg: Segment, attr: str, value) -> Segment | None:
    """Set one feature dimension of *seg* to *value*, resolving to an attested segment.

    Used by feature-agreement rules (e.g. a nasal taking a following stop's place). Returns
    None when no attested segment has the requested feature combination.
    """
    if isinstance(seg, Consonant):
        if attr not in ("place", "manner", "voicing"):
            return None
        feats = {"place": seg.place, "manner": seg.manner, "voicing": seg.voicing}
        feats[attr] = value
        return _CONSONANT_BY_FEATURES.get((feats["place"], feats["manner"], feats["voicing"]))
    if isinstance(seg, Vowel):
        if attr == "long":
            return _vowel_with_length(seg, bool(value))
        if attr not in ("height", "backness", "rounded"):
            return None
        feats = {"height": seg.height, "backness": seg.backness, "rounded": seg.rounded}
        feats[attr] = value
        return _VOWEL_BY_FEATURES.get((feats["height"], feats["backness"], feats["rounded"], seg.long))
    return None


# Feature dimensions an agreement variable can copy (word -> Segment attribute name).
DIMENSIONS = {
    "place": "place", "voicing": "voicing", "voice": "voicing", "manner": "manner",
    "height": "height", "backness": "backness", "rounded": "rounded", "round": "rounded",
    "length": "long",
}
_DIMENSIONS = DIMENSIONS  # backward-compatible alias


# --- Feature classes ----------------------------------------------------------------
# Vocabulary mapping descriptive words to feature constraints. A constraint is a
# (field name, set of allowed values); a class matches a segment when every constraint
# holds. Cover terms like "obstruent" expand to a set of manners.
_KIND = {"consonant": "consonant", "vowel": "vowel"}

_PLACE = {p.value.replace(" ", ""): p for p in Place}
_MANNER_WORDS: dict[str, set[Manner]] = {
    "plosive": {Manner.PLOSIVE},
    "stop": {Manner.PLOSIVE},
    "affricate": {Manner.AFFRICATE},
    "fricative": {Manner.FRICATIVE, Manner.LATERAL_FRICATIVE},
    "nasal": {Manner.NASAL},
    "trill": {Manner.TRILL},
    "tap": {Manner.TAP},
    "flap": {Manner.TAP},
    "approximant": {Manner.APPROXIMANT, Manner.LATERAL_APPROXIMANT},
    "lateral": {Manner.LATERAL_APPROXIMANT, Manner.LATERAL_FRICATIVE},
    "glide": {Manner.APPROXIMANT},
    "obstruent": {Manner.PLOSIVE, Manner.AFFRICATE, Manner.FRICATIVE, Manner.LATERAL_FRICATIVE},
    "sonorant": {
        Manner.NASAL,
        Manner.TRILL,
        Manner.TAP,
        Manner.LATERAL_APPROXIMANT,
        Manner.APPROXIMANT,
    },
    "liquid": {Manner.TRILL, Manner.TAP, Manner.LATERAL_APPROXIMANT},
}
_VOICING = {"voiced": Voicing.VOICED, "voiceless": Voicing.VOICELESS, "unvoiced": Voicing.VOICELESS}
_HEIGHT = {
    "close": Height.CLOSE,
    "high": Height.CLOSE,
    "near-close": Height.NEAR_CLOSE,
    "close-mid": Height.CLOSE_MID,
    "mid": Height.MID,
    "open-mid": Height.OPEN_MID,
    "near-open": Height.NEAR_OPEN,
    "open": Height.OPEN,
    "low": Height.OPEN,
}
_BACKNESS = {b.value: b for b in Backness}
_ROUNDED = {"rounded": True, "unrounded": False}
_LENGTH = {"long": True, "short": False}


@dataclass(frozen=True)
class FeatureClass:
    """A natural class parsed from words like ``voiceless plosive`` or ``long front vowel``.

    A word may also be a *capture*: ``αplace`` (or the ASCII alias ``@place``) does not
    constrain the segment but binds its place to the variable α, so a later ``[αplace]`` in
    the replacement can copy it — the mechanism behind feature-agreement assimilation.
    """

    constraints: tuple  # tuple of (field_name, frozenset_of_allowed_values)
    kind: str | None    # "consonant", "vowel", or None
    captures: tuple = ()  # tuple of (variable, attribute) pairs to bind on a match

    @classmethod
    def parse(cls, text: str) -> "FeatureClass":
        words = text.replace(",", " ").split()
        if not words:
            raise ValueError("empty feature class []")
        constraints: dict[str, set] = {}
        captures: list[tuple[str, str]] = []
        kind: str | None = None

        def add(field: str, value):
            constraints.setdefault(field, set())
            if isinstance(value, set):
                constraints[field] |= value
            else:
                constraints[field].add(value)

        for w in words:
            key = w.lower()
            if key and key[0] in "αβγ@" and key[1:] in _DIMENSIONS:
                var = "α" if key[0] == "@" else key[0]
                captures.append((var, _DIMENSIONS[key[1:]]))
                continue
            if key in _KIND:
                kind = _KIND[key]
            elif key in _VOICING:
                add("voicing", _VOICING[key])
            elif key in _MANNER_WORDS:
                add("manner", _MANNER_WORDS[key])
                kind = kind or "consonant"
            elif key in _PLACE:
                add("place", _PLACE[key])
                kind = kind or "consonant"
            elif key in _HEIGHT:
                add("height", _HEIGHT[key])
                kind = kind or "vowel"
            elif key in _BACKNESS:
                add("backness", _BACKNESS[key])
                kind = kind or "vowel"
            elif key in _ROUNDED:
                add("rounded", _ROUNDED[key])
                kind = kind or "vowel"
            elif key in _LENGTH:
                add("long", _LENGTH[key])
                kind = kind or "vowel"
            else:
                raise ValueError(f"unknown feature word {w!r} in [{text}]")

        frozen = tuple((f, frozenset(vs)) for f, vs in constraints.items())
        if not frozen and not kind and not captures:
            raise ValueError(f"feature class [{text}] constrains nothing")
        return cls(frozen, kind, tuple(captures))

    def matches(self, element: Element) -> bool:
        if not isinstance(element, Segment):
            return False
        if self.kind == "consonant" and not isinstance(element, Consonant):
            return False
        if self.kind == "vowel" and not isinstance(element, Vowel):
            return False
        for field, allowed in self.constraints:
            if getattr(element, field, None) not in allowed:
                return False
        return True

    def bindings(self, element: Element) -> dict:
        """The variable bindings this class captures from *element* (empty if none)."""
        if not isinstance(element, Segment) or not self.captures:
            return {}
        bound = {}
        for var, attr in self.captures:
            value = getattr(element, attr, None)
            if value is not None:
                bound[var] = value
        return bound


# --- Matchers -----------------------------------------------------------------------
class Matcher(Protocol):
    def matches(self, element: Element) -> bool: ...


@dataclass(frozen=True)
class LiteralMatcher:
    """Matches one specific segment by IPA symbol."""

    segment: Segment

    def matches(self, element: Element) -> bool:
        return isinstance(element, Segment) and element.ipa == self.segment.ipa


@dataclass(frozen=True)
class CategoryMatcher:
    """Matches any segment in a named category (e.g. ``V`` = the vowels)."""

    name: str
    members: frozenset  # frozenset of IPA symbols

    def matches(self, element: Element) -> bool:
        return isinstance(element, Segment) and element.ipa in self.members


@dataclass(frozen=True)
class BoundaryMatcher:
    """Matches a word edge (``#``)."""

    def matches(self, element: Element) -> bool:
        return element is BOUNDARY


@dataclass(frozen=True)
class AnyMatcher:
    """Matches any single segment (``.``), but not a word boundary."""

    def matches(self, element: Element) -> bool:
        return isinstance(element, Segment)
