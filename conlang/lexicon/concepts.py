"""The concept inventory: meanings to be given words, grouped by semantic field.

This is the onomasiological backbone — a curated, Swadesh-flavoured list of concepts that
languages reliably have words for. Each concept has a part of speech, a semantic field,
and a ``basicness`` (roughly how universal/frequent it is), which drives word length via
Zipf's law of abbreviation. The list is intentionally a representative core, not
exhaustive; adding a concept is a one-line edit.

Alongside the inventory sit three relational tables the generator uses to make the lexicon
cohere: :data:`COLEXIFICATION` (concepts often sharing one word), :data:`DERIVATIONS`
(words derivable from others), and :data:`COMPOUNDS` (words built from two roots).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    gloss: str
    pos: str       # "noun" | "verb" | "adjective" | "particle"
    field: str
    basicness: float  # 0..1, higher = more basic/frequent


# field -> (part of speech, [(gloss, basicness), ...])
_RAW: dict[str, tuple[str, list[tuple[str, float]]]] = {
    # Deictic/pronominal core (demonstratives treated as nominal for word-class purposes).
    "deixis": ("noun", [("I", 0.95), ("you", 0.90), ("we", 0.85), ("this", 0.85), ("that", 0.80)]),
    # Grammatical particles: negator, yes/no-question marker, relativizer (function words).
    "particle": ("particle", [("not", 0.95), ("Q", 0.85), ("REL", 0.80)]),
    "people": ("noun", [("person", 0.95), ("man", 0.90), ("woman", 0.90), ("child", 0.85)]),
    "body": ("noun", [
        ("head", 0.85), ("eye", 0.90), ("ear", 0.80), ("nose", 0.65), ("mouth", 0.80),
        ("tooth", 0.70), ("tongue", 0.65), ("hair", 0.65), ("hand", 0.85), ("arm", 0.70),
        ("foot", 0.80), ("leg", 0.70), ("face", 0.70), ("heart", 0.70), ("blood", 0.80),
        ("bone", 0.75), ("skin", 0.70),
    ]),
    "kinship": ("noun", [("mother", 0.90), ("father", 0.90), ("name", 0.80)]),
    "nature": ("noun", [
        ("sun", 0.90), ("moon", 0.85), ("star", 0.80), ("sky", 0.75), ("water", 0.95),
        ("fire", 0.90), ("stone", 0.85), ("earth", 0.80), ("mountain", 0.70),
        ("river", 0.75), ("tree", 0.85), ("wind", 0.70), ("rain", 0.75), ("wood", 0.60),
        ("bark", 0.50), ("day", 0.70), ("night", 0.70), ("month", 0.40),
    ]),
    "animals": ("noun", [("dog", 0.80), ("bird", 0.80), ("fish", 0.80), ("snake", 0.60)]),
    "food": ("noun", [
        ("meat", 0.75), ("fruit", 0.60), ("egg", 0.60), ("seed", 0.60), ("firewood", 0.40),
    ]),
    "motion": ("verb", [
        ("go", 0.90), ("come", 0.85), ("walk", 0.80), ("run", 0.75), ("fall", 0.70),
        ("fly", 0.65), ("swim", 0.60),
    ]),
    "cognition": ("verb", [
        ("see", 0.90), ("hear", 0.85), ("know", 0.80), ("think", 0.70), ("say", 0.85),
    ]),
    "action": ("verb", [
        ("eat", 0.90), ("drink", 0.85), ("give", 0.80), ("take", 0.75), ("make", 0.75),
        ("hunt", 0.60), ("cook", 0.60),
    ]),
    "quality": ("adjective", [
        ("big", 0.85), ("small", 0.85), ("long", 0.75), ("good", 0.85), ("bad", 0.80),
        ("hot", 0.75), ("cold", 0.75), ("new", 0.70), ("old", 0.70), ("full", 0.60),
    ]),
    "color": ("adjective", [
        ("red", 0.70), ("white", 0.75), ("black", 0.75), ("green", 0.60), ("blue", 0.50),
        ("yellow", 0.50),
    ]),
    "number": ("noun", [
        ("one", 0.85), ("two", 0.80), ("three", 0.70), ("four", 0.60), ("five", 0.55),
    ]),
}

# Products of derivation/compounding are concepts too; they are declared here so they have
# a field and part of speech, but they receive their forms from the source words.
_DERIVED_CONCEPTS = [
    Concept("hunter", "noun", "people", 0.45),
    Concept("stony", "adjective", "quality", 0.35),
    Concept("puppy", "noun", "animals", 0.35),
]
_COMPOUND_CONCEPTS = [
    Concept("waterfall", "noun", "nature", 0.40),
    Concept("blackbird", "noun", "animals", 0.30),
    Concept("nightbird", "noun", "animals", 0.30),
]


def _build_concepts() -> list[Concept]:
    out: list[Concept] = []
    for field, (pos, items) in _RAW.items():
        for gloss, basicness in items:
            out.append(Concept(gloss, pos, field, basicness))
    out.extend(_DERIVED_CONCEPTS)
    out.extend(_COMPOUND_CONCEPTS)
    return out


CONCEPTS: list[Concept] = _build_concepts()
BY_GLOSS: dict[str, Concept] = {c.gloss: c for c in CONCEPTS}
FIELDS: list[str] = list(_RAW.keys())

# Every concept (including derived/compound products) must live in a known field.
assert all(c.field in FIELDS for c in CONCEPTS), "a concept references an unknown field"


# --- Relational tables --------------------------------------------------------------
# Colexification: (source, target, probability the language merges them under one word).
# The source is the more basic concept; the target reuses its form when merged.
COLEXIFICATION: list[tuple[str, str, float]] = [
    ("tree", "wood", 0.45),
    ("fire", "firewood", 0.50),
    ("skin", "bark", 0.35),
    ("sun", "day", 0.30),
    ("moon", "month", 0.45),
    ("green", "blue", 0.35),
    ("person", "man", 0.25),
    ("water", "river", 0.30),
    # Body-part colexifications — among the most frequent cross-linguistically (CLICS).
    ("hand", "arm", 0.40),
    ("foot", "leg", 0.40),
    ("eye", "face", 0.30),
]

# Derivation: (base, product, relation, from_pos, to_pos). The relation matches a Stage 3
# derivational affix gloss; if the language has that affix the product is derived from the
# base, otherwise the product is coined as a fresh root.
DERIVATIONS: list[tuple[str, str, str, str, str]] = [
    ("hunt", "hunter", "AGENT", "verb", "noun"),
    ("stone", "stony", "HAVING", "noun", "adjective"),
    ("dog", "puppy", "DIMINUTIVE", "noun", "noun"),
]

# Compounding: (product, (part1, part2)). The product's form is the parts' forms joined.
COMPOUNDS: list[tuple[str, tuple[str, str]]] = [
    ("waterfall", ("water", "fall")),
    ("blackbird", ("black", "bird")),
    ("nightbird", ("night", "bird")),
]
