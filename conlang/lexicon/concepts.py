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
    "deixis": ("noun", [
        ("I", 0.95), ("you", 0.90), ("we", 0.85), ("this", 0.85), ("that", 0.80),
        ("who", 0.85), ("what", 0.85),  # interrogative pronouns (content questions)
    ]),
    # Grammatical particles: negator, yes/no-question marker, relativizer, coordinators.
    "particle": ("particle", [
        ("not", 0.95), ("Q", 0.85), ("REL", 0.80), ("and", 0.95), ("or", 0.75),
        # A dummy/periphrastic auxiliary ("do"/"be") for subject–auxiliary-inversion questions.
        ("AUX", 0.70),
    ]),
    "people": ("noun", [("person", 0.95), ("man", 0.90), ("woman", 0.90), ("child", 0.85)]),
    "body": ("noun", [
        ("head", 0.85), ("eye", 0.90), ("ear", 0.80), ("nose", 0.65), ("mouth", 0.80),
        ("tooth", 0.70), ("tongue", 0.65), ("hair", 0.65), ("hand", 0.85), ("arm", 0.70),
        ("foot", 0.80), ("leg", 0.70), ("face", 0.70), ("heart", 0.70), ("blood", 0.80),
        ("bone", 0.75), ("skin", 0.70), ("back", 0.70), ("belly", 0.65), ("neck", 0.60),
        ("liver", 0.50),
    ]),
    "kinship": ("noun", [
        ("mother", 0.90), ("father", 0.90), ("name", 0.80),
        ("brother", 0.80), ("sister", 0.80), ("son", 0.82), ("daughter", 0.82),
        ("uncle", 0.55), ("aunt", 0.55),
    ]),
    "nature": ("noun", [
        ("sun", 0.90), ("moon", 0.85), ("star", 0.80), ("sky", 0.75), ("water", 0.95),
        ("fire", 0.90), ("stone", 0.85), ("earth", 0.80), ("mountain", 0.70),
        ("river", 0.75), ("tree", 0.85), ("wind", 0.70), ("rain", 0.75), ("wood", 0.60),
        ("bark", 0.50), ("day", 0.70), ("night", 0.70), ("month", 0.40),
        ("sea", 0.65), ("cloud", 0.60), ("road", 0.65), ("hole", 0.55), ("forest", 0.60),
        ("leaf", 0.65), ("smoke", 0.50),
    ]),
    "animals": ("noun", [("dog", 0.80), ("bird", 0.80), ("fish", 0.80), ("snake", 0.60)]),
    "food": ("noun", [
        ("meat", 0.75), ("fruit", 0.60), ("egg", 0.60), ("seed", 0.60), ("firewood", 0.40),
    ]),
    # Cultural/social vocabulary — central to a community and, cross-linguistically, the most
    # readily *borrowed* stratum (so a natural target for a future loanword feature).
    "society": ("noun", [
        ("chief", 0.55), ("spirit", 0.55), ("war", 0.55), ("gift", 0.50), ("story", 0.55),
        ("work", 0.55), ("word", 0.65), ("god", 0.50),
    ]),
    # Made things — also frequently borrowed along with the artifact itself.
    "artifact": ("noun", [
        ("house", 0.80), ("door", 0.60), ("knife", 0.60), ("rope", 0.55), ("boat", 0.55),
        ("field", 0.55),
    ]),
    # Abstractions — fertile ground for polysemy and metaphor (way/road, time/day, word/speech).
    "abstract": ("noun", [
        ("time", 0.65), ("thing", 0.70), ("place", 0.55), ("dream", 0.55), ("language", 0.55),
    ]),
    "motion": ("verb", [
        ("go", 0.90), ("come", 0.85), ("walk", 0.80), ("run", 0.75), ("fall", 0.70),
        ("fly", 0.65), ("swim", 0.60), ("turn", 0.60), ("throw", 0.60),
    ]),
    "cognition": ("verb", [
        ("see", 0.90), ("hear", 0.85), ("know", 0.80), ("think", 0.70), ("say", 0.85),
        ("speak", 0.78), ("want", 0.70),
    ]),
    "existence": ("verb", [("die", 0.80), ("live", 0.75), ("sleep", 0.75)]),
    "action": ("verb", [
        ("eat", 0.90), ("drink", 0.85), ("give", 0.80), ("take", 0.75), ("make", 0.75),
        ("hunt", 0.60), ("cook", 0.60), ("hold", 0.70), ("cut", 0.70), ("burn", 0.65),
        ("kill", 0.65), ("grow", 0.60),
    ]),
    "quality": ("adjective", [
        ("big", 0.85), ("small", 0.85), ("long", 0.75), ("good", 0.85), ("bad", 0.80),
        ("hot", 0.75), ("cold", 0.75), ("new", 0.70), ("old", 0.70), ("full", 0.60),
        ("wide", 0.55), ("narrow", 0.50), ("short", 0.65), ("heavy", 0.55), ("dark", 0.55),
        ("dry", 0.55), ("sharp", 0.50), ("many", 0.70),
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
    Concept("speaker", "noun", "people", 0.40),  # AGENT of "speak"
    Concept("stony", "adjective", "quality", 0.35),
    Concept("puppy", "noun", "animals", 0.35),
    # A *stacked* derivation: petrify = stony + BECOME = (stone + HAVING) + BECOME, so the
    # word is built by two derivational steps when the language has both affixes.
    Concept("petrify", "verb", "action", 0.25),
]
_COMPOUND_CONCEPTS = [
    Concept("waterfall", "noun", "nature", 0.40),
    Concept("blackbird", "noun", "animals", 0.30),
    Concept("nightbird", "noun", "animals", 0.30),
    Concept("seabird", "noun", "animals", 0.30),
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

# A *conditional* pronoun, not part of the universal core: a separate inclusive 'we', minted
# only for a language that grammatically distinguishes clusivity (so it is kept out of CONCEPTS
# and coined on demand by the generator). The default 'we' then serves as the exclusive form.
WE_INCLUSIVE: Concept = Concept("we (incl)", "noun", "deixis", 0.80)


def _check_derivation_order() -> None:
    """A stacked derivation (whose base is itself a derivation product) must be listed after
    that base, so the base exists when the single-pass derivation builder reaches it."""
    seen_at = {prod: i for i, (_b, prod, *_r) in enumerate(DERIVATIONS)}
    for i, (base, prod, *_rest) in enumerate(DERIVATIONS):
        if base in seen_at:
            assert seen_at[base] < i, (
                f"derivation {prod!r} stacks on {base!r}, which must be listed earlier"
            )


# --- Relational tables --------------------------------------------------------------
# Colexification: (source, target, probability the language merges them under one word).
# The source is the more basic concept; the target reuses its form when merged. Entries are
# applied IN ORDER and rewrite the target's form, so a *chain* (A->B then B->C) yields a
# polysemy chain: if both links fire, one word covers A, B and C (sun = day = time). List a
# chained link AFTER the link that feeds it.
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
    # Polysemy hubs and metaphor (one source, sometimes several senses):
    ("tongue", "language", 0.50),  # tongue = language (near-universal)
    ("mouth", "door", 0.25),       # mouth = opening / doorway
    ("tree", "forest", 0.30),      # tree also -> forest (a second sense beside tree->wood)
    ("water", "sea", 0.25),        # water also -> sea (beside water->river)
    ("wind", "spirit", 0.30),      # breath / wind = spirit (Latin spiritus, Greek pneuma)
    ("word", "story", 0.30),       # word = speech / account
    # A polysemy CHAIN, applied after sun->day above: if both fire, one word = sun = day = time.
    ("day", "time", 0.35),
]

# Derivation: (base, product, relation, from_pos, to_pos). The relation matches a Stage 3
# derivational affix gloss; if the language has that affix the product is derived from the
# base, otherwise the product is coined as a fresh root.
DERIVATIONS: list[tuple[str, str, str, str, str]] = [
    ("hunt", "hunter", "AGENT", "verb", "noun"),
    ("speak", "speaker", "AGENT", "verb", "noun"),
    ("stone", "stony", "HAVING", "noun", "adjective"),
    ("dog", "puppy", "DIMINUTIVE", "noun", "noun"),
    # Antonyms by an opposite-forming affix: the marked pole of a polar pair derived from
    # the unmarked one, so the pair shares morphology (when the language lacks the affix
    # each stays an independent, suppletive root).
    ("good", "bad", "ANTONYM", "adjective", "adjective"),
    ("big", "small", "ANTONYM", "adjective", "adjective"),
    ("hot", "cold", "ANTONYM", "adjective", "adjective"),
    ("long", "short", "ANTONYM", "adjective", "adjective"),
    ("wide", "narrow", "ANTONYM", "adjective", "adjective"),
    # Derivation stacking: petrify is derived from *stony* (itself derived from stone), so a
    # language with both HAVING and BECOME builds it from two derivational affixes. Must come
    # after stone->stony so its base exists when this is processed.
    ("stony", "petrify", "BECOME", "adjective", "verb"),
]

_check_derivation_order()  # stacked derivations must follow their base's row

# Diachronic semantic shift: attested directional drifts of a word's meaning, applied during
# evolution (alongside sound change). (from_gloss, to_gloss, kind, base_probability) — the word
# for `from` may come to mean `to` in the daughter language. Both glosses are inventory concepts.
# These are classic shift *pathways* (perception->cognition, body->leader, euphemism, ...), not
# synchronic colexification; a sound change that makes a word homophonous boosts its drift.
SEMANTIC_SHIFTS: list[tuple[str, str, str, float]] = [
    ("see", "know", "perception -> cognition", 0.30),   # 'I see' = 'I understand'
    ("hear", "know", "perception -> cognition", 0.20),
    ("head", "chief", "metaphor: body -> leader", 0.25),
    ("hand", "five", "metonymy: hand -> numeral", 0.20),
    ("die", "sleep", "euphemism", 0.20),
    ("child", "fruit", "offspring -> produce", 0.15),
    ("spirit", "god", "breath -> deity", 0.20),
    # mouth->door also appears in COLEXIFICATION — deliberately: synchronic polysemy (one word,
    # two senses, in a single language) and diachronic drift (the meaning moving over time) are
    # distinct phenomena that the same metaphor can drive, not a duplicated entry.
    ("mouth", "door", "metaphor: opening", 0.15),
    ("eye", "seed", "metaphor: round thing", 0.15),
]

# Compounding: (product, (part1, part2)). The product's form is the parts' forms joined.
COMPOUNDS: list[tuple[str, tuple[str, str]]] = [
    ("waterfall", ("water", "fall")),
    ("blackbird", ("black", "bird")),
    ("nightbird", ("night", "bird")),
    ("seabird", ("sea", "bird")),
]
