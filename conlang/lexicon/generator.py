"""Build a lexicon for a language: coin, colexify, derive, and compound.

The passes run in order so that later ones can reuse the forms minted by earlier ones:

1. coin a fresh root for every primary concept (length set by Zipf's law of abbreviation),
2. merge colexified concept pairs with their attested probability,
3. derive products from their base words where the language has the matching affix,
4. join compound products from their parts.
"""

from __future__ import annotations

import random

from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import WordGenerator, Romanizer
from conlang.lexicon.concepts import (
    CONCEPTS,
    COLEXIFICATION,
    DERIVATIONS,
    COMPOUNDS,
    BY_GLOSS,
)
from conlang.lexicon.lexicon import Lexicon, LexicalEntry, Etymology


def _max_syllables(basicness: float) -> int:
    """Zipf's law of abbreviation: basic, frequent concepts get shorter words.

    This is a soft *cap* on a length distribution that already peaks at two syllables, not
    a hard floor — very basic words can still be one or two syllables. A hard one-syllable
    floor would crowd the tiny monosyllable space and force spelling collisions.
    """
    if basicness >= 0.70:
        return 2
    if basicness >= 0.45:
        return 3
    return 4


def build_lexicon(
    phonotactics: Phonotactics,
    rng: random.Random | None = None,
    *,
    romanizer: Romanizer | None = None,
    morphology=None,
    head_final: bool = True,
) -> Lexicon:
    """Build a dictionary for the language. ``head_final`` controls compound order: when
    False (a head-initial language) compounds put the head first (bird-black, not
    black-bird)."""
    rng = rng or random.Random()
    romanizer = romanizer or Romanizer()
    gen = WordGenerator(phonotactics, romanizer)
    # Headwords must be distinct in *spelling* as well as IPA: a dictionary with two
    # entries both spelled "he" is unusable, even if their IPA differs.
    used_ipa: set[str] = set()
    used_roman: set[str] = set()

    def coin(basicness: float):
        base_cap = _max_syllables(basicness)
        word = gen.word(rng, min_syllables=1, max_syllables=base_cap)
        for attempt in range(120):
            if word.ipa not in used_ipa and word.roman not in used_roman:
                break
            # Grow the length ceiling if the short space is exhausted, so we don't loop
            # forever or silently emit a duplicate headword.
            cap = base_cap + attempt // 40
            word = gen.word(rng, min_syllables=1, max_syllables=cap)
        used_ipa.add(word.ipa)
        used_roman.add(word.roman)
        form = tuple(s for syl in word.syllables for s in syl)
        return form, word.roman

    product_glosses = {prod for _, prod, *_ in DERIVATIONS} | {prod for prod, _ in COMPOUNDS}
    lex = Lexicon()

    def cls(pos: str) -> str:
        return _assign_class(rng, pos, morphology)

    # 1. Primary roots.
    for concept in CONCEPTS:
        if concept.gloss in product_glosses:
            continue
        form, roman = coin(concept.basicness)
        lex.entries[concept.gloss] = LexicalEntry(
            concept, form, roman, Etymology.ROOT, inflection_class=cls(concept.pos)
        )

    # 2. Colexification.
    for source, target, prob in COLEXIFICATION:
        src_entry = lex.entries.get(source)
        tgt_entry = lex.entries.get(target)
        if src_entry and tgt_entry and rng.random() < prob:
            lex.entries[target] = LexicalEntry(
                tgt_entry.concept, src_entry.form, src_entry.roman,
                Etymology.COLEXIFIED, note=f"= {source}",
                inflection_class=src_entry.inflection_class,  # shares form -> shares class
            )

    # 2b. Kinship system: roll how this language partitions kin terms.
    _apply_kinship(lex, rng)

    # 3. Derivation (falls back to a fresh root if the language lacks the affix).
    for base, prod, relation, from_pos, to_pos in DERIVATIONS:
        concept = BY_GLOSS[prod]
        base_entry = lex.entries.get(base)
        rule = _find_rule(morphology, relation, from_pos, to_pos)
        if base_entry is not None and rule is not None:
            form = tuple(rule.apply(base_entry.form))
            roman = romanizer.romanize([list(form)])
            # Derived/compound words may legitimately be homophonous with a root, so we
            # record their forms but do not force them to be unique.
            used_ipa.add("".join(s.ipa for s in form))
            used_roman.add(roman)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.DERIVED,
                note=f"from {base} ({relation})", inflection_class=cls(concept.pos),
            )
        else:
            form, roman = coin(concept.basicness)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.ROOT, inflection_class=cls(concept.pos)
            )

    # 4. Compounding (falls back to a fresh root if a part is missing).
    # COMPOUNDS list parts as (modifier, head). A head-final language keeps that order;
    # a head-initial one puts the head first, matching its phrasal head-directionality.
    for prod, parts in COMPOUNDS:
        concept = BY_GLOSS[prod]
        ordered = parts if head_final else (parts[1], parts[0])
        part_entries = [lex.entries.get(p) for p in ordered]
        if all(part_entries):
            form = tuple(seg for pe in part_entries for seg in pe.form)
            roman = romanizer.romanize([list(form)])
            used_ipa.add("".join(s.ipa for s in form))
            used_roman.add(roman)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.COMPOUND,
                note="+".join(ordered), inflection_class=cls(concept.pos),
            )
        else:
            form, roman = coin(concept.basicness)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.ROOT, inflection_class=cls(concept.pos)
            )

    return lex


def _merge_kin(lex: Lexicon, source: str, target: str, note: str) -> None:
    """Make *target* share *source*'s word (a colexification recording the kin merge)."""
    src = lex.entries.get(source)
    tgt = lex.entries.get(target)
    if src is None or tgt is None:
        return
    lex.entries[target] = LexicalEntry(
        tgt.concept, src.form, src.roman, Etymology.COLEXIFIED,
        note=note, inflection_class=src.inflection_class,
    )


def _apply_kinship(lex: Lexicon, rng: random.Random) -> None:
    """Roll a small kinship typology and apply it as kin-term colexifications.

    Two independent, attested axes:

    - **Sibling classification.** Most languages distinguish siblings by sex (brother vs.
      sister); a sizeable minority use a single sex-neutral 'sibling' term.
    - **Parent's-sibling merging.** In *classificatory* systems (Hawaiian/Iroquois) a
      parent's sibling shares the parental term (uncle = father, aunt = mother); *descriptive*
      systems keep them distinct. Simplification: with only generic uncle/aunt cover terms
      (no paternal/maternal or cross/parallel distinction), this is a generational-style
      approximation rather than strict bifurcate-merging (which keeps the cross-sibling apart).

    The two axes roll independently (a language can be classificatory yet sex-distinguishing,
    etc.). son/daughter are deliberately always distinct — no descendant-merging axis is
    modelled. Always consumes exactly two rng draws, so the choice is stable for a seed
    whether or not the kin terms exist.
    """
    sex_neutral_siblings = rng.random() < 0.30
    classificatory = rng.random() < 0.35
    if sex_neutral_siblings:  # one word for sibling regardless of sex
        _merge_kin(lex, "brother", "sister", note="= brother (sex-neutral sibling)")
    if classificatory:  # parent's sibling = parent
        _merge_kin(lex, "father", "uncle", note="= father (classificatory)")
        _merge_kin(lex, "mother", "aunt", note="= mother (classificatory)")


def _assign_class(rng: random.Random, pos: str, morphology) -> str:
    """Assign a word to one of its part-of-speech's inflection classes (uniform random).

    Consumes no RNG when there is only one class, so languages without declensions keep
    their previous lexicon output.
    """
    if morphology is None:
        return "1"
    classes = morphology.inflection_classes(pos)
    return rng.choice(classes) if len(classes) > 1 else "1"


def _find_rule(morphology, relation: str, from_pos: str, to_pos: str):
    if morphology is None:
        return None
    for rule in morphology.derivations:
        if rule.gloss == relation and rule.from_class == from_pos and rule.to_class == to_pos:
            return rule
    return None
