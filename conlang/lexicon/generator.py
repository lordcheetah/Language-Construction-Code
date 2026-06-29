"""Build a lexicon for a language: coin, colexify, derive, and compound.

The passes run in order so that later ones can reuse the forms minted by earlier ones:

1. coin a fresh root for every primary concept (length set by Zipf's law of abbreviation),
2. merge colexified concept pairs with their attested probability,
3. derive products from their base words where the language has the matching affix,
4. join compound products from their parts.
"""

from __future__ import annotations

import random
from dataclasses import replace

from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import WordGenerator, Romanizer
from conlang.lexicon.concepts import (
    CONCEPTS,
    COLEXIFICATION,
    DERIVATIONS,
    COMPOUNDS,
    BY_GLOSS,
    WE_INCLUSIVE,
)
from conlang.lexicon.lexicon import Lexicon, LexicalEntry, Etymology


# A language has a borrowed stratum with this probability; within it, each culturally-borrowable
# concept is a loanword with probability _BORROW_PROB. The borrowable fields are the cultural and
# abstract vocabulary that is, cross-linguistically, the most readily borrowed.
_LOAN_STRATUM_PROB = 0.40
_BORROW_PROB = 0.50
_BORROWABLE_FIELDS = frozenset({"society", "artifact", "abstract"})


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
    # A loanword stratum: many languages borrow a layer of cultural/abstract vocabulary from a
    # donor language. The donor shares this language's *phonemes* (so the loans stay writable in
    # its script) but has its own rolled *phonotactics*, giving loans a recognizably foreign
    # syllable shape (clusters/codas the native words may lack). Loans are coined from this donor
    # word generator. No loanword phonological *adaptation* is modelled — loans keep their
    # foreign shape rather than being nativised.
    loan_gen = None
    if rng.random() < _LOAN_STRATUM_PROB:
        # prefer_complex skews the donor toward clustered/coda-heavy syllables, so loans read
        # as foreign (more complex than a typical native profile), never simpler.
        donor_phonotactics = Phonotactics.random(phonotactics.inventory, rng, prefer_complex=True)
        loan_gen = WordGenerator(donor_phonotactics, romanizer)
    # Headwords must be distinct in *spelling* as well as IPA: a dictionary with two
    # entries both spelled "he" is unusable, even if their IPA differs.
    used_ipa: set[str] = set()
    used_roman: set[str] = set()

    def coin(basicness: float, generator: WordGenerator | None = None):
        g = generator or gen
        base_cap = _max_syllables(basicness)
        word = g.word(rng, min_syllables=1, max_syllables=base_cap)
        for attempt in range(120):
            if word.ipa not in used_ipa and word.roman not in used_roman:
                break
            # Grow the length ceiling if the short space is exhausted, so we don't loop
            # forever or silently emit a duplicate headword.
            cap = base_cap + attempt // 40
            word = g.word(rng, min_syllables=1, max_syllables=cap)
        used_ipa.add(word.ipa)
        used_roman.add(word.roman)
        form = tuple(s for syl in word.syllables for s in syl)
        return form, word.roman

    product_glosses = {prod for _, prod, *_ in DERIVATIONS} | {prod for prod, _ in COMPOUNDS}
    lex = Lexicon()

    # A per-language gender -> declension map, so which gender uses which class varies by
    # language (rather than masc always landing in the first declension).
    gender_classes = _gender_class_map(rng, morphology)

    def feat(pos: str) -> tuple[str, str | None]:
        return _assign_gender_and_class(rng, pos, morphology, gender_classes)

    # 1. Primary roots (a borrowable cultural/abstract concept may instead be a loanword,
    # coined from the donor generator and marked as such).
    for concept in CONCEPTS:
        if concept.gloss in product_glosses:
            continue
        borrowed = (loan_gen is not None and concept.field in _BORROWABLE_FIELDS
                    and rng.random() < _BORROW_PROB)
        form, roman = coin(concept.basicness, loan_gen if borrowed else None)
        klass, gender = feat(concept.pos)
        lex.entries[concept.gloss] = LexicalEntry(
            concept, form, roman,
            Etymology.LOANWORD if borrowed else Etymology.ROOT,
            note="borrowed" if borrowed else "",
            inflection_class=klass, gender=gender,
        )

    # 1b. A clusivity-marking language has a separate inclusive 'we' pronoun, distinct from the
    # default 'we' (which then serves as the exclusive). Coin a fresh root for it. Languages
    # without verb clusivity skip this and consume no RNG, so their lexicon output is unchanged.
    if _marks_clusivity(morphology) and "we" in lex.entries:
        form, roman = coin(WE_INCLUSIVE.basicness)
        klass, gender = feat(WE_INCLUSIVE.pos)
        lex.entries[WE_INCLUSIVE.gloss] = LexicalEntry(
            WE_INCLUSIVE, form, roman, Etymology.ROOT, inflection_class=klass, gender=gender,
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
                gender=src_entry.gender,                       # ...and gender
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
            klass, gender = feat(concept.pos)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.DERIVED,
                note=f"from {base} ({relation})", inflection_class=klass, gender=gender,
            )
        else:
            form, roman = coin(concept.basicness)
            klass, gender = feat(concept.pos)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.ROOT, inflection_class=klass, gender=gender
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
            klass, gender = feat(concept.pos)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.COMPOUND,
                note="+".join(ordered), inflection_class=klass, gender=gender,
            )
        else:
            form, roman = coin(concept.basicness)
            klass, gender = feat(concept.pos)
            lex.entries[prod] = LexicalEntry(
                concept, form, roman, Etymology.ROOT, inflection_class=klass, gender=gender
            )

    # 5. Suppletion: a few frequent words get a wholly irregular form in one inflectional cell
    # (go/went, person/people), coined as a fresh unrelated root. Rolled per language and only
    # where the language actually marks that category+value, so there's a cell to suppletivize.
    for gloss, wc, cat, val, prob in SUPPLETIONS:
        entry = lex.entries.get(gloss)
        if entry is None or not _marks_value(morphology, wc, cat, val):
            continue
        if rng.random() < prob:
            form, _roman = coin(BY_GLOSS[gloss].basicness)
            lex.entries[gloss] = replace(
                entry, suppletive_stems=(*entry.suppletive_stems, ((cat, val), form))
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
        note=note, inflection_class=src.inflection_class, gender=src.gender,
    )


# Lexical gender values and their rough prevalence (masculine the default/most common).
_GENDERS = ("masc", "fem", "neut")
_GENDER_WEIGHTS = (0.45, 0.35, 0.20)


# High-frequency concepts that cross-linguistically tend to have an irregular (suppletive)
# form in one inflectional cell, with the per-language probability of that suppletion:
# (concept gloss, word class, category, value, probability). Applied only where the language
# marks that category+value, so there is a real cell to fill with the irregular stem.
SUPPLETIONS = (
    ("go", "verb", "tense", "past", 0.40),     # go / went
    ("person", "noun", "number", "pl", 0.35),  # person / people
    # The marked core-case form of the 1sg pronoun: the accusative "me" (object) under
    # nominative-accusative alignment, or the marked agent form under ergative alignment —
    # the engine reuses one marked case ("acc") for whichever role its alignment singles out.
    ("I", "noun", "case", "acc", 0.30),
)


def _marks_value(morphology, word_class: str, category: str, value: str) -> bool:
    """True if this language's *word_class* paradigm marks *category* and that category has
    *value* — i.e. there is an inflectional cell a suppletive form could fill."""
    if morphology is None:
        return False
    par = morphology.paradigms.get(word_class)
    if par is None:
        return False
    cat = next((c for c in par.marked if c.name == category), None)
    return cat is not None and value in cat.values


def _marks_clusivity(morphology) -> bool:
    """True if this language's verb marks clusivity (1st-person inclusive vs exclusive),
    which is what licenses a separate inclusive 'we' pronoun in the lexicon.

    Simplification: pronoun clusivity is keyed off VERB agreement here. Typologically the
    pronoun is the more basic locus — many languages have inclusive/exclusive pronouns with no
    verb agreement at all (Mandarin 咱们/我们, Tok Pisin yumi/mipela) — so this proxy under-
    generates the split and never yields the common "pronoun clusivity, no verb agreement"
    type. It is the simple option while clusivity is modelled only as a verb category; if
    clusivity ever becomes an independent pronoun-system parameter, revisit this."""
    if morphology is None:
        return False
    verb = morphology.paradigms.get("verb")
    return verb is not None and any(c.name == "clusivity" for c in verb.marked)


def _roll_gender(rng: random.Random, pos: str, morphology) -> str | None:
    """A noun's lexical gender, if this language marks gender on nouns; else None."""
    if pos != "noun" or morphology is None:
        return None
    noun = morphology.paradigms.get("noun")
    if noun is None or not any(c.name == "gender" for c in noun.marked):
        return None
    return rng.choices(_GENDERS, weights=_GENDER_WEIGHTS, k=1)[0]


def _gender_class_map(rng: random.Random, morphology) -> dict[str, str] | None:
    """A per-language gender -> inflection-class map (declensions track gender), or None when
    gender isn't marked or there is only one noun class. The genders are shuffled and dealt
    round-robin onto the classes, so the partition (which genders share a declension) varies
    by language while still using every class."""
    if morphology is None:
        return None
    noun = morphology.paradigms.get("noun")
    if noun is None or not any(c.name == "gender" for c in noun.marked):
        return None
    classes = morphology.inflection_classes("noun")
    if len(classes) <= 1:
        return None
    genders = list(_GENDERS)
    rng.shuffle(genders)
    return {g: classes[i % len(classes)] for i, g in enumerate(genders)}


def _assign_gender_and_class(
    rng: random.Random, pos: str, morphology, gender_classes: dict[str, str] | None
) -> tuple[str, str | None]:
    """Assign a word's (inflection class, gender). In a gender-marking language with several
    declensions the noun's inflection class is *determined by* its gender via the per-language
    ``gender_classes`` map (declensions track gender, as in Latin or German); otherwise the
    class is random as before. Masculine is the base gender, so a masc noun takes no overt
    gender affix."""
    gender = _roll_gender(rng, pos, morphology)
    if gender is not None and gender_classes is not None:
        return gender_classes[gender], gender
    return _assign_class(rng, pos, morphology), gender


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
