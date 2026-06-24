"""Tests for the lexicon stage.

These check that the dictionary is built coherently: every concept gets a word, the three
non-coining mechanisms (colexification, derivation, compounding) actually reuse existing
forms, and word length tracks basicness (Zipf's law of abbreviation).
"""

import random

from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import Romanizer
from conlang.morphology.features import WORD_CLASSES, FeatureBundle, Typology
from conlang.morphology.affix import Affix, Position
from conlang.morphology.generator import MorphologySystem
from conlang.morphology.paradigm import DerivationRule
from conlang.phonology import data
from conlang.lexicon.concepts import (
    CONCEPTS,
    BY_GLOSS,
    FIELDS,
    COLEXIFICATION,
    DERIVATIONS,
    COMPOUNDS,
)
from conlang.lexicon.lexicon import Lexicon, Etymology
from conlang.lexicon.generator import build_lexicon, _max_syllables


def _phonotactics(seed: int):
    rng = random.Random(seed)
    inv = Inventory.random(rng)
    return Phonotactics.random(inv, rng), rng


# --- Concept data integrity ---------------------------------------------------------
def test_concept_glosses_unique():
    glosses = [c.gloss for c in CONCEPTS]
    assert len(glosses) == len(set(glosses))


def test_concept_basicness_in_range_and_fields_known():
    fields = set(FIELDS)
    for c in CONCEPTS:
        assert 0.0 <= c.basicness <= 1.0
        assert c.field in fields
        assert c.pos in ("noun", "verb", "adjective", "particle")


def test_relational_tables_reference_known_concepts():
    for src, tgt, _ in COLEXIFICATION:
        assert src in BY_GLOSS and tgt in BY_GLOSS
    for base, prod, _, _, _ in DERIVATIONS:
        assert base in BY_GLOSS and prod in BY_GLOSS
    for prod, parts in COMPOUNDS:
        assert prod in BY_GLOSS
        assert all(p in BY_GLOSS for p in parts)


# --- Build basics -------------------------------------------------------------------
def test_entries_are_assigned_valid_inflection_classes():
    # Each word's declension must be one the morphology actually defines for its POS.
    phono, rng = _phonotactics(9)
    romanizer = Romanizer()
    from conlang.morphology.generator import random_system

    system = random_system(phono, rng, romanizer=romanizer, typology=Typology.FUSIONAL)
    lex = build_lexicon(phono, rng, romanizer=romanizer, morphology=system)
    for entry in lex.entries.values():
        valid = system.inflection_classes(entry.concept.pos)
        assert entry.inflection_class in valid
    # at least one POS has multiple classes here, so some non-"1" assignment should occur
    assert any(e.inflection_class != "1" for e in lex.entries.values())


def test_every_concept_gets_a_nonempty_entry():
    phono, rng = _phonotactics(3)
    lex = build_lexicon(phono, rng)
    assert len(lex) == len(CONCEPTS)
    for c in CONCEPTS:
        entry = lex.get(c.gloss)
        assert entry is not None and len(entry.form) >= 1 and entry.roman


def test_build_is_reproducible():
    def run():
        phono, rng = _phonotactics(8)
        lex = build_lexicon(phono, rng)
        return [(g, e.ipa, e.etymology) for g, e in lex.entries.items()]

    assert run() == run()


# --- Colexification -----------------------------------------------------------------
def test_colexified_entries_share_their_source_form():
    # Search seeds until at least one colexification fires, then verify the shared form.
    for seed in range(40):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        colex = lex.of_etymology(Etymology.COLEXIFIED)
        for entry in colex:
            source_gloss = entry.note.lstrip("= ").strip()
            assert lex.get(source_gloss).ipa == entry.ipa
        if colex:
            return
    raise AssertionError("no colexification fired across 40 seeds (unexpected)")


# --- Kinship system -----------------------------------------------------------------
def test_classificatory_kinship_merges_uncle_with_father():
    # Search for a seed whose rolled kinship system is classificatory (uncle = father).
    for seed in range(40):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        uncle = lex.get("uncle")
        if uncle.etymology is Etymology.COLEXIFIED and "father" in uncle.note:
            assert uncle.form == lex.get("father").form  # shares the parental word
            assert lex.get("aunt").form == lex.get("mother").form  # aunt = mother too
            return
    raise AssertionError("no classificatory kinship system in 40 seeds (unexpected)")


def test_sex_neutral_siblings_share_one_word():
    for seed in range(40):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        sister = lex.get("sister")
        if sister.etymology is Etymology.COLEXIFIED and "brother" in sister.note:
            assert sister.form == lex.get("brother").form
            return
    raise AssertionError("no sex-neutral sibling system in 40 seeds (unexpected)")


def test_kinship_axes_are_independent():
    # The two axes roll independently: there exist languages that are classificatory but
    # still sex-distinguishing, and languages with sex-neutral siblings but descriptive
    # (distinct) uncle/aunt. A coupling of the two draws would make one of these impossible.
    classif_and_sexed = sexneutral_and_descriptive = False
    for seed in range(60):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        uncle_merged = lex.get("uncle").etymology is Etymology.COLEXIFIED
        sister_merged = lex.get("sister").etymology is Etymology.COLEXIFIED
        if uncle_merged and not sister_merged:
            classif_and_sexed = True
        if sister_merged and not uncle_merged:
            sexneutral_and_descriptive = True
    assert classif_and_sexed and sexneutral_and_descriptive


def test_son_and_daughter_are_always_distinct_roots():
    # No descendant-merging axis is modelled, so these stay independent in every language.
    for seed in range(20):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        assert lex.get("son").etymology is Etymology.ROOT
        assert lex.get("daughter").etymology is Etymology.ROOT
        assert lex.get("son").form != lex.get("daughter").form


def test_descriptive_kinship_keeps_kin_terms_distinct():
    # A seed whose kinship system merges nothing: uncle/aunt/sister are independent roots.
    for seed in range(40):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        if all(lex.get(g).etymology is Etymology.ROOT for g in ("uncle", "aunt", "sister")):
            assert lex.get("uncle").form != lex.get("father").form
            assert lex.get("sister").form != lex.get("brother").form
            return
    raise AssertionError("no fully descriptive kinship system in 40 seeds (unexpected)")


# --- Derivation ---------------------------------------------------------------------
def _system_with_agent_affix() -> MorphologySystem:
    affix = Affix((data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(), "AGENT")
    rule = DerivationRule(affix, from_class="verb", to_class="noun", gloss="AGENT")
    return MorphologySystem(Typology.AGGLUTINATIVE, {}, [rule])


def test_derivation_uses_the_base_word_when_affix_exists():
    phono, rng = _phonotactics(5)
    system = _system_with_agent_affix()
    lex = build_lexicon(phono, rng, morphology=system)
    hunter = lex.get("hunter")
    hunt = lex.get("hunt")
    assert hunter.etymology is Etymology.DERIVED
    # hunter = hunt + agent suffix /a/
    assert hunter.ipa == hunt.ipa + "a"
    assert "hunt" in hunter.note


def test_derivation_falls_back_to_root_without_affix():
    phono, rng = _phonotactics(5)
    lex = build_lexicon(phono, rng, morphology=None)  # no derivational affixes
    assert lex.get("hunter").etymology is Etymology.ROOT


def test_antonym_derivation_relates_polar_pairs():
    # When the language has an opposite-forming affix, the marked pole of an antonym pair is
    # derived from the unmarked one (so the pair shares morphology): bad = opposite-of-good.
    anton = DerivationRule(
        Affix((data.consonant("m"),), Position.PREFIX, FeatureBundle.of(), "ANTONYM"),
        from_class="adjective", to_class="adjective", gloss="ANTONYM",
    )
    system = MorphologySystem(Typology.AGGLUTINATIVE, {}, [anton])
    phono, rng = _phonotactics(5)
    lex = build_lexicon(phono, rng, morphology=system)
    for marked, base in (("bad", "good"), ("small", "big"), ("cold", "hot")):
        entry = lex.get(marked)
        assert entry.etymology is Etymology.DERIVED
        assert entry.ipa == "m" + lex.get(base).ipa   # prefix opposite-marker
        assert base in entry.note


def test_antonyms_are_independent_roots_without_the_affix():
    phono, rng = _phonotactics(5)
    lex = build_lexicon(phono, rng, morphology=None)  # no derivational affixes
    bad, good = lex.get("bad"), lex.get("good")
    assert bad.etymology is Etymology.ROOT
    assert bad.ipa != good.ipa  # an unrelated, suppletive antonym


def test_antonym_affix_is_reachable_from_a_generated_morphology():
    # The opposite-forming affix is in the template pool, so some seed rolls it organically;
    # the dictionary then shows the antonym derived end-to-end (not just with injected rules).
    from conlang.morphology.generator import random_system

    found = False
    for seed in range(40):
        phono, rng = _phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        if not any(r.gloss == "ANTONYM" for r in system.derivations):
            continue
        lex = build_lexicon(phono, random.Random(seed), morphology=system)
        if lex.get("bad").etymology is Etymology.DERIVED:
            assert "good" in lex.get("bad").note
            found = True
            break
    assert found, "no seed in range organically rolled the antonym derivation"


def test_having_and_diminutive_derivation_paths():
    having = DerivationRule(
        Affix((data.consonant("l"),), Position.SUFFIX, FeatureBundle.of(), "HAVING"),
        from_class="noun", to_class="adjective", gloss="HAVING",
    )
    dim = DerivationRule(
        Affix((data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(), "DIMINUTIVE"),
        from_class="noun", to_class="noun", gloss="DIMINUTIVE",
    )
    system = MorphologySystem(Typology.AGGLUTINATIVE, {}, [having, dim])
    phono, rng = _phonotactics(6)
    lex = build_lexicon(phono, rng, morphology=system)
    assert lex.get("stony").etymology is Etymology.DERIVED
    assert lex.get("stony").ipa == lex.get("stone").ipa + "l"
    assert lex.get("puppy").etymology is Etymology.DERIVED
    assert lex.get("puppy").ipa == lex.get("dog").ipa + "i"


# --- Compounding --------------------------------------------------------------------
def test_compound_is_the_concatenation_of_its_parts():
    phono, rng = _phonotactics(7)
    lex = build_lexicon(phono, rng)  # head_final default -> modifier+head
    waterfall = lex.get("waterfall")
    assert waterfall.etymology is Etymology.COMPOUND
    assert waterfall.ipa == lex.get("water").ipa + lex.get("fall").ipa
    assert waterfall.note == "water+fall"


def test_compound_order_follows_head_directionality():
    phono, rng = _phonotactics(7)
    lex = build_lexicon(phono, rng, head_final=False)  # head-initial -> head+modifier
    waterfall = lex.get("waterfall")
    assert waterfall.ipa == lex.get("fall").ipa + lex.get("water").ipa
    assert waterfall.note == "fall+water"


# --- Zipf's law of abbreviation -----------------------------------------------------
def test_basicness_caps_word_length():
    assert _max_syllables(0.95) == 2  # basic: short, but not forced monosyllabic
    assert _max_syllables(0.60) == 3
    assert _max_syllables(0.40) == 4


def test_root_headwords_are_uniquely_spelled():
    # The romanization is the dictionary headword; coined roots must not collide in
    # spelling even when the romanizer merges distinct IPA vowels.
    for seed in range(15):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        romans = [e.roman for e in lex.entries.values() if e.etymology is Etymology.ROOT]
        assert len(romans) == len(set(romans)), f"duplicate root headword at seed {seed}"


def test_basic_words_are_shorter_on_average():
    # Zipf's law of abbreviation is a statistical tendency, so aggregate across seeds rather
    # than betting on any single language.
    basic, rare = [], []
    for seed in range(12):
        phono, rng = _phonotactics(seed)
        lex = build_lexicon(phono, rng)
        roots = [e for e in lex.entries.values() if e.etymology is Etymology.ROOT]
        basic += [len(e.form) for e in roots if e.concept.basicness >= 0.85]
        rare += [len(e.form) for e in roots if e.concept.basicness <= 0.55]
    assert sum(basic) / len(basic) < sum(rare) / len(rare)


# --- Lexicon container --------------------------------------------------------------
def test_by_field_groups_and_orders():
    phono, rng = _phonotactics(2)
    lex = build_lexicon(phono, rng)
    grouped = lex.by_field()
    assert "nature" in grouped
    # within a field, basicness is non-increasing
    for entries in grouped.values():
        bs = [e.concept.basicness for e in entries]
        assert bs == sorted(bs, reverse=True)
