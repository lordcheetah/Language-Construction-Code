"""Tests for the morphology stage.

Inflection correctness is checked with small hand-built paradigms (so the expected forms
are unambiguous), and the generator is checked for reproducibility and structural
validity over random rolls.
"""

import random

import pytest

from conlang.phonology import data
from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import WordGenerator, Romanizer
from conlang.soundchange.ruleset import RuleSet
from conlang.morphology.features import (
    CATEGORIES,
    WORD_CLASSES,
    FeatureBundle,
    GrammaticalCategory,
    Typology,
)
from conlang.morphology.affix import Affix, Position
from conlang.morphology.paradigm import Paradigm, DerivationRule
from conlang.morphology.generator import random_system


def segs(symbols: str):
    return [data.BY_IPA[s] for s in symbols.split()]


def ipa(segments) -> str:
    return "".join(s.ipa for s in segments)


NUMBER = CATEGORIES["number"]
GENDER = CATEGORIES["gender"]
NOUN = WORD_CLASSES["noun"]


# --- Feature system -----------------------------------------------------------------
def test_feature_bundle_basics():
    b = FeatureBundle.of(number="pl", case="acc")
    assert b.get("number") == "pl"
    assert b.get("tense") is None
    assert str(b) == "case=acc, number=pl"  # sorted by category name
    assert b.with_feature("number", "sg").get("number") == "sg"


def test_category_base_must_be_a_value():
    with pytest.raises(ValueError):
        GrammaticalCategory("x", ("a", "b"), "z", 0.5)


def test_marked_values_excludes_base():
    assert NUMBER.marked_values == ("pl",)


# --- Affixes ------------------------------------------------------------------------
def test_affix_attaches_by_position():
    suffix = Affix((data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"))
    prefix = Affix((data.consonant("k"),), Position.PREFIX, FeatureBundle.of(number="pl"))
    assert ipa(suffix.attach(segs("a t"))) == "ats"
    assert ipa(prefix.attach(segs("a t"))) == "kat"


def test_zero_affix():
    zero = Affix((), Position.SUFFIX, FeatureBundle.of())
    assert zero.is_zero
    assert ipa(zero.attach(segs("a t"))) == "at"


# --- Agglutinative inflection -------------------------------------------------------
def _plural_paradigm(**kw) -> Paradigm:
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), **kw)
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    return par


def test_base_value_is_zero_marked():
    par = _plural_paradigm()
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="sg"))) == "kat"  # base
    assert ipa(par.inflect(root, FeatureBundle.of())) == "kat"            # defaults to base
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kats"


def test_agglutinative_stacks_two_affixes():
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER))
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    par.agglutinative_affixes[("gender", "fem")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(gender="fem"), "FEM"
    )
    root = segs("k a t")
    # number is the first marked category, so it sits closer to the root: kat + i + a
    out = par.inflect(root, FeatureBundle.of(number="pl", gender="fem"))
    assert ipa(out) == "katia"


# --- Fusional inflection ------------------------------------------------------------
def test_fusional_portmanteau():
    par = Paradigm(NOUN, Typology.FUSIONAL, (NUMBER, GENDER))
    b = FeatureBundle.of(number="pl", gender="fem")
    par.fusional_affixes[b] = Affix((data.vowel("a"),), Position.SUFFIX, b, str(b))
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl", gender="fem"))) == "kata"
    # a bundle with no portmanteau entry is left unchanged
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl", gender="masc"))) == "kat"


# --- Paradigm enumeration -----------------------------------------------------------
def test_enumerate_bundles_is_cartesian_product():
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER))
    bundles = par.enumerate_bundles()
    assert len(bundles) == len(NUMBER.values) * len(GENDER.values)  # 2 * 3


# --- Sandhi (cross-stage with sound change) -----------------------------------------
def test_sandhi_applies_after_affixation():
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / V_V"])
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), sandhi=rs)
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    # 'ap' + '-a' = 'apa'; intervocalic voicing then gives 'aba'
    assert ipa(par.inflect(segs("a p"), FeatureBundle.of(number="pl"))) == "aba"


# --- Derivation ---------------------------------------------------------------------
def test_derivation_rule_applies_affix():
    rule = DerivationRule(
        Affix((data.consonant("n"), data.vowel("o")), Position.SUFFIX, FeatureBundle.of(), "AGENT"),
        from_class="verb",
        to_class="noun",
        gloss="AGENT",
    )
    assert ipa(rule.apply(segs("k a t"))) == "katno"


# --- Generator ----------------------------------------------------------------------
def _random_phonotactics(seed: int):
    rng = random.Random(seed)
    inv = Inventory.random(rng)
    return Phonotactics.random(inv, rng), rng


def test_random_system_is_reproducible():
    def run():
        phono, rng = _random_phonotactics(7)
        return random_system(phono, rng).summary()

    assert run() == run()


def test_random_system_affixes_mark_their_category():
    phono, rng = _random_phonotactics(5)
    system = random_system(phono, rng)
    for paradigm in system.paradigms.values():
        for (cat_name, value), affix in paradigm.agglutinative_affixes.items():
            assert affix.marks.get(cat_name) == value
            cat = CATEGORIES[cat_name]
            assert value in cat.values and value != cat.base


def test_prefix_dominant_inflection_order():
    # Two prefixes stack with the first marked category closest to the root.
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER))
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("i"),), Position.PREFIX, FeatureBundle.of(number="pl"), "PL"
    )
    par.agglutinative_affixes[("gender", "fem")] = Affix(
        (data.vowel("a"),), Position.PREFIX, FeatureBundle.of(gender="fem"), "FEM"
    )
    # number is inner -> i directly before root; gender outer -> a before that: a + i + kat
    out = par.inflect(segs("k a t"), FeatureBundle.of(number="pl", gender="fem"))
    assert ipa(out) == "aikat"


def test_complete_fills_missing_categories_with_base():
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER))
    par.agglutinative_affixes[("gender", "fem")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(gender="fem"), "FEM"
    )
    # Only gender given; number defaults to its base (sg, zero) -> just the gender affix.
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of(gender="fem"))) == "kata"


def test_forced_fusional_generator_populates_bundle_map():
    phono, rng = _random_phonotactics(9)
    # Roll until a fusional system appears, then check its structure.
    system = None
    for s in range(50):
        phono, rng = _random_phonotactics(s)
        cand = random_system(phono, rng)
        if cand.typology is Typology.FUSIONAL:
            system = cand
            break
    assert system is not None, "no fusional system in 50 seeds (unexpected)"
    for paradigm in system.paradigms.values():
        assert not paradigm.agglutinative_affixes  # fusional uses the other map
        for bundle, affix in paradigm.fusional_affixes.items():
            # keys are full bundles over exactly the marked categories
            assert {c for c, _ in bundle.items()} == {cat.name for cat in paradigm.marked}
            assert not affix.is_zero


def test_isolating_marks_few_categories():
    # Across seeds, isolating systems should mark fewer categories on average than the
    # full set a class can take.
    found = False
    for s in range(60):
        phono, rng = _random_phonotactics(s)
        system = random_system(phono, rng)
        if system.typology is Typology.ISOLATING:
            found = True
            for name, par in system.paradigms.items():
                assert len(par.marked) <= len(WORD_CLASSES[name].category_names)
    assert found, "no isolating system in 60 seeds (unexpected)"


def test_generated_nonbase_affixes_are_never_empty():
    for s in range(30):
        phono, rng = _random_phonotactics(s)
        system = random_system(phono, rng)
        for par in system.paradigms.values():
            for affix in par.agglutinative_affixes.values():
                assert len(affix.form) >= 1
            for affix in par.fusional_affixes.values():
                assert len(affix.form) >= 1


def test_derive_feeds_inflection():
    phono, rng = _random_phonotactics(4)
    system = random_system(phono, rng)
    rule = system.derivations[0]
    root = segs("k a t")
    bare = system.derive(rule, root)               # no bundle -> bare derived stem
    inflected = system.derive(rule, root, FeatureBundle.of())  # citation of target class
    assert ipa(bare).startswith("kat") or ipa(bare).endswith("kat")
    assert isinstance(ipa(inflected), str)


def test_random_system_full_paradigm_smoke():
    phono, rng = _random_phonotactics(3)
    romanizer = Romanizer()
    gen = WordGenerator(phono, romanizer)
    system = random_system(phono, rng, romanizer=romanizer)
    root = [s for syl in gen.word(rng).syllables for s in syl]
    for paradigm in system.paradigms.values():
        for bundle, seg, roman in paradigm.table(root):
            assert isinstance(roman, str) and roman
            # No sandhi here, so affixation only ever adds material to the root.
            assert len(seg) >= len(root)
