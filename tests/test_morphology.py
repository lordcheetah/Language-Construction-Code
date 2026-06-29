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
from conlang.morphology.paradigm import (
    Paradigm, DerivationRule, InflectionClass, StemAlternation,
)
from conlang.morphology.generator import random_system, MorphologySystem


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


# --- Inflection classes (declensions) -----------------------------------------------
def test_inflection_classes_use_different_affixes():
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,))
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    par.extra_classes["2"] = InflectionClass()
    par.extra_classes["2"].agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    root = segs("k a t")
    assert par.class_ids() == ["1", "2"]
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"), "1")) == "kats"
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"), "2")) == "kati"
    # default (None) and the base value behave as before
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kats"
    assert ipa(par.inflect(root, FeatureBundle.of(number="sg"), "2")) == "kat"


def test_unknown_inflection_class_falls_back_to_default():
    par = _plural_paradigm()  # single class
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of(number="pl"), "99")) == "kats"


def test_random_system_inflection_classes_are_well_formed():
    phono, rng = _random_phonotactics(9)
    system = random_system(phono, rng, typology=Typology.FUSIONAL)
    for name, par in system.paradigms.items():
        cids = par.class_ids()
        assert cids[0] == "1" and len(cids) == len(set(cids))
        assert system.inflection_classes(name) == cids
        # every declared extra class actually carries affixes (when the class marks anything)
        if par.marked:
            for cid in cids[1:]:
                assert par.extra_classes[cid].fusional_affixes


def test_isolating_has_a_single_inflection_class():
    phono, rng = _random_phonotactics(2)
    system = random_system(phono, rng, typology=Typology.ISOLATING)
    assert all(par.class_ids() == ["1"] for par in system.paradigms.values())


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


# --- Stem allomorphy ----------------------------------------------------------------
def test_bound_stem_alternation_leaves_citation_form_untouched():
    # Final voiceless stop voices in the bound stem; the citation (all-base) form is the root.
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = _plural_paradigm(stem_alternation=StemAlternation(rs))
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of())) == "kat"            # citation: no change
    assert ipa(par.inflect(root, FeatureBundle.of(number="sg"))) == "kat"  # base value: no change
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kads"  # bound stem kad + s


def test_stem_alternation_can_be_triggered_by_a_single_category():
    # Number-triggered umlaut: the stem's final vowel raises only when number is non-base.
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER),
                   stem_alternation=StemAlternation(
                       RuleSet.from_rules(["a > e / _#"]), trigger_category="number"))
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    par.agglutinative_affixes[("gender", "fem")] = Affix(
        (data.consonant("n"),), Position.SUFFIX, FeatureBundle.of(gender="fem"), "FEM"
    )
    root = segs("m a")
    assert ipa(par.inflect(root, FeatureBundle.of())) == "ma"                       # citation
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "mes"           # umlaut + PL
    # gender alone is non-base but number is base -> the trigger category stays base -> no umlaut
    assert ipa(par.inflect(root, FeatureBundle.of(gender="fem"))) == "man"


def test_stem_alternation_composes_with_sandhi():
    # Allomorphy fires on the bare stem (pre-affix); sandhi fires after affixation. Both run.
    alt = StemAlternation(RuleSet.from_rules(["a > e / _#"]))                 # stem: final a -> e
    sandhi = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / V_V"])    # boundary voicing
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), sandhi=sandhi,
                   stem_alternation=alt)
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("p"), data.vowel("a")), Position.SUFFIX,
        FeatureBundle.of(number="pl"), "PL",
    )
    # 'ma' -> bound stem 'me' -> + 'pa' = 'mepa' -> intervocalic p voices -> 'meba'
    assert ipa(par.inflect(segs("m a"), FeatureBundle.of(number="pl"))) == "meba"


def test_affix_conditioned_alternation_fires_only_before_a_vowel_initial_suffix():
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    alt = StemAlternation(rs, condition="before_vowel")
    # a vowel-initial plural suffix /-a/ -> the stem lenites (kad + a)
    v_par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), stem_alternation=alt)
    v_par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(v_par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "kada"
    # a consonant-initial plural suffix /-s/ -> the stem stays strong (kat + s)
    c_par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), stem_alternation=alt)
    c_par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(c_par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "kats"


def test_before_consonant_condition_is_the_mirror_image():
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    alt = StemAlternation(rs, condition="before_consonant")
    # consonant-initial suffix /-s/ -> the stem lenites (kad + s)
    c_par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), stem_alternation=alt)
    c_par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(c_par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "kads"
    # vowel-initial suffix /-a/ -> the stem stays strong (kat + a)
    v_par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,), stem_alternation=alt)
    v_par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(v_par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "kata"


def test_conditioned_alternation_needs_a_suffix_not_just_a_prefix():
    # Overtly inflected, but the marker is a PREFIX, so nothing follows the stem's right edge
    # -> a before_vowel alternation stays strong (no lenition word-finally).
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER,),
                   stem_alternation=StemAlternation(rs, condition="before_vowel"))
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("i"),), Position.PREFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "ikat"  # strong


def test_affix_conditioning_keys_off_the_innermost_suffix():
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    alt = StemAlternation(rs, condition="before_vowel")
    # inner suffix (number) vowel-initial, outer (gender) consonant-initial -> lenites
    vc = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER), stem_alternation=alt)
    vc.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL")
    vc.agglutinative_affixes[("gender", "fem")] = Affix(
        (data.consonant("n"),), Position.SUFFIX, FeatureBundle.of(gender="fem"), "FEM")
    assert ipa(vc.inflect(segs("k a t"), FeatureBundle.of(number="pl", gender="fem"))) == "kadin"
    # inner suffix consonant-initial, outer vowel-initial -> stays strong (the inner one conditions)
    cv = Paradigm(NOUN, Typology.AGGLUTINATIVE, (NUMBER, GENDER), stem_alternation=alt)
    cv.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL")
    cv.agglutinative_affixes[("gender", "fem")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(gender="fem"), "FEM")
    assert ipa(cv.inflect(segs("k a t"), FeatureBundle.of(number="pl", gender="fem"))) == "katsa"


def test_conditioned_alternation_under_fusional_typology():
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = Paradigm(NOUN, Typology.FUSIONAL, (NUMBER,),
                   stem_alternation=StemAlternation(rs, condition="before_vowel"))
    par.fusional_affixes[FeatureBundle.of(number="pl")] = Affix(
        (data.vowel("a"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "kada"  # lenites


def test_generated_stem_alternation_can_be_affix_conditioned():
    for seed in range(40):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        for par in system.paradigms.values():
            alt = par.stem_alternation
            if alt is not None and alt.condition == "before_vowel":
                return
    raise AssertionError("no affix-conditioned stem alternation in 40 seeds (unexpected)")


def test_overtly_inflected_root_that_does_not_match_is_unchanged():
    # The bound-stem rule targets a final stop; a vowel-final root meets the inflection
    # trigger but not the rule's structural description, so it does NOT alternate (partial,
    # phonologically-conditioned allomorphy).
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = _plural_paradigm(stem_alternation=StemAlternation(rs))
    assert ipa(par.inflect(segs("m a"), FeatureBundle.of(number="pl"))) == "mas"  # no voicing


def test_stem_alternation_applies_under_fusional_typology():
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = Paradigm(NOUN, Typology.FUSIONAL, (NUMBER,), stem_alternation=StemAlternation(rs))
    par.fusional_affixes[FeatureBundle.of(number="pl")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of())) == "kat"            # citation
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of(number="pl"))) == "kadi"  # bound stem


def test_stem_alternation_applies_across_inflection_classes():
    # An unbound (classes=None) bound stem applies in every declension.
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = _plural_paradigm(stem_alternation=StemAlternation(rs))
    par.extra_classes["2"] = InflectionClass(
        {("number", "pl"): Affix((data.vowel("i"),), Position.SUFFIX,
                                 FeatureBundle.of(number="pl"), "PL")}
    )
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"), "1")) == "kads"  # decl 1: kad+s
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"), "2")) == "kadi"  # decl 2: kad+i


def test_class_bound_stem_alternation_applies_only_to_its_classes():
    # A class-bound alternation (a strong/weak split): only declension "2" mutates its stem;
    # declension "1" keeps the plain root, even though both are overtly inflected.
    rs = RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    par = _plural_paradigm(stem_alternation=StemAlternation(rs, classes=("2",)))
    par.extra_classes["2"] = InflectionClass(
        {("number", "pl"): Affix((data.vowel("i"),), Position.SUFFIX,
                                 FeatureBundle.of(number="pl"), "PL")}
    )
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"), "1")) == "kats"  # weak: root kept
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"), "2")) == "kadi"  # strong: kad+i
    # the default class is "1" when none is named -> also unaltered here
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kats"


def test_class_bound_alternation_can_be_generated():
    # some seed organically rolls a class-bound stem alternation (a proper subset of classes)
    found = False
    for seed in range(400):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        for par in system.paradigms.values():
            sa = par.stem_alternation
            if sa is not None and sa.classes is not None:
                assert 1 <= len(sa.classes) < len(par.class_ids())  # a proper, non-empty subset
                assert set(sa.classes) <= set(par.class_ids())
                found = True
        if found:
            break
    assert found, "no seed organically rolled a class-bound stem alternation"


def test_suppletive_stem_replaces_the_whole_word_for_its_cell():
    # Full-form suppletion (go/went): the past cell yields a wholly irregular word, no affix;
    # other cells inflect regularly even with the suppletion table present.
    TENSE = CATEGORIES["tense"]
    par = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (TENSE,))
    par.agglutinative_affixes[("tense", "past")] = Affix(
        (data.consonant("d"),), Position.SUFFIX, FeatureBundle.of(tense="past"), "PST"
    )
    root = segs("g o")
    went = tuple(segs("w e n t"))
    sup = ((("tense", "past"), went),)
    assert ipa(par.inflect(root, FeatureBundle.of(tense="pres"))) == "go"     # base, no affix
    assert ipa(par.inflect(root, FeatureBundle.of(tense="past"))) == "god"    # regular past
    # with the table: past is suppletive (the stored form verbatim), present is still regular
    assert ipa(par.inflect(root, FeatureBundle.of(tense="past"), suppletive_stems=sup)) == "went"
    assert ipa(par.inflect(root, FeatureBundle.of(tense="pres"), suppletive_stems=sup)) == "go"


def test_suppletion_is_inert_when_the_marked_value_does_not_match():
    # a suppletion keyed to past never fires on a future-tense cell
    TENSE = CATEGORIES["tense"]
    par = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (TENSE,))
    par.agglutinative_affixes[("tense", "fut")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(tense="fut"), "FUT"
    )
    sup = ((("tense", "past"), tuple(segs("w e n t"))),)
    assert ipa(par.inflect(segs("g o"), FeatureBundle.of(tense="fut"), suppletive_stems=sup)) == "gos"


def test_generated_stem_alternation_is_reachable():
    # Some seed organically rolls stem allomorphy on at least one paradigm.
    found = False
    for seed in range(25):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        if any(par.stem_alternation is not None for par in system.paradigms.values()):
            found = True
            break
    assert found, "no seed in range organically rolled a stem alternation"


# --- Dual number --------------------------------------------------------------------
def test_dual_number_inflects_three_ways():
    num = GrammaticalCategory("number", ("sg", "dual", "pl"), "sg", 0.70)
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (num,))
    par.agglutinative_affixes[("number", "dual")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="dual"), "DU"
    )
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="sg"))) == "kat"     # base (zero)
    assert ipa(par.inflect(root, FeatureBundle.of(number="dual"))) == "kati"  # dual affix
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kats"    # plural affix
    # the full paradigm enumerates all three number values
    assert {b.get("number") for b in par.enumerate_bundles()} == {"sg", "dual", "pl"}


def test_fusional_dual_inflects():
    num = GrammaticalCategory("number", ("sg", "dual", "pl"), "sg", 0.70)
    par = Paradigm(NOUN, Typology.FUSIONAL, (num,))
    par.fusional_affixes[FeatureBundle.of(number="dual")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="dual"), "DU"
    )
    par.fusional_affixes[FeatureBundle.of(number="pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="sg"))) == "kat"     # citation
    assert ipa(par.inflect(root, FeatureBundle.of(number="dual"))) == "kati"
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kats"


def test_dual_differs_across_inflection_classes():
    num = GrammaticalCategory("number", ("sg", "dual", "pl"), "sg", 0.70)
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (num,))
    par.agglutinative_affixes[("number", "dual")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="dual"), "DU"
    )
    par.extra_classes["2"] = InflectionClass(
        {("number", "dual"): Affix((data.vowel("u"),), Position.SUFFIX,
                                   FeatureBundle.of(number="dual"), "DU")}
    )
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="dual"), "1")) == "kati"
    assert ipa(par.inflect(root, FeatureBundle.of(number="dual"), "2")) == "katu"


def test_requesting_an_unsupported_number_value_falls_back_to_base():
    # a sg/pl paradigm asked for 'dual' yields the citation form, not a silent mismatch
    par = _plural_paradigm()
    assert ipa(par.inflect(segs("k a t"), FeatureBundle.of(number="dual"))) == "kat"


def test_generated_dual_is_consistent_across_word_classes():
    # Some seed rolls a dual; when it does, EVERY number-marking class uses sg/dual/pl.
    for seed in range(60):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        number_cats = [next((c for c in par.marked if c.name == "number"), None)
                       for par in system.paradigms.values()]
        number_cats = [c for c in number_cats if c is not None]
        if any("dual" in c.values for c in number_cats):
            assert all("dual" in c.values for c in number_cats)  # consistent system-wide
            return
    raise AssertionError("no dual number system in 60 seeds (unexpected)")


def test_paucal_number_inflects_as_its_own_value():
    num = GrammaticalCategory("number", ("sg", "paucal", "pl"), "sg", 0.70)
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (num,))
    par.agglutinative_affixes[("number", "paucal")] = Affix(
        (data.vowel("u"),), Position.SUFFIX, FeatureBundle.of(number="paucal"), "PAUC")
    par.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL")
    root = segs("k a t")
    assert ipa(par.inflect(root, FeatureBundle.of(number="paucal"))) == "katu"
    assert ipa(par.inflect(root, FeatureBundle.of(number="pl"))) == "kats"
    assert {b.get("number") for b in par.enumerate_bundles()} == {"sg", "paucal", "pl"}


def test_four_way_number_inflects_each_value_distinctly():
    # a composed sg/dual/paucal/pl system: each non-singular value has its own affix
    num = GrammaticalCategory("number", ("sg", "dual", "paucal", "pl"), "sg", 0.70)
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (num,))
    affixes = {"dual": ("i", "DU"), "paucal": ("u", "PAUC"), "pl": ("s", "PL")}
    for value, (sym, gloss) in affixes.items():
        seg = data.vowel(sym) if sym in "aiueo" else data.consonant(sym)
        par.agglutinative_affixes[("number", value)] = Affix(
            (seg,), Position.SUFFIX, FeatureBundle.of(number=value), gloss)
    root = segs("k a t")
    forms = {v: ipa(par.inflect(root, FeatureBundle.of(number=v)))
             for v in ("sg", "dual", "paucal", "pl")}
    assert forms == {"sg": "kat", "dual": "kati", "paucal": "katu", "pl": "kats"}
    assert len(set(forms.values())) == 4  # all four number values are distinct


def test_trial_number_inflects_as_its_own_value():
    num = GrammaticalCategory("number", ("sg", "dual", "trial", "pl"), "sg", 0.70)
    par = Paradigm(NOUN, Typology.AGGLUTINATIVE, (num,))
    par.agglutinative_affixes[("number", "trial")] = Affix(
        (data.consonant("t"),), Position.SUFFIX, FeatureBundle.of(number="trial"), "TRI")
    assert ipa(par.inflect(segs("k a"), FeatureBundle.of(number="trial"))) == "kat"
    assert "trial" in {b.get("number") for b in par.enumerate_bundles()}


def test_generated_trial_always_implies_a_dual():
    # the implicational universal: no language has a trial without a dual
    found = False
    for seed in range(300):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        noun = system.paradigms.get("noun")
        cat = next((c for c in noun.marked if c.name == "number"), None) if noun else None
        if cat is not None and "trial" in cat.values:
            assert "dual" in cat.values                       # trial requires a dual
            assert cat.values.index("dual") < cat.values.index("trial")  # ordered dual < trial
            found = True
    assert found, "no trial number system in 300 seeds (unexpected)"


def test_generator_can_roll_paucal_ordered_in_the_number_system():
    # paucal rolls independently of dual; wherever it appears it sits after sg and any dual,
    # before pl (the values are ordered sg < dual < paucal < pl).
    for seed in range(120):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        noun = system.paradigms.get("noun")
        cat = next((c for c in noun.marked if c.name == "number"), None) if noun else None
        if cat is not None and "paucal" in cat.values:
            vals = cat.values
            assert vals[0] == "sg" and vals[-1] == "pl"
            if "dual" in vals:
                assert vals.index("dual") < vals.index("paucal")
            return
    raise AssertionError("no paucal number system in 120 seeds (unexpected)")


# --- Derivation ---------------------------------------------------------------------
def test_derivation_rule_applies_affix():
    rule = DerivationRule(
        Affix((data.consonant("n"), data.vowel("o")), Position.SUFFIX, FeatureBundle.of(), "AGENT"),
        from_class="verb",
        to_class="noun",
        gloss="AGENT",
    )
    assert ipa(rule.apply(segs("k a t"))) == "katno"


def test_generator_can_roll_clusivity():
    # some seed's verb marks clusivity (a 1st-person inclusive/exclusive category)
    for seed in range(60):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        verb = system.paradigms.get("verb")
        if verb and any(c.name == "clusivity" for c in verb.marked):
            return
    raise AssertionError("no clusivity in 60 seeds (unexpected)")


def test_generator_can_roll_object_agreement():
    # some seed's verb marks object agreement (polypersonal), a minority feature
    for seed in range(40):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        verb = system.paradigms.get("verb")
        if verb and any(c.name in ("object_person", "object_number") for c in verb.marked):
            return
    raise AssertionError("no object agreement in 40 seeds (unexpected)")


def test_zero_derivation_is_a_conversion():
    # a zero-marked derivation changes word class with no affix: the form is unchanged
    rule = DerivationRule(
        Affix((), Position.SUFFIX, FeatureBundle.of(), "BECOME"),
        from_class="adjective", to_class="verb", gloss="BECOME",
    )
    assert rule.affix.is_zero
    assert ipa(rule.apply(segs("k a t"))) == "kat"


def test_generator_zero_derivations_are_restricted_to_conversion_relations():
    # zero-derivation is reachable, and only AGENT/RESULT/BECOME ever go zero — never a zero
    # ANTONYM (good==bad) or DIMINUTIVE/HAVING (which would collapse their contrast).
    found = False
    for seed in range(60):
        phono, _ = _random_phonotactics(seed)
        system = random_system(phono, random.Random(seed))
        for r in system.derivations:
            if r.affix.is_zero:
                found = True
                assert r.gloss in {"AGENT", "RESULT", "BECOME"}
    assert found, "no zero (conversion) derivation in 60 seeds (unexpected)"


def test_zero_derivation_feeds_inflection():
    # a zero AGENT conversion (verb -> noun) whose product then inflects normally
    noun = Paradigm(WORD_CLASSES["noun"], Typology.AGGLUTINATIVE, (NUMBER,))
    noun.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    rule = DerivationRule(
        Affix((), Position.SUFFIX, FeatureBundle.of(), "AGENT"),
        from_class="verb", to_class="noun", gloss="AGENT",
    )
    system = MorphologySystem(Typology.AGGLUTINATIVE, {"noun": noun}, [rule])
    # 'kat' -> agent noun (zero, still 'kat') -> plural 'kats'
    out = system.derive(rule, segs("k a t"), FeatureBundle.of(number="pl"))
    assert ipa(out) == "kats"


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
