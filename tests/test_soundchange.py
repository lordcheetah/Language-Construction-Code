"""Tests for the sound-change stage.

These exercise the two things that matter: rules fire in the right *contexts* (and only
there), and feature-class rules resolve to attested segments. Like the phonology tests
they lean on invariants and small worked examples rather than brittle full-word strings.
"""

import random

import pytest

from conlang.phonology import data
from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import WordGenerator
from conlang.soundchange.matcher import FeatureClass, apply_delta
from conlang.soundchange.rule import SoundChange
from conlang.soundchange.ruleset import RuleSet


def segs(symbols: str):
    return [data.BY_IPA[s] for s in symbols.split()]


def ipa(segments) -> str:
    return "".join(s.ipa for s in segments)


# --- Feature classes ----------------------------------------------------------------
def test_feature_class_matches_natural_class():
    fc = FeatureClass.parse("voiceless plosive")
    assert fc.matches(data.consonant("p"))
    assert fc.matches(data.consonant("k"))
    assert not fc.matches(data.consonant("b"))  # voiced
    assert not fc.matches(data.consonant("s"))  # fricative
    assert not fc.matches(data.vowel("a"))


def test_feature_class_cover_terms():
    obstruent = FeatureClass.parse("obstruent")
    assert obstruent.matches(data.consonant("t"))
    assert obstruent.matches(data.consonant("z"))
    assert not obstruent.matches(data.consonant("m"))  # nasal is a sonorant


def test_feature_class_rejects_unknown_word():
    with pytest.raises(ValueError):
        FeatureClass.parse("squishy plosive")


def test_apply_delta_voicing_resolves_to_attested_segment():
    assert apply_delta(data.consonant("p"), "+voiced").ipa == "b"
    assert apply_delta(data.consonant("k"), "+voiced").ipa == "g"
    assert apply_delta(data.consonant("d"), "-voiced").ipa == "t"


def test_apply_delta_length_constructs_long_vowel():
    long_a = apply_delta(data.vowel("a"), "+long")
    assert long_a.ipa == "aː" and long_a.long is True
    assert apply_delta(long_a, "-long").ipa == "a"


# --- Single rules: context sensitivity ----------------------------------------------
def test_intervocalic_voicing_only_between_vowels():
    rule = SoundChange.parse("p > b / V_V", RuleSet().categories)
    # apa -> aba, but #pa and ap# are untouched
    assert ipa(rule.apply(segs("a p a"))) == "aba"
    assert ipa(rule.apply(segs("p a"))) == "pa"
    assert ipa(rule.apply(segs("a p"))) == "ap"


def test_final_devoicing_word_finally():
    rule = SoundChange.parse("[voiced obstruent] > [-voiced] / _#", RuleSet().categories)
    assert ipa(rule.apply(segs("a d"))) == "at"
    assert ipa(rule.apply(segs("a d a"))) == "ada"  # not word-final -> unchanged


def test_deletion():
    rule = SoundChange.parse("h > 0 / V_V", RuleSet().categories)
    assert ipa(rule.apply(segs("a h a"))) == "aa"
    assert ipa(rule.apply(segs("h a"))) == "ha"


def test_rule_applies_simultaneously_not_iteratively():
    # /a a a/ with 'a > i / a_' : each site judged against the ORIGINAL, so only the
    # 2nd and 3rd a (each preceded by an original a) change, giving 'a i i' -- a left-to-
    # right feeding pass would instead give 'a i a' or cascade. We assert simultaneity.
    cats = RuleSet().categories
    rule = SoundChange.parse("a > i / a_", cats)
    assert ipa(rule.apply(segs("a a a"))) == "aii"


def test_boundary_in_left_context():
    rule = SoundChange.parse("k > 0 / #_", RuleSet().categories)
    assert ipa(rule.apply(segs("k a t"))) == "at"
    assert ipa(rule.apply(segs("a k a"))) == "aka"


def test_place_assimilation_with_feature_class_context():
    # n -> m before a bilabial (here /p/)
    rule = SoundChange.parse("n > m / _[bilabial]", RuleSet().categories)
    assert ipa(rule.apply(segs("a n p a"))) == "ampa"
    assert ipa(rule.apply(segs("a n t a"))) == "anta"  # /t/ is alveolar, no change


def test_compact_environment_with_multicodepoint_affricate():
    # /t͡ʃ/ carries a tie bar; compact tokenizing must keep it as one token.
    rule = SoundChange.parse("a > e / t͡ʃ_", RuleSet().categories)
    assert ipa(rule.apply(segs("t͡ʃ a"))) == "t͡ʃe"
    assert ipa(rule.apply(segs("t a"))) == "ta"


def test_empty_and_total_deletion():
    rule = SoundChange.parse("a > 0 / V_", RuleSet().categories)
    assert rule.apply([]) == []                       # empty word
    assert ipa(rule.apply(segs("a a a"))) == "a"      # delete every a preceded by a vowel


def test_impossible_feature_delta_leaves_segment_unchanged():
    # No attested voiced glottal fricative, so [+voiced] on /h/ is a documented no-op.
    rule = SoundChange.parse("h > [+voiced] / V_V", RuleSet().categories)
    assert ipa(rule.apply(segs("a h a"))) == "aha"
    # but a resolvable delta does change: /q/ -> /ɢ/ now that the series is complete
    voicing = SoundChange.parse("q > [+voiced] / V_V", RuleSet().categories)
    assert ipa(voicing.apply(segs("a q a"))) == "aɢa"


# --- Rulesets: ordering, categories, derivation -------------------------------------
def test_ruleset_rule_ordering_feeds():
    # Voicing feeds nothing here, but order matters: voicing then devoicing.
    rs = RuleSet.parse(
        """
        K = p t k
        K > [+voiced] / V_V
        [voiced obstruent] > [-voiced] / _#
        """
    )
    # 'apak' : intervocalic p->b giving 'abak'; final k is voiceless already -> 'abak'
    assert ipa(rs.apply(segs("a p a k"))) == "abak"
    # 'adad' : final devoicing d->t -> 'adat'
    assert ipa(rs.apply(segs("a d a d"))) == "adat"


def test_ruleset_unknown_category_symbol_errors():
    with pytest.raises(ValueError):
        RuleSet.parse("K = p t Q\nK > [+voiced] / V_V")


def test_ruleset_comments_and_blank_lines_ignored():
    rs = RuleSet.parse(
        """
        # a comment
        p > b / V_V

        # another
        """
    )
    assert len(rs.rules) == 1


def test_from_rules_uses_default_categories():
    rs = RuleSet.from_rules(["p > b / V_V", "h > 0 / V_V"])
    assert len(rs.rules) == 2
    assert ipa(rs.apply(segs("a p a h a"))) == "abaa"


def test_derivation_trace_records_changes():
    rs = RuleSet.parse("p > b / V_V\nb > 0 / _#")
    deriv = rs.derive(segs("a p a b"))
    assert deriv.original_ipa == "apab"
    assert deriv.final_ipa == "aba"  # p->b intervocalically, final b deleted
    assert deriv.changed
    assert "apab" in deriv.trace()


def test_evolve_lexicon_keeps_ancestor_link():
    rng = random.Random(7)
    inv = Inventory.random(rng)
    phono = Phonotactics.random(inv, rng)
    gen = WordGenerator(phono)
    proto = gen.lexicon(15, rng)
    rs = RuleSet.parse(_REALISTIC)
    evolved = rs.evolve_lexicon(proto, gen.romanizer)
    assert len(evolved) == len(proto)
    for e in evolved:
        assert e.original in proto
        assert e.roman and e.ipa


def test_full_evolution_is_reproducible():
    def run():
        rng = random.Random(2024)
        inv = Inventory.random(rng)
        phono = Phonotactics.random(inv, rng)
        gen = WordGenerator(phono)
        proto = gen.lexicon(10, rng)
        rs = RuleSet.parse(_REALISTIC)
        return [str(e) for e in rs.evolve_lexicon(proto, gen.romanizer)]

    assert run() == run()


_REALISTIC = """
K = p t k
K > [+voiced] / V_V
[voiced obstruent] > [-voiced] / _#
h > 0 / V_V
"""
