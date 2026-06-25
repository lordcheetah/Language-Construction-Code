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


# --- Epenthesis (insertion) ---------------------------------------------------------
def test_epenthesis_inserts_between_consonants():
    rule = SoundChange.parse("0 > ə / C_C", RuleSet().categories)
    assert ipa(rule.apply(segs("a k t a"))) == "akəta"   # break up the /kt/ cluster
    assert ipa(rule.apply(segs("a t a"))) == "ata"        # no cluster -> no insertion


def test_word_initial_and_final_epenthesis():
    initial = SoundChange.parse("0 > a / #_C", RuleSet().categories)
    assert ipa(initial.apply(segs("s t a"))) == "asta"
    final = SoundChange.parse("0 > a / C_#", RuleSet().categories)
    assert ipa(final.apply(segs("t a k"))) == "taka"


def test_insertion_must_be_a_literal():
    with pytest.raises(ValueError):
        SoundChange.parse("0 > [+voiced] / V_V", RuleSet().categories)


def test_insertion_requires_an_environment():
    with pytest.raises(ValueError):
        SoundChange.parse("0 > ə /", RuleSet().categories)  # would insert in every gap


def test_unbalanced_bracket_gives_a_clear_error():
    with pytest.raises(ValueError, match="unbalanced"):
        SoundChange.parse("p > b / _[voiced", RuleSet().categories)


# --- Optional environment elements --------------------------------------------------
def test_optional_environment_element():
    # Delete a plosive that is word-final, optionally across one more consonant.
    rule = SoundChange.parse("[plosive] > 0 / _(C)#", RuleSet().categories)
    assert ipa(rule.apply(segs("a t"))) == "a"      # t_#  (optional C skipped)
    assert ipa(rule.apply(segs("a t k"))) == "a"     # both: k_# and t_(k)#
    assert ipa(rule.apply(segs("a t a"))) == "ata"   # not before a boundary -> unchanged


def test_optional_element_matches_zero_or_one():
    rule = SoundChange.parse("a > e / _(C)i", RuleSet().categories)
    assert ipa(rule.apply(segs("a i"))) == "ei"      # optional C skipped: a_i
    assert ipa(rule.apply(segs("a t i"))) == "eti"   # optional C present: a_(t)i
    assert ipa(rule.apply(segs("a t t i"))) == "atti"  # two consonants: no match


# --- Feature-agreement (alpha) assimilation -----------------------------------------
def test_nasal_place_assimilation():
    rule = SoundChange.parse("[nasal] > [αplace] / _[αplace plosive]", RuleSet().categories)
    assert ipa(rule.apply(segs("a n p a"))) == "ampa"   # n -> m before bilabial /p/
    assert ipa(rule.apply(segs("a n k a"))) == "aŋka"   # n -> ŋ before velar /k/
    assert ipa(rule.apply(segs("a m k a"))) == "aŋka"   # m -> ŋ before velar /k/
    assert ipa(rule.apply(segs("a n t a"))) == "anta"   # already alveolar -> unchanged


def test_agreement_ascii_alias_and_voicing():
    # @ is an ASCII alias for α; agreement also works on voicing.
    rule = SoundChange.parse("s > [@voicing] / _[@voicing plosive]", RuleSet().categories)
    assert ipa(rule.apply(segs("a s b a"))) == "azba"   # s voices before voiced /b/
    assert ipa(rule.apply(segs("a s p a"))) == "aspa"   # stays voiceless before /p/


def test_agreement_left_context():
    # Capture from the left context instead of the right.
    rule = SoundChange.parse("[nasal] > [αplace] / [αplace plosive]_", RuleSet().categories)
    assert ipa(rule.apply(segs("a p n a"))) == "apma"   # n -> m after bilabial /p/


def test_unbound_agreement_variable_is_rejected():
    with pytest.raises(ValueError):
        SoundChange.parse("[nasal] > [αplace] / V_V", RuleSet().categories)  # α never captured


def test_unattested_agreement_target_is_left_unchanged():
    # No attested nasal at the glottal place, so assimilation to /h/ can't resolve.
    rule = SoundChange.parse("[nasal] > [αplace] / _[αplace fricative]", RuleSet().categories)
    assert ipa(rule.apply(segs("a n h a"))) == "anha"


def test_agreement_composes_with_an_optional_element():
    rule = SoundChange.parse("[nasal] > [αplace] / _(s)[αplace plosive]", RuleSet().categories)
    assert ipa(rule.apply(segs("a n p a"))) == "ampa"      # optional /s/ skipped
    assert ipa(rule.apply(segs("a n s p a"))) == "amspa"   # optional /s/ present


def test_optional_on_the_left_context():
    rule = SoundChange.parse("a > e / i(C)_", RuleSet().categories)
    assert ipa(rule.apply(segs("i a"))) == "ie"     # optional C skipped: i_a
    assert ipa(rule.apply(segs("i t a"))) == "ite"  # optional C present: i(t)_a


# --- Unbounded wildcards (Kleene * / +) ---------------------------------------------
def test_star_matches_zero_or_more_in_the_right_context():
    rule = SoundChange.parse("a > e / _C*i", RuleSet().categories)
    assert ipa(rule.apply(segs("a i"))) == "ei"        # zero consonants
    assert ipa(rule.apply(segs("a t i"))) == "eti"     # one
    assert ipa(rule.apply(segs("a s t i"))) == "esti"  # several
    assert ipa(rule.apply(segs("a t a"))) == "ata"     # no following i -> no change


def test_plus_requires_at_least_one():
    rule = SoundChange.parse("a > e / _C+i", RuleSet().categories)
    assert ipa(rule.apply(segs("a i"))) == "ai"        # zero consonants -> NOT matched
    assert ipa(rule.apply(segs("a t i"))) == "eti"     # one consonant -> matched
    assert ipa(rule.apply(segs("a s t i"))) == "esti"  # several


def test_star_on_the_left_context():
    rule = SoundChange.parse("a > e / iC*_", RuleSet().categories)
    assert ipa(rule.apply(segs("i a"))) == "ie"        # i then zero C then _
    assert ipa(rule.apply(segs("i s t a"))) == "iste"  # i then several C then _
    assert ipa(rule.apply(segs("u t a"))) == "uta"     # no preceding i -> no change


def test_star_composes_with_a_boundary():
    # delete an /a/ separated from the word edge by any number of consonants
    rule = SoundChange.parse("a > 0 / _C*#", RuleSet().categories)
    assert ipa(rule.apply(segs("t a"))) == "t"        # a then zero C then #
    assert ipa(rule.apply(segs("t a k"))) == "tk"     # a then one C then #
    # in 'tata' the final a deletes (a#); the first a is followed by a vowel, so it stays
    assert ipa(rule.apply(segs("t a t a"))) == "tat"


def test_star_over_a_feature_class():
    rule = SoundChange.parse("a > e / _[plosive]*i", RuleSet().categories)
    assert ipa(rule.apply(segs("a t k i"))) == "etki"  # several plosives then i
    assert ipa(rule.apply(segs("a i"))) == "ei"        # zero plosives
    assert ipa(rule.apply(segs("a s i"))) == "asi"     # /s/ isn't a plosive -> no match


def test_star_composes_with_feature_agreement():
    # a nasal assimilates to a following plosive's place across any intervening consonants
    rule = SoundChange.parse("[nasal] > [αplace] / _C*[αplace plosive]", RuleSet().categories)
    assert ipa(rule.apply(segs("a n k a"))) == "aŋka"     # n -> ŋ before velar k (zero C)
    assert ipa(rule.apply(segs("a n s k a"))) == "aŋska"  # ... across an intervening /s/
    assert ipa(rule.apply(segs("a n t a"))) == "anta"     # already alveolar -> unchanged


def test_star_fires_at_every_matching_site_in_one_pass():
    rule = SoundChange.parse("a > e / _C*i", RuleSet().categories)
    # two independent 'a_i' sites both change; the medial 'a' in 'atati' is blocked by the
    # intervening vowel (no all-consonant path to an i), so it stays.
    assert ipa(rule.apply(segs("a t i a t i"))) == "etieti"
    assert ipa(rule.apply(segs("a t a t i"))) == "ateti"


def test_malformed_quantifiers_are_rejected():
    cats = RuleSet().categories
    for bad in ("a > e / _*i", "a > e / _C**i", "a > e / _#*i", "a > e / _(C)*i"):
        with pytest.raises(ValueError):
            SoundChange.parse(bad, cats)


# --- Multi-segment (window) rules: metathesis, gemination, splits -------------------
def test_metathesis_swaps_two_segments():
    rule = SoundChange.parse("[plosive] [liquid] > 2 1", RuleSet().categories)
    assert ipa(rule.apply(segs("a t r a"))) == "arta"   # tr -> rt
    assert ipa(rule.apply(segs("p l a"))) == "lpa"        # pl -> lp
    assert ipa(rule.apply(segs("a r t a"))) == "arta"     # rt is not stop+liquid: unchanged


def test_long_distance_metathesis_swaps_across_a_span():
    # two liquids swap across any intervening segments (. = any segment)
    rule = SoundChange.parse("[liquid] .* [liquid] > 3 2 1", RuleSet().categories)
    assert ipa(rule.apply(segs("l a r"))) == "ral"           # across one vowel
    assert ipa(rule.apply(segs("m i l a k r o"))) == "miraklo"  # milagro <-> miraclo
    assert ipa(rule.apply(segs("l r"))) == "rl"               # empty span -> adjacent swap
    assert ipa(rule.apply(segs("p a t a"))) == "pata"         # no liquid pair -> unchanged


def test_long_distance_metathesis_with_a_class_gap():
    # the gap can be a natural class: liquids swap only across vowels, not a consonant
    rule = SoundChange.parse("[liquid] V* [liquid] > 3 2 1", RuleSet().categories)
    assert ipa(rule.apply(segs("l a a r"))) == "raal"   # across two vowels
    assert ipa(rule.apply(segs("l t r"))) == "ltr"      # /t/ isn't a vowel -> no span, no swap


def test_window_rule_rejects_two_variable_slots():
    with pytest.raises(ValueError):
        SoundChange.parse("[liquid] V* C* [liquid] > 4 1", RuleSet().categories)


def test_long_distance_metathesis_is_greedy_outermost():
    rule = SoundChange.parse("[liquid] .* [liquid] > 3 2 1", RuleSet().categories)
    # 'r a l a r' has two candidate liquid pairs; greedy takes the OUTERMOST (r...r), which
    # swaps the two identical /r/ -> unchanged. (A shortest-match would give 'l a r a r'.)
    assert ipa(rule.apply(segs("r a l a r"))) == "ralar"


def test_long_distance_metathesis_respects_the_environment():
    rule = SoundChange.parse("[liquid] .* [liquid] > 3 2 1 / _ a", RuleSet().categories)
    assert ipa(rule.apply(segs("l a r a"))) == "rala"   # the pair is followed by /a/ -> swap
    assert ipa(rule.apply(segs("l a r t"))) == "lart"   # followed by /t/ -> no swap


def test_any_matcher_in_a_single_segment_rule_with_a_boundary():
    rule = SoundChange.parse(". > k / #_", RuleSet().categories)
    assert ipa(rule.apply(segs("a t a"))) == "kta"  # only the word-initial segment -> /k/


def test_metathesis_windows_do_not_overlap():
    # 'abab' under ab -> ba becomes 'baba', not a cascade: each window is taken once.
    rule = SoundChange.parse("a b > 2 1", RuleSet().categories)
    assert ipa(rule.apply(segs("a b a b"))) == "baba"


def test_gemination_doubles_a_segment_by_backreference():
    rule = SoundChange.parse("[voiceless plosive] > 1 1 / V_V", RuleSet().categories)
    assert ipa(rule.apply(segs("a t a"))) == "atta"
    assert ipa(rule.apply(segs("t a"))) == "ta"             # needs a vowel on both sides
    assert ipa(rule.apply(segs("a t a k a"))) == "attakka"  # every eligible stop doubles


def test_window_rule_can_reduce_or_split():
    reduce = SoundChange.parse("s k > k", RuleSet().categories)   # cluster simplification
    assert ipa(reduce.apply(segs("a s k a"))) == "aka"
    split = SoundChange.parse("[plosive] > ʔ 1 / #_", RuleSet().categories)  # prothesis
    assert ipa(split.apply(segs("p a"))) == "ʔpa"
    assert ipa(split.apply(segs("a p a"))) == "apa"             # only word-initial


def test_window_deletion_removes_a_cluster():
    rule = SoundChange.parse("h h > 0", RuleSet().categories)
    assert ipa(rule.apply(segs("a h h a"))) == "aa"


def test_all_literal_window_swap():
    # A two-segment target with an all-literal (no backref) reordered output.
    rule = SoundChange.parse("s k > k s", RuleSet().categories)
    assert ipa(rule.apply(segs("a s k a"))) == "aksa"


def test_window_rule_orders_inside_a_ruleset():
    # A window rule feeds a later single-segment rule: cluster reduction creates an
    # intervocalic stop, which then voices.
    rs = RuleSet.parse(
        """
        s k > k
        [voiceless plosive] > [+voiced] / V_V
        """
    )
    # 'aska': sk -> k giving 'aka'; then intervocalic k -> g -> 'aga'.
    assert ipa(rs.apply(segs("a s k a"))) == "aga"


def test_backreference_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        SoundChange.parse("p > 2", RuleSet().categories)        # only one target segment
    with pytest.raises(ValueError):
        SoundChange.parse("p t > 3 1", RuleSet().categories)    # no third segment


def test_feature_change_in_a_window_replacement_is_rejected():
    with pytest.raises(ValueError):
        SoundChange.parse("p t > [+voiced]", RuleSet().categories)


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
