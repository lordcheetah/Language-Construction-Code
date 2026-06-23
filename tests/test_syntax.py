"""Tests for the syntax stage.

Linearization is checked with hand-built morphology + parameters so the expected ordered,
inflected sentence is unambiguous; the harmonic correlations are checked statistically.
"""

import random

from conlang.phonology import data
from conlang.morphology.features import CATEGORIES, WORD_CLASSES, FeatureBundle, Typology
from conlang.morphology.affix import Affix, Position
from conlang.morphology.paradigm import Paradigm
from conlang.morphology.generator import MorphologySystem
from conlang.syntax.parameters import (
    WordOrder,
    Side,
    Adposition,
    Alignment,
    SyntaxParameters,
    derive_correlates,
)
from conlang.syntax.structure import Lexeme, NounPhrase, Clause, AdpositionalPhrase
from conlang.syntax.linearizer import Linearizer
from conlang.syntax.generator import random_syntax

NUMBER = CATEGORIES["number"]
CASE = CATEGORIES["case"]


def lex(symbols: str, word_class: str, gloss: str) -> Lexeme:
    return Lexeme(tuple(data.BY_IPA[s] for s in symbols.split()), word_class, gloss)


def forms(sentence):
    return [w.ipa for w in sentence.words]


# --- A controlled language: nouns mark number+case, verbs agree in number ------------
def _system() -> MorphologySystem:
    noun = Paradigm(WORD_CLASSES["noun"], Typology.AGGLUTINATIVE, (NUMBER, CASE))
    noun.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    noun.agglutinative_affixes[("case", "acc")] = Affix(
        (data.consonant("n"),), Position.SUFFIX, FeatureBundle.of(case="acc"), "ACC"
    )
    verb = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (NUMBER,))
    verb.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("u"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    return MorphologySystem(Typology.AGGLUTINATIVE, {"noun": noun, "verb": verb})


def _params(order=WordOrder.SVO, alignment=Alignment.NOMINATIVE_ACCUSATIVE, **kw):
    defaults = dict(
        adposition=Adposition.PREPOSITION,
        adjective=Side.AFTER,
        genitive=Side.AFTER,
        relative=Side.AFTER,
    )
    defaults.update(kw)
    return SyntaxParameters(basic_order=order, alignment=alignment, **defaults)


WOMAN = lex("m i", "noun", "woman")
BIRD = lex("p o", "noun", "bird")
SEE = lex("t a", "verb", "see")
BIG = lex("r a", "adjective", "big")


# --- Word order ---------------------------------------------------------------------
def test_is_vo_classification():
    assert WordOrder.SVO.is_vo and WordOrder.VSO.is_vo and WordOrder.VOS.is_vo
    assert not WordOrder.SOV.is_vo and not WordOrder.OVS.is_vo and not WordOrder.OSV.is_vo


def test_constituent_order_follows_basic_order():
    lin_svo = Linearizer(_params(WordOrder.SVO), _system())
    lin_sov = Linearizer(_params(WordOrder.SOV), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    # SVO: woman, see, bird+ACC ; SOV: woman, bird+ACC, see
    assert forms(lin_svo.linearize(clause)) == ["mi", "ta", "pon"]
    assert forms(lin_sov.linearize(clause)) == ["mi", "pon", "ta"]


# --- Alignment / case ---------------------------------------------------------------
def test_nominative_accusative_marks_the_object():
    lin = Linearizer(_params(alignment=Alignment.NOMINATIVE_ACCUSATIVE), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    # A (woman) = unmarked nominative; O (bird) = accusative -n
    assert forms(lin.linearize(clause)) == ["mi", "ta", "pon"]


def test_ergative_absolutive_marks_the_agent():
    lin = Linearizer(_params(alignment=Alignment.ERGATIVE_ABSOLUTIVE), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    # A (woman) = marked ergative -n; O (bird) = unmarked absolutive
    assert forms(lin.linearize(clause)) == ["min", "ta", "po"]


def test_intransitive_subject_is_unmarked_in_both_alignments():
    clause = Clause(NounPhrase(WOMAN), SEE)  # no object
    for alignment in Alignment:
        lin = Linearizer(_params(alignment=alignment), _system())
        assert forms(lin.linearize(clause)) == ["mi", "ta"]  # S unmarked, sg verb


# --- Agreement ----------------------------------------------------------------------
def test_verb_agrees_with_subject_number():
    lin = Linearizer(_params(), _system())
    clause = Clause(NounPhrase(WOMAN, number="pl"), SEE)  # intransitive plural subject
    # woman+PL = mis ; verb agrees plural = ta+u
    assert forms(lin.linearize(clause)) == ["mis", "tau"]


# --- Intra-NP order -----------------------------------------------------------------
def test_adjective_placement_follows_parameter():
    clause = Clause(NounPhrase(WOMAN, adjective=BIG), SEE)
    after = Linearizer(_params(adjective=Side.AFTER), _system())
    before = Linearizer(_params(adjective=Side.BEFORE), _system())
    # No adjective paradigm -> adjective surfaces as its bare root /ra/.
    assert forms(after.linearize(clause))[:2] == ["mi", "ra"]
    assert forms(before.linearize(clause))[:2] == ["ra", "mi"]


# --- Graceful when the language marks little ----------------------------------------
def test_caseless_language_relies_on_word_order():
    bare = MorphologySystem(Typology.ISOLATING, {})  # no paradigms at all
    lin = Linearizer(_params(WordOrder.SOV), bare)
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    # Nothing inflects; SOV order alone distinguishes the arguments.
    assert forms(lin.linearize(clause)) == ["mi", "po", "ta"]


def _glosses(sentence):
    return [w.gloss for w in sentence.words]


# --- Gloss honesty (must reflect realized morphology) -------------------------------
def test_gloss_omits_unmarked_categories():
    # Language marks number+case -> object glossed ACC, plural subject glossed PL.
    lin = Linearizer(_params(), _system())
    clause = Clause(NounPhrase(WOMAN, number="pl"), SEE, NounPhrase(BIRD))
    g = _glosses(lin.linearize(clause))
    assert "woman.PL" in g and "bird.ACC" in g


def test_gloss_honest_in_caseless_numberless_language():
    bare = MorphologySystem(Typology.ISOLATING, {})  # marks nothing
    lin = Linearizer(_params(), bare)
    clause = Clause(NounPhrase(WOMAN, number="pl"), SEE, NounPhrase(BIRD))
    g = _glosses(lin.linearize(clause))
    # No category is marked, so no PL/ACC tags may appear -- the gloss must not lie.
    assert g == ["woman", "see", "bird"]


def test_verb_gloss_only_tags_marked_agreement():
    # Verb paradigm marks number but not person -> "see.PL", never "see.3SG".
    lin = Linearizer(_params(), _system())
    clause = Clause(NounPhrase(WOMAN, number="pl"), SEE)
    assert _glosses(lin.linearize(clause)) == ["woman.PL", "see.PL"]


# --- Ergative agreement controller --------------------------------------------------
def test_ergative_verb_agrees_with_absolutive_object():
    lin = Linearizer(_params(alignment=Alignment.ERGATIVE_ABSOLUTIVE), _system())
    # singular agent, plural object: an ergative language agrees with the absolutive (O).
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, number="pl"))
    forms_out = forms(lin.linearize(clause))
    # agent woman = ergative -n (min); verb agrees with plural object = ta+u (tau)
    assert "min" in forms_out and "tau" in forms_out


# --- Intra-NP stacking: genitive outside adjective ----------------------------------
def test_genitive_is_placed_outside_adjective():
    np = NounPhrase(WOMAN, adjective=BIG, genitive=NounPhrase(BIRD))
    before = Linearizer(_params(adjective=Side.BEFORE, genitive=Side.BEFORE), _system())
    clause = Clause(np, SEE)
    # Both prenominal: genitive outermost -> [gen, adj, noun, ...]
    first_three = forms(before.linearize(clause))[:3]
    assert first_three == ["po", "ra", "mi"]  # bird(gen) big woman


# --- Adpositional phrase uses the adposition parameter ------------------------------
def test_adposition_placement_follows_parameter():
    NEAR = lex("k u", "adposition", "near")
    pp = AdpositionalPhrase(NEAR, NounPhrase(BIRD), "near")
    clause = Clause(NounPhrase(WOMAN), SEE, obliques=[pp])
    prep = Linearizer(_params(adposition=Adposition.PREPOSITION), _system())
    post = Linearizer(_params(adposition=Adposition.POSTPOSITION), _system())
    # preposition: adposition before its NP; postposition: after.
    assert forms(prep.linearize(clause))[-2:] == ["ku", "po"]
    assert forms(post.linearize(clause))[-2:] == ["po", "ku"]


# --- Fusional morphology through the linearizer -------------------------------------
def test_fusional_morphology_linearizes():
    noun = Paradigm(WORD_CLASSES["noun"], Typology.FUSIONAL, (CASE,))
    b = FeatureBundle.of(case="acc")
    noun.agglutinative_affixes  # (unused for fusional)
    noun.fusional_affixes[b] = Affix(
        (data.consonant("n"),), Position.SUFFIX, b, str(b)
    )
    system = MorphologySystem(Typology.FUSIONAL, {"noun": noun})
    lin = Linearizer(_params(), system)
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    # object accusative via portmanteau -n; subject unmarked; verb (no paradigm) bare.
    assert forms(lin.linearize(clause)) == ["mi", "ta", "pon"]


# --- Generator ----------------------------------------------------------------------
def test_random_syntax_reproducible():
    a = random_syntax(random.Random(7))
    b = random_syntax(random.Random(7))
    assert a == b


def test_vo_languages_tend_prepositional():
    # Harmonic correlation is statistical; assert the tendency over many rolls.
    prep = 0
    n = 300
    for s in range(n):
        params = derive_correlates(WordOrder.SVO, random.Random(s))
        if params.adposition is Adposition.PREPOSITION:
            prep += 1
    assert prep / n > 0.7  # ~0.85 expected


def test_ov_languages_tend_postpositional():
    post = 0
    n = 300
    for s in range(n):
        params = derive_correlates(WordOrder.SOV, random.Random(s))
        if params.adposition is Adposition.POSTPOSITION:
            post += 1
    assert post / n > 0.7
