"""Tests for the syntax stage.

Linearization is checked with hand-built morphology + parameters so the expected ordered,
inflected sentence is unambiguous; the harmonic correlations are checked statistically.
"""

import random

import pytest

from conlang.phonology import data
from conlang.morphology.features import (
    CATEGORIES, WORD_CLASSES, FeatureBundle, GrammaticalCategory, Typology,
)
from conlang.morphology.affix import Affix, Position
from conlang.morphology.paradigm import Paradigm
from conlang.morphology.generator import MorphologySystem
from conlang.syntax.parameters import (
    WordOrder,
    Side,
    Adposition,
    Alignment,
    Negation,
    PolarQuestion,
    DitransitiveAlignment,
    SyntaxParameters,
    derive_correlates,
)
from conlang.syntax.structure import (
    Lexeme, NounPhrase, Clause, AdpositionalPhrase, RelativeClause, Coordination, Role,
)
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


# --- Sentence types: negation, questions, imperative --------------------------------
NEG = lex("n a", "particle", "not")
QPART = lex("k a", "particle", "Q")
RELPART = lex("r a", "particle", "REL")
_PARTICLES = {"neg": NEG, "q": QPART, "rel": RELPART}


def _lin(system=None, **param_kw):
    return Linearizer(_params(**param_kw), system or _system(), particles=_PARTICLES)


def test_negation_with_a_particle_before_or_after_the_verb():
    clause = Clause(NounPhrase(WOMAN), SEE, negated=True)
    before = forms(_lin(negation=Negation.PARTICLE_BEFORE_VERB).linearize(clause))
    after = forms(_lin(negation=Negation.PARTICLE_AFTER_VERB).linearize(clause))
    assert before == ["mi", "na", "ta"]   # neg directly before the verb
    assert after == ["mi", "ta", "na"]     # neg directly after the verb


def test_verbal_negation_marks_the_verb_not_a_particle():
    verb = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (CATEGORIES["polarity"],))
    verb.agglutinative_affixes[("polarity", "negative")] = Affix(
        (data.consonant("n"),), Position.SUFFIX, FeatureBundle.of(polarity="negative"), "NEG"
    )
    system = MorphologySystem(Typology.AGGLUTINATIVE, {"verb": verb})
    sent = _lin(system, negation=Negation.VERBAL).linearize(
        Clause(NounPhrase(WOMAN), SEE, negated=True)
    )
    assert "na" not in forms(sent)                     # no separate negator particle
    assert any(w.ipa == "tan" for w in sent.words)     # see + negative suffix
    assert any("NEG" in w.gloss for w in sent.words)


def test_polar_question_particle_position():
    clause = Clause(NounPhrase(WOMAN), SEE, mood="interrogative")
    final = forms(_lin(polar_question=PolarQuestion.PARTICLE_FINAL).linearize(clause))
    initial = forms(_lin(polar_question=PolarQuestion.PARTICLE_INITIAL).linearize(clause))
    inton = forms(_lin(polar_question=PolarQuestion.INTONATION).linearize(clause))
    assert final[-1] == "ka" and initial[0] == "ka"
    assert "ka" not in inton  # intonation-only: no overt marker


def test_verbal_strategy_falls_back_to_a_particle_without_polarity_marking():
    # The language "chose" verbal negation but its verb doesn't mark polarity -> particle.
    clause = Clause(NounPhrase(WOMAN), SEE, negated=True)
    forms_out = forms(_lin(negation=Negation.VERBAL).linearize(clause))  # _system() lacks polarity
    assert "na" in forms_out  # the negator particle is used as a fallback


def test_negation_is_never_silently_lost():
    # No verbal polarity and no particle supplied: negation must still show in the gloss.
    lin = Linearizer(_params(negation=Negation.PARTICLE_BEFORE_VERB), _system())  # no particles
    sent = lin.linearize(Clause(NounPhrase(WOMAN), SEE, negated=True))
    assert any("NEG" in w.gloss for w in sent.words)


def test_declarative_affirmative_is_unchanged():
    # Adding mood/polarity to the verb bundle must not alter a plain declarative.
    lin = _lin()
    plain = forms(lin.linearize(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))))
    assert plain == ["mi", "ta", "pon"]  # same as the basic SVO transitive result


def test_negation_and_question_combine():
    clause = Clause(NounPhrase(WOMAN), SEE, negated=True, mood="interrogative")
    out = forms(_lin(negation=Negation.PARTICLE_AFTER_VERB,
                     polar_question=PolarQuestion.PARTICLE_FINAL).linearize(clause))
    assert "na" in out and out[-1] == "ka"  # both a negator and a clause-final Q


def test_imperative_drops_the_subject_and_is_second_person():
    verb = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (CATEGORIES["person"],))
    verb.agglutinative_affixes[("person", "2")] = Affix(
        (data.vowel("u"),), Position.SUFFIX, FeatureBundle.of(person="2"), "2"
    )
    system = MorphologySystem(Typology.AGGLUTINATIVE, {"verb": verb})
    sent = _lin(system).linearize(Clause(NounPhrase(WOMAN), SEE, mood="imperative"))
    assert len(sent.words) == 1                       # subject dropped, just the verb
    assert sent.words[0].ipa == "tau"                 # see + 2nd-person suffix
    assert "2" in sent.words[0].gloss


# --- Content (wh-) questions --------------------------------------------------------
WHO = lex("w o", "noun", "who")


def test_wh_in_situ_keeps_the_questioned_argument_in_place():
    # In-situ: "who" stays in subject position (SVO -> who sees bird).
    clause = Clause(NounPhrase(WHO), SEE, NounPhrase(BIRD), questioned=Role.SUBJECT)
    out = forms(_lin(wh_fronting=False).linearize(clause))
    assert out == ["wo", "ta", "pon"]


def test_wh_fronting_moves_the_object_question_to_the_front():
    # Object wh-word fronts past the subject (SVO -> what woman see).
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(WHO), questioned=Role.OBJECT)
    base = forms(_lin(wh_fronting=False).linearize(clause))
    fronted = forms(_lin(wh_fronting=True).linearize(clause))
    # the object wh-word keeps its accusative case (won = wo + -n), fronted or not
    assert base == ["mi", "ta", "won"]        # in situ: subject verb object
    assert fronted == ["won", "mi", "ta"]      # fronted: object-wh first


def test_content_question_takes_no_polar_particle():
    clause = Clause(NounPhrase(WHO), SEE, NounPhrase(BIRD), questioned=Role.SUBJECT)
    out = forms(_lin(wh_fronting=True, polar_question=PolarQuestion.PARTICLE_FINAL).linearize(clause))
    assert "ka" not in out  # the yes/no marker is not used for a content question


def test_wh_fronting_correlates_with_word_order():
    vo = sum(derive_correlates(WordOrder.SVO, random.Random(s)).wh_fronting for s in range(200))
    ov = sum(derive_correlates(WordOrder.SOV, random.Random(s)).wh_fronting for s in range(200))
    assert vo > ov  # VO languages front wh-words more often


def test_ergative_object_wh_is_unmarked():
    # Under ergative alignment the object is absolutive (unmarked), even when questioned.
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(WHO), questioned=Role.OBJECT)
    out = forms(_lin(alignment=Alignment.ERGATIVE_ABSOLUTIVE, wh_fronting=True).linearize(clause))
    assert out[0] == "wo"  # absolutive object-wh fronted, no accusative -n


def test_content_question_composes_with_negation():
    clause = Clause(NounPhrase(WHO), SEE, NounPhrase(BIRD),
                    questioned=Role.SUBJECT, negated=True)
    out = forms(_lin(wh_fronting=True, negation=Negation.PARTICLE_AFTER_VERB).linearize(clause))
    assert "wo" in out and "na" in out  # the wh-word and the negator both surface


# --- Relative clauses ---------------------------------------------------------------
def test_postnominal_subject_gap_takes_a_relativizer():
    # "woman [REL __ sees bird]" — postnominal: gap + relativizer; embedded subject omitted.
    np = NounPhrase(WOMAN, relative=RelativeClause(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD)), Role.SUBJECT))
    toks = [t.ipa for t in _lin(relative=Side.AFTER)._noun_phrase(np, "nom")]
    assert toks == ["mi", "ra", "ta", "pon"]  # head, REL, see, bird.ACC


def test_postnominal_object_gap():
    np = NounPhrase(BIRD, relative=RelativeClause(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD)), Role.OBJECT))
    toks = [t.ipa for t in _lin(relative=Side.AFTER)._noun_phrase(np, "nom")]
    assert toks == ["po", "ra", "mi", "ta"]  # head, REL, woman, see (object gapped)


def test_prenominal_relative_is_participial_no_relativizer():
    # Prenominal RCs are participial cross-linguistically: no relativizer.
    np = NounPhrase(WOMAN, relative=RelativeClause(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD)), Role.SUBJECT))
    toks = [t.ipa for t in _lin(relative=Side.BEFORE)._noun_phrase(np, "nom")]
    assert toks == ["ta", "pon", "mi"]   # embedded (see bird) then head; no "ra"
    assert "ra" not in toks


def test_embedded_verb_agrees_with_a_plural_head():
    # The load-bearing claim: the embedded verb agrees with the gapped head's number.
    embedded = Clause(NounPhrase(WOMAN, number="pl"), SEE, NounPhrase(BIRD))
    np = NounPhrase(WOMAN, number="pl", relative=RelativeClause(embedded, Role.SUBJECT))
    toks = [t.ipa for t in _lin(relative=Side.AFTER)._noun_phrase(np, "nom")]
    assert "tau" in toks  # see + plural agreement (-u), controlled by the plural head


def test_relative_clause_can_be_negated():
    embedded = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD), negated=True)
    np = NounPhrase(WOMAN, relative=RelativeClause(embedded, Role.SUBJECT))
    toks = [t.ipa for t in
            _lin(relative=Side.AFTER, negation=Negation.PARTICLE_AFTER_VERB)._noun_phrase(np, "nom")]
    assert "na" in toks  # the embedded clause's negator surfaces


def test_postnominal_without_relativizer_is_gap_only():
    lin = Linearizer(_params(relative=Side.AFTER), _system())  # no particles supplied
    np = NounPhrase(WOMAN, relative=RelativeClause(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD)), Role.SUBJECT))
    toks = [t.ipa for t in lin._noun_phrase(np, "nom")]
    assert toks == ["mi", "ta", "pon"]  # head + embedded, no relativizer


# --- Coordination -------------------------------------------------------------------
AND = lex("w a", "particle", "and")
OR = lex("y o", "particle", "or")
_COORD_PARTICLES = {**_PARTICLES, "and": AND, "or": OR}


def _coord_lin(**param_kw):
    return Linearizer(_params(**param_kw), _system(), particles=_COORD_PARTICLES)


def test_coordinated_subject_takes_a_medial_conjunction_and_plural_agreement():
    lin = _coord_lin()
    subj = Coordination([NounPhrase(WOMAN), NounPhrase(BIRD)], "and")
    clause = Clause(subj, SEE)  # intransitive
    # "woman AND bird" + verb agrees PLURAL: mi wa po tau
    assert forms(lin.linearize(clause)) == ["mi", "wa", "po", "tau"]


def test_each_conjunct_of_a_coordinated_object_is_case_marked():
    lin = _coord_lin()
    obj = Coordination([NounPhrase(BIRD), NounPhrase(WOMAN)], "and")
    clause = Clause(NounPhrase(WOMAN), SEE, obj)  # SVO
    # subject sg -> verb sg (ta); both objects accusative: pon wa min
    assert forms(lin.linearize(clause)) == ["mi", "ta", "pon", "wa", "min"]


def test_disjunction_does_not_force_plural_agreement():
    lin = _coord_lin()
    subj = Coordination([NounPhrase(WOMAN), NounPhrase(BIRD)], "or")
    clause = Clause(subj, SEE)
    # "or" leaves the verb singular (first-conjunct number): mi yo po ta
    assert forms(lin.linearize(clause)) == ["mi", "yo", "po", "ta"]


def test_coordination_without_a_conjunction_particle_is_asyndetic():
    lin = Linearizer(_params(), _system(), particles=_PARTICLES)  # no "and" supplied
    subj = Coordination([NounPhrase(WOMAN), NounPhrase(BIRD)], "and")
    clause = Clause(subj, SEE)
    # juxtaposition, still plural agreement: mi po tau
    assert forms(lin.linearize(clause)) == ["mi", "po", "tau"]


def test_three_conjuncts_repeat_the_medial_coordinator():
    lin = _coord_lin()
    subj = Coordination(
        [NounPhrase(WOMAN), NounPhrase(BIRD), NounPhrase(WOMAN)], "and"
    )
    clause = Clause(subj, SEE)
    # A and B and C, coordinator between every pair, verb plural: mi wa po wa mi tau
    assert forms(lin.linearize(clause)) == ["mi", "wa", "po", "wa", "mi", "tau"]


def test_coordinated_absolutive_object_controls_ergative_agreement():
    lin = Linearizer(
        _params(alignment=Alignment.ERGATIVE_ABSOLUTIVE), _system(),
        particles=_COORD_PARTICLES,
    )
    obj = Coordination([NounPhrase(BIRD), NounPhrase(WOMAN)], "and")
    clause = Clause(NounPhrase(WOMAN), SEE, obj)  # transitive
    forms_out = forms(lin.linearize(clause))
    # Under ergative alignment the verb agrees with the (coordinated, plural) absolutive
    # object, so the verb is plural (tau); the agent is ergative-marked (-n).
    assert "tau" in forms_out          # plural agreement from the coordinated object
    assert "min" in forms_out          # ergative agent woman+ACC-as-ergative


def test_coordination_outside_subject_object_fails_clearly():
    lin = _coord_lin()
    coord = Coordination([NounPhrase(WOMAN), NounPhrase(BIRD)], "and")
    np = NounPhrase(WOMAN, genitive=coord)  # a coordinated possessor is unsupported
    with pytest.raises(TypeError):
        lin.linearize(Clause(np, SEE))


def test_compound_sentence_coordinates_whole_clauses():
    lin = _coord_lin()
    compound = Coordination(
        [Clause(NounPhrase(WOMAN), SEE), Clause(NounPhrase(BIRD), SEE)], "and"
    )
    # "woman see AND bird see" — each clause has singular agreement: mi ta wa po ta
    assert forms(lin.linearize(compound)) == ["mi", "ta", "wa", "po", "ta"]


# --- Dual number flows to the surface and the gloss ---------------------------------
_DUAL_NUMBER = GrammaticalCategory("number", ("sg", "dual", "pl"), "sg", 0.70)


def _dual_system() -> MorphologySystem:
    noun = Paradigm(WORD_CLASSES["noun"], Typology.AGGLUTINATIVE, (_DUAL_NUMBER, CASE))
    noun.agglutinative_affixes[("number", "dual")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="dual"), "DU"
    )
    noun.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    noun.agglutinative_affixes[("case", "acc")] = Affix(
        (data.consonant("n"),), Position.SUFFIX, FeatureBundle.of(case="acc"), "ACC"
    )
    verb = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (_DUAL_NUMBER,))
    verb.agglutinative_affixes[("number", "dual")] = Affix(
        (data.vowel("u"),), Position.SUFFIX, FeatureBundle.of(number="dual"), "DU"
    )
    return MorphologySystem(Typology.AGGLUTINATIVE, {"noun": noun, "verb": verb})


def test_dual_number_is_glossed_du_through_the_linearizer():
    lin = Linearizer(_params(WordOrder.SVO), _dual_system())
    out = lin.linearize(Clause(NounPhrase(WOMAN, number="dual"), SEE))  # intransitive
    glosses = [w.gloss for w in out.words]
    assert "woman.DU" in glosses                       # the noun shows the dual, not bare/PL
    assert any(g == "see.DU" for g in glosses)          # the verb agrees DU, not SG
    assert forms(out) == ["mii", "tau"]                 # woman+DU /i/, see+DU /u/


def test_requesting_dual_in_a_non_dual_language_degrades_to_base():
    lin = Linearizer(_params(WordOrder.SVO), _system())  # number is sg/pl only
    out = lin.linearize(Clause(NounPhrase(WOMAN, number="dual"), SEE))
    # 'dual' isn't in this language -> coerced to the base (sg): bare form, no DU in the gloss
    assert forms(out)[0] == "mi"
    assert not any("DU" in w.gloss for w in out.words)


def test_paucal_number_is_glossed_pauc_through_the_linearizer():
    num = GrammaticalCategory("number", ("sg", "paucal", "pl"), "sg", 0.70)
    noun = Paradigm(WORD_CLASSES["noun"], Typology.AGGLUTINATIVE, (num,))
    noun.agglutinative_affixes[("number", "paucal")] = Affix(
        (data.vowel("u"),), Position.SUFFIX, FeatureBundle.of(number="paucal"), "PAUC")
    system = MorphologySystem(Typology.AGGLUTINATIVE,
                              {"noun": noun, "verb": _system().paradigms["verb"]})
    out = Linearizer(_params(WordOrder.SVO), system).linearize(
        Clause(NounPhrase(WOMAN, number="paucal"), SEE))
    assert "woman.PAUC" in [w.gloss for w in out.words]  # glossed PAUC, not PL/DU


def test_requesting_paucal_in_a_non_paucal_language_degrades_to_base():
    # the controlled system is sg/pl only; a paucal request coerces to the base (sg), not pl
    lin = Linearizer(_params(WordOrder.SVO), _system())
    out = lin.linearize(Clause(NounPhrase(WOMAN, number="paucal"), SEE))
    assert forms(out)[0] == "mi"
    assert not any("PAUC" in w.gloss for w in out.words)


# --- Gender agreement (driven by the noun's lexical gender) --------------------------
def test_adjective_agrees_with_the_nouns_gender():
    gender = CATEGORIES["gender"]

    def gender_paradigm(wc):
        p = Paradigm(WORD_CLASSES[wc], Typology.AGGLUTINATIVE, (gender,))
        p.agglutinative_affixes[("gender", "fem")] = Affix(
            (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(gender="fem"), "FEM")
        return p

    system = MorphologySystem(Typology.AGGLUTINATIVE, {
        "noun": gender_paradigm("noun"), "adjective": gender_paradigm("adjective"),
        "verb": Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, ()),
    })
    lin = Linearizer(_params(WordOrder.SVO), system)
    fem_cat = Lexeme(tuple(data.BY_IPA[x] for x in "k a t".split()), "noun", "cat", None, "fem")
    masc_cat = Lexeme(tuple(data.BY_IPA[x] for x in "k a t".split()), "noun", "cat", None, "masc")
    big = Lexeme(tuple(data.BY_IPA[x] for x in "r a".split()), "adjective", "big")
    # feminine noun -> the noun and its agreeing adjective both take the -s gender affix
    fem = forms(lin.linearize(Clause(NounPhrase(fem_cat, adjective=big), SEE)))
    assert "kats" in fem and "ras" in fem
    # masculine (the base value) -> no gender affix on either
    masc = forms(lin.linearize(Clause(NounPhrase(masc_cat, adjective=big), SEE)))
    assert "kat" in masc and "ra" in masc


# --- Object agreement (polypersonal) ------------------------------------------------
OBJ_PERSON = CATEGORIES["object_person"]
OBJ_NUMBER = CATEGORIES["object_number"]
THEY = lex("d e", "noun", "they")


def _polypersonal_system() -> MorphologySystem:
    # the verb agrees with the subject (person+number) AND the object (object_person+number)
    verb = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE,
                    (PERSON, NUMBER, OBJ_PERSON, OBJ_NUMBER))
    verb.agglutinative_affixes[("person", "1")] = Affix(
        (data.vowel("o"),), Position.SUFFIX, FeatureBundle.of(person="1"), "1")
    verb.agglutinative_affixes[("object_person", "1")] = Affix(
        (data.consonant("m"),), Position.SUFFIX, FeatureBundle.of(object_person="1"), "O1")
    verb.agglutinative_affixes[("object_number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(object_number="pl"), "OPL")
    return MorphologySystem(Typology.AGGLUTINATIVE, {"verb": verb, "noun": _system().paradigms["noun"]})


def _the_verb(sentence):
    return next(w for w in sentence.words if w.gloss.startswith("see"))


def test_verb_cross_references_subject_and_object():
    lin = Linearizer(_params(WordOrder.SVO), _polypersonal_system())
    # "I see them": 1sg subject acting on a 3pl object -> see.1SG>3PL, form ta+o+s
    v = _the_verb(lin.linearize(Clause(NounPhrase(I_PRON, person="1"), SEE,
                                       NounPhrase(THEY, number="pl"))))
    assert v.gloss == "see.1SG>3PL" and v.roman == "taos"
    # "woman sees me": 3sg subject, 1sg object -> see.3SG>1SG, form ta+m
    v2 = _the_verb(lin.linearize(Clause(NounPhrase(WOMAN), SEE,
                                        NounPhrase(I_PRON, person="1"))))
    assert v2.gloss == "see.3SG>1SG" and v2.roman == "tam"


def test_object_agreement_absent_on_an_intransitive_verb():
    lin = Linearizer(_params(WordOrder.SVO), _polypersonal_system())
    v = _the_verb(lin.linearize(Clause(NounPhrase(WOMAN), SEE)))  # no object
    assert ">" not in v.gloss     # nothing to cross-reference in the gloss
    assert v.roman == "ta"        # ...and no spurious object affix in the form (base = zero)


def test_object_agreement_cross_references_a_coordinated_object_as_plural():
    lin = Linearizer(_params(WordOrder.SVO), _polypersonal_system(),
                     particles={"and": lex("w a", "particle", "and")})
    obj = Coordination([NounPhrase(WOMAN), NounPhrase(BIRD)], "and")  # 3rd-person plural
    v = _the_verb(lin.linearize(Clause(NounPhrase(WOMAN), SEE, obj)))
    assert v.gloss == "see.3SG>3PL"  # subject 3sg, conjoined object 3pl


def test_object_agreement_under_ergative_indexes_object_then_agent():
    lin = Linearizer(_params(WordOrder.SVO, alignment=Alignment.ERGATIVE_ABSOLUTIVE),
                     _polypersonal_system())
    # primary agreement is the absolutive object (3pl); the object-agreement slot is the agent (1sg)
    v = _the_verb(lin.linearize(Clause(NounPhrase(I_PRON, person="1"), SEE,
                                       NounPhrase(THEY, number="pl"))))
    assert v.gloss == "see.3PL>1SG"  # object>agent under ergative (primary>cross-referenced)


# --- Verb-second (V2) ---------------------------------------------------------------
def test_verb_second_puts_the_verb_second_overriding_the_base_order():
    # underlying SOV, but a V2 main clause surfaces as subject-verb-object
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    assert forms(lin.linearize(clause)) == ["mi", "ta", "pon"]  # S V O.ACC, verb second
    # without V2 the same clause is verb-final
    base = Linearizer(_params(WordOrder.SOV), _system())
    assert forms(base.linearize(clause)) == ["mi", "pon", "ta"]  # S O.ACC V


def test_v2_fronts_a_questioned_object_with_the_verb_still_second():
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True, wh_fronting=True), _system())
    what = lex("s i", "noun", "what")
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(what), questioned=Role.OBJECT)
    # O V S: the wh-object fronts, the verb is second, the subject follows
    assert forms(lin.linearize(clause)) == ["sin", "ta", "mi"]


def test_v2_polar_question_is_verb_first():
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True), _system(), particles=_PARTICLES)
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD), mood="interrogative")
    out = forms(lin.linearize(clause))
    # verb-first (V1): the verb precedes the subject (a clause-final Q particle may follow)
    assert out[0] == "ta" and out.index("ta") < out.index("mi")


def test_v2_disables_pro_drop():
    # a V2 main clause needs an overt first constituent, so the pronoun subject is kept
    system = _agreeing_system()
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True, pro_drop=True), system)
    clause = Clause(NounPhrase(I_PRON, person="1"), SEE, NounPhrase(BIRD))
    assert "ni" in forms(lin.linearize(clause))  # subject not dropped


def test_v2_applies_only_to_matrix_clauses():
    # V2 is a main-clause phenomenon: an embedded (matrix=False) clause keeps the base order.
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))
    matrix = [t.ipa for t in lin._core_tokens(clause, matrix=True)]
    embedded = [t.ipa for t in lin._core_tokens(clause, matrix=False)]
    assert matrix == ["mi", "ta", "pon"]    # V2: subject, verb (second), object.ACC
    assert embedded == ["mi", "pon", "ta"]  # embedded: base SOV, verb final


def test_v2_questioned_subject_keeps_the_subject_first():
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True, wh_fronting=True), _system())
    who = lex("s u", "noun", "who")
    clause = Clause(NounPhrase(who), SEE, NounPhrase(BIRD), questioned=Role.SUBJECT)
    # the wh-subject is already the front; verb second: who, see, bird.ACC
    assert forms(lin.linearize(clause)) == ["su", "ta", "pon"]


def test_v2_ditransitive_is_subject_verb_recipient_theme():
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True), _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    # S V then the rest (recipient.DAT, theme.ACC): mi ta dul pon
    assert forms(lin.linearize(clause)) == ["mi", "ta", "dul", "pon"]


def test_v2_keeps_the_negated_verb_group_in_second_position():
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True), _system(), particles=_PARTICLES)
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD), negated=True)
    out = forms(lin.linearize(clause))
    assert out[0] == "mi" and out[-1] == "pon"  # subject first, object last
    assert "na" in out[1:-1] and "ta" in out[1:-1]  # the verb group (NEG + verb) is in between


def test_v2_applies_to_each_clause_of_a_compound():
    lin = Linearizer(_params(WordOrder.SOV, verb_second=True), _system(),
                     particles={"and": lex("w a", "particle", "and")})
    compound = Coordination(
        [Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD)),
         Clause(NounPhrase(BIRD), SEE, NounPhrase(WOMAN))], "and"
    )
    # each conjunct is independently V2 (S V O): mi ta pon AND po ta min
    assert forms(lin.linearize(compound)) == ["mi", "ta", "pon", "wa", "po", "ta", "min"]


# --- Differential object marking ----------------------------------------------------
def test_dom_marks_only_definite_objects():
    lin = Linearizer(_params(WordOrder.SVO, differential_object_marking=True), _system())
    definite = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness="def"))
    indefinite = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness="indef"))
    assert forms(lin.linearize(definite)) == ["mi", "ta", "pon"]    # definite -> accusative -n
    assert forms(lin.linearize(indefinite)) == ["mi", "ta", "po"]   # indefinite -> unmarked


def test_dom_off_marks_every_object_uniformly():
    lin = Linearizer(_params(WordOrder.SVO, differential_object_marking=False), _system())
    for d in ("def", "indef", None):
        clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness=d))
        assert forms(lin.linearize(clause)) == ["mi", "ta", "pon"]  # always accusative


def test_dom_has_no_effect_under_ergative_alignment():
    # the object is absolutive (unmarked) under ergative, so DOM has nothing to strip
    lin = Linearizer(
        _params(WordOrder.SVO, alignment=Alignment.ERGATIVE_ABSOLUTIVE,
                differential_object_marking=True),
        _system(),
    )
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness="indef"))
    out = forms(lin.linearize(clause))
    assert "min" in out and "po" in out  # ergative agent marked, absolutive object bare


def test_dom_applies_to_a_ditransitive_theme():
    lin = Linearizer(_params(WordOrder.SVO, differential_object_marking=True), _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness="indef"),
                    indirect_object=NounPhrase(CHILD))
    out = forms(lin.linearize(clause))
    # recipient stays dative; the indefinite theme is left unmarked (po, not pon)
    assert "dul" in out and "po" in out and "pon" not in out


def test_dom_leaves_a_bare_object_unmarked():
    lin = Linearizer(_params(WordOrder.SVO, differential_object_marking=True), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))  # no definiteness
    assert forms(lin.linearize(clause)) == ["mi", "ta", "po"]  # bare -> unmarked


def test_dom_does_not_touch_the_subject():
    # the (ergative-marked) transitive subject is unaffected by DOM; only the object differs
    lin = Linearizer(_params(WordOrder.SVO, alignment=Alignment.ERGATIVE_ABSOLUTIVE,
                             differential_object_marking=True), _system())
    out = forms(lin.linearize(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness="indef"))))
    assert out[0] == "min"  # subject keeps its ergative marking regardless of DOM


def test_dom_marks_a_fully_definite_coordinated_object():
    lin = Linearizer(_params(WordOrder.SVO, differential_object_marking=True), _system(),
                     particles={"and": lex("w a", "particle", "and")})
    both_def = Coordination(
        [NounPhrase(BIRD, definiteness="def"), NounPhrase(WOMAN, definiteness="def")], "and"
    )
    out = forms(lin.linearize(Clause(NounPhrase(WOMAN), SEE, both_def)))
    assert "pon" in out and "min" in out  # both conjuncts keep the accusative (all definite)
    has_indef = Coordination(
        [NounPhrase(BIRD, definiteness="def"), NounPhrase(WOMAN, definiteness="indef")], "and"
    )
    out2 = forms(lin.linearize(Clause(NounPhrase(WOMAN), SEE, has_indef)))
    assert "po" in out2 and "mi" in out2  # a non-definite conjunct -> the whole object unmarked


def test_dom_strips_the_accusative_from_a_wh_object():
    what = lex("s i", "noun", "what")
    on = Linearizer(_params(WordOrder.SVO, differential_object_marking=True), _system())
    off = Linearizer(_params(WordOrder.SVO, differential_object_marking=False), _system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(what), questioned=Role.OBJECT)
    # without DOM the wh-object is accusative (sin); under DOM (non-specific) it is unmarked (si)
    assert "sin" in forms(off.linearize(clause))
    assert "si" in forms(on.linearize(clause)) and "sin" not in forms(on.linearize(clause))


def test_dom_coexists_with_free_articles():
    # a definite object in an article+DOM language gets BOTH a determiner and the accusative
    lin = Linearizer(
        _params(WordOrder.SVO, articles=True, differential_object_marking=True), _system(),
        particles={"art_def": lex("d o", "particle", "that"),
                   "art_indef": lex("m o", "particle", "one")},
    )
    out = lin.linearize(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD, definiteness="def")))
    glosses = [w.gloss for w in out.words]
    assert "DEF" in glosses                                  # the determiner word
    assert any("bird" in g and "ACC" in g for g in glosses)  # and the accusative case


# --- Pro-drop -----------------------------------------------------------------------
PERSON = CATEGORIES["person"]
I_PRON = lex("n i", "noun", "I")


def _agreeing_system() -> MorphologySystem:
    # A verb with rich agreement (marks both person and number), which licenses pro-drop.
    system = _system()  # noun marks number+case; verb (replaced below) marks person+number
    verb = Paradigm(WORD_CLASSES["verb"], Typology.AGGLUTINATIVE, (PERSON, NUMBER))
    verb.agglutinative_affixes[("person", "1")] = Affix(
        (data.vowel("o"),), Position.SUFFIX, FeatureBundle.of(person="1"), "1"
    )
    verb.agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("u"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    system.paradigms["verb"] = verb
    return system


def test_verb_agrees_with_the_subject_person():
    lin = Linearizer(_params(WordOrder.SVO), _agreeing_system())  # pro_drop off (default)
    first = forms(lin.linearize(Clause(NounPhrase(I_PRON, person="1"), SEE, NounPhrase(BIRD))))
    third = forms(lin.linearize(Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))))
    assert first[1] == "tao"  # 1st-person subject -> verb takes the -o agreement suffix
    assert third[1] == "ta"   # full NP -> 3rd person (base, zero affix)


def test_pro_drop_omits_a_pronominal_subject_under_rich_agreement():
    lin = Linearizer(_params(WordOrder.SVO, pro_drop=True), _agreeing_system())
    clause = Clause(NounPhrase(I_PRON, person="1"), SEE, NounPhrase(BIRD))
    # 'I' is dropped; the verb's 1st-person agreement recovers it: just V O -> tao pon
    assert forms(lin.linearize(clause)) == ["tao", "pon"]


def test_pro_drop_keeps_a_full_noun_phrase_subject():
    lin = Linearizer(_params(WordOrder.SVO, pro_drop=True), _agreeing_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD))  # person=None -> not a pronoun
    assert forms(lin.linearize(clause))[0] == "mi"  # subject present


def test_pro_drop_does_not_drop_a_questioned_subject():
    lin = Linearizer(_params(WordOrder.SVO, pro_drop=True), _agreeing_system())
    who = lex("s i", "noun", "who")
    clause = Clause(NounPhrase(who, person="3"), SEE, NounPhrase(BIRD), questioned=Role.SUBJECT)
    assert "si" in forms(lin.linearize(clause))  # a questioned subject can't be dropped


def test_pro_drop_requires_rich_agreement_to_drop():
    # _system()'s verb marks only number, so person can't be recovered -> no drop.
    lin = Linearizer(_params(WordOrder.SVO, pro_drop=True), _system())
    clause = Clause(NounPhrase(I_PRON, person="1"), SEE, NounPhrase(BIRD))
    assert "ni" in forms(lin.linearize(clause))  # subject kept


def test_pro_drop_keeps_the_subject_under_ergative_alignment():
    # Under ergative alignment the transitive verb agrees with the object, not the subject,
    # so a dropped subject would be unrecoverable: it must be kept.
    lin = Linearizer(
        _params(WordOrder.SVO, pro_drop=True, alignment=Alignment.ERGATIVE_ABSOLUTIVE),
        _agreeing_system(),
    )
    clause = Clause(NounPhrase(I_PRON, person="1"), SEE, NounPhrase(BIRD))
    out = forms(lin.linearize(clause))
    assert any(w.startswith("ni") for w in out)  # subject kept (ergative-marked) despite pro-drop
    assert len(out) == 3  # subject, verb, object all present


def test_pro_drop_works_for_a_second_person_subject():
    lin = Linearizer(_params(WordOrder.SVO, pro_drop=True), _agreeing_system())
    you = lex("k a", "noun", "you")
    clause = Clause(NounPhrase(you, person="2"), SEE, NounPhrase(BIRD))
    # 'you' dropped; verb has no 2nd-person suffix in this toy system, so it is the bare base
    out = forms(lin.linearize(clause))
    assert "ka" not in out and out == ["ta", "pon"]


# --- Ditransitives ------------------------------------------------------------------
def _ditrans_system() -> MorphologySystem:
    system = _system()  # noun marks number (PL=-s) and case (ACC=-n)
    system.paradigms["noun"].agglutinative_affixes[("case", "dat")] = Affix(
        (data.consonant("l"),), Position.SUFFIX, FeatureBundle.of(case="dat"), "DAT"
    )
    return system


CHILD = lex("d u", "noun", "child")


def test_indirective_marks_recipient_dative_and_theme_accusative():
    params = _params(WordOrder.SVO, ditransitive=DitransitiveAlignment.INDIRECTIVE)
    lin = Linearizer(params, _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    # S V [recipient.DAT theme.ACC]: mi ta dul pon
    assert forms(lin.linearize(clause)) == ["mi", "ta", "dul", "pon"]


def test_secundative_swaps_the_object_cases():
    params = _params(WordOrder.SVO, ditransitive=DitransitiveAlignment.SECUNDATIVE)
    lin = Linearizer(params, _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    # recipient is now the primary (accusative) object, theme the dative: mi ta dun pol
    assert forms(lin.linearize(clause)) == ["mi", "ta", "dun", "pol"]


def test_recipient_precedes_theme_and_sits_in_the_object_region():
    params = _params(WordOrder.SOV, ditransitive=DitransitiveAlignment.INDIRECTIVE)
    lin = Linearizer(params, _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    # SOV: subject, [recipient.DAT theme.ACC], verb -> mi dul pon ta
    assert forms(lin.linearize(clause)) == ["mi", "dul", "pon", "ta"]


def test_caseless_ditransitive_relies_on_order_without_crashing():
    bare = MorphologySystem(Typology.ISOLATING, {})  # nothing inflects
    lin = Linearizer(_params(WordOrder.SVO), bare)
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    # no case marking: bare forms, recipient before theme, distinguished only by order
    assert forms(lin.linearize(clause)) == ["mi", "ta", "du", "po"]


def test_ergative_ditransitive_keeps_the_theme_absolutive():
    params = _params(WordOrder.SVO, alignment=Alignment.ERGATIVE_ABSOLUTIVE,
                     ditransitive=DitransitiveAlignment.INDIRECTIVE)
    lin = Linearizer(params, _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    # agent woman = ergative (marked -n); theme bird = absolutive (unmarked); recipient dative
    assert forms(lin.linearize(clause)) == ["min", "ta", "dul", "po"]


def test_relativizing_the_theme_keeps_the_recipient():
    # "the woman gives __ to the child" — gapping the theme must not delete the recipient.
    lin = Linearizer(_params(WordOrder.SVO), _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD))
    toks = [t.ipa for t in lin._core_tokens(clause, omit=Role.OBJECT)]
    assert toks == ["mi", "ta", "dul"]   # subject, verb, recipient.DAT — theme gapped, IO kept


def test_wh_fronting_the_theme_moves_only_the_theme_not_the_recipient():
    params = _params(WordOrder.SVO, wh_fronting=True,
                     ditransitive=DitransitiveAlignment.INDIRECTIVE)
    lin = Linearizer(params, _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, NounPhrase(BIRD),
                    indirect_object=NounPhrase(CHILD), questioned=Role.OBJECT)
    # only the questioned theme fronts; the recipient stays in situ: pon mi ta dul
    assert forms(lin.linearize(clause)) == ["pon", "mi", "ta", "dul"]


def test_coordinated_theme_in_a_ditransitive_marks_each_conjunct():
    lin = Linearizer(_params(WordOrder.SVO), _ditrans_system(),
                     particles={"and": lex("w a", "particle", "and")})
    theme = Coordination([NounPhrase(BIRD), NounPhrase(WOMAN)], "and")
    clause = Clause(NounPhrase(WOMAN), SEE, theme, indirect_object=NounPhrase(CHILD))
    # recipient.DAT then both themes accusative joined by "and": mi ta dul pon wa min
    assert forms(lin.linearize(clause)) == ["mi", "ta", "dul", "pon", "wa", "min"]


def test_recipient_without_a_direct_object_is_rejected():
    lin = Linearizer(_params(), _ditrans_system())
    clause = Clause(NounPhrase(WOMAN), SEE, indirect_object=NounPhrase(CHILD))
    with pytest.raises(ValueError):
        lin.linearize(clause)


# --- Free-word articles -------------------------------------------------------------
_ARTICLE_PARTICLES = {
    "art_def": lex("d o", "particle", "that"),    # definite article (from a demonstrative)
    "art_indef": lex("m o", "particle", "one"),   # indefinite article (from 'one')
}


def _article_lin(order=WordOrder.SVO):
    return Linearizer(_params(order, articles=True), _system(), particles=_ARTICLE_PARTICLES)


def test_definite_free_article_precedes_its_noun():
    clause = Clause(NounPhrase(WOMAN, definiteness="def"), SEE)
    out = _article_lin().linearize(clause)
    assert [w.gloss for w in out.words][:2] == ["DEF", "woman"]
    assert forms(out)[:2] == ["do", "mi"]  # the article 'do' sits at the NP's left edge


def test_indefinite_free_article():
    clause = Clause(NounPhrase(WOMAN, definiteness="indef"), SEE)
    assert forms(_article_lin().linearize(clause))[:2] == ["mo", "mi"]


def test_a_bare_noun_phrase_takes_no_article():
    clause = Clause(NounPhrase(WOMAN), SEE)  # no definiteness
    assert forms(_article_lin().linearize(clause)) == ["mi", "ta"]


def test_articles_off_emits_no_article_word():
    lin = Linearizer(_params(WordOrder.SVO, articles=False), _system(),
                     particles=_ARTICLE_PARTICLES)
    clause = Clause(NounPhrase(WOMAN, definiteness="def"), SEE)
    assert forms(lin.linearize(clause)) == ["mi", "ta"]  # no article; _system has no def affix


def test_free_article_suppresses_the_definiteness_affix():
    # A noun paradigm that marks definiteness with a -d suffix.
    DEFN = CATEGORIES["definiteness"]
    system = _system()
    noun = Paradigm(WORD_CLASSES["noun"], Typology.AGGLUTINATIVE, (DEFN,))
    noun.agglutinative_affixes[("definiteness", "def")] = Affix(
        (data.consonant("d"),), Position.SUFFIX, FeatureBundle.of(definiteness="def"), "DEF"
    )
    system.paradigms["noun"] = noun
    clause = Clause(NounPhrase(WOMAN, definiteness="def"), SEE)
    # without articles the noun takes the affix: woman+DEF = mid
    off = Linearizer(_params(WordOrder.SVO, articles=False), system, particles=_ARTICLE_PARTICLES)
    assert forms(off.linearize(clause))[0] == "mid"
    # with articles the affix is suppressed (bare 'mi') and the article carries definiteness
    on = Linearizer(_params(WordOrder.SVO, articles=True), system, particles=_ARTICLE_PARTICLES)
    out = on.linearize(clause)
    assert forms(out)[:2] == ["do", "mi"]
    assert not any(".DEF" in w.gloss for w in out.words)  # no double-marking on the noun


def test_articles_on_but_no_article_particle_falls_back_gracefully():
    # articles=True but no art particles supplied: degrade to the affix path, never crash,
    # never lose definiteness (mirrors the relativizer/negator fallbacks).
    lin = Linearizer(_params(WordOrder.SVO, articles=True), _system())  # no particles
    clause = Clause(NounPhrase(WOMAN, definiteness="def"), SEE)
    # _system's noun doesn't mark definiteness, so it surfaces bare — but no crash, no article
    assert forms(lin.linearize(clause)) == ["mi", "ta"]


def test_free_article_sits_outside_an_adjective():
    clause = Clause(NounPhrase(WOMAN, adjective=BIG, definiteness="def"), SEE)
    out = forms(_article_lin().linearize(clause))
    # DET ADJ N (adjective after the noun by _params default): article at the NP's left edge
    assert out[0] == "do" and out[1:3] == ["mi", "ra"]


# --- Stem allomorphy flows through to the surface -----------------------------------
def test_stem_allomorphy_surfaces_in_a_linearized_sentence():
    from conlang.morphology.paradigm import StemAlternation
    from conlang.soundchange.ruleset import RuleSet

    system = _system()  # noun marks number (PL=-s) and case
    system.paradigms["noun"].stem_alternation = StemAlternation(
        RuleSet.from_rules(["[voiceless plosive] > [+voiced] / _#"])
    )
    lin = Linearizer(_params(), system)
    cat = lex("k a t", "noun", "cat")
    # citation (singular subject, unmarked) keeps the root /kat/
    assert forms(lin.linearize(Clause(NounPhrase(cat), SEE)))[0] == "kat"
    # plural is overtly inflected -> bound stem /kad/ + PL /-s/ = /kads/
    assert forms(lin.linearize(Clause(NounPhrase(cat, number="pl"), SEE)))[0] == "kads"


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
def test_inflection_class_flows_through_to_the_surface():
    # Two nouns differing only in declension must surface differently in a sentence.
    from conlang.morphology.paradigm import InflectionClass

    noun = Paradigm(WORD_CLASSES["noun"], Typology.AGGLUTINATIVE, (NUMBER,))
    noun.agglutinative_affixes[("number", "pl")] = Affix(
        (data.consonant("s"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    noun.extra_classes["2"] = InflectionClass()
    noun.extra_classes["2"].agglutinative_affixes[("number", "pl")] = Affix(
        (data.vowel("i"),), Position.SUFFIX, FeatureBundle.of(number="pl"), "PL"
    )
    lin = Linearizer(_params(), MorphologySystem(Typology.AGGLUTINATIVE, {"noun": noun}))

    woman1 = Lexeme(WOMAN.root, "noun", "woman", "1")
    woman2 = Lexeme(WOMAN.root, "noun", "woman", "2")
    s1 = lin.linearize(Clause(NounPhrase(woman1, number="pl"), SEE))
    s2 = lin.linearize(Clause(NounPhrase(woman2, number="pl"), SEE))
    assert forms(s1)[0] == "mis" and forms(s2)[0] == "mii"  # different declensions


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


def test_sentence_type_params_are_deterministic():
    a = derive_correlates(WordOrder.SOV, random.Random(0))
    b = derive_correlates(WordOrder.SOV, random.Random(0))
    assert (a.negation, a.polar_question) == (b.negation, b.polar_question)


def test_question_position_correlates_with_word_order():
    # VO languages lean toward clause-initial Q particles; OV toward clause-final.
    vo_initial = sum(
        derive_correlates(WordOrder.SVO, random.Random(s)).polar_question
        is PolarQuestion.PARTICLE_INITIAL for s in range(200))
    ov_final = sum(
        derive_correlates(WordOrder.SOV, random.Random(s)).polar_question
        is PolarQuestion.PARTICLE_FINAL for s in range(200))
    assert vo_initial > 50 and ov_final > 80  # the leanings hold in aggregate


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
