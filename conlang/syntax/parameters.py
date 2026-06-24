"""Word-order parameters and their cross-linguistic correlations.

The central typological parameter is the basic order of Subject, Object, and Verb. From
whether the verb precedes the object (VO) or follows it (OV) flow a set of *harmonic*
correlations identified by Greenberg and quantified by Dryer: VO languages tend to be
prepositional with the noun before its genitive and relative clause; OV languages tend to
be postpositional with those orders reversed. :func:`derive_correlates` builds a plausible
parameter set from the basic order, applying those tendencies with realistic noise (the
correlations are statistical, not absolute — adjective order in particular barely
correlates at all).

Alignment (how the core arguments S, A, O are case-marked) is independent of word order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class WordOrder(Enum):
    """Basic order of Subject (S), Object (O), and Verb (V)."""

    SOV = ("S", "O", "V")
    SVO = ("S", "V", "O")
    VSO = ("V", "S", "O")
    VOS = ("V", "O", "S")
    OVS = ("O", "V", "S")
    OSV = ("O", "S", "V")

    @property
    def sequence(self) -> tuple[str, ...]:
        return self.value

    @property
    def is_vo(self) -> bool:
        """True if the verb precedes the object (the head-initial correlate)."""
        seq = self.value
        return seq.index("V") < seq.index("O")


class Side(Enum):
    """Where a modifier sits relative to its head noun."""

    BEFORE = "before"
    AFTER = "after"


class Adposition(Enum):
    PREPOSITION = "preposition"   # head-initial: adposition before its noun
    POSTPOSITION = "postposition"  # head-final: adposition after its noun


class Alignment(Enum):
    """How the core arguments are grouped for case marking.

    - NOMINATIVE_ACCUSATIVE: intransitive subject (S) and transitive agent (A) pattern
      together (nominative); the patient (O) is set apart (accusative).
    - ERGATIVE_ABSOLUTIVE: S and O pattern together (absolutive); A is set apart
      (ergative).
    """

    NOMINATIVE_ACCUSATIVE = "nominative-accusative"
    ERGATIVE_ABSOLUTIVE = "ergative-absolutive"


class DitransitiveAlignment(Enum):
    """How the two objects of a ditransitive (give X to Y) are case-marked.

    - INDIRECTIVE: the theme (X) is the normal direct object; the recipient (Y) is set apart
      in the dative. The dominant pattern (English "to", German/Latin dative).
    - SECUNDATIVE: the recipient (Y) is the primary object (takes the direct-object case);
      the theme (X) is set apart (here, the dative as a secondary/oblique case). Attested in
      e.g. many Bantu and some Mesoamerican languages.
    """

    INDIRECTIVE = "indirective (dative recipient)"
    SECUNDATIVE = "secundative (primary-object recipient)"


class Negation(Enum):
    """How a clause is negated."""

    PARTICLE_BEFORE_VERB = "negative particle before the verb"
    PARTICLE_AFTER_VERB = "negative particle after the verb"
    VERBAL = "negative marked on the verb"  # uses the verb's polarity inflection


class PolarQuestion(Enum):
    """How a yes/no question is marked."""

    PARTICLE_INITIAL = "question particle, clause-initial"
    PARTICLE_FINAL = "question particle, clause-final"
    INTONATION = "intonation only (no overt marker)"


@dataclass(frozen=True)
class SyntaxParameters:
    basic_order: WordOrder
    adposition: Adposition
    adjective: Side       # noun-adjective vs adjective-noun
    genitive: Side        # possessor relative to head noun
    relative: Side        # relative clause relative to head noun
    alignment: Alignment
    negation: Negation = Negation.PARTICLE_BEFORE_VERB
    polar_question: PolarQuestion = PolarQuestion.PARTICLE_FINAL
    wh_fronting: bool = False  # content questions move the wh-word clause-initial
    ditransitive: DitransitiveAlignment = DitransitiveAlignment.INDIRECTIVE
    pro_drop: bool = False  # a pronominal subject may be dropped when agreement recovers it
    articles: bool = False  # definiteness shown by a free article word, not an affix
    differential_object_marking: bool = False  # only a prominent (definite) object is case-marked

    def describe(self) -> str:
        wh = "fronted" if self.wh_fronting else "in situ"
        return (
            f"  basic order: {self.basic_order.name}\n"
            f"  adpositions: {self.adposition.value}\n"
            f"  adjective:   {self.adjective.value} the noun\n"
            f"  genitive:    {self.genitive.value} the noun\n"
            f"  relative:    {self.relative.value} the noun\n"
            f"  alignment:   {self.alignment.value}\n"
            f"  negation:    {self.negation.value}\n"
            f"  questions:   {self.polar_question.value}; wh-words {wh}\n"
            f"  ditransitive: {self.ditransitive.value}\n"
            f"  pro-drop:    {'yes' if self.pro_drop else 'no'} (null pronominal subjects)\n"
            f"  articles:    {'free word' if self.articles else 'none / affixal'}\n"
            f"  object case: {'differential (definite only)' if self.differential_object_marking else 'uniform'}"
        )


def derive_correlates(
    basic_order: WordOrder,
    rng: random.Random,
    *,
    alignment: Alignment | None = None,
) -> SyntaxParameters:
    """Build a harmonic parameter set for *basic_order*, with statistical noise."""

    def lean(prob_first: float, first, second):
        return first if rng.random() < prob_first else second

    if basic_order.is_vo:  # head-initial tendencies
        adposition = lean(0.85, Adposition.PREPOSITION, Adposition.POSTPOSITION)
        genitive = lean(0.80, Side.AFTER, Side.BEFORE)
        relative = lean(0.90, Side.AFTER, Side.BEFORE)
    else:  # head-final tendencies
        adposition = lean(0.85, Adposition.POSTPOSITION, Adposition.PREPOSITION)
        genitive = lean(0.80, Side.BEFORE, Side.AFTER)
        relative = lean(0.65, Side.BEFORE, Side.AFTER)

    # Adjective order barely correlates with VO/OV, so it is nearly a free coin flip.
    adjective = lean(0.5, Side.AFTER, Side.BEFORE)

    if alignment is None:
        alignment = lean(
            0.75, Alignment.NOMINATIVE_ACCUSATIVE, Alignment.ERGATIVE_ABSOLUTIVE
        )

    # Both correlate with head-direction (Dryer): a question particle is clause-final in
    # OV languages and clause-initial in VO ones; negation leans pre-verbal in VO, post-
    # verbal in OV. Verbal negation is order-independent.
    if basic_order.is_vo:
        negation = rng.choices(
            [Negation.PARTICLE_BEFORE_VERB, Negation.PARTICLE_AFTER_VERB, Negation.VERBAL],
            weights=[0.45, 0.15, 0.40], k=1)[0]
        polar_question = rng.choices(
            [PolarQuestion.PARTICLE_INITIAL, PolarQuestion.PARTICLE_FINAL, PolarQuestion.INTONATION],
            weights=[0.40, 0.25, 0.35], k=1)[0]
    else:
        negation = rng.choices(
            [Negation.PARTICLE_BEFORE_VERB, Negation.PARTICLE_AFTER_VERB, Negation.VERBAL],
            weights=[0.15, 0.45, 0.40], k=1)[0]
        polar_question = rng.choices(
            [PolarQuestion.PARTICLE_INITIAL, PolarQuestion.PARTICLE_FINAL, PolarQuestion.INTONATION],
            weights=[0.10, 0.55, 0.35], k=1)[0]

    # Wh-fronting is more common in VO languages; in-situ dominates OV (and is common overall).
    wh_fronting = rng.random() < (0.55 if basic_order.is_vo else 0.20)

    # Indirective (dative recipient) is the cross-linguistically dominant ditransitive type.
    ditransitive = lean(
        0.80, DitransitiveAlignment.INDIRECTIVE, DitransitiveAlignment.SECUNDATIVE
    )

    # Null-subject languages are common; the linearizer only actually drops a pronoun when
    # the verb's agreement is rich enough to recover it, so this is a permission, not a rule.
    pro_drop = rng.random() < 0.45

    # Many languages have no articles at all; those that do often use free words.
    articles = rng.random() < 0.40

    # Differential object marking (mark only prominent/definite objects) is common but a
    # minority — Spanish, Turkish, Hindi, Hebrew, etc.
    differential_object_marking = rng.random() < 0.25

    return SyntaxParameters(
        basic_order=basic_order,
        adposition=adposition,
        adjective=adjective,
        genitive=genitive,
        relative=relative,
        alignment=alignment,
        negation=negation,
        polar_question=polar_question,
        wh_fronting=wh_fronting,
        ditransitive=ditransitive,
        pro_drop=pro_drop,
        articles=articles,
        differential_object_marking=differential_object_marking,
    )
