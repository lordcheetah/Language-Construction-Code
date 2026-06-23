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


@dataclass(frozen=True)
class SyntaxParameters:
    basic_order: WordOrder
    adposition: Adposition
    adjective: Side       # noun-adjective vs adjective-noun
    genitive: Side        # possessor relative to head noun
    relative: Side        # relative clause relative to head noun
    alignment: Alignment

    def describe(self) -> str:
        return (
            f"  basic order: {self.basic_order.name}\n"
            f"  adpositions: {self.adposition.value}\n"
            f"  adjective:   {self.adjective.value} the noun\n"
            f"  genitive:    {self.genitive.value} the noun\n"
            f"  relative:    {self.relative.value} the noun\n"
            f"  alignment:   {self.alignment.value}"
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

    return SyntaxParameters(
        basic_order=basic_order,
        adposition=adposition,
        adjective=adjective,
        genitive=genitive,
        relative=relative,
        alignment=alignment,
    )
