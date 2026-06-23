"""Sound change: evolving a proto-language into daughter languages.

Stage 2 of the toolkit. A *sound change* is a rule that rewrites segments in a word
according to their phonological context, written in the familiar SCA-style notation::

    target > replacement / environment

The engine operates on sequences of :class:`~conlang.phonology.features.Segment` rather
than raw text, so rules can target **natural classes by feature** — the payoff of the
Stage 1 feature system. For example::

    [voiceless plosive] > [+voiced] / V_V     # intervocalic voicing: p t k -> b d g
    [voiced obstruent]  > [-voiced] / _#      # final devoicing
    h > 0 / V_V                               # intervocalic /h/ loss

Rules are applied in order (a :class:`RuleSet`), so feeding and bleeding relationships
between changes work the way they do in historical linguistics. Apply a ruleset to a
whole lexicon to derive a daughter language.
"""

from conlang.soundchange.matcher import FeatureClass
from conlang.soundchange.rule import SoundChange
from conlang.soundchange.ruleset import RuleSet, Derivation

__all__ = ["FeatureClass", "SoundChange", "RuleSet", "Derivation"]
