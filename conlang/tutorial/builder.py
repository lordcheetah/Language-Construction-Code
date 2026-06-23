"""A builder that assembles a language one stage at a time from tutorial choices.

Each ``set_*``/``roll_*`` method records one stage; a missing argument means "roll a
plausible random one". When every stage is in place, :meth:`to_language` produces the
final :class:`~conlang.language.Language`. The whole thing runs off one seeded RNG, so a
tutorial session is reproducible from its seed just like ``Language.generate``.

Call the stage methods in dependency order (inventory -> phonotactics -> morphology ->
syntax -> lexicon -> writing), which is what :class:`~conlang.tutorial.session.TutorialSession`
does. A method whose prerequisite is missing will auto-roll it from the RNG; calling stages
out of order can therefore roll a stage you meant to choose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import Romanizer
from conlang.morphology.generator import random_system, MorphologySystem
from conlang.morphology.features import Typology
from conlang.syntax.parameters import WordOrder, SyntaxParameters, derive_correlates
from conlang.lexicon.generator import build_lexicon
from conlang.lexicon.lexicon import Lexicon
from conlang.lexicon.numerals import build_numerals
from conlang.writing.generator import build_writing_system
from conlang.writing.system import WritingSystem, WritingSystemType
from conlang.language import Language

# Named inventory sizes -> (consonant target, vowel target).
_INVENTORY_SIZES = {"small": (12, 3), "medium": (20, 5), "large": (30, 8)}

# Named syllable-structure complexities -> templates.
_SYLLABLE_PROFILES = {
    "simple": ["CV"],
    "moderate": ["(C)V", "(C)V(C)"],
    "complex": ["(C)(C)V(C)", "(C)V(C)"],
}


@dataclass
class LanguageBuilder:
    seed: int | None
    rng: random.Random
    romanizer: Romanizer = field(default_factory=Romanizer)
    inventory: Inventory | None = None
    phonotactics: Phonotactics | None = None
    morphology: MorphologySystem | None = None
    syntax: SyntaxParameters | None = None
    lexicon: Lexicon | None = None
    writing: WritingSystem | None = None

    @classmethod
    def start(cls, seed: int | None = None) -> "LanguageBuilder":
        if seed is None:
            seed = random.Random().randrange(2**63)
        return cls(seed=seed, rng=random.Random(seed))

    # --- Stage decisions -------------------------------------------------------------
    def roll_inventory(self, size: str | None = None) -> None:
        if size in _INVENTORY_SIZES:
            c, v = _INVENTORY_SIZES[size]
            self.inventory = Inventory.random(self.rng, consonant_target=c, vowel_target=v)
        else:
            self.inventory = Inventory.random(self.rng)

    def set_phonotactics(self, complexity: str | None = None) -> None:
        inv = self._require_inventory()
        if complexity in _SYLLABLE_PROFILES:
            self.phonotactics = Phonotactics.from_notation(inv, _SYLLABLE_PROFILES[complexity])
        else:
            self.phonotactics = Phonotactics.random(inv, self.rng)

    def roll_morphology(self, typology: Typology | None = None) -> None:
        phono = self._require_phonotactics()
        self.morphology = random_system(
            phono, self.rng, romanizer=self.romanizer, typology=typology
        )

    def set_syntax(self, order: WordOrder | None = None) -> None:
        order = order or self.rng.choice(list(WordOrder))
        self.syntax = derive_correlates(order, self.rng)

    def roll_lexicon(self) -> None:
        phono = self._require_phonotactics()
        if self.morphology is None:
            self.roll_morphology()
        if self.syntax is None:
            self.set_syntax()
        self.lexicon = build_lexicon(
            phono, self.rng,
            romanizer=self.romanizer, morphology=self.morphology,
            head_final=not self.syntax.basic_order.is_vo,
        )

    def roll_writing(self, wtype: WritingSystemType | None = None) -> None:
        inv = self._require_inventory()
        self.writing = build_writing_system(inv, self.rng, wtype=wtype)

    # --- Result ----------------------------------------------------------------------
    def to_language(self) -> Language:
        """Assemble the language, rolling any stage the tutorial happened to skip."""
        if self.inventory is None:
            self.roll_inventory()
        if self.phonotactics is None:
            self.set_phonotactics()
        if self.morphology is None:
            self.roll_morphology()
        if self.syntax is None:
            self.set_syntax()
        if self.lexicon is None:
            self.roll_lexicon()
        if self.writing is None:
            self.roll_writing()
        numerals = build_numerals(
            self.lexicon, self.phonotactics, self.rng,
            romanizer=self.romanizer, head_final=not self.syntax.basic_order.is_vo,
        )
        return Language(
            inventory=self.inventory,
            phonotactics=self.phonotactics,
            morphology=self.morphology,
            syntax=self.syntax,
            lexicon=self.lexicon,
            writing=self.writing,
            numerals=numerals,
            romanizer=self.romanizer,
            seed=self.seed,
        )

    # --- Guards ----------------------------------------------------------------------
    def _require_inventory(self) -> Inventory:
        if self.inventory is None:
            self.roll_inventory()
        return self.inventory

    def _require_phonotactics(self) -> Phonotactics:
        if self.phonotactics is None:
            self.set_phonotactics()
        return self.phonotactics
