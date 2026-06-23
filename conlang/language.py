"""The :class:`Language` aggregate — a whole conlang in one object.

Each engine stage produces a piece of a language; this ties them together so they share
one inventory, one romanizer, and one consistent set of typological choices. The pieces
are wired in dependency order by :meth:`Language.generate`:

    phonology -> morphology -> syntax -> lexicon -> writing system

Because the whole thing is rolled from a single seeded RNG, **a language is fully
determined by its seed** — ``Language.generate(seed=42)`` always reproduces the same
language, which is the simplest possible persistence story (store the seed).

Beyond holding the pieces, the aggregate offers the cross-stage operations the rest of the
toolkit (and the planned capstones) need: :meth:`make_sentence` builds and inflects a
clause from dictionary glosses, :meth:`evolve` runs the lexicon through a sound change to
yield daughter forms, and :meth:`to_dict` exports a JSON-friendly snapshot.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import WordGenerator, Romanizer, Word
from conlang.morphology.generator import random_system, MorphologySystem
from conlang.syntax.generator import random_syntax
from conlang.syntax.parameters import SyntaxParameters
from conlang.syntax.structure import Lexeme, NounPhrase, Clause, Role
from conlang.syntax.linearizer import Linearizer, Sentence
from conlang.lexicon.generator import build_lexicon
from conlang.lexicon.lexicon import Lexicon
from conlang.lexicon.numerals import build_numerals, NumeralSystem
from conlang.soundchange.ruleset import RuleSet
from conlang.writing.generator import build_writing_system
from conlang.writing.system import WritingSystem

# Bumped whenever a change to any stage alters what a given seed produces. Stored in
# snapshots so a seed can be paired with the generator version that minted it.
# v2: morphology gained inflection classes (changes RNG consumption and lexicon output).
# v3: clause-level sentence types added negator/question particles to the lexicon.
# v4: a numeral system is rolled per language.
# v5: a relativizer particle was added to the lexicon.
# v6: interrogative pronouns (who/what) were added to the lexicon.
GENERATOR_VERSION = 6


@dataclass
class Language:
    """A complete constructed language: all six engine stages, wired together."""

    inventory: Inventory
    phonotactics: Phonotactics
    morphology: MorphologySystem
    syntax: SyntaxParameters
    lexicon: Lexicon
    writing: WritingSystem
    numerals: NumeralSystem
    romanizer: Romanizer
    seed: int | None = None

    # --- Construction ----------------------------------------------------------------
    @classmethod
    def generate(
        cls,
        seed: int | None = None,
        *,
        rng: random.Random | None = None,
        sandhi: RuleSet | None = None,
    ) -> "Language":
        """Roll a complete, internally consistent language from one seeded RNG.

        When neither ``seed`` nor ``rng`` is given, a concrete random seed is drawn and
        recorded, so even a "random" language stays reproducible via its :attr:`seed`.
        """
        if rng is None:
            if seed is None:
                seed = random.Random().randrange(2**63)
            rng = random.Random(seed)
        romanizer = Romanizer()

        inventory = Inventory.random(rng)
        phonotactics = Phonotactics.random(inventory, rng)
        morphology = random_system(phonotactics, rng, romanizer=romanizer, sandhi=sandhi)
        syntax = random_syntax(rng)
        lexicon = build_lexicon(
            phonotactics, rng,
            romanizer=romanizer, morphology=morphology,
            head_final=not syntax.basic_order.is_vo,
        )
        writing = build_writing_system(inventory, rng)
        numerals = build_numerals(
            lexicon, phonotactics, rng,
            romanizer=romanizer, head_final=not syntax.basic_order.is_vo,
        )

        return cls(
            inventory=inventory,
            phonotactics=phonotactics,
            morphology=morphology,
            syntax=syntax,
            lexicon=lexicon,
            writing=writing,
            numerals=numerals,
            romanizer=romanizer,
            seed=seed,
        )

    # --- Word & sentence building ----------------------------------------------------
    @property
    def word_generator(self) -> WordGenerator:
        return WordGenerator(self.phonotactics, self.romanizer)

    def word(self, rng: random.Random, **kwargs) -> Word:
        return self.word_generator.word(rng, **kwargs)

    def _lexeme(self, gloss: str, *, expect_pos: str | None = None) -> Lexeme:
        entry = self.lexicon.get(gloss)
        if entry is None:
            raise KeyError(f"no word for {gloss!r} in this language's lexicon")
        if expect_pos is not None and entry.concept.pos != expect_pos:
            raise ValueError(
                f"{gloss!r} is a {entry.concept.pos}, but a {expect_pos} is required here"
            )
        return Lexeme(entry.form, entry.concept.pos, gloss, entry.inflection_class)

    def make_sentence(
        self,
        subject: str,
        verb: str,
        obj: str | None = None,
        *,
        subject_number: str = "sg",
        subject_definiteness: str | None = None,
        subject_adjective: str | None = None,
        object_number: str = "sg",
        object_definiteness: str | None = None,
        tense: str = "pres",
        negated: bool = False,
        mood: str = "declarative",
        question: str | None = None,
    ) -> Sentence:
        """Build and inflect a clause from dictionary glosses, then linearize it.

        This is the whole stack in one call: the lexicon supplies the roots, the morphology
        inflects them, and the syntax orders, case-marks, negates, and marks sentence type.
        ``mood`` is "declarative", "interrogative" (yes/no question), or "imperative".
        ``question`` ("subject" or "object") makes a content (wh-) question on that role,
        whose gloss should be a wh-pronoun ("who"/"what"). Only core arguments can be
        questioned, and wh-fronting moves just the wh-word — no auxiliary inversion.
        """
        if question is not None and question not in ("subject", "object"):
            raise ValueError("question must be 'subject', 'object', or None")
        if question is not None and mood == "imperative":
            raise ValueError("an imperative clause cannot also be a content question")
        subj = NounPhrase(
            self._lexeme(subject, expect_pos="noun"),
            adjective=self._lexeme(subject_adjective, expect_pos="adjective")
            if subject_adjective else None,
            number=subject_number,
            definiteness=subject_definiteness,
        )
        obj_np = None
        if obj is not None:
            obj_np = NounPhrase(
                self._lexeme(obj, expect_pos="noun"),
                number=object_number, definiteness=object_definiteness,
            )
        questioned = {"subject": Role.SUBJECT, "object": Role.OBJECT}.get(question)
        clause = Clause(
            subj, self._lexeme(verb, expect_pos="verb"), obj_np,
            tense=tense, negated=negated, mood=mood, questioned=questioned,
        )
        linearizer = Linearizer(
            self.syntax, self.morphology, self.romanizer, particles=self._particles()
        )
        return linearizer.linearize(clause)

    def _particles(self) -> dict:
        """The language's grammatical particles (negator, yes/no marker) as Lexemes."""
        out = {}
        for key, gloss in (("neg", "not"), ("q", "Q"), ("rel", "REL")):
            entry = self.lexicon.get(gloss)
            if entry is not None:
                out[key] = Lexeme(entry.form, "particle", gloss, entry.inflection_class)
        return out

    def speak(self, word: Word, *, voice=None, rng: random.Random | None = None) -> list[float]:
        """Synthesize a word to audio samples (in [-1, 1]).

        Lazily imports the speech capstone so it stays an optional layer.
        """
        from conlang.speech.synth import Synthesizer

        return Synthesizer(voice, rng).synthesize_word(word)

    # --- Diachrony (Stage 2 integration) ---------------------------------------------
    def evolve(self, rules: RuleSet | list[str] | str) -> dict[str, tuple[str, str]]:
        """Run every dictionary word through a sound change; return daughter forms.

        ``rules`` may be a :class:`RuleSet`, a list of rule strings, or a multi-line
        ruleset block. Returns ``{gloss: (roman, ipa)}`` for the evolved lexicon.

        Scope: only *citation* forms are evolved. Bound morphology (affixes), derived
        stems, and inflected surface forms are not — a fuller diachrony would evolve those
        too. Compounds are stored as one flat segment sequence, so a context-sensitive rule
        can apply across the compound seam.
        """
        ruleset = _as_ruleset(rules)
        out: dict[str, tuple[str, str]] = {}
        for gloss, entry in self.lexicon.entries.items():
            evolved = ruleset.apply(entry.form)
            roman = self.romanizer.romanize([list(evolved)])
            ipa = "".join(s.ipa for s in evolved)
            out[gloss] = (roman, ipa)
        return out

    # --- Export ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        """A JSON-serializable snapshot (strings only) for display and downstream tools.

        This is *lossy* — it is not a reload format. To reconstruct the exact language,
        re-run ``Language.generate(seed)`` with the same ``generator_version``; the seed
        is the canonical persistence handle (a stage change bumps the version and may make
        an old seed produce a different language).
        """
        return {
            "seed": self.seed,
            "generator_version": GENERATOR_VERSION,
            "phonology": {
                "consonants": [c.ipa for c in self.inventory.consonants],
                "vowels": [v.ipa for v in self.inventory.vowels],
                "syllables": [str(t) for t in self.phonotactics.templates],
            },
            "morphology": {
                "typology": self.morphology.typology.value,
                "marks": {
                    name: [c.name for c in par.marked]
                    for name, par in self.morphology.paradigms.items()
                },
            },
            "syntax": {
                "order": self.syntax.basic_order.name,
                "alignment": self.syntax.alignment.value,
                "adposition": self.syntax.adposition.value,
            },
            "writing": {"type": self.writing.type.value},
            "numerals": {
                "base": self.numerals.base,
                "samples": {
                    str(n): self.numerals.number(n).roman
                    for n in (1, 2, 5, 10, 42, 100) if n <= self.numerals.max_value
                },
            },
            "lexicon": {
                gloss: {
                    "roman": e.roman,
                    "ipa": e.ipa,
                    "pos": e.concept.pos,
                    "field": e.concept.field,
                    "etymology": e.etymology.value,
                    "note": e.note,
                    "inflection_class": e.inflection_class,
                }
                for gloss, e in self.lexicon.entries.items()
            },
        }

    # --- Display ---------------------------------------------------------------------
    def summary(self) -> str:
        return "\n".join([
            self.inventory.summary(),
            f"  Syllables:  {', '.join(str(t) for t in self.phonotactics.templates)}",
            "",
            self.morphology.summary(),
            "",
            "Syntax:",
            self.syntax.describe(),
            "",
            self.writing.summary(),
            "",
            f"Numerals: base {self.numerals.base}; "
            + ", ".join(f"{n}={self.numerals.number(n).roman}"
                        for n in (1, 2, 3, 10, 100) if n <= self.numerals.max_value),
            "",
            f"Lexicon: {len(self.lexicon)} words across "
            f"{len(self.lexicon.by_field())} semantic fields",
        ])


def _as_ruleset(rules: RuleSet | list[str] | str) -> RuleSet:
    if isinstance(rules, RuleSet):
        return rules
    if isinstance(rules, list):
        return RuleSet.from_rules(rules)
    return RuleSet.parse(rules)
