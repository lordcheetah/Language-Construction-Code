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
from conlang.syntax.structure import Lexeme, NounPhrase, Clause, Coordination, Role
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
# v7: coordinator particles (and/or) were added to the lexicon.
# v8: a privative derivation can derive antonyms (bad/small/cold) from their base words.
# v9: paradigms may roll stem allomorphy (a bound stem), changing morphology RNG.
# v10: syntax rolls a ditransitive alignment, shifting the shared RNG before lexicon/writing.
# v11: syntax rolls a pro-drop parameter (another shared-RNG draw before lexicon/writing).
# v12: syntax rolls a free-article parameter (another shared-RNG draw before lexicon/writing).
# v13: the lexicon rolls a kinship system and adds kin terms (changes lexicon RNG/output).
# v14: the writing system rolls a layout direction (an rng draw before numerals are built).
# v15: morphology may roll a dual number (sg/dual/pl), changing morphology RNG.
# v16: syntax rolls differential object marking (another shared-RNG draw before lexicon/writing).
# v17: numerals may roll irregular/suppletive teens, changing numeral RNG/output.
# v18: derivations may be zero-marked (conversion), changing morphology RNG.
# v19: stem alternations may be affix-conditioned (before a vowel-initial suffix), changing RNG.
# v20: syntax rolls verb-second (another shared-RNG draw before lexicon/writing).
# v21: the verb may mark object agreement (object_person/number categories), changing morphology RNG.
# v22: nouns get a lexical gender that drives their inflection class (class<->gender link).
# v23: a stacked derivation (petrify <- stony <- stone) was added to the lexicon.
# v24: morphology may roll a paucal number value (alongside/instead of dual), changing RNG.
# v25: the verb may mark clusivity (a 1st-person inclusive/exclusive category), changing RNG.
# v26: morphology may roll a trial number value (requires a dual), changing RNG.
# v27: syntax rolls a suffixed-definite-article parameter (another shared-RNG draw before lexicon/writing).
# v28: a clusivity-marking language coins a separate inclusive 'we' pronoun (extra lexicon RNG draw).
# v29: a multi-class language may make a stem alternation class-bound (extra morphology RNG draw).
# v30: frequent words may roll a suppletive stem (go/went) — extra lexicon RNG draws before writing.
# v31: syntax may roll subject-aux-inversion questions + an AUX particle is coined (shifts lexicon RNG).
# v32: the concept inventory was expanded (new fields + colexification/derivation entries), changing the lexicon.
GENERATOR_VERSION = 32

# Person of the lexicon's PERSONAL pronouns, for subject-verb agreement and pro-drop
# licensing. Demonstratives (this/that) are deliberately excluded: they are 3rd person by
# default but resist null realization, so they are treated as ordinary full NPs.
_PRONOUN_PERSON = {"I": "1", "we": "1", "you": "2"}


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
        return Lexeme(entry.form, entry.concept.pos, gloss, entry.inflection_class,
                      entry.gender, entry.suppletive_stems)

    def _subject_lexeme(self, gloss: str, clusivity: str | None) -> Lexeme:
        """The subject's lexeme, choosing the separate inclusive 'we' when the subject is an
        inclusive first person and the language has that pronoun. The gloss stays "we" (the
        .INCL/.EXCL tag is added at linearization), so only the *form* differs from exclusive
        'we'. Falls back to the ordinary lookup for every other subject.

        Scope: the split is offered only in subject position (make_sentence exposes
        ``subject_clusivity`` but no object/recipient counterpart), so an inclusive 'we' used
        as an object or possessor falls back to the plain (exclusive) form. Subjects are the
        common case; extending it would just mean threading clusivity onto the other NPs."""
        if gloss == "we" and clusivity == "inclusive":
            incl = self.lexicon.get("we (incl)")
            if incl is not None:
                return Lexeme(incl.form, "noun", "we", incl.inflection_class, incl.gender,
                              incl.suppletive_stems)
        return self._lexeme(gloss, expect_pos="noun")

    def make_sentence(
        self,
        subject: str,
        verb: str,
        obj: str | None = None,
        *,
        subject_number: str = "sg",
        subject_definiteness: str | None = None,
        subject_adjective: str | None = None,
        subject_clusivity: str | None = None,
        object_number: str = "sg",
        object_definiteness: str | None = None,
        recipient: str | None = None,
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
        clause = self._build_clause(
            subject, verb, obj,
            subject_number=subject_number, subject_definiteness=subject_definiteness,
            subject_adjective=subject_adjective, subject_clusivity=subject_clusivity,
            object_number=object_number, object_definiteness=object_definiteness,
            recipient=recipient, tense=tense, negated=negated, mood=mood, question=question,
        )
        return self._linearizer().linearize(clause)

    def make_compound(self, *clauses: dict, coordinator: str = "and") -> Sentence:
        """Coordinate several clauses into one compound sentence ("… and …" / "… or …").

        Each clause is a dict of :meth:`make_sentence` keyword arguments (``subject``,
        ``verb`` and the rest). The coordinator ("and"/"or") joins them medially.
        """
        if coordinator not in ("and", "or"):
            raise ValueError("coordinator must be 'and' or 'or'")
        if len(clauses) < 2:
            raise ValueError("a compound sentence needs at least two clauses")
        built = [self._build_clause(**spec) for spec in clauses]
        return self._linearizer().linearize(Coordination(built, coordinator))

    def _linearizer(self) -> Linearizer:
        return Linearizer(
            self.syntax, self.morphology, self.romanizer, particles=self._particles()
        )

    def _build_clause(
        self,
        subject: str,
        verb: str,
        obj: str | None = None,
        *,
        subject_number: str = "sg",
        subject_definiteness: str | None = None,
        subject_adjective: str | None = None,
        subject_clusivity: str | None = None,
        object_number: str = "sg",
        object_definiteness: str | None = None,
        recipient: str | None = None,
        tense: str = "pres",
        negated: bool = False,
        mood: str = "declarative",
        question: str | None = None,
    ) -> Clause:
        if question is not None and question not in ("subject", "object"):
            raise ValueError("question must be 'subject', 'object', or None")
        if question is not None and mood == "imperative":
            raise ValueError("an imperative clause cannot also be a content question")
        if recipient is not None and obj is None:
            raise ValueError("a recipient (indirect object) needs a direct object too")
        subj = NounPhrase(
            self._subject_lexeme(subject, subject_clusivity),
            adjective=self._lexeme(subject_adjective, expect_pos="adjective")
            if subject_adjective else None,
            number=subject_number,
            definiteness=subject_definiteness,
            person=_PRONOUN_PERSON.get(subject),  # a pronoun subject carries its person
            clusivity=subject_clusivity,           # ...and, for "we", its clusivity
        )
        obj_np = None
        if obj is not None:
            obj_np = NounPhrase(
                self._lexeme(obj, expect_pos="noun"),
                number=object_number, definiteness=object_definiteness,
            )
        io_np = None
        if recipient is not None:
            io_np = NounPhrase(self._lexeme(recipient, expect_pos="noun"))
        questioned = {"subject": Role.SUBJECT, "object": Role.OBJECT}.get(question)
        return Clause(
            subj, self._lexeme(verb, expect_pos="verb"), obj_np,
            indirect_object=io_np,
            tense=tense, negated=negated, mood=mood, questioned=questioned,
        )

    def _particles(self) -> dict:
        """The language's grammatical particles (negator, yes/no marker, …) as Lexemes."""
        out = {}
        # Coordinators are keyed by their own gloss ("and"/"or") to match Coordination.coordinator.
        # Free articles reuse the demonstrative ("that" -> definite) and "one" (-> indefinite),
        # the usual grammaticalization sources, so they need no new lexemes.
        for key, gloss in (("neg", "not"), ("q", "Q"), ("rel", "REL"),
                           ("and", "and"), ("or", "or"),
                           ("art_def", "that"), ("art_indef", "one")):
            entry = self.lexicon.get(gloss)
            if entry is not None:
                out[key] = Lexeme(entry.form, "particle", gloss, entry.inflection_class)
        # The interrogative auxiliary is verb-class so it inflects (carries the verb's tense and
        # agreement) when it fronts in a subject–auxiliary-inversion question.
        aux = self.lexicon.get("AUX")
        if aux is not None:
            out["aux"] = Lexeme(aux.form, "verb", "AUX", aux.inflection_class)
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
            "writing": {
                "type": self.writing.type.value,
                "direction": self.writing.direction.value,
            },
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
                    "gender": e.gender,
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
