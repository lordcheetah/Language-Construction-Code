"""Turn a clause into an ordered, inflected, glossed sentence.

This is where the stages meet. For each clause the linearizer:

1. assigns a core case to every argument from the language's **alignment** (see
   :func:`_core_case`),
2. inflects each noun and the verb with the Stage 3 **morphology** — applying case,
   number, definiteness, tense, and subject agreement, but only for the categories the
   language actually marks (a caseless language simply leans on word order instead),
3. orders modifiers within each noun phrase and the S/V/O constituents of the clause by
   the language's **word-order parameters**.

The result is a :class:`Sentence` of :class:`GlossedWord` tokens that can be printed as an
interlinear gloss.

Note on alignment: the morphology marks core arguments with an unmarked case (``nom``)
and a marked case (``acc``); alignment decides which argument receives the marked one
(the object under accusative alignment, the agent under ergative alignment). This reuses
the two-way core-case contrast rather than introducing separate erg/abs forms.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence

from conlang.phonology.features import Segment
from conlang.phonology.wordgen import Romanizer
from conlang.morphology.features import FeatureBundle
from conlang.morphology.generator import MorphologySystem
from conlang.syntax.parameters import (
    SyntaxParameters, Side, Adposition, Alignment, Negation, PolarQuestion,
    DitransitiveAlignment,
)
from conlang.syntax.structure import (
    Clause, NounPhrase, AdpositionalPhrase, RelativeClause, Coordination, Lexeme, Role,
)


def _display_width(text: str) -> int:
    """Visual width: count non-combining codepoints (so affricate tie-bars don't inflate it)."""
    return sum(1 for ch in text if not unicodedata.combining(ch))


def _ljust(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


@dataclass(frozen=True)
class GlossedWord:
    roman: str
    ipa: str
    gloss: str


@dataclass(frozen=True)
class Sentence:
    words: tuple[GlossedWord, ...]

    @property
    def text(self) -> str:
        return " ".join(w.roman for w in self.words)

    @property
    def ipa(self) -> str:
        return " ".join(w.ipa for w in self.words)

    def interlinear(self) -> str:
        """Three vertically-aligned lines: surface text, IPA, and gloss.

        Each word is one column whose width is the widest of its three cells, so the
        surface form, its IPA, and its gloss line up beneath one another.
        """
        cols = [(w.roman, f"/{w.ipa}/", w.gloss) for w in self.words]
        col_widths = [max(_display_width(cell) for cell in col) for col in cols]
        lines = []
        for line_idx in range(3):
            lines.append(
                "  ".join(_ljust(col[line_idx], col_widths[c]) for c, col in enumerate(cols))
            )
        return "\n".join(lines)


class Linearizer:
    def __init__(
        self,
        params: SyntaxParameters,
        morphology: MorphologySystem,
        romanizer: Romanizer | None = None,
        particles: dict[str, Lexeme] | None = None,
    ) -> None:
        self.params = params
        self.morphology = morphology
        self.romanizer = romanizer or Romanizer()
        # Grammatical function words by role: "neg" (negator), "q" (yes/no marker).
        self.particles = particles or {}

    # --- Public API ------------------------------------------------------------------
    def linearize(self, item: "Clause | Coordination") -> Sentence:
        if isinstance(item, Coordination):  # a compound sentence (coordinated clauses)
            return Sentence(tuple(self._coordinate(item, self._clause_tokens)))
        return Sentence(tuple(self._clause_tokens(item)))

    def _clause_tokens(self, clause: Clause) -> list[GlossedWord]:
        words = self._core_tokens(clause)
        return self._mark_polar_question(clause, words)

    def _coordinate(self, coord: Coordination, render) -> list[GlossedWord]:
        """Join conjuncts with the medial coordinator particle (asyndetic if it's absent)."""
        conj = self._particle(coord.coordinator, coord.coordinator.upper())
        words: list[GlossedWord] = []
        for i, item in enumerate(coord.conjuncts):
            if i > 0 and conj is not None:
                words.append(conj)
            words.extend(render(item))
        return words

    def _core_tokens(self, clause: Clause, omit: Role | None = None) -> list[GlossedWord]:
        """The ordered S/V/O (+ recipient, obliques) tokens of a clause, optionally omitting
        one role (used to leave a gap when the clause is relativized)."""
        if clause.indirect_object is not None and clause.object is None:
            raise ValueError("an indirect object (recipient) requires a direct object")
        transitive = clause.is_transitive
        constituents: dict[str, list[GlossedWord]] = {"V": self._verb_group(clause)}
        if not clause.is_imperative and omit is not Role.SUBJECT:
            constituents["S"] = self._argument(
                clause.subject, self._core_case(Role.SUBJECT, transitive)
            )
        if transitive:
            recipient_case, theme_case = self._object_cases(clause, transitive)
            if omit is not Role.OBJECT:  # gapping the theme leaves the recipient in place
                constituents["O"] = self._argument(clause.object, theme_case)
            if clause.indirect_object is not None:
                constituents["IO"] = self._argument(clause.indirect_object, recipient_case)

        order = list(self.params.basic_order.sequence)
        # The recipient sits immediately before the theme (a common default; morphological
        # case, not position, distinguishes them). Inserted before wh-fronting so that
        # questioning the theme moves only the theme, leaving the recipient in situ.
        if "IO" in constituents:
            order.insert(order.index("O"), "IO")
        # A content (wh-) question may move the questioned constituent (S or O) to the front.
        if clause.questioned is not None and self.params.wh_fronting:
            q_slot = "S" if clause.questioned is Role.SUBJECT else "O"
            if q_slot in order:
                order.remove(q_slot)
                order.insert(0, q_slot)

        words: list[GlossedWord] = []
        for slot in order:
            if slot in constituents:
                words.extend(constituents[slot])
        # Obliques are placed clause-finally here (a simplification); their *internal*
        # adposition order does follow the language's pre/postposition parameter.
        for pp in clause.obliques:
            words.extend(self._adpositional_phrase(pp))
        return words

    def _object_cases(self, clause: Clause, transitive: bool) -> tuple[str, str]:
        """The (recipient, theme) cases. For a monotransitive the theme takes the normal
        object case and the recipient slot is unused. For a ditransitive the two split by
        the language's alignment: indirective = dative recipient + object-case theme;
        secundative = object-case recipient + dative theme."""
        object_case = self._core_case(Role.OBJECT, transitive)
        if clause.indirect_object is None:
            return object_case, object_case
        if self.params.ditransitive is DitransitiveAlignment.SECUNDATIVE:
            return object_case, "dat"   # recipient is the primary object; theme set apart
        return "dat", object_case       # indirective: recipient set apart in the dative

    # --- Case / alignment ------------------------------------------------------------
    def _core_case(self, role: Role, transitive: bool) -> str:
        if not transitive:
            unmarked = True  # intransitive subject S is always the unmarked core case
        elif role is Role.SUBJECT:  # transitive agent A
            unmarked = self.params.alignment is Alignment.NOMINATIVE_ACCUSATIVE
        else:  # transitive patient O
            unmarked = self.params.alignment is Alignment.ERGATIVE_ABSOLUTIVE
        return "nom" if unmarked else "acc"

    # --- Noun phrases ----------------------------------------------------------------
    def _argument(self, arg: "NounPhrase | Coordination", case: str) -> list[GlossedWord]:
        """A core argument: a single noun phrase, or several coordinated by "and"/"or".

        Every conjunct takes the same core case; verb agreement reads the coordination's
        resolved number (plural for "and") via :attr:`Coordination.number`.
        """
        if isinstance(arg, Coordination):
            return self._coordinate(arg, lambda np: self._noun_phrase(np, case))
        return self._noun_phrase(arg, case)

    def _marked(self, word_class: str) -> set[str]:
        paradigm = self.morphology.paradigms.get(word_class)
        return {c.name for c in paradigm.marked} if paradigm else set()

    def _noun_phrase(self, np: NounPhrase, case: str) -> list[GlossedWord]:
        if isinstance(np, Coordination):
            # Coordination is supported only as a core argument (see _argument); a coordinated
            # possessor or relative-clause head is out of scope — fail clearly, not obscurely.
            raise TypeError(
                "coordination is only supported in subject/object position, not as a "
                "possessor or relative-clause head"
            )
        noun_bundle = FeatureBundle.of(
            **_drop_none(case=case, number=np.number, definiteness=np.definiteness)
        )
        marked = self._marked(np.head.word_class)
        gloss = np.head.gloss + _grammatical_tags(marked, np.number, case, np.definiteness)
        noun = self._inflected_word(np.head, noun_bundle, gloss)

        tokens: list[GlossedWord] = [noun]
        if np.adjective is not None:
            # Adjectives agree with their head noun in number (and case where marked).
            adj_bundle = FeatureBundle.of(number=np.number, case=case)
            adj_marked = self._marked(np.adjective.word_class)
            adj_gloss = np.adjective.gloss + _grammatical_tags(adj_marked, np.number, case, None)
            adj = self._inflected_word(np.adjective, adj_bundle, adj_gloss)
            tokens = _place(self.params.adjective, adj, tokens)
        if np.genitive is not None:
            # The genitive (possessor) is placed *outside* the adjective — the common
            # cross-linguistic stacking — because it is applied after the adjective.
            gen = self._noun_phrase(np.genitive, "gen")
            tokens = _place(self.params.genitive, gen, tokens)
        if np.relative is not None:
            tokens = _place(self.params.relative, self._relative_clause(np.relative), tokens)
        return tokens

    def _relative_clause(self, rc: RelativeClause) -> list[GlossedWord]:
        """The embedded clause with the head's role left as a gap.

        A postnominal relative takes a relativizer (gap + complementizer, like English
        "that"); a prenominal one is participial cross-linguistically (Japanese, Turkish)
        and takes none.
        """
        inner = self._core_tokens(rc.clause, omit=rc.role)
        if self.params.relative is Side.AFTER:
            rel = self._particle("rel", "REL")
            if rel is not None:
                return [rel, *inner]
        return inner

    def _adpositional_phrase(self, pp: AdpositionalPhrase) -> list[GlossedWord]:
        inner = self._noun_phrase(pp.np, "nom")  # oblique NP left in its base/unmarked form
        roman = self.romanizer.romanize([list(pp.adposition.root)])
        adp = GlossedWord(roman, pp.adposition.ipa, pp.relation or pp.adposition.gloss)
        if self.params.adposition is Adposition.PREPOSITION:
            return [adp, *inner]
        return [*inner, adp]

    def _verb_group(self, clause: Clause) -> list[GlossedWord]:
        """The verb, plus a negative particle when negation isn't marked on the verb."""
        verb = self._verb(clause)
        if not clause.negated or self._verbal_negation(clause):
            return [verb]  # verbal negation is already in the verb's bundle + gloss
        neg = self._particle("neg", "NEG")
        if neg is None:
            # Negation can be realized neither on the verb nor as a particle; keep it
            # visible in the gloss rather than silently dropping it.
            return [GlossedWord(verb.roman, verb.ipa, verb.gloss + ".NEG")]
        if self.params.negation is Negation.PARTICLE_AFTER_VERB:
            return [verb, neg]
        return [neg, verb]  # before the verb (also the fallback position)

    def _verb(self, clause: Clause) -> GlossedWord:
        # Agreement controller: the absolutive argument under ergative alignment (object of
        # a transitive, else subject), otherwise the subject (nominative).
        if clause.is_transitive and self.params.alignment is Alignment.ERGATIVE_ABSOLUTIVE:
            agr = clause.object
        else:
            agr = clause.subject
        # Imperative addresses the listener: 2nd person, number from the (dropped) subject.
        person = "2" if clause.is_imperative else "3"
        mood = "imperative" if clause.is_imperative else "indicative"
        polarity = "negative" if self._verbal_negation(clause) else "affirmative"
        bundle = FeatureBundle.of(
            tense=clause.tense, person=person, number=agr.number,
            mood=mood, polarity=polarity,
        )
        marked = self._marked(clause.verb.word_class)
        gloss = clause.verb.gloss + _verb_tags(
            marked, person, agr.number, clause.tense, mood, polarity
        )
        return self._inflected_word(clause.verb, bundle, gloss)

    # --- Sentence-type helpers -------------------------------------------------------
    def _verbal_negation(self, clause: Clause) -> bool:
        if not clause.negated or "polarity" not in self._marked(clause.verb.word_class):
            return False
        # Mark on the verb when the language chose verbal negation, or as a fallback when
        # no negator particle is available (so a VERBAL roll on a polarity-less verb still
        # degrades gracefully to a particle elsewhere).
        return self.params.negation is Negation.VERBAL or "neg" not in self.particles

    def _mark_polar_question(self, clause: Clause, words: list[GlossedWord]) -> list[GlossedWord]:
        # Polar marking applies only to yes/no questions, not content (wh-) questions.
        if clause.mood != "interrogative" or clause.questioned is not None:
            return words
        q = self._particle("q", "Q")
        if q is None or self.params.polar_question is PolarQuestion.INTONATION:
            return words  # intonation only: no overt marker
        if self.params.polar_question is PolarQuestion.PARTICLE_INITIAL:
            return [q, *words]
        return [*words, q]  # clause-final

    def _particle(self, key: str, gloss: str) -> GlossedWord | None:
        lexeme = self.particles.get(key)
        if lexeme is None:
            return None
        roman = self.romanizer.romanize([list(lexeme.root)])
        return GlossedWord(roman, lexeme.ipa, gloss)

    # --- Inflection helper -----------------------------------------------------------
    def _inflected_word(self, lexeme: Lexeme, bundle: FeatureBundle, gloss: str) -> GlossedWord:
        paradigm = self.morphology.paradigms.get(lexeme.word_class)
        if paradigm is None:
            segments: Sequence[Segment] = lexeme.root
        else:
            segments = paradigm.inflect(lexeme.root, bundle, lexeme.inflection_class)
        roman = self.romanizer.romanize([list(segments)])
        ipa = "".join(s.ipa for s in segments)
        return GlossedWord(roman, ipa, gloss)


# --- Module helpers -----------------------------------------------------------------
def _grammatical_tags(marked: set[str], number: str, case: str, definiteness: str | None) -> str:
    """Leipzig-style tag suffix, emitting a tag only for categories the language marks.

    This keeps the gloss honest: a caseless language never shows ``.ACC``, a numberless
    one never shows ``.PL``. The unmarked nominative is conventionally left unglossed.
    """
    tags = []
    if "number" in marked and number == "pl":
        tags.append("PL")
    if "case" in marked and case and case != "nom":
        tags.append(case.upper())
    if "definiteness" in marked and definiteness == "def":
        tags.append("DEF")
    elif "definiteness" in marked and definiteness == "indef":
        tags.append("INDEF")
    return "." + ".".join(tags) if tags else ""


def _verb_tags(marked: set[str], person: str, number: str, tense: str, mood: str, polarity: str) -> str:
    """Agreement/tense/mood/polarity gloss for the verb, gated on what it actually marks."""
    agr = ""
    if "person" in marked:
        agr += person
    if "number" in marked:
        agr += "PL" if number == "pl" else "SG"
    tags = [agr] if agr else []
    if "tense" in marked and tense and tense != "pres":
        tags.append(tense.upper())
    if "mood" in marked and mood == "imperative":
        tags.append("IMP")
    if "polarity" in marked and polarity == "negative":
        tags.append("NEG")
    return "." + ".".join(tags) if tags else ""


def _place(side: Side, new, anchor: list[GlossedWord]) -> list[GlossedWord]:
    """Place *new* (a token or list of tokens) before/after the *anchor* tokens."""
    new_list = new if isinstance(new, list) else [new]
    return [*new_list, *anchor] if side is Side.BEFORE else [*anchor, *new_list]


def _drop_none(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}
