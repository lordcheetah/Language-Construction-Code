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
from conlang.syntax.parameters import SyntaxParameters, Side, Adposition, Alignment
from conlang.syntax.structure import Clause, NounPhrase, AdpositionalPhrase, Lexeme, Role


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
    ) -> None:
        self.params = params
        self.morphology = morphology
        self.romanizer = romanizer or Romanizer()

    # --- Public API ------------------------------------------------------------------
    def linearize(self, clause: Clause) -> Sentence:
        transitive = clause.is_transitive
        subject_case = self._core_case(Role.SUBJECT, transitive)
        constituents: dict[str, list[GlossedWord]] = {
            "S": self._noun_phrase(clause.subject, subject_case),
            "V": [self._verb(clause)],
        }
        if transitive:
            object_case = self._core_case(Role.OBJECT, transitive)
            constituents["O"] = self._noun_phrase(clause.object, object_case)

        words: list[GlossedWord] = []
        for slot in self.params.basic_order.sequence:
            if slot in constituents:
                words.extend(constituents[slot])
        # Obliques are placed clause-finally here (a simplification); their *internal*
        # adposition order does follow the language's pre/postposition parameter.
        for pp in clause.obliques:
            words.extend(self._adpositional_phrase(pp))
        return Sentence(tuple(words))

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
    def _marked(self, word_class: str) -> set[str]:
        paradigm = self.morphology.paradigms.get(word_class)
        return {c.name for c in paradigm.marked} if paradigm else set()

    def _noun_phrase(self, np: NounPhrase, case: str) -> list[GlossedWord]:
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
        return tokens

    def _adpositional_phrase(self, pp: AdpositionalPhrase) -> list[GlossedWord]:
        inner = self._noun_phrase(pp.np, "nom")  # oblique NP left in its base/unmarked form
        roman = self.romanizer.romanize([list(pp.adposition.root)])
        adp = GlossedWord(roman, pp.adposition.ipa, pp.relation or pp.adposition.gloss)
        if self.params.adposition is Adposition.PREPOSITION:
            return [adp, *inner]
        return [*inner, adp]

    def _verb(self, clause: Clause) -> GlossedWord:
        # Agreement controller: the absolutive argument under ergative alignment (object of
        # a transitive, else subject), otherwise the subject (nominative).
        if clause.is_transitive and self.params.alignment is Alignment.ERGATIVE_ABSOLUTIVE:
            agr = clause.object
        else:
            agr = clause.subject
        bundle = FeatureBundle.of(tense=clause.tense, person="3", number=agr.number)

        marked = self._marked(clause.verb.word_class)
        gloss = clause.verb.gloss + _verb_tags(marked, agr.number, clause.tense)
        return self._inflected_word(clause.verb, bundle, gloss)

    # --- Inflection helper -----------------------------------------------------------
    def _inflected_word(self, lexeme: Lexeme, bundle: FeatureBundle, gloss: str) -> GlossedWord:
        paradigm = self.morphology.paradigms.get(lexeme.word_class)
        if paradigm is None:
            segments: Sequence[Segment] = lexeme.root
        else:
            segments = paradigm.inflect(lexeme.root, bundle)
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


def _verb_tags(marked: set[str], number: str, tense: str) -> str:
    """Agreement/tense gloss for the verb, gated on what the verb actually marks."""
    agr = ""
    if "person" in marked:
        agr += "3"
    if "number" in marked:
        agr += "PL" if number == "pl" else "SG"
    tags = [agr] if agr else []
    if "tense" in marked and tense and tense != "pres":
        tags.append(tense.upper())
    return "." + ".".join(tags) if tags else ""


def _place(side: Side, new, anchor: list[GlossedWord]) -> list[GlossedWord]:
    """Place *new* (a token or list of tokens) before/after the *anchor* tokens."""
    new_list = new if isinstance(new, list) else [new]
    return [*new_list, *anchor] if side is Side.BEFORE else [*anchor, *new_list]


def _drop_none(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}
