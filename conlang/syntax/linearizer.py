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

    def _core_tokens(
        self, clause: Clause, omit: Role | None = None, matrix: bool = True
    ) -> list[GlossedWord]:
        """The ordered S/V/O (+ recipient, obliques) tokens of a clause, optionally omitting
        one role (used to leave a gap when the clause is relativized). ``matrix`` is False for
        an embedded clause, where verb-second does not apply (V2 is a main-clause phenomenon)."""
        if clause.indirect_object is not None and clause.object is None:
            raise ValueError("an indirect object (recipient) requires a direct object")
        if clause.topic is not None and not any(pp is clause.topic for pp in clause.obliques):
            raise ValueError("clause.topic must be one of the clause's obliques")
        transitive = clause.is_transitive
        constituents: dict[str, list[GlossedWord]] = {"V": self._verb_group(clause)}
        drop_subject = clause.is_imperative or self._pro_drops_subject(clause)
        if not drop_subject and omit is not Role.SUBJECT:
            constituents["S"] = self._argument(
                clause.subject, self._core_case(Role.SUBJECT, transitive)
            )
        if transitive:
            recipient_case, theme_case = self._object_cases(clause, transitive)
            if omit is not Role.OBJECT:  # gapping the theme leaves the recipient in place
                constituents["O"] = self._argument(clause.object, theme_case)
            if clause.indirect_object is not None:
                constituents["IO"] = self._argument(clause.indirect_object, recipient_case)

        # A topicalized oblique is fronted to clause-initial position; the rest stay final.
        # Topicalization is main-clause only (like V2), and a content-question wh-fronting takes
        # precedence over it (only one constituent can occupy the pre-verbal position).
        fronted_topic = clause.topic if matrix else None
        if (fronted_topic is not None and self.params.verb_second
                and clause.questioned is not None and self.params.wh_fronting):
            fronted_topic = None  # the wh-word fronts; this oblique stays clause-final
        topic = self._adpositional_phrase(fronted_topic) if fronted_topic is not None else []

        if matrix and self.params.verb_second and not clause.is_imperative and "V" in constituents:
            if topic:
                # the fronted oblique is the V2 first constituent, so the verb comes second
                order = ["V", *(s for s in ("S", "IO", "O") if s in constituents)]
            else:
                order = self._verb_second_order(clause, constituents)
        else:
            order = list(self.params.basic_order.sequence)
            # The recipient sits immediately before the theme (a common default; case, not
            # position, distinguishes them). Inserted before wh-fronting so questioning the
            # theme moves only the theme, leaving the recipient in situ.
            if "IO" in constituents:
                order.insert(order.index("O"), "IO")
            # A content (wh-) question may move the questioned constituent (S or O) to the front.
            if clause.questioned is not None and self.params.wh_fronting:
                q_slot = "S" if clause.questioned is Role.SUBJECT else "O"
                if q_slot in order:
                    order.remove(q_slot)
                    order.insert(0, q_slot)

        words: list[GlossedWord] = list(topic)  # the fronted topic, if any, comes first
        for slot in order:
            if slot in constituents:
                words.extend(constituents[slot])
        # The remaining obliques are placed clause-finally (a simplification); their *internal*
        # adposition order does follow the language's pre/postposition parameter.
        for pp in clause.obliques:
            if pp is not fronted_topic:
                words.extend(self._adpositional_phrase(pp))
        return words

    def _verb_second_order(self, clause: Clause, constituents: dict) -> list[str]:
        """The constituent order for a verb-second main clause: one fronted constituent, then
        the finite verb, then the rest (subject, recipient, object). A polar (yes/no) question
        is verb-first (V1, German "Siehst du …?"); a content question fronts the wh-word; an
        ordinary declarative fronts the subject (the unmarked topic).

        A fronted oblique topic is handled upstream in :meth:`_core_tokens` (which overrides
        this method), so here only the subject or a questioned object fronts. Simplifications:
        no plain-object topicalization (the other common German first constituent), and the
        post-verbal remainder is a fixed subject-recipient-object order rather than the base
        order's midfield. Embedded clauses are out of scope (handled by ``matrix=False``)."""
        present = [s for s in ("S", "IO", "O") if s in constituents]
        if clause.mood == "interrogative" and clause.questioned is None:
            return ["V", *present]  # polar question: verb-first
        front = "O" if (clause.questioned is Role.OBJECT and self.params.wh_fronting) else "S"
        if front not in present:
            front = present[0] if present else "S"  # defensive: chosen front must exist
        return [front, "V", *[s for s in present if s != front]]

    def _pro_drops_subject(self, clause: Clause) -> bool:
        """True if a pronominal subject should be left null (pro-drop).

        Only a pronoun (a subject NP with a person) can drop, never a questioned subject,
        and only when the verb's agreement marks both person and number — so the dropped
        argument is recoverable from the verb (the rich-agreement licensing condition). The
        verb must agree with the *subject*: under ergative alignment a transitive verb agrees
        with the absolutive object instead, so its subject cannot be recovered and is kept. A
        verb-second clause keeps the subject too — V2 needs an overt first constituent.
        """
        if not self.params.pro_drop or self.params.verb_second:
            return False
        if clause.questioned is Role.SUBJECT:
            return False
        if getattr(clause.subject, "person", None) is None:
            return False
        if clause.is_transitive and self.params.alignment is Alignment.ERGATIVE_ABSOLUTIVE:
            return False  # the verb agrees with the object here, not the subject
        marked = self._marked(clause.verb.word_class)
        return "person" in marked and "number" in marked

    def _object_cases(self, clause: Clause, transitive: bool) -> tuple[str, str]:
        """The (recipient, theme) cases. For a monotransitive the theme takes the normal
        object case and the recipient slot is unused. For a ditransitive the two split by
        the language's alignment: indirective = dative recipient + object-case theme;
        secundative = object-case recipient + dative theme."""
        object_case = self._core_case(Role.OBJECT, transitive)
        theme_case = self._dom_adjusted(clause.object, object_case)
        if clause.indirect_object is None:
            return object_case, theme_case
        if self.params.ditransitive is DitransitiveAlignment.SECUNDATIVE:
            return object_case, "dat"   # recipient is the primary object; theme set apart
        return "dat", theme_case        # indirective: recipient set apart in the dative

    def _dom_adjusted(self, obj, case: str) -> str:
        """Differential object marking: a low-prominence object — here an indefinite or bare
        one — is left in the unmarked case even where the alignment would mark it accusative.
        Definite objects keep the overt accusative (the marked/prominent half of the split).

        A bare object (no definiteness) and an interrogative object (a wh-word, which has no
        definiteness) both count as non-prominent and are left unmarked. A coordinated object
        is prominent only if every conjunct is definite.
        """
        if (case == "acc"
                and self.params.differential_object_marking
                and not self._is_prominent_object(obj)):
            return "nom"
        return case

    def _is_prominent_object(self, obj) -> bool:
        if isinstance(obj, Coordination):
            return all(getattr(c, "definiteness", None) == "def" for c in obj.conjuncts)
        return getattr(obj, "definiteness", None) == "def"

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

    def _number_for(self, word_class: str, number: str) -> str:
        """Coerce a requested number to one the word class actually has, so a value the
        language lacks (e.g. ``dual`` in a sg/pl language) degrades to the base consistently
        in both the inflected form and the gloss. A class that doesn't mark number passes the
        value through (it has no overt effect either way)."""
        paradigm = self.morphology.paradigms.get(word_class)
        if paradigm is not None:
            for cat in paradigm.marked:
                if cat.name == "number":
                    return number if number in cat.values else cat.base
        return number

    def _noun_phrase(self, np: NounPhrase, case: str) -> list[GlossedWord]:
        if isinstance(np, Coordination):
            # Coordination is supported only as a core argument (see _argument); a coordinated
            # possessor or relative-clause head is out of scope — fail clearly, not obscurely.
            raise TypeError(
                "coordination is only supported in subject/object position, not as a "
                "possessor or relative-clause head"
            )
        # An article language realizes definiteness with a separate determiner, so the noun
        # itself is not inflected for it (no double-marking). A suffixing language binds the
        # *definite* article onto the noun as a postnominal enclitic (Scandinavian hus-et)
        # rather than placing it as a free prenominal word; the indefinite stays prenominal.
        # Simplification: the enclitic always hosts on the noun head — matching Scandinavian,
        # but not the Romanian/Bulgarian pattern where a prenominal adjective hosts it instead
        # (frumoas-a carte), nor the Scandinavian double definiteness with adjectives.
        article = self._article(np.definiteness)
        suffixing = (article is not None and self.params.suffixed_article
                     and np.definiteness == "def")
        bundle_def = None if article is not None else np.definiteness
        number = self._number_for(np.head.word_class, np.number)
        gender = np.head.gender  # the noun's lexical gender (inflects the noun if it marks gender)
        noun_bundle = FeatureBundle.of(
            **_drop_none(case=case, number=number, definiteness=bundle_def, gender=gender)
        )
        marked = self._marked(np.head.word_class)
        gloss = np.head.gloss + _grammatical_tags(marked, number, case, bundle_def)
        # A 1st-person pronoun in a clusivity-marking language shows its inclusive/exclusive
        # value (a separate 'we' word, or just disambiguating the gloss). The gate is whether
        # the language HAS the contrast (verb marks clusivity), not whether the verb realizes
        # it in this clause — so the tag stays on the pronoun even where verb agreement drops
        # clusivity (ergative-transitive, singular). The pronoun's reference is inherently
        # in/exclusive regardless of agreement, so tagging it there is intentional, not a leak.
        if (np.person == "1" and getattr(np, "clusivity", None) in ("inclusive", "exclusive")
                and "clusivity" in self._marked("verb")):
            gloss += ".INCL" if np.clusivity == "inclusive" else ".EXCL"
        noun = self._inflected_word(np.head, noun_bundle, gloss)
        if suffixing:
            noun = self._enclitic(noun, article)  # hus + et -> huset (one bound word)

        tokens: list[GlossedWord] = [noun]
        if np.adjective is not None:
            # Adjectives agree with their head noun in number, gender, and case where marked.
            adj_number = self._number_for(np.adjective.word_class, np.number)
            adj_bundle = FeatureBundle.of(**_drop_none(number=adj_number, case=case, gender=gender))
            adj_marked = self._marked(np.adjective.word_class)
            adj_gloss = np.adjective.gloss + _grammatical_tags(adj_marked, adj_number, case, None)
            adj = self._inflected_word(np.adjective, adj_bundle, adj_gloss)
            tokens = _place(self.params.adjective, adj, tokens)
        if np.genitive is not None:
            # The genitive (possessor) is placed *outside* the adjective — the common
            # cross-linguistic stacking — because it is applied after the adjective.
            gen = self._noun_phrase(np.genitive, "gen")
            tokens = _place(self.params.genitive, gen, tokens)
        if np.relative is not None:
            tokens = _place(self.params.relative, self._relative_clause(np.relative), tokens)
        if article is not None and not suffixing:
            tokens = [article, *tokens]  # the determiner sits at the noun phrase's left edge
        return tokens

    def _enclitic(self, host: GlossedWord, clitic: GlossedWord) -> GlossedWord:
        """Bind a clitic onto its host noun as a single suffixed word (hus + et -> huset).

        The article's phonology and gloss tag are simply concatenated onto the (already
        inflected) noun, modelling the Scandinavian/Romanian suffixed definite article as one
        orthographic word — distinct from a free-standing determiner. No morphophonological
        fusion at the seam is applied; that would be the job of the sound-change layer.
        """
        return GlossedWord(
            host.roman + clitic.roman, host.ipa + clitic.ipa, host.gloss + "-" + clitic.gloss
        )

    def _article(self, definiteness: str | None) -> GlossedWord | None:
        """A free article word for a determiner language, else None.

        Definite and indefinite articles commonly grammaticalize from the demonstrative and
        the numeral 'one'; this reuses those full words (supplied as the ``art_def``/
        ``art_indef`` particles) rather than a reduced/clitic form. Returns None when the
        language has no free articles, the NP is bare, or the particle is unavailable — in
        which case the caller falls back to the morphological definiteness affix.
        """
        if not self.params.articles or definiteness not in ("def", "indef"):
            return None
        if definiteness == "def":
            return self._particle("art_def", "DEF")
        return self._particle("art_indef", "INDEF")

    def _relative_clause(self, rc: RelativeClause) -> list[GlossedWord]:
        """The embedded clause with the head's role left as a gap.

        A postnominal relative takes a relativizer (gap + complementizer, like English
        "that"); a prenominal one is participial cross-linguistically (Japanese, Turkish)
        and takes none.
        """
        inner = self._core_tokens(rc.clause, omit=rc.role, matrix=False)  # V2 is main-clause only
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
        ergative = clause.is_transitive and self.params.alignment is Alignment.ERGATIVE_ABSOLUTIVE
        agr = clause.object if ergative else clause.subject
        # Imperative addresses the listener (2nd person); otherwise the verb agrees with the
        # subject's person — a pronoun carries 1/2/3, a full NP defaults to 3rd person.
        person = "2" if clause.is_imperative else (getattr(agr, "person", None) or "3")
        number = self._number_for(clause.verb.word_class, agr.number)
        mood = "imperative" if clause.is_imperative else "indicative"
        polarity = "negative" if self._verbal_negation(clause) else "affirmative"
        marked = self._marked(clause.verb.word_class)
        # Object (polypersonal) agreement: cross-reference the *other* core argument — the
        # object under nom-acc, the agent under ergative (where the primary slot is the object).
        obj_person = obj_number = None
        if clause.is_transitive and {"object_person", "object_number"} & marked:
            other = clause.subject if ergative else clause.object
            obj_person = getattr(other, "person", None) or "3"
            obj_number = "pl" if getattr(other, "number", "sg") != "sg" else "sg"
        # Clusivity is a property of a 1st-person NON-SINGULAR subject ("we" inclusive vs
        # exclusive): vacuous for any other person/number, and only coherent when the verb's
        # primary agreement is the subject (so it's suppressed under ergative alignment, where
        # a transitive verb agrees with the object instead).
        clusivity = None
        if ("clusivity" in marked and agr is clause.subject and not clause.is_imperative
                and getattr(clause.subject, "person", None) == "1"
                and getattr(clause.subject, "number", "sg") != "sg"):
            clusivity = getattr(clause.subject, "clusivity", None) or "exclusive"
        bundle = FeatureBundle.of(
            tense=clause.tense, person=person, number=number, mood=mood, polarity=polarity,
            **_drop_none(clusivity=clusivity, object_person=obj_person, object_number=obj_number),
        )
        gloss = clause.verb.gloss + _verb_tags(
            marked, person, number, clause.tense, mood, polarity, obj_person, obj_number, clusivity
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
            segments = paradigm.inflect(
                lexeme.root, bundle, lexeme.inflection_class, lexeme.suppletive_stems
            )
        roman = self.romanizer.romanize([list(segments)])
        ipa = "".join(s.ipa for s in segments)
        return GlossedWord(roman, ipa, gloss)


# --- Module helpers -----------------------------------------------------------------
# Leipzig number-value tags (singular is the unmarked default, left unglossed). The two call
# sites use different `.get` fallbacks deliberately: the noun tagger gates out singular before
# the lookup so its fallback is the generic plural ("PL"); the verb tagger renders singular too,
# so its fallback is "SG".
_NUMBER_TAG = {"pl": "PL", "dual": "DU", "trial": "TRI", "paucal": "PAUC"}


def _grammatical_tags(marked: set[str], number: str, case: str, definiteness: str | None) -> str:
    """Leipzig-style tag suffix, emitting a tag only for categories the language marks.

    This keeps the gloss honest: a caseless language never shows ``.ACC``, a numberless
    one never shows ``.PL``. The unmarked nominative is conventionally left unglossed.
    """
    tags = []
    if "number" in marked and number not in (None, "sg"):
        tags.append(_NUMBER_TAG.get(number, "PL"))
    if "case" in marked and case and case != "nom":
        tags.append(case.upper())
    if "definiteness" in marked and definiteness == "def":
        tags.append("DEF")
    elif "definiteness" in marked and definiteness == "indef":
        tags.append("INDEF")
    return "." + ".".join(tags) if tags else ""


def _verb_tags(
    marked: set[str], person: str, number: str, tense: str, mood: str, polarity: str,
    object_person: str | None = None, object_number: str | None = None,
    clusivity: str | None = None,
) -> str:
    """Agreement/tense/mood/polarity gloss for the verb, gated on what it actually marks.

    Object (polypersonal) agreement is written after the primary agreement with a ``>``
    (primary-controller > cross-referenced argument). Under nominative-accusative that reads
    as agent>patient — ``see.1SG>3PL`` for "I see them"; under ergative the primary slot is the
    absolutive object, so it reads object>agent. A 1st-person subject's clusivity follows as
    ``.INCL``/``.EXCL`` (``see.1PL.INCL``).
    """
    agr = ""
    if "person" in marked:
        agr += person
    if "number" in marked:
        agr += _NUMBER_TAG.get(number, "SG")
    obj = ""
    if "object_person" in marked and object_person is not None:
        obj += object_person
    if "object_number" in marked and object_number is not None:
        obj += "PL" if object_number == "pl" else "SG"
    if obj:
        agr = f"{agr}>{obj}" if agr else f">{obj}"
    tags = [agr] if agr else []
    if "clusivity" in marked and clusivity is not None:
        tags.append("INCL" if clusivity == "inclusive" else "EXCL")
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
