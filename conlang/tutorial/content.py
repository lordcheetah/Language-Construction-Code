"""The tutorial's steps: what each stage teaches and the choices it offers.

Each :class:`Step` carries a short lesson (grounded in the Language Construction Kit), a
set of :class:`Choice` options (always including a random roll), a function that applies
the chosen option to the :class:`LanguageBuilder`, and a function that reports what was
produced. Summary functions never consume the builder's RNG — sample words use a separate
display RNG — so a run is reproducible whether or not anyone reads the summaries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from conlang.phonology.wordgen import WordGenerator
from conlang.morphology.features import Typology
from conlang.syntax.parameters import WordOrder
from conlang.writing.system import WritingSystemType
from conlang.lexicon.lexicon import Etymology
from conlang.tutorial.builder import LanguageBuilder


@dataclass(frozen=True)
class Choice:
    key: str
    label: str
    note: str = ""


@dataclass
class Step:
    id: str
    title: str
    teaching: str
    choices: tuple[Choice, ...]
    apply: Callable[[LanguageBuilder, str], None]
    summary: Callable[[LanguageBuilder], str]


_RANDOM = Choice("random", "Roll a random one", "let the generator pick something plausible")


def _sample_words(builder: LanguageBuilder, n: int = 4) -> str:
    """A few example words, generated off a display RNG so the builder stays untouched."""
    gen = WordGenerator(builder.phonotactics, builder.romanizer)
    rng = random.Random((builder.seed or 0) ^ 0x5151)
    return ", ".join(gen.word(rng, min_syllables=1, max_syllables=3).roman for _ in range(n))


# --- Step builders ------------------------------------------------------------------
def _phonology_step() -> Step:
    def apply(b: LanguageBuilder, key: str) -> None:
        b.roll_inventory(key if key != "random" else None)

    def summary(b: LanguageBuilder) -> str:
        inv = b.inventory
        cons = " ".join(c.ipa for c in inv.consonants)
        vows = " ".join(v.ipa for v in inv.vowels)
        return (f"  -> {len(inv.consonants)} consonants / {len(inv.vowels)} vowels\n"
                f"     {cons}\n     {vows}")

    return Step(
        id="phonology",
        title="1. Sounds (phonology)",
        teaching=(
            "Every language is built from a set of distinct speech sounds, its phonemes. "
            "Consonants are organised by where and how they are made (place and manner), "
            "vowels by tongue height and position. A larger inventory packs more contrasts "
            "into shorter words; a smaller one leans on longer words. Most languages sit "
            "around 20-25 consonants and 5 vowels."
        ),
        choices=(
            Choice("small", "Small inventory", "~12 consonants, 3 vowels — sparse and economical"),
            Choice("medium", "Medium inventory", "~20 consonants, 5 vowels — the common middle"),
            Choice("large", "Large inventory", "~30 consonants, 8 vowels — many fine contrasts"),
            _RANDOM,
        ),
        apply=apply,
        summary=summary,
    )


def _phonotactics_step() -> Step:
    def apply(b: LanguageBuilder, key: str) -> None:
        b.set_phonotactics(key if key != "random" else None)

    def summary(b: LanguageBuilder) -> str:
        templates = ", ".join(str(t) for t in b.phonotactics.templates)
        return f"  -> syllable shapes: {templates}\n     sample words: {_sample_words(b)}"

    return Step(
        id="phonotactics",
        title="2. Syllables (phonotactics)",
        teaching=(
            "Phonotactics is the rule for how sounds combine into syllables, written as "
            "templates like CV or (C)(C)V(C), where parentheses mark an optional slot. A "
            "strict CV language sounds open and flowing; one that allows consonant clusters "
            "sounds denser. Well-formed clusters rise in sonority toward the vowel and fall "
            "away from it."
        ),
        choices=(
            Choice("simple", "Simple (CV)", "open syllables only, like Hawaiian or Japanese"),
            Choice("moderate", "Moderate ((C)V(C))", "optional onset and coda, a single consonant each"),
            Choice("complex", "Complex (clusters)", "consonant clusters allowed, like English"),
            _RANDOM,
        ),
        apply=apply,
        summary=summary,
    )


def _morphology_step() -> Step:
    _TYPOLOGY = {
        "isolating": Typology.ISOLATING,
        "agglutinative": Typology.AGGLUTINATIVE,
        "fusional": Typology.FUSIONAL,
    }

    def apply(b: LanguageBuilder, key: str) -> None:
        b.roll_morphology(_TYPOLOGY.get(key))

    return Step(
        id="morphology",
        title="3. Word structure (morphology)",
        teaching=(
            "Morphology is how words carry grammar like number, tense, or case. An "
            "isolating language keeps words simple and leans on separate words and order. "
            "An agglutinative one stacks clear affixes, one tidy piece per meaning. A "
            "fusional one fuses several meanings into a single ending, so one suffix can "
            "mark, say, plural and accusative at once."
        ),
        choices=(
            Choice("isolating", "Isolating", "very few affixes; grammar from word order (Mandarin, Vietnamese)"),
            Choice("agglutinative", "Agglutinative", "stacked, transparent affixes (Turkish, Japanese)"),
            Choice("fusional", "Fusional", "fused, portmanteau endings (Latin, Russian)"),
            _RANDOM,
        ),
        apply=apply,
        summary=lambda b: "  -> " + b.morphology.summary().replace("\n", "\n     "),
    )


def _syntax_step() -> Step:
    _ORDER = {"sov": WordOrder.SOV, "svo": WordOrder.SVO, "vso": WordOrder.VSO}

    def apply(b: LanguageBuilder, key: str) -> None:
        b.set_syntax(_ORDER.get(key))

    return Step(
        id="syntax",
        title="4. Word order (syntax)",
        teaching=(
            "Syntax sets the order of Subject, Object, and Verb, plus a cascade of related "
            "choices. SOV and SVO are by far the most common. Word order tends to be "
            "harmonic: a verb-object language usually puts prepositions before nouns and the "
            "noun before its genitive, while an object-verb language mirrors all of that. The "
            "tutorial derives those correlates from your choice."
        ),
        choices=(
            Choice("sov", "Subject-Object-Verb", "the world's most common order (Japanese, Turkish)"),
            Choice("svo", "Subject-Verb-Object", "like English and Mandarin"),
            Choice("vso", "Verb-Subject-Object", "verb first (Irish, Arabic)"),
            _RANDOM,
        ),
        apply=apply,
        summary=lambda b: "  -> \n" + "\n".join("     " + ln for ln in b.syntax.describe().splitlines()),
    )


def _lexicon_step() -> Step:
    def summary(b: LanguageBuilder) -> str:
        lex = b.lexicon
        der = len(lex.of_etymology(Etymology.DERIVED))
        comp = len(lex.of_etymology(Etymology.COMPOUND))
        colex = len(lex.of_etymology(Etymology.COLEXIFIED))
        samples = ", ".join(
            f"{g}={lex.get(g).roman}" for g in ("water", "fire", "see", "big") if lex.get(g)
        )
        return (f"  -> {len(lex)} words. {samples}\n"
                f"     word formation: {der} derived, {comp} compounds, {colex} colexified")

    return Step(
        id="lexicon",
        title="5. Vocabulary (the lexicon)",
        teaching=(
            "Now the words themselves. We give each core concept a word, but real languages "
            "do not coin everything from scratch: some meanings share one word (tree and "
            "wood), some words are derived from others (hunt becomes hunter), and some are "
            "compounds (water plus fall makes waterfall). The most frequent words tend to be "
            "the shortest."
        ),
        choices=(Choice("build", "Build the dictionary", "generate vocabulary across all semantic fields"),),
        apply=lambda b, key: b.roll_lexicon(),
        summary=summary,
    )


def _writing_step() -> Step:
    _TYPE = {t.value: t for t in WritingSystemType}

    def apply(b: LanguageBuilder, key: str) -> None:
        b.roll_writing(_TYPE.get(key))

    return Step(
        id="writing",
        title="6. The script (writing system)",
        teaching=(
            "Finally, a way to write it. An alphabet has a letter per sound. An abjad writes "
            "only consonants and lets the reader supply the vowels (Arabic, Hebrew). An "
            "abugida hangs a vowel mark on each consonant (Devanagari). A syllabary has one "
            "glyph per syllable (Japanese kana). The glyphs here are featural, so sounds that "
            "are alike get shapes that are alike."
        ),
        choices=(
            Choice("alphabet", "Alphabet", "a glyph for every consonant and vowel"),
            Choice("abjad", "Abjad", "consonants only"),
            Choice("abugida", "Abugida", "consonant glyphs with vowel diacritics"),
            Choice("syllabary", "Syllabary", "a glyph per consonant-vowel syllable"),
            _RANDOM,
        ),
        apply=apply,
        summary=lambda b: "  -> " + b.writing.summary().replace("\n", "\n     "),
    )


def build_steps() -> list[Step]:
    return [
        _phonology_step(),
        _phonotactics_step(),
        _morphology_step(),
        _syntax_step(),
        _lexicon_step(),
        _writing_step(),
    ]
