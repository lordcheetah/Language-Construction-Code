"""Gather everything the GUI shows for one language into a plain, testable object.

``build_view(seed)`` regenerates the language from its seed (so the GUI is as reproducible
as every other front-end) and precomputes the display strings — overview, dictionary rows,
glossed sentences, numbers, and the sample word to render in the native script. It holds no
Tkinter state and imports no GUI toolkit, so the whole view is unit-testable and the
cross-front-end determinism invariant (same seed → same content) can be asserted directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conlang.language import Language
from conlang.speech.respell import respell

# The showcase content, mirroring the `generate` CLI so the two front-ends agree.
VOCAB = ["I", "water", "fire", "sun", "woman", "eye", "tree", "see", "eat", "big", "one"]
SENTENCES = [
    ("the woman sees a bird",
     dict(subject="woman", verb="see", obj="bird",
          subject_definiteness="def", object_definiteness="indef")),
    ("the children run",
     dict(subject="child", verb="run", subject_number="pl", subject_definiteness="def")),
    ("the big dog eats meat",
     dict(subject="dog", verb="eat", obj="meat",
          subject_adjective="big", subject_definiteness="def")),
]
NUMBERS = (1, 2, 3, 7, 10, 11, 24, 100)
# Words to render as a written sentence in the script tab (skips any the lexicon lacks).
_SENTENCE_WORDS = ("person", "see", "bird")


@dataclass(frozen=True)
class VocabRow:
    gloss: str
    roman: str
    ipa: str
    say: str  # TTS-friendly respelling


@dataclass(frozen=True)
class SentenceBlock:
    english: str
    interlinear: str


@dataclass
class LanguageView:
    language: Language
    seed: int
    overview: str
    vocab: list[VocabRow]
    sentences: list[SentenceBlock]
    numbers: list[tuple[int, str]]
    script_type: str
    direction: str
    sample_gloss: str
    _sample_segments: list = field(default_factory=list)

    # Glyph objects stay on the writing system; the app renders them via drawops.
    def word_units(self):
        """(label, glyph) units for the sample word in the native script."""
        return self.language.writing.render_segments(self._sample_segments)

    def chart_cells(self):
        return self.language.writing.chart_cells()

    def sentence_words_units(self):
        """(label, glyph) unit lists for each word of the sample written sentence."""
        entries = [self.language.lexicon.get(g) for g in _SENTENCE_WORDS]
        entries = [e for e in entries if e is not None]
        if not entries:
            entries = list(self.language.lexicon.entries.values())[:3]
        return [self.language.writing.render_segments(list(e.form)) for e in entries]


def build_view(seed: int | None = None, *, sandhi: bool = False) -> LanguageView:
    rules = None
    if sandhi:
        from conlang.soundchange.ruleset import RuleSet
        from conlang.cli import _DEMO_RULES
        rules = RuleSet.parse(_DEMO_RULES)
    lang = Language.generate(seed, sandhi=rules)

    vocab = []
    for gloss in VOCAB:
        e = lang.lexicon.get(gloss)
        if e is not None:
            vocab.append(VocabRow(gloss, e.roman, e.ipa, respell(e.form)))

    sentences = []
    for english, kw in SENTENCES:
        try:
            sentence = lang.make_sentence(**kw)
        except (KeyError, ValueError):
            continue
        sentences.append(SentenceBlock(english, sentence.interlinear()))

    numbers = [(n, lang.numerals.number(n).roman)
               for n in NUMBERS if n <= lang.numerals.max_value]

    sample = lang.lexicon.get("woman") or next(iter(lang.lexicon.entries.values()))

    return LanguageView(
        language=lang,
        seed=lang.seed,
        overview=lang.summary(),
        vocab=vocab,
        sentences=sentences,
        numbers=numbers,
        script_type=lang.writing.type.value,
        direction=lang.writing.direction.value,
        sample_gloss=sample.gloss,
        _sample_segments=list(sample.form),
    )
