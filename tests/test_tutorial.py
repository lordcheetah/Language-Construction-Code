"""Tests for the tutorial capstone: the builder, the pure session flow, and the runners."""

import random

import pytest

from conlang.language import Language
from conlang.morphology.features import Typology
from conlang.syntax.parameters import WordOrder, Adposition
from conlang.writing.system import WritingSystemType
from conlang.lexicon.concepts import CONCEPTS
from conlang.tutorial.builder import LanguageBuilder
from conlang.tutorial.content import build_steps
from conlang.tutorial.session import TutorialSession
from conlang.tutorial.runner import run_demo, run_interactive


# --- Builder ------------------------------------------------------------------------
def test_builder_choices_are_honored():
    b = LanguageBuilder.start(7)
    b.roll_inventory("small")
    b.set_phonotactics("simple")
    b.roll_morphology(Typology.AGGLUTINATIVE)
    b.set_syntax(WordOrder.SOV)
    b.roll_lexicon()
    b.roll_writing(WritingSystemType.ABUGIDA)
    lang = b.to_language()

    assert len(lang.inventory.consonants) <= 16          # "small" stays small
    assert [str(t) for t in lang.phonotactics.templates] == ["CV"]
    assert lang.morphology.typology is Typology.AGGLUTINATIVE
    assert lang.syntax.basic_order is WordOrder.SOV
    assert lang.syntax.adposition is Adposition.POSTPOSITION  # SOV is harmonic -> postpositions
    assert lang.writing.type is WritingSystemType.ABUGIDA
    assert len(lang.lexicon) == len(CONCEPTS)


def test_builder_records_a_reproducible_seed():
    b = LanguageBuilder.start()  # no seed
    assert b.seed is not None


def test_to_language_rolls_skipped_stages():
    lang = LanguageBuilder.start(3).to_language()  # nothing chosen at all
    assert lang.inventory and lang.morphology and lang.writing and len(lang.lexicon) > 0


# --- Session (pure flow) ------------------------------------------------------------
def test_session_runs_all_steps_in_order():
    session = TutorialSession(LanguageBuilder.start(1), build_steps())
    steps = build_steps()
    assert session.progress() == (1, len(steps))
    while not session.is_complete:
        session.choose("random" if "random" in session.valid_keys() else session.current.choices[0].key)
    assert [h[0] for h in session.history] == [s.id for s in steps]
    assert isinstance(session.language(), Language)


def test_session_rejects_invalid_choice():
    session = TutorialSession(LanguageBuilder.start(1), build_steps())
    with pytest.raises(ValueError):
        session.choose("nonsense")


def test_session_language_before_finish_raises():
    session = TutorialSession(LanguageBuilder.start(1), build_steps())
    with pytest.raises(ValueError):
        session.language()


def test_scripted_choices_drive_the_session():
    session = TutorialSession(LanguageBuilder.start(5), build_steps())
    session.choose("medium")       # phonology
    session.choose("complex")      # phonotactics
    session.choose("fusional")     # morphology
    session.choose("svo")          # syntax
    session.choose("build")        # lexicon
    session.choose("syllabary")    # writing
    lang = session.language()
    assert lang.morphology.typology is Typology.FUSIONAL
    assert lang.syntax.basic_order is WordOrder.SVO
    assert lang.writing.type is WritingSystemType.SYLLABARY


# --- Runners ------------------------------------------------------------------------
def test_run_demo_completes_and_emits_output():
    lines: list[str] = []
    session = TutorialSession(LanguageBuilder.start(2), build_steps())
    lang = run_demo(session, write=lines.append)
    assert isinstance(lang, Language)
    text = "\n".join(lines)
    # the teaching content and the per-stage results both appear
    assert "phonology" in text.lower() and "syllab" in text.lower()
    assert f"seed {lang.seed}" in text


def test_run_demo_with_scripted_choices():
    session = TutorialSession(LanguageBuilder.start(4), build_steps())
    lang = run_demo(session, write=lambda s: None, choices={"morphology": "isolating", "writing": "alphabet"})
    assert lang.morphology.typology is Typology.ISOLATING
    assert lang.writing.type is WritingSystemType.ALPHABET


def test_run_interactive_drives_from_input_and_quits():
    # Feed numbered answers; pick option 1 at every step.
    answers = iter(["1", "1", "1", "1", "1", "1"])
    session = TutorialSession(LanguageBuilder.start(6), build_steps())
    lang = run_interactive(session, read_line=lambda prompt: next(answers), write=lambda s: None)
    assert isinstance(lang, Language)

    # 'q' aborts and returns None.
    session2 = TutorialSession(LanguageBuilder.start(6), build_steps())
    assert run_interactive(session2, read_line=lambda prompt: "q", write=lambda s: None) is None


def test_demo_and_interactive_match_for_same_seed_and_choices():
    # The headline invariant: the front-end doesn't affect the language. Both pick random
    # everywhere ('r' in interactive == random default in demo; the lexicon step proceeds).
    d = run_demo(TutorialSession(LanguageBuilder.start(31), build_steps()), write=lambda s: None)
    i = run_interactive(
        TutorialSession(LanguageBuilder.start(31), build_steps()),
        read_line=lambda prompt: "r", write=lambda s: None,
    )
    assert d.to_dict() == i.to_dict()


def test_reading_summaries_does_not_change_the_language():
    def run(read_summaries: bool):
        session = TutorialSession(LanguageBuilder.start(21), build_steps())
        while not session.is_complete:
            step = session.current
            key = "random" if "random" in session.valid_keys() else step.choices[0].key
            session.choose(key)
            if read_summaries:
                step.summary(session.builder)  # must use a display RNG, not builder.rng
        return session.language().to_dict()

    assert run(True) == run(False)


def test_interactive_eof_falls_back_to_random_and_completes():
    def reader(prompt):
        raise EOFError

    session = TutorialSession(LanguageBuilder.start(2), build_steps())
    lang = run_interactive(session, read_line=reader, write=lambda s: None)
    assert isinstance(lang, Language)


def test_vso_and_abjad_choice_mappings():
    session = TutorialSession(LanguageBuilder.start(1), build_steps())
    for key in ("large", "moderate", "agglutinative", "vso", "build", "abjad"):
        session.choose(key)
    lang = session.language()
    assert lang.syntax.basic_order is WordOrder.VSO
    assert lang.writing.type is WritingSystemType.ABJAD
    assert len(lang.inventory.consonants) >= 20  # "large" inventory


def test_run_interactive_reprompts_on_bad_input():
    seen: list[str] = []
    replies = iter(["99", "abc", "2"])  # two bad, then valid
    session = TutorialSession(LanguageBuilder.start(1), [build_steps()[0]])  # just one step
    run_interactive(session, read_line=lambda prompt: next(replies), write=seen.append)
    assert any("please enter" in line for line in seen)
    assert session.is_complete
