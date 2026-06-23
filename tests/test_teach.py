"""Tests for the teaching app: SM-2 scheduling, the course logic, persistence, and the runner."""

import random

import pytest

from conlang.language import Language
from conlang.lexicon.concepts import CONCEPTS
from conlang.teach.srs import CardState, review, PASS_THRESHOLD
from conlang.teach.cards import Direction, build_deck
from conlang.teach.course import Course, Question
from conlang.teach.progress import course_to_dict, course_from_dict
from conlang.teach.runner import run_session, grammar_notes


# --- SM-2 scheduler -----------------------------------------------------------------
def test_first_correct_reviews_schedule_1_then_6_days():
    s = review(CardState(), 5, today=0)
    assert s.repetitions == 1 and s.interval == 1 and s.due == 1
    s = review(s, 5, today=1)
    assert s.repetitions == 2 and s.interval == 6 and s.due == 7


def test_interval_grows_by_ease_after_the_second_review():
    s = CardState(ease=2.5, interval=6, repetitions=2, due=0)
    s2 = review(s, 4, today=10)
    assert s2.interval == round(6 * s2.ease)  # ~15
    assert s2.due == 10 + s2.interval


def test_a_lapse_resets_interval_and_repetitions():
    mature = CardState(ease=2.5, interval=30, repetitions=5, due=0)
    lapsed = review(mature, 1, today=100)
    assert lapsed.repetitions == 0 and lapsed.interval == 1 and lapsed.due == 101
    assert lapsed.ease < mature.ease  # ease dropped


def test_ease_never_falls_below_floor():
    s = CardState(ease=1.3)
    for _ in range(5):
        s = review(s, 0, today=0)
    assert s.ease >= 1.3


def test_quality_out_of_range_raises():
    with pytest.raises(ValueError):
        review(CardState(), 7, today=0)


def test_interval_grows_over_many_correct_reviews():
    s, day = CardState(), 0
    intervals = []
    for _ in range(5):
        s = review(s, 5, today=day)
        intervals.append(s.interval)
        day += s.interval
    # 1, 6, then geometric growth by ease -> strictly increasing after the second review
    assert intervals[0] == 1 and intervals[1] == 6
    assert intervals[2] < intervals[3] < intervals[4]


def test_passing_grade_boundary():
    assert review(CardState(), PASS_THRESHOLD, 0).repetitions == 1
    assert review(CardState(), PASS_THRESHOLD - 1, 0).repetitions == 0


# --- Deck ---------------------------------------------------------------------------
def test_deck_has_two_cards_per_word_ordered_by_basicness():
    lang = Language.generate(7)
    deck = build_deck(lang)
    teachable = [c for c in CONCEPTS if c.pos != "particle"]  # particles aren't drilled
    assert len(deck) == 2 * len(teachable)
    assert all(card.pos != "particle" for card in deck)
    recognitions = [c for c in deck if c.direction is Direction.RECOGNITION]
    basics = [_basicness(lang, c.gloss) for c in recognitions]
    assert basics == sorted(basics, reverse=True)  # most basic first


def _basicness(lang, gloss):
    return lang.lexicon.get(gloss).concept.basicness


# --- Course -------------------------------------------------------------------------
def test_questions_are_well_formed_multiple_choice():
    course = Course(Language.generate(3), new_per_day=8)
    rng = random.Random(1)
    q = course.next_question(today=0, rng=rng)
    assert isinstance(q, Question)
    assert len(q.options) == 4
    assert q.options[q.answer_index]  # the correct option exists
    assert len(set(q.options)) == 4   # no duplicate options


def test_new_card_is_not_introduced_until_answered():
    # Quitting at a new prompt must leak nothing (lazy introduction).
    course = Course(Language.generate(3), new_per_day=8)
    q = course.next_question(today=0, rng=random.Random(0))
    assert q.is_new
    assert course.stats(0)["introduced"] == 0   # shown, but not yet recorded
    course.record(q.card_id, 5, today=0)
    assert course.stats(0)["introduced"] == 1
    assert course.states[q.card_id].due == 1    # correct answer -> due tomorrow


def test_every_question_has_four_distinct_options():
    course = Course(Language.generate(9), new_per_day=8)
    rng = random.Random(4)
    for _ in range(30):
        q = course.next_question(today=0, rng=rng)
        course.record(q.card_id, 5, today=0)
        assert len(q.options) == 4 and len(set(q.options)) == 4
        assert q.options[q.answer_index]


def test_new_card_budget_is_respected():
    course = Course(Language.generate(3), new_per_day=3)
    rng = random.Random(0)
    introduced = 0
    for _ in range(3):
        q = course.next_question(today=0, rng=rng, introduce_new=True)
        assert q.is_new
        course.record(q.card_id, 5, today=0)
        introduced += 1
    # with no due cards and new disabled, there is nothing to study
    assert course.next_question(today=0, rng=rng, introduce_new=False) is None


def test_wrong_answer_brings_card_back_tomorrow():
    course = Course(Language.generate(5), new_per_day=8)
    rng = random.Random(2)
    q = course.next_question(today=0, rng=rng)
    course.record(q.card_id, 1, today=0)        # lapse
    assert course.states[q.card_id].due == 1
    assert q.card_id in [i for i in course.introduced if course.states[i].due <= 1]


# --- Persistence --------------------------------------------------------------------
def test_progress_round_trips_through_json():
    course = Course(Language.generate(11), new_per_day=5)
    rng = random.Random(0)
    for _ in range(6):  # study a handful of cards
        q = course.next_question(today=0, rng=rng)
        course.record(q.card_id, 4, today=0)

    restored = course_from_dict(course_to_dict(course))
    assert restored.language.seed == 11
    assert restored.new_per_day == 5
    assert restored.introduced == course.introduced
    assert restored.states == course.states


def test_resume_cannot_recover_a_seedless_course():
    data = course_to_dict(Course(Language.generate(1)))
    data["seed"] = None
    with pytest.raises(ValueError):
        course_from_dict(data)


def test_resume_across_days_surfaces_due_cards():
    course = Course(Language.generate(13), new_per_day=4)
    rng = random.Random(0)
    for _ in range(4):  # study day 0, all correct -> next due day 1
        q = course.next_question(today=0, rng=rng)
        course.record(q.card_id, 5, today=0)
    resumed = course_from_dict(course_to_dict(course))
    assert resumed.stats(0)["due"] == 0   # nothing due the same day
    assert resumed.stats(1)["due"] == 4   # all four come back the next day


def test_version_mismatch_is_rejected():
    data = course_to_dict(Course(Language.generate(1)))
    data["generator_version"] = 999
    with pytest.raises(ValueError):
        course_from_dict(data)


def test_corrupt_state_is_tolerated_or_reported():
    course = Course(Language.generate(1), new_per_day=2)
    q = course.next_question(today=0, rng=random.Random(0))
    course.record(q.card_id, 4, today=0)
    data = course_to_dict(course)
    # extra/unknown keys are ignored, types coerced from strings
    cid = next(iter(data["states"]))
    data["states"][cid]["bogus"] = 1
    data["states"][cid]["interval"] = "1"
    restored = course_from_dict(data)
    assert restored.states[cid].interval == 1


# --- Runner -------------------------------------------------------------------------
def test_grammar_notes_describe_the_language():
    notes = "\n".join(grammar_notes(Language.generate(4)))
    assert "Word order" in notes and "alignment" in notes


def test_run_session_grades_and_records():
    # new_per_day must cover max_cards, since freshly-failed cards only return the next day.
    course = Course(Language.generate(8), new_per_day=8)
    transcript: list[str] = []
    # Always answer "1"; some right, some wrong — the session must just run and record.
    answers = iter(["1"] * 50)
    run_session(
        course, today=0, rng=random.Random(0),
        read_line=lambda prompt: next(answers), write=transcript.append, max_cards=6,
    )
    text = "\n".join(transcript)
    assert "Reviewed 6 cards" in text
    assert course.stats(0)["introduced"] == 6  # six new cards were studied


def test_run_session_quits_on_q():
    course = Course(Language.generate(8), new_per_day=5)
    lines: list[str] = []
    run_session(
        course, today=0, rng=random.Random(0),
        read_line=lambda prompt: "q", write=lines.append, max_cards=20,
    )
    assert any("Stopping here" in line for line in lines)
