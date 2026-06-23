"""The interactive study session — the only part that does input/output.

It opens with a few grammar notes and a worked example so the learner sees the language in
use, then drills due and new vocabulary as multiple-choice questions, recording each answer
through the :class:`Course`. A correct answer grades well in SM-2 (so the card's interval
grows); a wrong one lapses it (so it returns tomorrow).
"""

from __future__ import annotations

import random
from typing import Callable

from conlang.teach.course import Course

_CORRECT_QUALITY = 4
_LAPSE_QUALITY = 1


def grammar_notes(language) -> list[str]:
    s = language.syntax
    adp = "before" if s.adposition.value == "preposition" else "after"
    lines = [
        "A few notes on the grammar:",
        f"  - Word order is {s.basic_order.name} (Subject, Object, Verb).",
        f"  - Adpositions go {adp} the noun; adjectives go {s.adjective.value} the noun.",
        f"  - Core arguments use {s.alignment.value} alignment.",
    ]
    try:
        sentence = language.make_sentence(
            "woman", "see", "bird", subject_definiteness="def", object_definiteness="indef"
        )
        lines.append('  - For example, "the woman sees a bird" is:')
        for line in sentence.interlinear().splitlines():
            lines.append("      " + line)
    except (KeyError, ValueError):  # pragma: no cover - lexicon always has these glosses
        pass
    return lines


def run_session(
    course: Course,
    today: int,
    *,
    rng: random.Random,
    read_line: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    max_cards: int = 20,
) -> Course:
    write("Study session — type the number of the right answer, or 'q' to stop.\n")
    for line in grammar_notes(course.language):
        write(line)

    st = course.stats(today)
    write(
        f"\n{st['introduced']} words started, {st['due']} due today, "
        f"{st['mature']} learned, {st['new_remaining']} still new.\n"
    )

    new_count = reviewed = correct = 0
    while reviewed < max_cards:
        question = course.next_question(
            today, rng, introduce_new=new_count < course.new_per_day
        )
        if question is None:
            write("That's everything due for now. Well done!")
            break
        if question.is_new:
            new_count += 1

        ask = ("What does this word mean?" if question.direction == "recognition"
               else "How do you say this?")
        write(("[new] " if question.is_new else "") + ask)
        write(f"    {question.prompt}")
        for i, option in enumerate(question.options, 1):
            write(f"      {i}. {option}")

        choice = _read_choice(len(question.options), read_line, write)
        if choice is None:
            write("\nStopping here. Progress is saved.")
            break

        was_correct = (choice - 1) == question.answer_index
        course.record(question.card_id, _CORRECT_QUALITY if was_correct else _LAPSE_QUALITY, today)
        reviewed += 1
        answer = question.options[question.answer_index]
        if was_correct:
            correct += 1
            write(f"    Correct.  {question.prompt} /{question.pronunciation}/ = {answer}\n")
        else:
            write(f"    Not quite — the answer is {answer} (/{question.pronunciation}/)\n")

    write(f"\nReviewed {reviewed} cards, {correct} correct.")
    if reviewed > correct:
        write("Cards you missed will come back tomorrow.")
    return course


def _read_choice(n_options: int, read_line, write) -> int | None:
    while True:
        try:
            raw = read_line("> ").strip().lower()
        except EOFError:
            return None
        if raw in ("q", "quit"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= n_options:
            return int(raw)
        write(f"  (enter a number 1-{n_options}, or 'q' to stop)")
