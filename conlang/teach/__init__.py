"""Teach: learn a generated conlang with a spaced-repetition vocabulary trainer.

The third and largest capstone. It turns a :class:`~conlang.language.Language` into a
course: flashcards for every dictionary word, in both directions (see the word and recall
its meaning; see the meaning and recall the word), scheduled by the classic SM-2
spaced-repetition algorithm so that words you find hard come back sooner.

The design mirrors the tutorial — pure logic separated from I/O:

- ``srs``      — the SM-2 scheduler (:class:`CardState`, :func:`review`).
- ``cards``    — the :class:`Card` model and a deck built from a language, frequent first.
- ``course``   — :class:`Course`: which cards are due, introducing new ones, posing
  multiple-choice questions, and recording answers. No input/output.
- ``progress`` — saving and loading review state as JSON (the language itself is recovered
  from its seed).
- ``runner``   — the interactive study session.

Time is always passed in as a day number, so the scheduling is deterministic and testable.
"""

from conlang.teach.srs import CardState, review
from conlang.teach.cards import Card, Direction, build_deck
from conlang.teach.course import Course, Question
from conlang.teach.progress import course_to_dict, course_from_dict, save_course, load_course

__all__ = [
    "CardState",
    "review",
    "Card",
    "Direction",
    "build_deck",
    "Course",
    "Question",
    "course_to_dict",
    "course_from_dict",
    "save_course",
    "load_course",
]
