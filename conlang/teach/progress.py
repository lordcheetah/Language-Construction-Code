"""Saving and loading study progress.

Only the *review state* needs to be stored: the language itself is recovered from its seed
(``Language.generate(seed)``), so a save file is small and the deck is rebuilt on load. The
generator version is recorded so a future incompatible change can be detected rather than
silently rebuilding a different language under the same seed.
"""

from __future__ import annotations

import json

from conlang.language import Language, GENERATOR_VERSION
from conlang.teach.srs import CardState
from conlang.teach.course import Course


def course_to_dict(course: Course) -> dict:
    return {
        "seed": course.language.seed,
        "generator_version": GENERATOR_VERSION,
        "new_per_day": course.new_per_day,
        "introduced": sorted(course.introduced),
        "states": {
            card_id: {
                "ease": s.ease, "interval": s.interval,
                "repetitions": s.repetitions, "due": s.due,
            }
            for card_id, s in course.states.items()
        },
    }


def course_from_dict(data: dict) -> Course:
    seed = data["seed"]
    if seed is None:
        raise ValueError("cannot resume a course with no seed (the language is unrecoverable)")
    saved_version = data.get("generator_version")
    if saved_version is not None and saved_version != GENERATOR_VERSION:
        raise ValueError(
            f"saved progress was made with generator version {saved_version}, "
            f"but this build is version {GENERATOR_VERSION}; the language would differ"
        )
    language = Language.generate(seed)
    states = {
        card_id: _state_from_fields(fields)
        for card_id, fields in data.get("states", {}).items()
    }
    return Course(
        language,
        new_per_day=data.get("new_per_day", 8),
        states=states,
        introduced=set(data.get("introduced", [])),
    )


def _state_from_fields(fields: dict) -> CardState:
    """Build a CardState from saved fields, tolerating extra/missing keys and coercing types."""
    try:
        return CardState(
            ease=float(fields.get("ease", 2.5)),
            interval=int(fields.get("interval", 0)),
            repetitions=int(fields.get("repetitions", 0)),
            due=int(fields.get("due", 0)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"corrupt card state in saved progress: {fields!r}") from exc


def save_course(course: Course, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(course_to_dict(course), fh, ensure_ascii=False, indent=2)


def load_course(path: str) -> Course:
    with open(path, encoding="utf-8") as fh:
        return course_from_dict(json.load(fh))
