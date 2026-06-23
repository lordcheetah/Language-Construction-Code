"""The course: which card to study next, and recording the answer.

A :class:`Course` holds the deck and every card's :class:`CardState`. It introduces new
cards gradually, surfaces cards whose review is due, and poses each as a multiple-choice
:class:`Question` with plausible distractors drawn from the same semantic field. All of its
methods take the day number (and an RNG for the question) explicitly, so the logic is pure
and reproducible — the interactive runner supplies today's date and a seeded RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from conlang.teach.cards import Card, Direction, build_deck
from conlang.teach.srs import CardState, review


@dataclass(frozen=True)
class Question:
    card_id: str
    direction: str
    prompt: str
    pronunciation: str        # IPA of the conlang word
    options: tuple[str, ...]
    answer_index: int
    is_new: bool              # first time this card is being shown


class Course:
    def __init__(
        self,
        language,
        *,
        new_per_day: int = 8,
        states: dict[str, CardState] | None = None,
        introduced: set[str] | None = None,
    ) -> None:
        self.language = language
        self.new_per_day = new_per_day
        self.deck = build_deck(language)
        self.by_id = {c.id: c for c in self.deck}
        self.states: dict[str, CardState] = dict(states or {})
        self.introduced: set[str] = set(introduced or [])

    # --- Selection -------------------------------------------------------------------
    def due_ids(self, today: int) -> list[str]:
        due = [i for i in self.introduced if self.states[i].due <= today]
        return sorted(due, key=lambda i: self.states[i].due)

    def _next_new_id(self) -> str | None:
        for card in self.deck:
            if card.id not in self.introduced:
                return card.id
        return None

    def next_question(
        self, today: int, rng: random.Random, *, introduce_new: bool = True
    ) -> Question | None:
        due = self.due_ids(today)
        if due:
            return self._make_question(due[0], rng, is_new=False)
        if introduce_new:
            new_id = self._next_new_id()
            if new_id is not None:
                # The card is only recorded as "introduced" once it is actually answered
                # (see record), so quitting at a new prompt leaks nothing.
                return self._make_question(new_id, rng, is_new=True)
        return None

    def _make_question(self, card_id: str, rng: random.Random, *, is_new: bool) -> Question:
        card = self.by_id[card_id]
        correct = card.answer
        same_dir = [c for c in self.deck if c.direction is card.direction]
        same_field = [c for c in same_dir if c.field == card.field]
        pool = same_field if _distinct_answers(same_field, correct) >= 3 else same_dir

        candidates = [a for a in dict.fromkeys(c.answer for c in pool) if a != correct]
        rng.shuffle(candidates)
        if len(candidates) < 3:  # top up from the whole direction so we always get 4 options
            extra = [a for a in dict.fromkeys(c.answer for c in same_dir)
                     if a != correct and a not in candidates]
            rng.shuffle(extra)
            candidates += extra
        options = [correct, *candidates[:3]]
        rng.shuffle(options)
        return Question(
            card_id=card_id,
            direction=card.direction.value,
            prompt=card.prompt,
            pronunciation=card.ipa,
            options=tuple(options),
            answer_index=options.index(correct),
            is_new=is_new,
        )

    # --- Recording -------------------------------------------------------------------
    def record(self, card_id: str, quality: int, today: int) -> None:
        current = self.states.get(card_id, CardState(due=today))
        self.states[card_id] = review(current, quality, today)
        self.introduced.add(card_id)

    # --- Progress --------------------------------------------------------------------
    def stats(self, today: int) -> dict:
        introduced = len(self.introduced)
        return {
            "total": len(self.deck),
            "introduced": introduced,
            "new_remaining": len(self.deck) - introduced,
            "due": len(self.due_ids(today)),
            "mature": sum(1 for s in self.states.values() if s.is_mature()),
        }


def _distinct_answers(cards: list[Card], correct: str) -> int:
    return len({c.answer for c in cards if c.answer != correct})
