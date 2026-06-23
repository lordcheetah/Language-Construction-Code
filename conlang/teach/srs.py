"""The SM-2 spaced-repetition scheduler.

SM-2 (SuperMemo 2) is the classic algorithm behind Anki and most flashcard apps. Each card
keeps an *ease factor*, the current *interval* in days, and a count of consecutive correct
*repetitions*. After a review the learner grades recall from 0-5; the grade nudges the ease
and either advances the interval (correct) or resets it (a lapse). It is fully
deterministic — given a card's state, a grade, and today's day number, the next state is
fixed — which is exactly what makes it testable here.
"""

from __future__ import annotations

from dataclasses import dataclass

# A passing grade. Below this, the card lapses and is shown again the next day.
PASS_THRESHOLD = 3
MIN_EASE = 1.3
DEFAULT_EASE = 2.5


@dataclass(frozen=True)
class CardState:
    ease: float = DEFAULT_EASE
    interval: int = 0       # days until the next review
    repetitions: int = 0    # consecutive correct recalls
    due: int = 0            # day number on which the card is next due

    def is_mature(self) -> bool:
        """A card is "learned" once its interval has grown past about three weeks."""
        return self.interval >= 21


def review(state: CardState, quality: int, today: int) -> CardState:
    """Return the card's new state after a review graded *quality* (0-5) on day *today*."""
    if not 0 <= quality <= 5:
        raise ValueError("quality must be between 0 and 5")

    # Ease is always adjusted by the SM-2 formula, then floored.
    ease = state.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease = max(MIN_EASE, ease)

    if quality < PASS_THRESHOLD:  # a lapse: start the interval over
        repetitions = 0
        interval = 1
    else:
        repetitions = state.repetitions + 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = max(1, round(state.interval * ease))

    return CardState(
        ease=round(ease, 3),
        interval=interval,
        repetitions=repetitions,
        due=today + interval,
    )
