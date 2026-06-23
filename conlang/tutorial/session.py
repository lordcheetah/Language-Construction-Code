"""The tutorial flow as pure logic — no input or output.

A :class:`TutorialSession` walks the ordered steps, applies a chosen option to the builder
at each one, and records the choices. Keeping this free of I/O makes the whole flow
testable and lets different front-ends (an interactive prompt, a scripted demo, or a future
GUI) drive the same logic.
"""

from __future__ import annotations

from conlang.tutorial.builder import LanguageBuilder
from conlang.tutorial.content import Step
from conlang.language import Language


class TutorialSession:
    def __init__(self, builder: LanguageBuilder, steps: list[Step]) -> None:
        self.builder = builder
        self.steps = steps
        self._index = 0
        self.history: list[tuple[str, str]] = []  # (step id, chosen key)

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_complete(self) -> bool:
        return self._index >= len(self.steps)

    @property
    def current(self) -> Step | None:
        return None if self.is_complete else self.steps[self._index]

    def valid_keys(self) -> set[str]:
        step = self.current
        return {c.key for c in step.choices} if step else set()

    def progress(self) -> tuple[int, int]:
        return (min(self._index + 1, len(self.steps)), len(self.steps))

    def choose(self, key: str) -> None:
        if self.is_complete:
            raise ValueError("the tutorial is already complete")
        step = self.current
        if key not in self.valid_keys():
            raise ValueError(f"{key!r} is not a valid choice for step {step.id!r}")
        step.apply(self.builder, key)
        self.history.append((step.id, key))
        self._index += 1

    def language(self) -> Language:
        if not self.is_complete:
            raise ValueError("the tutorial is not finished yet")
        return self.builder.to_language()
