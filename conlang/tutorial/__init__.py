"""Tutorial: an interactive, guided walkthrough of creating a language.

The second capstone. It teaches the Language Construction Kit ideas one stage at a time,
letting the learner make a deliberate choice at each step (or roll a random, plausible
one) and watching a real :class:`~conlang.language.Language` take shape.

The design separates *what the tutorial knows* from *how it talks to the user*:

- :class:`LanguageBuilder` accumulates the per-stage decisions and produces the language.
- :class:`Step` / :class:`Choice` hold the teaching content and the options.
- :class:`TutorialSession` is the pure flow logic (no input/output), so it is testable.
- ``runner`` is the thin interactive (and non-interactive ``demo``) I/O layer.
"""

from conlang.tutorial.builder import LanguageBuilder
from conlang.tutorial.content import Step, Choice, build_steps
from conlang.tutorial.session import TutorialSession
from conlang.tutorial.runner import run_interactive, run_demo

__all__ = [
    "LanguageBuilder",
    "Step",
    "Choice",
    "build_steps",
    "TutorialSession",
    "run_interactive",
    "run_demo",
]
