"""I/O front-ends for a :class:`TutorialSession`.

``run_interactive`` prompts a user step by step. ``run_demo`` plays the whole tutorial
without input (choosing random, or a scripted choice per step) — handy for a non-interactive
walkthrough and for tests. Both are thin layers over the session's pure logic.
"""

from __future__ import annotations

from typing import Callable

from conlang.tutorial.session import TutorialSession
from conlang.language import Language


def _print_step(session: TutorialSession, write: Callable[[str], None]) -> None:
    step = session.current
    n, total = session.progress()
    write("")
    write(f"{step.title}   [{n}/{total}]")
    write(step.teaching)
    write("")
    for i, choice in enumerate(step.choices, 1):
        tail = f" — {choice.note}" if choice.note else ""
        write(f"  {i}. {choice.label}{tail}")
    if len(step.choices) == 1:
        write("  (press Enter to continue)")


def _finish(session: TutorialSession, write: Callable[[str], None]) -> Language:
    language = session.language()
    write("\n" + "=" * 60)
    write("Your language is complete.\n")
    write(language.summary())
    try:
        sentence = language.make_sentence(
            "woman", "see", "bird", subject_definiteness="def", object_definiteness="indef"
        )
        write('\nA sentence in your language ("the woman sees a bird"):')
        for line in sentence.interlinear().splitlines():
            write("  " + line)
    except (KeyError, ValueError):  # pragma: no cover - lexicon always has these glosses
        pass
    write(f"\nThis whole language is seed {language.seed}.")
    write(f"Recreate or explore it any time:  python -m conlang generate --seed {language.seed}")
    return language


def run_demo(
    session: TutorialSession,
    *,
    write: Callable[[str], None] = print,
    choices: dict[str, str] | None = None,
) -> Language:
    """Play through every step non-interactively (random by default)."""
    while not session.is_complete:
        step = session.current
        _print_step(session, write)
        key = (choices or {}).get(step.id) or _default_key(session)
        chosen = next(c for c in step.choices if c.key == key)
        write(f"  -> chose: {chosen.label}")
        session.choose(key)
        write(step.summary(session.builder))
    return _finish(session, write)


def run_interactive(
    session: TutorialSession,
    *,
    read_line: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> Language | None:
    """Prompt the user through the tutorial. Returns the finished language, or None if quit."""
    write("Welcome — let's build a language together, one stage at a time.")
    write("At each step, pick a number, or type 'r' for a random choice, or 'q' to quit.")

    while not session.is_complete:
        step = session.current
        _print_step(session, write)
        key = _prompt_choice(session, read_line, write)
        if key is None:
            write("\nStopped. Nothing saved.")
            return None
        session.choose(key)
        write(step.summary(session.builder))

    return _finish(session, write)


# --- Helpers ------------------------------------------------------------------------
def _default_key(session: TutorialSession) -> str:
    """Prefer a random roll; fall back to the first (and maybe only) option."""
    keys = session.valid_keys()
    return "random" if "random" in keys else session.current.choices[0].key


def _prompt_choice(session, read_line, write) -> str | None:
    step = session.current
    while True:
        try:
            raw = read_line("> ").strip().lower()
        except EOFError:  # non-interactive stream: take a random/default choice and move on
            return _default_key(session)
        if raw in ("q", "quit"):
            return None
        if len(step.choices) == 1:
            return step.choices[0].key  # one option: any input (Enter/r/1/key) proceeds
        if raw in ("r", "random") and "random" in session.valid_keys():
            return "random"
        if raw.isdigit() and 1 <= int(raw) <= len(step.choices):
            return step.choices[int(raw) - 1].key
        # also accept the literal choice key
        if raw in session.valid_keys():
            return raw
        write("  (please enter one of the listed numbers, 'r' for random, or 'q' to quit)")
