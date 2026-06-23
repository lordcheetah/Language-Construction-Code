"""Command-line interface for the conlang toolkit.

Currently exposes the phonology stage. Two modes mirror the project's guiding principle
of *guided and random*:

    python -m conlang phonology --random [--seed N]   # roll a whole language
    python -m conlang phonology --inventory "p t k a i u" --templates "(C)V(C)"
"""

from __future__ import annotations

import argparse
import random
import sys

from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import WordGenerator


def _build_phonology(args, rng: random.Random) -> tuple[Inventory, Phonotactics]:
    if args.inventory:
        inv = Inventory.from_ipa(args.inventory)
    elif args.random:
        inv = Inventory.random(rng)
    else:
        raise SystemExit("phonology: pass --random or --inventory \"...\"")

    if args.templates:
        phono = Phonotactics.from_notation(inv, args.templates.split(","))
    else:
        phono = Phonotactics.random(inv, rng)
    return inv, phono


def cmd_phonology(args) -> int:
    # One RNG stream drives the whole pipeline so a seed fully determines the output.
    rng = random.Random(args.seed)
    inv, phono = _build_phonology(args, rng)

    print(inv.summary())
    print(f"  Syllables:  {', '.join(str(t) for t in phono.templates)}")
    print()

    gen = WordGenerator(phono)
    words = gen.lexicon(args.count, rng)
    print(f"Sample lexicon ({len(words)} words):")
    width = max((len(w.roman) for w in words), default=0)
    for w in words:
        print(f"  {w.roman:<{width}}  /{w.ipa}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conlang", description="Generate constructed languages, one stage at a time."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("phonology", help="generate a phoneme inventory and sample words")
    p.add_argument("--random", action="store_true", help="roll a random plausible language")
    p.add_argument("--inventory", help='explicit IPA inventory, e.g. "p t k a i u"')
    p.add_argument(
        "--templates",
        help='comma-separated syllable templates, e.g. "(C)V,(C)V(C)"',
    )
    p.add_argument("--count", type=int, default=20, help="number of sample words (default 20)")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    p.set_defaults(func=cmd_phonology)
    return parser


def main(argv: list[str] | None = None) -> int:
    # IPA is UTF-8; the Windows console defaults to a legacy codepage that can't encode
    # symbols like /ʔ/. Reconfigure stdout where the runtime supports it.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover - environment dependent
            pass

    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
