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
from conlang.soundchange.ruleset import RuleSet

# An illustrative ruleset used by `soundchange --demo`: intervocalic voicing, then final
# devoicing (a classic feeding/bleeding pair), plus intervocalic /h/-loss.
_DEMO_RULES = """
# Demo: lenition then final devoicing
K = p t k
K > [+voiced] / V_V
[voiced obstruent] > [-voiced] / _#
h > 0 / V_V
"""


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


def cmd_soundchange(args) -> int:
    rng = random.Random(args.seed)
    # Build a proto-language; default to random when no explicit inventory is given.
    if not args.inventory and not args.random:
        args.random = True
    inv, phono = _build_phonology(args, rng)

    # Assemble the ruleset from a file, inline --rule flags, or the built-in demo.
    if args.rules_file:
        with open(args.rules_file, encoding="utf-8") as fh:
            ruleset = RuleSet.parse(fh.read())
    elif args.rule:
        ruleset = RuleSet.from_rules(args.rule)
    elif args.demo:
        ruleset = RuleSet.parse(_DEMO_RULES)
    else:
        raise SystemExit("soundchange: pass --rules-file FILE, one or more --rule, or --demo")

    gen = WordGenerator(phono)
    proto = gen.lexicon(args.count, rng)
    evolved = ruleset.evolve_lexicon(proto, gen.romanizer)

    print(inv.summary())
    print(f"\nApplied {len(ruleset.rules)} sound change(s):")
    for r in ruleset.rules:
        print(f"  {r.source}")
    print(f"\nProto -> Daughter ({len(evolved)} words):")
    proto_col = [f"{e.original.roman} /{e.original.ipa}/" for e in evolved]
    width = max((len(s) for s in proto_col), default=0)
    for e, proto_str in zip(evolved, proto_col):
        mark = "" if e.ipa == e.original.ipa else "  *"
        print(f"  {proto_str:<{width}}  ->  {e.roman} /{e.ipa}/{mark}")

    if args.trace:
        print("\nDerivations (changed words only):")
        for e in evolved:
            proto_segments = [s for syl in e.original.syllables for s in syl]
            deriv = ruleset.derive(proto_segments)
            if deriv.changed:
                print(f"\n{e.original.roman}:")
                print(deriv.trace())
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

    s = sub.add_parser(
        "soundchange",
        help="evolve a generated proto-lexicon through sound changes into a daughter language",
    )
    s.add_argument("--random", action="store_true", help="random proto-language (default)")
    s.add_argument("--inventory", help='explicit proto IPA inventory, e.g. "p t k a i u"')
    s.add_argument("--templates", help='comma-separated syllable templates')
    s.add_argument("--rules-file", help="path to a ruleset file")
    s.add_argument(
        "--rule",
        action="append",
        help='an inline rule, e.g. "p > b / V_V" (repeatable)',
    )
    s.add_argument("--demo", action="store_true", help="use a built-in example ruleset")
    s.add_argument("--trace", action="store_true", help="show step-by-step derivations")
    s.add_argument("--count", type=int, default=20, help="proto-lexicon size (default 20)")
    s.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    s.set_defaults(func=cmd_soundchange)
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
