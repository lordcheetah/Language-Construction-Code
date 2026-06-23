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
from conlang.phonology.wordgen import WordGenerator, Romanizer
from conlang.soundchange.ruleset import RuleSet
from conlang.morphology.generator import random_system
from conlang.morphology.features import FeatureBundle
from conlang.syntax.generator import random_syntax
from conlang.syntax.linearizer import Linearizer
from conlang.syntax.structure import Lexeme, NounPhrase, Clause, AdpositionalPhrase
from conlang.lexicon.generator import build_lexicon
from conlang.lexicon.lexicon import Etymology
from conlang.writing.generator import build_writing_system
from conlang.writing.system import WritingSystemType
from conlang.language import Language
from conlang.phonology import data as _phon_data
from conlang.speech.synth import Synthesizer, Voice
from conlang.tutorial.builder import LanguageBuilder
from conlang.tutorial.content import build_steps
from conlang.tutorial.session import TutorialSession
from conlang.tutorial.runner import run_interactive, run_demo
from conlang.teach.course import Course
from conlang.teach.progress import load_course, save_course
from conlang.teach.runner import run_session as run_study_session

# Sample sentences for the `generate` showcase (glosses must exist in the lexicon).
_SHOWCASE_SENTENCES = [
    ("the woman sees a bird",
     dict(subject="woman", verb="see", obj="bird",
          subject_definiteness="def", object_definiteness="indef")),
    ("the children run",
     dict(subject="child", verb="run", subject_number="pl", subject_definiteness="def")),
    ("the big dog eats meat",
     dict(subject="dog", verb="eat", obj="meat",
          subject_adjective="big", subject_definiteness="def")),
]
_SHOWCASE_VOCAB = ["I", "water", "fire", "sun", "woman", "eye", "tree", "see", "eat", "big", "one"]

# Small English gloss banks so generated sentences can be read interlinearly.
_NOUN_GLOSSES = ["dog", "woman", "stone", "river", "bird", "child", "fire", "tree"]
_TVERB_GLOSSES = ["see", "eat", "carry", "find", "hear"]
_IVERB_GLOSSES = ["sleep", "run", "arrive"]
_ADJ_GLOSSES = ["big", "red", "old", "cold"]
_ADP_GLOSSES = ["near"]

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


def cmd_morphology(args) -> int:
    rng = random.Random(args.seed)
    args.random = not args.inventory  # default to a random inventory
    inv, phono = _build_phonology(args, rng)
    romanizer = Romanizer()
    gen = WordGenerator(phono, romanizer)

    sandhi = RuleSet.parse(_DEMO_RULES) if args.sandhi else None
    system = random_system(phono, rng, romanizer=romanizer, sandhi=sandhi)

    print(inv.summary())
    print()
    print(system.summary())
    if args.sandhi:
        print("  (sandhi applied at morpheme boundaries)")

    for class_name, paradigm in system.paradigms.items():
        root_word = gen.word(rng, min_syllables=1, max_syllables=2)
        root = [s for syl in root_word.syllables for s in syl]
        cids = paradigm.class_ids()
        header = f"\n{class_name.capitalize()} paradigm for {root_word.roman} /{root_word.ipa}/"
        if len(cids) > 1:
            header += f"  (declensions {' | '.join(cids)})"
        print(header + ":")
        bundles = paradigm.enumerate_bundles()
        shown = bundles[: args.max_rows]
        bw = max((len(str(b)) for b in shown), default=0)
        for bundle in shown:
            forms = [paradigm.romanize(paradigm.inflect(root, bundle, cid)) for cid in cids]
            print(f"  {str(bundle):<{bw}}  " + "  |  ".join(forms))
        if len(bundles) > len(shown):
            print(f"  ... ({len(bundles) - len(shown)} more cells)")

    if system.derivations:
        print("\nDerivations (derived stem shown in its citation form):")
        for d in system.derivations:
            # Use a root of the correct source class, then inflect the derived stem with
            # its target paradigm (citation form) -- derivation feeding inflection.
            src_word = gen.word(rng, min_syllables=1, max_syllables=2)
            src_root = [s for syl in src_word.syllables for s in syl]
            derived = system.derive(d, src_root, FeatureBundle.of())
            roman = romanizer.romanize([list(derived)])
            ipa_form = "".join(s.ipa for s in derived)
            print(
                f"  {d.gloss:<11} {d.from_class} {src_word.roman} /{src_word.ipa}/"
                f"  ->  {d.to_class} {roman} /{ipa_form}/"
            )
    return 0


def _build_lexicon(gen, rng, glosses, word_class) -> dict:
    lex = {}
    for gloss in glosses:
        word = gen.word(rng, min_syllables=1, max_syllables=2)
        root = tuple(s for syl in word.syllables for s in syl)
        lex[gloss] = Lexeme(root, word_class, gloss)
    return lex


def cmd_syntax(args) -> int:
    rng = random.Random(args.seed)
    args.random = not args.inventory
    inv, phono = _build_phonology(args, rng)
    romanizer = Romanizer()
    gen = WordGenerator(phono, romanizer)

    sandhi = RuleSet.parse(_DEMO_RULES) if args.sandhi else None
    morphology = random_system(phono, rng, romanizer=romanizer, sandhi=sandhi)
    params = random_syntax(rng)

    nouns = _build_lexicon(gen, rng, _NOUN_GLOSSES, "noun")
    tverbs = _build_lexicon(gen, rng, _TVERB_GLOSSES, "verb")
    iverbs = _build_lexicon(gen, rng, _IVERB_GLOSSES, "verb")
    adjs = _build_lexicon(gen, rng, _ADJ_GLOSSES, "adjective")
    adps = _build_lexicon(gen, rng, _ADP_GLOSSES, "adposition")
    particle_lex = _build_lexicon(gen, rng, ["not", "Q"], "particle")
    particles = {"neg": particle_lex["not"], "q": particle_lex["Q"]}
    lin = Linearizer(params, morphology, romanizer, particles=particles)

    print("Syntax parameters:")
    print(params.describe())
    print(f"\nMorphology: {morphology.typology.value}")
    for name, par in morphology.paradigms.items():
        marked = ", ".join(c.name for c in par.marked) or "(none)"
        print(f"  {name} marks: {marked}")

    clauses = [
        Clause(NounPhrase(nouns["dog"], definiteness="def"), iverbs["sleep"]),
        Clause(
            NounPhrase(nouns["woman"], definiteness="def"),
            tverbs["see"],
            NounPhrase(nouns["bird"], definiteness="indef"),
        ),
        Clause(
            NounPhrase(nouns["child"], adjective=adjs["big"], number="pl", definiteness="def"),
            tverbs["carry"],
            NounPhrase(nouns["stone"], number="pl"),
        ),
        Clause(
            NounPhrase(nouns["bird"], definiteness="def"),
            iverbs["sleep"],
            obliques=[
                AdpositionalPhrase(adps["near"], NounPhrase(nouns["river"], definiteness="def"), "near")
            ],
        ),
        # Sentence types: negation, a yes/no question, and an imperative.
        Clause(NounPhrase(nouns["dog"], definiteness="def"), iverbs["sleep"], negated=True),
        Clause(
            NounPhrase(nouns["woman"], definiteness="def"),
            tverbs["see"], NounPhrase(nouns["bird"], definiteness="indef"),
            mood="interrogative",
        ),
        Clause(NounPhrase(nouns["child"]), tverbs["carry"], NounPhrase(nouns["stone"]),
               mood="imperative"),
    ]

    print("\nSample sentences:")
    for clause in clauses:
        sentence = lin.linearize(clause)
        english = _english_gloss(clause)
        print(f"\n  “{english}”")
        for line in sentence.interlinear().splitlines():
            print(f"    {line}")
    return 0


def _english_gloss(clause) -> str:
    parts = []
    if clause.mood != "imperative":
        parts.append(clause.subject.gloss)
    if clause.negated:
        parts.append("not")
    parts.append(clause.verb.gloss)
    if clause.object is not None:
        parts.append(clause.object.gloss)
    for pp in clause.obliques:
        parts.append(f"{pp.relation} {pp.np.gloss}")
    text = " ".join(parts)
    if clause.mood == "interrogative":
        text += "?"
    elif clause.mood == "imperative":
        text += "!"
    return text


def cmd_lexicon(args) -> int:
    rng = random.Random(args.seed)
    args.random = not args.inventory
    inv, phono = _build_phonology(args, rng)
    romanizer = Romanizer()
    sandhi = RuleSet.parse(_DEMO_RULES) if args.sandhi else None
    morphology = random_system(phono, rng, romanizer=romanizer, sandhi=sandhi)
    # Compound order follows the language's head-directionality (head-final unless VO).
    params = random_syntax(rng)
    lexicon = build_lexicon(
        phono, rng, romanizer=romanizer, morphology=morphology,
        head_final=not params.basic_order.is_vo,
    )

    print(inv.summary())
    print(f"\nLexicon: {len(lexicon)} words across {len(lexicon.by_field())} semantic fields\n")
    print(lexicon.glossary())

    # Highlight the non-trivial etymologies the generator produced.
    for label, etymology in (
        ("Colexified (shared words)", Etymology.COLEXIFIED),
        ("Derived", Etymology.DERIVED),
        ("Compounds", Etymology.COMPOUND),
    ):
        items = lexicon.of_etymology(etymology)
        if items:
            print(f"\n{label}:")
            for e in items:
                print(f"  {e.gloss:<12} {e.roman} /{e.ipa}/   [{e.note}]")
    return 0


def cmd_writing(args) -> int:
    import os

    rng = random.Random(args.seed)
    args.random = not args.inventory
    inv, phono = _build_phonology(args, rng)
    gen = WordGenerator(phono, Romanizer())

    wtype = WritingSystemType(args.type) if args.type else None
    ws = build_writing_system(inv, rng, wtype=wtype)

    print(inv.summary())
    print()
    print(ws.summary())
    print(
        f"  style: stroke {ws.style.stroke_width}, slant {ws.style.slant}°, "
        f"voicing mark = {ws.style.voiced_mark}"
    )

    word = gen.word(rng, min_syllables=2, max_syllables=3)
    segments = [s for syl in word.syllables for s in syl]

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, "chart.svg")
    word_path = os.path.join(out_dir, "word.svg")
    with open(chart_path, "w", encoding="utf-8") as fh:
        fh.write(ws.chart_svg())
    with open(word_path, "w", encoding="utf-8") as fh:
        fh.write(ws.word_svg(segments))

    print(f"\nSample word: {word.roman} /{word.ipa}/")
    print(f"Wrote glyph chart -> {chart_path}")
    print(f"Wrote sample word -> {word_path}")
    return 0


def cmd_generate(args) -> int:
    import json
    import os

    sandhi = RuleSet.parse(_DEMO_RULES) if args.sandhi else None
    lang = Language.generate(args.seed, sandhi=sandhi)

    if args.json:
        print(json.dumps(lang.to_dict(), ensure_ascii=False, indent=2))
        return 0

    seed_label = args.seed if args.seed is not None else "random"
    print(f"# A constructed language (seed: {seed_label})\n")
    print(lang.summary())

    if args.dictionary:
        print("\n" + lang.lexicon.glossary())
    else:
        print("\nSample vocabulary:")
        for gloss in _SHOWCASE_VOCAB:
            e = lang.lexicon.get(gloss)
            if e:
                print(f"  {gloss:<8} {e.roman} /{e.ipa}/")

    print("\nSample sentences:")
    for english, kw in _SHOWCASE_SENTENCES:
        try:
            sentence = lang.make_sentence(**kw)
        except KeyError:
            continue
        print(f"\n  “{english}”")
        for line in sentence.interlinear().splitlines():
            print(f"    {line}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        chart_path = os.path.join(args.out, "chart.svg")
        word_path = os.path.join(args.out, "word.svg")
        sample = lang.lexicon.get("woman") or next(iter(lang.lexicon.entries.values()))
        with open(chart_path, "w", encoding="utf-8") as fh:
            fh.write(lang.writing.chart_svg())
        with open(word_path, "w", encoding="utf-8") as fh:
            fh.write(lang.writing.word_svg(list(sample.form)))
        print(f"\nWrote script -> {chart_path}, {word_path} (the word for '{sample.gloss}')")
    return 0


def cmd_speak(args) -> int:
    import os

    rng = random.Random(args.seed)
    voice = Voice(f0=args.f0, rate=args.rate)
    synth = Synthesizer(voice, rng)

    if args.ipa:
        try:
            segments = [_phon_data.BY_IPA[s] for s in args.ipa.split()]
        except KeyError as exc:
            raise SystemExit(f"speak: unknown IPA symbol {exc}")
        label = f"/{args.ipa.replace(' ', '')}/"
    else:
        lang = Language.generate(args.seed)
        if args.gloss:
            entry = lang.lexicon.get(args.gloss)
            if entry is None:
                raise SystemExit(f"speak: no word for {args.gloss!r} in the lexicon")
            segments = list(entry.form)
            label = f"{args.gloss}: {entry.roman} /{entry.ipa}/"
        else:
            word = lang.word(rng, min_syllables=2, max_syllables=3)
            segments = [s for syl in word.syllables for s in syl]
            label = f"{word.roman} /{word.ipa}/"

    samples = synth.synthesize(segments)
    out = args.out
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    synth.write_wav(out, samples)

    seconds = len(samples) / voice.sample_rate
    print(f"Spoke {label}")
    print(f"  {seconds:.2f}s @ {voice.sample_rate} Hz, F0 {voice.f0:.0f} Hz -> {out}")
    return 0


def cmd_tutorial(args) -> int:
    session = TutorialSession(LanguageBuilder.start(args.seed), build_steps())
    if args.demo:
        run_demo(session)
    else:
        run_interactive(session)
    return 0


def cmd_learn(args) -> int:
    import os
    from datetime import date

    today = args.day if args.day is not None else date.today().toordinal()
    new_per_day = args.new if args.new is not None else 8

    if args.save and os.path.exists(args.save):
        try:
            course = load_course(args.save)
        except ValueError as exc:
            raise SystemExit(f"learn: cannot resume {args.save}: {exc}")
        path = args.save
        if args.seed is not None and args.seed != course.language.seed:
            print(f"(note: {args.save} holds seed {course.language.seed}; ignoring --seed {args.seed})")
        if args.new is not None:  # let an explicit --new change the daily budget on resume
            course.new_per_day = args.new
    else:
        seed = args.seed if args.seed is not None else random.Random().randrange(2**63)
        course = Course(Language.generate(seed), new_per_day=new_per_day)
        path = args.save or os.path.join("out", f"learn-{course.language.seed}.json")

    print(f"Learning the language with seed {course.language.seed}.\n")
    rng = random.Random(((course.language.seed or 0) ^ today) & (2**63 - 1))
    run_study_session(course, today, rng=rng, max_cards=args.max)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_course(course, path)
    print(f"Progress saved -> {path}")
    print(f"Resume any time:  python -m conlang learn --save {path}")
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

    m = sub.add_parser(
        "morphology",
        help="roll a morphological system and show inflection paradigms",
    )
    m.add_argument("--inventory", help='explicit IPA inventory (else random)')
    m.add_argument("--templates", help="comma-separated syllable templates")
    m.add_argument("--sandhi", action="store_true", help="apply demo sound changes at boundaries")
    m.add_argument("--max-rows", type=int, default=16, help="max paradigm cells to show (default 16)")
    m.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    m.set_defaults(func=cmd_morphology)

    y = sub.add_parser(
        "syntax",
        help="roll word-order + alignment parameters and build glossed sample sentences",
    )
    y.add_argument("--inventory", help="explicit IPA inventory (else random)")
    y.add_argument("--templates", help="comma-separated syllable templates")
    y.add_argument("--sandhi", action="store_true", help="apply demo sound changes at boundaries")
    y.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    y.set_defaults(func=cmd_syntax)

    x = sub.add_parser(
        "lexicon",
        help="generate a dictionary organized by semantic field, with etymologies",
    )
    x.add_argument("--inventory", help="explicit IPA inventory (else random)")
    x.add_argument("--templates", help="comma-separated syllable templates")
    x.add_argument("--sandhi", action="store_true", help="apply demo sound changes at boundaries")
    x.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    x.set_defaults(func=cmd_lexicon)

    w = sub.add_parser(
        "writing",
        help="generate a native script (SVG glyph chart + sample word)",
    )
    w.add_argument("--inventory", help="explicit IPA inventory (else random)")
    w.add_argument("--templates", help="comma-separated syllable templates")
    w.add_argument(
        "--type",
        choices=[t.value for t in WritingSystemType],
        help="force a script type (else rolled)",
    )
    w.add_argument("--out", default="out", help="output directory for SVG files (default out/)")
    w.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    w.set_defaults(func=cmd_writing)

    g = sub.add_parser(
        "generate",
        help="roll a complete language (all six stages) and show a full overview",
    )
    g.add_argument("--sandhi", action="store_true", help="apply sound changes at morpheme boundaries")
    g.add_argument("--dictionary", action="store_true", help="print the full glossary by field")
    g.add_argument("--json", action="store_true", help="emit a JSON snapshot instead of the overview")
    g.add_argument("--out", help="also write the native script (chart.svg, word.svg) to this dir")
    g.add_argument("--seed", type=int, default=None, help="seed (a language is fully determined by it)")
    g.set_defaults(func=cmd_generate)

    sp = sub.add_parser(
        "speak",
        help="synthesize a word to a WAV with the built-in formant synthesizer",
    )
    sp.add_argument("--ipa", help='speak an explicit IPA string, e.g. "p a t a"')
    sp.add_argument("--gloss", help="speak the language's word for this concept (e.g. water)")
    sp.add_argument("--out", default="out/speech.wav", help="output WAV path")
    sp.add_argument("--f0", type=float, default=120.0, help="fundamental pitch in Hz (default 120)")
    sp.add_argument("--rate", type=float, default=1.0, help="speech speed multiplier (default 1.0)")
    sp.add_argument("--seed", type=int, default=None, help="seed for the language and noise")
    sp.set_defaults(func=cmd_speak)

    tut = sub.add_parser(
        "tutorial",
        help="an interactive, guided walkthrough that teaches you to build a language",
    )
    tut.add_argument("--demo", action="store_true", help="play through non-interactively (random choices)")
    tut.add_argument("--seed", type=int, default=None, help="seed for reproducible choices")
    tut.set_defaults(func=cmd_tutorial)

    ln = sub.add_parser(
        "learn",
        help="learn a generated language with spaced-repetition vocabulary drills",
    )
    ln.add_argument("--seed", type=int, default=None, help="which language to learn (new course)")
    ln.add_argument("--save", help="progress file (resumes if it exists; default out/learn-<seed>.json)")
    ln.add_argument("--new", type=int, default=None, help="new cards to introduce per session (default 8)")
    ln.add_argument("--max", type=int, default=20, help="max cards to review this session (default 20)")
    ln.add_argument("--day", type=int, default=None, help="override today's day number (testing)")
    ln.set_defaults(func=cmd_learn)
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
