# Language Construction Code

A toolkit for generating constructed languages (conlangs), built on the principles in
Mark Rosenfelder's (zompist.com) *Language Construction Kit*, *Advanced Language
Construction Kit*, *The Conlanger's Lexipedia*, and *The Syntax Construction Kit*.

## Design philosophy

- **The engine is algorithmic.** Phonology, phonotactics, sound change, morphology, and
  word generation are deterministic algorithms — weighted random sampling over IPA
  feature space, constrained by cross-linguistic frequency data and typological
  universals. This makes output reproducible (seedable), fast, and offline.
- **AI is reserved for judgment.** Large-language-model / image-model calls are an
  *optional* later layer, used only where genuine aesthetic or semantic judgment helps:
  polishing a lexicon for sound-symbolic cohesion, seeding semantic fields, and
  designing glyphs/pictograms for a writing system. The core never requires a network.
- **Guided *and* random.** Every stage can be driven by explicit user choices (a
  conlanger's workbench) or rolled automatically into a typologically plausible language
  (push-button conlang).

## Roadmap (LCK pillars)

| Stage | Module | Status |
|-------|--------|--------|
| 1. Phonology      | `conlang.phonology` — inventory, phonotactics, word generation | **done** |
| 2. Sound change   | `conlang.soundchange` — SCA-style proto → daughter evolution    | **done** |
| 3. Morphology     | `conlang.morphology` — inflection, derivation, affixes          | **done** |
| 4. Syntax         | `conlang.syntax` — word order, alignment, glossed sentences     | **done** |
| 5. Lexicon        | `conlang.lexicon` — vocabulary, semantic fields, etymology      | **done** |
| 6. Orthography    | `conlang.writing` — featural SVG glyphs, 4 script types         | **done** |

### Capstone applications (after the engine stages)

These sit on top of the whole engine and ship last:

- **Text-to-speech** *(done)* — `conlang.speech`: a pure-Python, dependency-free formant
  synthesizer. It reads the same phonological features the engine uses (vowel
  height/backness → formants, consonant manner → buzz/noise/burst, voicing → glottal
  source) and writes a 16-bit WAV. It synthesizes the whole word in one pass through
  time-varying resonators with **formant transitions** — a vowel glides toward each
  neighbouring consonant's locus, so place of articulation is actually audible. Robotic but
  real, offline, and deterministic. Backlog: parallel-formant amplitudes, a vowel-sensitive
  velar locus, and an optional `espeak-ng` backend.
- **Tutorial** *(done)* — `conlang.tutorial`: an interactive, guided walkthrough that
  teaches the LCK ideas one stage at a time. At each step you make a real choice (or roll a
  random one) and watch a `Language` take shape, ending with a sample sentence and the seed
  to reproduce it. The flow logic is I/O-free, so it is fully testable.
- **Teaching app** *(done)* — `conlang.teach`: learn a *generated* conlang with a
  spaced-repetition vocabulary trainer. Every dictionary word becomes flashcards in both
  directions, scheduled by the SM-2 algorithm and quizzed as multiple choice, with grammar
  notes and a worked example each session. Progress persists as JSON keyed by the seed (the
  language is recovered from it), so you can resume across days.

### Deferred (Stage 2 "advanced" backlog)

The sound-change engine handles substitution, feature-class transforms, deletion, ordered
(feeding/bleeding) rule application, and — added since the first pass — **epenthesis**
(`0 > ə / C_C`), **optional environment elements** (`(C)`), **feature-agreement
assimilation** with α-features (`[nasal] > [αplace] / _[αplace plosive]`), and **multi-segment
(window) rules** with positional backreferences for metathesis (`[stop] [liquid] > 2 1`),
gemination (`[voiceless plosive] > 1 1 / V_V`), and cluster reduction/prothesis (`s k > k`,
`[plosive] > ʔ 1 / #_`). Still to add when needed: unbounded wildcards (`C*`, "any segment")
and long-distance (non-adjacent) metathesis.

### Deferred (Stage 3 "advanced" backlog)

Morphology handles inflection (agglutinative + fusional), basic class-changing
derivation, optional sandhi, **inflection classes** (declensions/conjugations: several
affix sets per word class, the extras sharing a backbone with the base via partial
syncretism; each word is assigned a class), and — added since the first pass — **stem
allomorphy** (a bound/oblique stem distinct from the citation root, formed by a final-edge
mutation — final-stop voicing, vowel raising/umlaut — and used whenever the word is overtly
inflected, optionally restricted to one triggering category). Still to add: alternation
conditioned by the following affix (e.g. only before vowel-initial endings), class-bound
stems, true analytic-particle isolating morphology, extra number values (dual/paucal) and
clusivity, zero-derivation (conversion), derivation stacking, and tying inflection class to
gender / stem shape (class assignment is currently random rather than phonologically
conditioned).

### Deferred (Stage 4 "advanced" backlog)

Syntax models constituent order, harmonic correlates, alignment (with a two-way
core-case simplification), subject/absolutive agreement, intra-NP modifier order,
adpositional phrases, and — added since the first pass — **clause-level sentence types**:
negation (a verbal marker or a negator particle), polar (yes/no) questions (a clause-
initial/final particle or intonation), and imperatives (subject dropped, 2nd person),
with the strategy harmonically tied to word order, and **relative clauses** (a gap
strategy with the head's role omitted; postnominal relatives take a relativizer particle,
prenominal ones are participial; placement follows the `relative` parameter), and
**content (wh-) questions** (an argument is an interrogative pronoun; wh-fronting vs
in-situ, VO-biased; the wh-word keeps its case), **coordination** (conjoined noun-phrase
arguments and compound sentences with a medial "and"/"or" particle; a conjoined subject
resolves to plural agreement, each conjunct keeps the same core case, and an absent
coordinator degrades to asyndetic juxtaposition), and **ditransitives** (a recipient /
indirect object marked by the language's ditransitive alignment — indirective: dative
recipient + accusative theme; secundative: the cases swapped — the recipient placed before
the theme, with object-gapping and wh-fronting affecting only the theme), and **pro-drop**
(a pronominal subject is left null when the verb's agreement is rich enough — marks both
person and number — to recover it; the verb now agrees with the subject's person, and an
ergative language keeps the subject since its transitive verb agrees with the object), and
**free-word articles** (definiteness shown by a separate determiner — grammaticalized from
the demonstrative and the numeral 'one' — at the noun phrase's left edge, suppressing the
definiteness affix so there is no double-marking; falls back to morphology when absent).
Still to build: verb-second, oblique/adjunct wh and auxiliary inversion, object agreement
and differential object marking, and postnominal/suffixed articles. Oblique phrases are
still placed clause-finally rather than by a positional parameter.

### Deferred (Stage 5 "advanced" backlog)

The lexicon builds a semantic-field dictionary with colexification, derivation, and
compounding, sized by Zipf's law of abbreviation, plus a **numeral system** (a rolled base
— decimal/vigesimal/quinary/duodecimal — with words composed past five, e.g. "two-ten-four"
= 24, order tied to head-direction), **antonyms that share morphology** (an
opposite-forming adjective affix derives the marked pole of a polar pair — bad/small/cold —
from the unmarked one when the language has it, otherwise each pole is a suppletive root),
and a **kinship system** (kin terms plus a rolled typology: siblings distinguished by sex or
merged into one sex-neutral term, and classificatory parent's-sibling merging where uncle =
father and aunt = mother, vs. descriptive distinct terms). Still to add: irregular teens /
suppletive numerals and sub-bases, polysemy networks (chains, not just binary
colexification), suppletion in basic vocabulary, semantic shift tied to sound change, and
loanword strata / register.

### Deferred (Stage 6 "advanced" backlog)

The writing system generates featural SVG glyphs for four script types, plus **numeral
glyphs** (Maya-style bars-and-dots digits in positional notation) and **punctuation** (a
daṇḍa-like sentence stop, a half-height clause pause, and an interpunct word divider, with
`sentence_svg` laying out a written sentence). Still to add: ligatures/positional forms,
explicit cluster stacking (beyond the coda virama), connecting/cursive strokes, and
right-to-left or vertical layout direction. True logographic/pictographic scripts are a
separate, larger effort.

## How this is built

Development uses a **plan → execute → review** loop with Claude subagents: an
architecture/plan pass, hands-on execution of each module, then a review subagent that
critiques each module for both linguistic accuracy and code quality before it's accepted.

## Quick start

```bash
pip install -e .

# Roll a COMPLETE language (all six stages) from one seed — the push-button entry point
python -m conlang generate --seed 42
python -m conlang generate --seed 42 --out out   # + write the native script as SVG
python -m conlang generate --json                # JSON snapshot (a language = its seed)

python -m conlang phonology --random            # roll a random plausible inventory + words
python -m conlang phonology --random --seed 42  # reproducible

# Evolve a generated proto-lexicon into a daughter language
python -m conlang soundchange --demo --seed 42 --trace
python -m conlang soundchange --seed 3 --rule "[voiceless plosive] > [+voiced] / V_V"

# Roll a morphological system and show inflection paradigms
python -m conlang morphology --seed 5
python -m conlang morphology --seed 5 --sandhi   # apply sound changes at boundaries

# Roll word order + alignment and build glossed sample sentences
python -m conlang syntax --seed 8

# Generate a dictionary by semantic field, with colexification/derivation/compounds
python -m conlang lexicon --seed 8

# Generate a native script as SVG (featural glyphs) -> out/chart.svg, out/word.svg
python -m conlang writing --seed 8
python -m conlang writing --seed 3 --type abugida

# Speak a word with the built-in formant synthesizer -> a WAV file
python -m conlang speak --ipa "p a t a k a" --out out/word.wav
python -m conlang speak --seed 42 --gloss water --out out/water.wav

# Learn to build a language with the interactive, guided tutorial
python -m conlang tutorial
python -m conlang tutorial --demo --seed 7   # non-interactive walkthrough

# Learn a generated language with spaced-repetition vocabulary drills (resumable)
python -m conlang learn --seed 42
```

## Layout

```
conlang/
  phonology/
    features.py      # IPA feature system + Segment model
    data.py          # IPA charts with cross-linguistic frequencies
    inventory.py     # phoneme inventory: random-plausible & guided
    phonotactics.py  # syllable templates, onset/coda constraints, clusters
    wordgen.py       # frequency-weighted word/root generator + romanization
  soundchange/
    matcher.py       # feature classes, natural-class matching, reverse feature lookup
    rule.py          # parse + apply one "target > replacement / environment" rule
    ruleset.py       # ordered rules, category defs, derivation over a lexicon
  morphology/
    features.py      # grammatical categories, FeatureBundle, WordClass, Typology
    affix.py         # Affix: form + position + marked features; attach to a stem
    paradigm.py      # agglutinative + fusional inflection, paradigm tables, derivation
    generator.py     # roll a plausible morphological system (typology, affixes, sandhi)
  syntax/
    parameters.py    # word order, head-directionality correlates, alignment
    structure.py     # Lexeme, NounPhrase, AdpositionalPhrase, Clause
    linearizer.py    # case by alignment, agreement, ordering -> glossed sentence
    generator.py     # roll plausible syntax parameters
  lexicon/
    concepts.py      # semantic-field concept inventory + colexification/derivation/compounds
    lexicon.py       # LexicalEntry (etymology) + Lexicon container (lookups, glossary)
    generator.py     # coin roots (Zipf length), colexify, derive, compound
  writing/
    glyph.py         # stroke primitives (line/path/circle) + Glyph -> SVG
    featural.py      # deterministic featural glyph from a segment's features
    system.py        # WritingSystem (alphabet/abjad/abugida/syllabary); word + chart SVG
    generator.py     # roll a script type + per-language style
  speech/
    phones.py        # map a segment's features -> an acoustic plan (formants/noise/bursts)
    synth.py         # formant synthesizer: glottal source, resonators, noise -> WAV
  tutorial/
    builder.py       # LanguageBuilder: accumulate per-stage choices -> Language
    content.py       # the teaching steps (LCK lessons) + choices
    session.py       # pure flow logic (no I/O, testable)
    runner.py        # interactive prompt + non-interactive demo
  teach/
    srs.py           # SM-2 spaced-repetition scheduler
    cards.py         # flashcards + deck built from a language (frequent first)
    course.py        # due selection, new cards, multiple-choice questions (pure logic)
    progress.py      # save/load review state as JSON (keyed by seed)
    runner.py        # interactive study session
  language.py        # the Language aggregate: all six stages in one object + generate(seed)
  cli.py             # command-line interface (guided + random)
tests/               # pytest suite
```
