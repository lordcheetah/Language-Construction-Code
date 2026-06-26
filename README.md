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
  neighbouring consonant's locus, so place of articulation is actually audible — and the
  **velar pinch**, where a velar's F2 locus assimilates toward the adjacent vowel (higher by
  a front vowel, lower by a back one). Robotic but real, offline, and deterministic. Backlog:
  parallel-formant amplitudes and an optional `espeak-ng` backend.
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
assimilation** with α-features (`[nasal] > [αplace] / _[αplace plosive]`), **multi-segment
(window) rules** with positional backreferences for metathesis (`[stop] [liquid] > 2 1`),
gemination (`[voiceless plosive] > 1 1 / V_V`), and cluster reduction/prothesis (`s k > k`,
`[plosive] > ʔ 1 / #_`), **unbounded wildcards** — Kleene `X*` (zero or more) and `X+`
(one or more) over a class, e.g. assimilation across any number of consonants
(`[nasal] > [αplace] / _C*[αplace plosive]`), and **long-distance metathesis** — a window
target may include one variable-width `*` slot (and `.` matches any segment), so two segments
swap across a span (`[liquid] .* [liquid] > 3 2 1`, the *miraclo*→*milagro* alternation). The
sound-change engine is now feature-complete.

### Deferred (Stage 3 "advanced" backlog)

Morphology handles inflection (agglutinative + fusional), basic class-changing
derivation, optional sandhi, **inflection classes** (declensions/conjugations: several
affix sets per word class, the extras sharing a backbone with the base via partial
syncretism; each word is assigned a class), and — added since the first pass — **stem
allomorphy** (a bound/oblique stem distinct from the citation root, formed by a final-edge
mutation — final-stop voicing, vowel raising/umlaut — and used whenever the word is overtly
inflected, optionally restricted to one triggering category) and optional extra **number
values** (a minority of languages roll a dual, a trial, and/or a paucal — a trial requires a
dual, per the implicational universal — composing into anything from sg/pl up to a five-way
sg/dual/trial/paucal/pl system, applied consistently across nouns, verbs and adjectives and
glossed `DU`/`TRI`/`PAUC`), and **zero-derivation** (conversion: ~15% of
eligible derivations — AGENT/RESULT/BECOME — are zero-marked, so the word changes class with
no affix and the product is homophonous with its base, English *to water*). Stem allomorphy
can also be **affix-conditioned** — a mutation that fires only before a vowel- (or consonant-)
initial suffix, the Finnish-gradation / Celtic-mutation pattern (it keys off the innermost,
stem-adjacent suffix and stays strong word-finally). The verb can also mark **clusivity** —
a 1st-person inclusive ("you and I") vs exclusive ("they and I") distinction, glossed
`1PL.INCL`/`1PL.EXCL`, shown only for a non-singular 1st-person subject the verb agrees with.
Still to add: class-bound stems, true analytic-particle isolating morphology, and a separate
inclusive/exclusive *pronoun*. **Derivation stacking**
is supported — a word can be built by two derivational steps (e.g. *stone* → *stony* [HAVING] →
*petrify* [BECOME], carrying both affixes when the language has them). A gender-marking language
now assigns each noun a **lexical gender** and ties its inflection
class to that gender (declensions track gender, Latin-style — a per-language gender→declension
map dealt round-robin so different languages pair them differently), and adjectives agree
with their head noun's gender.

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
ergative language keeps the subject since its transitive verb agrees with the object),
**free-word articles** (definiteness shown by a separate determiner — grammaticalized from
the demonstrative and the numeral 'one' — at the noun phrase's left edge, suppressing the
definiteness affix so there is no double-marking; falls back to morphology when absent), and
**differential object marking** (only a prominent/definite object takes the accusative; an
indefinite or bare object — and a non-specific wh-object — is left unmarked, a coordinated
object marked only when every conjunct is definite), and **verb-second** (the finite verb
sits second in a main clause, overriding the base order — the subject fronts in a plain
declarative, a wh-word in a content question, and a polar question is verb-first; relative
clauses keep the base order since V2 is main-clause only), and **object agreement** (a
minority of languages are polypersonal — the verb cross-references its object too via
object_person/object_number, glossed `see.1SG>3PL`; under ergative the primary slot indexes
the absolutive object and the secondary the agent), and **oblique topicalization** (a
`Clause.topic` — one of the clause's obliques — fronts to clause-initial position; it is the
canonical verb-second first constituent, so under V2 the verb follows it, while a content
question's wh-fronting takes precedence over it; main-clause only, so a relative clause keeps
its oblique clause-final). Still to build: auxiliary inversion and postnominal/suffixed
articles. Non-topic oblique phrases are still placed clause-finally rather than by a
positional parameter.

### Deferred (Stage 5 "advanced" backlog)

The lexicon builds a semantic-field dictionary with colexification, derivation, and
compounding, sized by Zipf's law of abbreviation, plus a **numeral system** (a rolled base
— decimal/vigesimal/quinary/duodecimal — with words composed past five, e.g. "two-ten-four"
= 24, order tied to head-direction), **antonyms that share morphology** (an
opposite-forming adjective affix derives the marked pole of a polar pair — bad/small/cold —
from the unmarked one when the language has it, otherwise each pole is a suppletive root),
and a **kinship system** (kin terms plus a rolled typology: siblings distinguished by sex or
merged into one sex-neutral term, and classificatory parent's-sibling merging where uncle =
father and aunt = mother, vs. descriptive distinct terms), and **irregular teens** (a base-≥10
language may give the first few numbers above the base — `base+1 … base+3` — their own
suppletive root, English *eleven/twelve*, reused compositionally in larger numbers). Still to
add: suppletive decades and sub-bases (French *quatre-vingts*), polysemy networks (chains, not
just binary colexification), suppletion in basic vocabulary, semantic shift tied to sound
change, and loanword strata / register.

### Deferred (Stage 6 "advanced" backlog)

The writing system generates featural SVG glyphs for four script types, plus **numeral
glyphs** (Maya-style bars-and-dots digits in positional notation), **punctuation** (a
daṇḍa-like sentence stop, a half-height clause pause, and an interpunct word divider, with
`sentence_svg` laying out a written sentence), and a rolled **layout direction**
(left-to-right, right-to-left, or top-to-bottom — running text flows accordingly while the
reference chart and numerals stay left-to-right). Still to add: ligatures/positional forms,
explicit cluster stacking (beyond the coda virama), and connecting/cursive strokes. True
logographic/pictographic scripts are a separate, larger effort.

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
