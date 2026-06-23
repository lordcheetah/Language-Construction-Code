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
| 5. Lexicon        | `conlang.lexicon` — vocabulary & semantic fields                | planned |
| 6. Orthography    | `conlang.writing` — romanization, glyph/pictogram generation    | planned |

### Capstone applications (after the engine stages)

These sit on top of the whole engine and ship last:

- **Tutorial** — an interactive, guided walkthrough that teaches a user to *create* a
  language, driving the engine stage by stage.
- **Text-to-speech** — pronounce generated words from their IPA. Offline-first via
  `espeak-ng` (which accepts IPA input directly), in keeping with the project philosophy.
- **Teaching app** *(the big one)* — learn a *generated* conlang: lessons, flashcards,
  and spaced repetition built from the language's phonology, lexicon, and grammar.

### Deferred (Stage 2 "advanced" backlog)

The sound-change engine handles substitution, feature-class transforms, deletion, and
ordered (feeding/bleeding) rule application. Still to add when needed: epenthesis /
insertion (`0 > V / …`), feature-agreement assimilation (α-features, "assimilate to the
following place" in one rule), optional / variable-length environment elements (`(C)`,
wildcards), and multi-segment targets (gemination, metathesis).

### Deferred (Stage 3 "advanced" backlog)

Morphology handles inflection (agglutinative + fusional) and basic class-changing
derivation, with optional sandhi. The largest fidelity gaps to add later: inflection
classes / declensions / conjugations (multiple affix sets per word class), stem
allomorphy, true analytic-particle isolating morphology (free grammatical words rather
than affixes), extra number values (dual/paucal) and clusivity, zero-derivation
(conversion), and derivation stacking.

### Deferred (Stage 4 "advanced" backlog)

Syntax models constituent order, harmonic correlates, alignment (with a two-way
core-case simplification), subject/absolutive agreement, intra-NP modifier order, and
adpositional phrases. Still to build: relative clauses, clause-level negation and
questions, verb-second, pro-drop, object agreement and differential object marking,
free-word articles, coordination, and ditransitives. Oblique phrases are currently placed
clause-finally rather than by a positional parameter.

## How this is built

Development uses a **plan → execute → review** loop with Claude subagents: an
architecture/plan pass, hands-on execution of each module, then a review subagent that
critiques each module for both linguistic accuracy and code quality before it's accepted.

## Quick start

```bash
pip install -e .
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
  cli.py             # command-line interface (guided + random)
tests/               # pytest suite
```
