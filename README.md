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
| 3. Morphology     | `conlang.morphology` — inflection, derivation, affixes          | planned |
| 4. Syntax         | `conlang.syntax` — word order, grammar rules                    | planned |
| 5. Lexicon        | `conlang.lexicon` — vocabulary & semantic fields                | planned |
| 6. Orthography    | `conlang.writing` — romanization, glyph/pictogram generation    | planned |

### Deferred (Stage 2 "advanced" backlog)

The sound-change engine handles substitution, feature-class transforms, deletion, and
ordered (feeding/bleeding) rule application. Still to add when needed: epenthesis /
insertion (`0 > V / …`), feature-agreement assimilation (α-features, "assimilate to the
following place" in one rule), optional / variable-length environment elements (`(C)`,
wildcards), and multi-segment targets (gemination, metathesis).

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
  cli.py             # command-line interface (guided + random)
tests/               # pytest suite
```
