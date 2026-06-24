"""Roll a typologically plausible morphological system.

Choices are made the way the rest of the toolkit makes them — weighted sampling guided
by cross-linguistic tendencies:

- A **typology** is chosen (agglutinative is the most common, then fusional, then
  isolating).
- For each word class, the categories it marks are drawn by their ``commonness`` (an
  isolating language marks far fewer).
- A dominant **affix position** is picked per language, biased toward suffixing — the
  strong cross-linguistic preference.
- Affix forms are short morphemes generated from the language's own inventory, so the
  morphology sounds like the phonology.

Accidental syncretism (two cells sharing a form) is allowed, because it is realistic.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from conlang.phonology.features import Segment
from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import Romanizer
from conlang.morphology.features import (
    CATEGORIES,
    WORD_CLASSES,
    FeatureBundle,
    GrammaticalCategory,
    Typology,
)
from conlang.morphology.affix import Affix, Position
from conlang.morphology.paradigm import (
    Paradigm, DerivationRule, InflectionClass, StemAlternation,
)

# Derivation templates: (from_class, to_class, gloss).
_DERIVATION_TEMPLATES = [
    ("verb", "noun", "AGENT"),
    ("verb", "noun", "RESULT"),
    ("noun", "adjective", "HAVING"),
    ("adjective", "verb", "BECOME"),
    ("noun", "noun", "DIMINUTIVE"),
    # An antonym/opposite-forming affix (like English un-/dis-, Esperanto mal-): derives the
    # marked pole of a polar adjective pair from the unmarked one (bad = opposite-of-good).
    ("adjective", "adjective", "ANTONYM"),
]


@dataclass
class MorphologySystem:
    typology: Typology
    paradigms: dict[str, Paradigm]
    derivations: list[DerivationRule] = field(default_factory=list)

    def inflect(self, word_class: str, root, bundle: FeatureBundle, inflection_class=None):
        return self.paradigms[word_class].inflect(root, bundle, inflection_class)

    def inflection_classes(self, word_class: str) -> list[str]:
        par = self.paradigms.get(word_class)
        return par.class_ids() if par is not None else ["1"]

    def derive(self, rule: DerivationRule, root, bundle: FeatureBundle | None = None):
        """Apply a derivation, then inflect the derived stem with its target class.

        This is derivation *feeding* inflection: e.g. a verb root -> agent noun, which
        then takes that language's noun inflection. If the target class has no paradigm
        (or no bundle is given), the bare derived stem is returned.
        """
        stem = rule.apply(root)
        target = self.paradigms.get(rule.to_class)
        if target is None or bundle is None:
            return stem
        return target.inflect(stem, bundle)

    def summary(self) -> str:
        lines = [f"Morphology: {self.typology.value}"]
        for name, par in self.paradigms.items():
            marked = ", ".join(c.name for c in par.marked) or "(none)"
            n = (
                len(par.fusional_affixes)
                if self.typology is Typology.FUSIONAL
                else len(par.agglutinative_affixes)
            )
            classes = len(par.class_ids())
            cls_note = f", {classes} inflection classes" if classes > 1 else ""
            lines.append(f"  {name}: marks {marked}  [{n} affixes{cls_note}]")
        if self.derivations:
            der = ", ".join(f"{d.gloss}({d.from_class}->{d.to_class})" for d in self.derivations)
            lines.append(f"  derivation: {der}")
        return "\n".join(lines)


def random_system(
    phonotactics: Phonotactics,
    rng: random.Random | None = None,
    *,
    romanizer: Romanizer | None = None,
    sandhi: object | None = None,
    classes: tuple[str, ...] = ("noun", "verb", "adjective"),
    typology: Typology | None = None,
) -> MorphologySystem:
    rng = rng or random.Random()
    romanizer = romanizer or Romanizer()
    inventory = phonotactics.inventory

    if typology is None:
        typology = rng.choices(
            [Typology.AGGLUTINATIVE, Typology.FUSIONAL, Typology.ISOLATING],
            weights=[0.45, 0.35, 0.20],
            k=1,
        )[0]
    # Suffixing is cross-linguistically dominant; pick a per-language bias.
    dominant = Position.SUFFIX if rng.random() < 0.7 else Position.PREFIX

    paradigms: dict[str, Paradigm] = {}
    for class_name in classes:
        word_class = WORD_CLASSES[class_name]
        marked = _choose_marked_categories(rng, word_class.categories(), typology)
        paradigms[class_name] = _build_paradigm(
            rng, word_class, typology, marked, dominant, inventory, romanizer, sandhi
        )

    derivations = _build_derivations(rng, dominant, inventory)
    return MorphologySystem(typology, paradigms, derivations)


# --- Helpers ------------------------------------------------------------------------
def _choose_marked_categories(
    rng: random.Random, categories: list[GrammaticalCategory], typology: Typology
) -> tuple[GrammaticalCategory, ...]:
    scale = 0.25 if typology is Typology.ISOLATING else 1.0
    chosen = [c for c in categories if rng.random() < c.commonness * scale]
    # A non-isolating language should mark at least one category per class.
    if not chosen and typology is not Typology.ISOLATING and categories:
        chosen = [max(categories, key=lambda c: c.commonness)]
    return tuple(chosen)


def _build_paradigm(
    rng, word_class, typology, marked, dominant, inventory, romanizer, sandhi
) -> Paradigm:
    par = Paradigm(
        word_class=word_class,
        typology=typology,
        marked=marked,
        romanizer=romanizer,
        sandhi=sandhi,
    )
    if not marked:
        return par  # nothing to inflect -> no affixes, and no stem alternation to trigger

    n_classes = _choose_class_count(rng, typology)
    # The first inflection class fills the paradigm's own affix fields ("1"). Each extra
    # class is a *perturbation* of it — most endings are shared (syncretism across
    # declensions), a minority differ — so the classes look related, as real declensions do
    # rather than like unrelated languages.
    base_aggl, base_fus = _build_affix_set(rng, typology, marked, dominant, inventory)
    par.agglutinative_affixes, par.fusional_affixes = base_aggl, base_fus
    for k in range(2, n_classes + 1):
        aggl, fus = _perturb_affix_set(
            rng, base_aggl, base_fus, typology, dominant, inventory
        )
        par.extra_classes[str(k)] = InflectionClass(aggl, fus)
    par.stem_alternation = _build_stem_alternation(rng)
    return par


# Stem-allomorphy templates: a final-edge mutation forming the bound stem. Each is a
# Stage-2 rule fired (word-finally on the bare stem) whenever the word is overtly inflected.
_STEM_ALTERNATIONS = (
    "[voiceless plosive] > [+voiced] / _#",   # final-stop voicing (intervocalic-lenition feel)
    "[voiced plosive] > [-voiced] / _#",       # final-stop devoicing
    "a > e / _#",                              # final-vowel raising / umlaut, low -> mid
    "o > u / _#",                              # ... mid -> high (back)
    "e > i / _#",                              # ... mid -> high (front)
)


def _build_stem_alternation(rng: random.Random) -> StemAlternation | None:
    """Roll a bound-stem allomorphy rule for some languages (most have none)."""
    if rng.random() >= 0.30:
        return None
    from conlang.soundchange.ruleset import RuleSet  # local: keep Stage 2 a soft dependency

    rule = rng.choice(_STEM_ALTERNATIONS)
    return StemAlternation(RuleSet.from_rules([rule]))


def _perturb_affix_set(rng, base_aggl, base_fus, typology, dominant, inventory):
    """A new inflection class derived from a base: keep most cells, regenerate a minority.

    At least one cell is forced to differ so the class is genuinely distinct from the base.
    """
    base = base_fus if typology is Typology.FUSIONAL else base_aggl
    items = list(base.items())
    forced = rng.randrange(len(items)) if items else -1
    out: dict = {}
    for idx, (key, affix) in enumerate(items):
        if idx == forced or rng.random() < 0.45:  # this declension differs here
            form = _random_affix_form(rng, inventory)
            out[key] = Affix(form, dominant, affix.marks, gloss=affix.gloss)
        else:
            out[key] = affix  # shared with the base declension
    return (({}, out) if typology is Typology.FUSIONAL else (out, {}))


def _choose_class_count(rng: random.Random, typology: Typology) -> int:
    """How many inflection classes a word class has. Fusional languages have declensions;
    agglutinative ones are usually regular; isolating ones have a single (trivial) class."""
    if typology is Typology.ISOLATING:
        return 1
    if typology is Typology.AGGLUTINATIVE:
        return rng.choices([1, 2], weights=[0.8, 0.2], k=1)[0]
    return rng.choices([1, 2, 3, 4], weights=[0.2, 0.35, 0.3, 0.15], k=1)[0]


def _build_affix_set(rng, typology, marked, dominant, inventory):
    """Build one inflection class's affixes: (agglutinative dict, fusional dict)."""
    agglutinative: dict = {}
    fusional: dict = {}
    if typology is Typology.FUSIONAL:
        names = [c.name for c in marked]
        for combo in itertools.product(*[c.values for c in marked]):
            bundle = FeatureBundle.from_dict(dict(zip(names, combo)))
            if all(val == cat.base for cat, val in zip(marked, combo)):
                continue  # all-base combination is the zero (citation) form
            form = _random_affix_form(rng, inventory)
            fusional[bundle] = Affix(form, dominant, bundle, gloss=str(bundle))
    else:  # agglutinative / isolating: one affix per marked (non-base) value
        for cat in marked:
            for value in cat.marked_values:
                form = _random_affix_form(rng, inventory)
                marks = FeatureBundle.of(**{cat.name: value})
                agglutinative[(cat.name, value)] = Affix(
                    form, dominant, marks, gloss=value.upper()
                )
    return agglutinative, fusional


def _build_derivations(rng, dominant, inventory) -> list[DerivationRule]:
    k = rng.randint(2, 3)
    templates = rng.sample(_DERIVATION_TEMPLATES, k=min(k, len(_DERIVATION_TEMPLATES)))
    rules = []
    for from_class, to_class, gloss in templates:
        form = _random_affix_form(rng, inventory)
        affix = Affix(form, dominant, FeatureBundle.of(), gloss=gloss)
        rules.append(DerivationRule(affix, from_class, to_class, gloss))
    return rules


def _random_affix_form(rng: random.Random, inventory: Inventory) -> tuple[Segment, ...]:
    """Generate a short, non-empty affix shape (sub-syllabic to one syllable).

    A marked (non-base) affix must never be empty — a zero form would silently collapse
    the marked cell into the citation form. We pick from shapes the inventory can supply,
    and fall back to a single available segment, raising only if the inventory is empty.
    """
    if not inventory.segments:
        raise ValueError("cannot generate an affix from an empty inventory")

    # Only offer shapes the inventory can actually fill (e.g. no C-slots without consonants).
    shapes = ["V", "C", "CV", "VC", "CVC"]
    weights = [0.25, 0.15, 0.35, 0.15, 0.10]
    usable = [
        (sh, w)
        for sh, w in zip(shapes, weights)
        if ("C" not in sh or inventory.consonants) and ("V" not in sh or inventory.vowels)
    ]
    shape = rng.choices([s for s, _ in usable], weights=[w for _, w in usable], k=1)[0]

    segments: list[Segment] = []
    for slot in shape:
        pool = inventory.consonants if slot == "C" else inventory.vowels
        s_weights = [max(s.frequency, 1e-9) for s in pool]
        segments.append(rng.choices(pool, weights=s_weights, k=1)[0])
    if not segments:  # shape was empty-safe but produced nothing; guarantee one segment
        segments.append(rng.choice(inventory.segments))
    return tuple(segments)
