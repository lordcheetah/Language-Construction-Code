"""Tests for the phonology stage.

These check the two things that matter for a generator: it is *reproducible* (a seed
fully determines output) and it is *plausible* (the typological universals actually
hold). They are deliberately property-style — asserting invariants over many random
rolls rather than pinning exact strings — so they keep their value as the data evolves.
"""

import random

import pytest

from conlang.phonology import data
from conlang.phonology.features import Consonant, Voicing, Manner, sonority
from conlang.phonology.inventory import Inventory, _OBSTRUENT_MANNERS
from conlang.phonology.phonotactics import Phonotactics, SyllableTemplate
from conlang.phonology.wordgen import WordGenerator, Romanizer
from conlang import cli


# --- Data integrity -----------------------------------------------------------------
def test_ipa_symbols_unique():
    symbols = [s.ipa for s in data.ALL_SEGMENTS]
    assert len(symbols) == len(set(symbols))


def test_frequencies_in_range():
    assert all(0.0 <= s.frequency <= 1.0 for s in data.ALL_SEGMENTS)


# --- Inventory ----------------------------------------------------------------------
def test_random_inventory_is_reproducible():
    a = Inventory.random(random.Random(42))
    b = Inventory.random(random.Random(42))
    assert a.summary() == b.summary()


def test_random_inventory_has_universal_nasals():
    for seed in range(25):
        inv = Inventory.random(random.Random(seed))
        assert inv.has("m") and inv.has("n")


def test_voiced_obstruent_implies_voiceless():
    for seed in range(40):
        inv = Inventory.random(random.Random(seed))
        present = {(c.place, c.manner, c.voicing) for c in inv.consonants}
        for c in inv.consonants:
            if c.voicing is Voicing.VOICED and c.manner in _OBSTRUENT_MANNERS:
                assert (c.place, c.manner, Voicing.VOICELESS) in present, (
                    f"{c.ipa} present without its voiceless counterpart (seed {seed})"
                )


def test_velar_nasal_implies_velar_stop():
    for seed in range(60):
        inv = Inventory.random(random.Random(seed))
        if inv.has("ŋ"):
            assert inv.has("k") or inv.has("g")


def test_vowel_systems_are_canonical_sizes():
    for seed in range(30):
        inv = Inventory.random(random.Random(seed))
        assert 3 <= len(inv.vowels) <= 9


def test_from_ipa_roundtrip():
    inv = Inventory.from_ipa("p t k a i u")
    assert inv.size == 6
    assert inv.has("p") and inv.has("u")


def test_from_ipa_rejects_unknown_symbol():
    with pytest.raises(KeyError):
        Inventory.from_ipa("p t Q")  # capital Q is not in the chart


# --- Phonotactics -------------------------------------------------------------------
def test_template_parsing():
    t = SyllableTemplate.parse("(C)(C)V(C)")
    kinds = [(s.kind, s.optional) for s in t.slots]
    assert kinds == [("C", True), ("C", True), ("V", False), ("C", True)]


def test_template_requires_a_nucleus():
    with pytest.raises(ValueError):
        SyllableTemplate.parse("CC")


def test_vowels_are_the_sonority_peak():
    # Every vowel must outrank every consonant, including glides/approximants.
    max_consonant = max(sonority(c) for c in data.CONSONANTS)
    assert all(sonority(v) > max_consonant for v in data.VOWELS)


def test_syllable_clusters_are_well_formed():
    inv = Inventory.from_ipa("p t k s ʃ l r m n a i u")
    phono = Phonotactics.from_notation(inv, ["(C)(C)V(C)(C)"])
    ssd = phono.min_sonority_distance
    rng = random.Random(7)
    for _ in range(400):
        syl = phono.random_syllable(rng)
        vowel_idx = next(i for i, s in enumerate(syl) if s.is_vowel)
        onset = syl[:vowel_idx]
        coda = syl[vowel_idx + 1 :]
        for i in range(len(onset) - 1):
            a, b = onset[i], onset[i + 1]
            assert sonority(b) - sonority(a) >= ssd  # strictly rising w/ min distance
            assert a.ipa != b.ipa  # OCP: no geminate cluster
            # no stop/affricate followed by a nasal in the onset (no /pm/, /kn/)
            assert not (
                a.manner in (Manner.PLOSIVE, Manner.AFFRICATE) and b.manner is Manner.NASAL
            )
        for i in range(len(coda) - 1):
            a, b = coda[i], coda[i + 1]
            assert sonority(a) - sonority(b) >= ssd  # strictly falling w/ min distance
            assert a.ipa != b.ipa


def test_syllable_always_has_a_vowel():
    inv = Inventory.from_ipa("p t k a i u")
    phono = Phonotactics.from_notation(inv, ["(C)V(C)"])
    rng = random.Random(3)
    for _ in range(100):
        syl = phono.random_syllable(rng)
        assert any(s.is_vowel for s in syl)


# --- Word generation ----------------------------------------------------------------
def test_lexicon_words_are_distinct():
    inv = Inventory.from_ipa("p t k s m n l a i u o e")
    phono = Phonotactics.from_notation(inv, ["(C)V", "(C)V(C)"])
    gen = WordGenerator(phono)
    words = gen.lexicon(50, random.Random(11))
    assert len({w.ipa for w in words}) == len(words)


def test_word_generation_reproducible():
    inv = Inventory.random(random.Random(5))
    phono = Phonotactics.random(inv, random.Random(5))
    gen = WordGenerator(phono)
    a = [str(w) for w in gen.lexicon(20, random.Random(99))]
    b = [str(w) for w in gen.lexicon(20, random.Random(99))]
    assert a == b


def test_romanizer_overrides():
    rom = Romanizer(overrides={"ʃ": "x"})
    sh = data.consonant("ʃ")
    assert rom.spell(sh) == "x"


def test_cli_seeded_run_is_reproducible(capsys):
    args = cli.build_parser().parse_args(
        ["phonology", "--random", "--seed", "123", "--count", "10"]
    )
    cli.cmd_phonology(args)
    first = capsys.readouterr().out
    cli.cmd_phonology(args)
    second = capsys.readouterr().out
    assert first == second
    assert "Inventory" in first


def test_full_pipeline_smoke():
    rng = random.Random(2024)
    inv = Inventory.random(rng)
    phono = Phonotactics.random(inv, rng)
    gen = WordGenerator(phono)
    words = gen.lexicon(10, rng)
    assert len(words) == 10
    assert all(w.roman and w.ipa for w in words)


# --- IPA pronunciation guide --------------------------------------------------------
def test_every_producible_symbol_has_a_pronunciation_hint():
    # the guide must cover every segment the generator can emit, so a word never contains a
    # symbol the learner can't look up
    from conlang.phonology.ipa_guide import PRONUNCIATION
    missing = [s.ipa for s in data.ALL_SEGMENTS if s.ipa not in PRONUNCIATION]
    assert not missing, f"no pronunciation hint for: {missing}"


def test_describe_accepts_symbol_or_segment():
    from conlang.phonology.ipa_guide import describe
    assert "ship" in describe("ʃ")
    assert "ship" in describe(data.consonant("ʃ"))  # segment object, same hint


def test_describe_falls_back_to_features_for_an_unlisted_segment():
    # a hypothetical segment with no curated hint is still described from its features
    from conlang.phonology.features import Consonant, Place, Manner, Voicing
    from conlang.phonology.ipa_guide import describe
    fake = Consonant("ǂ", 0.0, Place.PALATAL, Manner.PLOSIVE, Voicing.VOICELESS)
    desc = describe(fake)
    assert "palatal" in desc and "plosive" in desc


def test_pronunciation_key_lists_each_phoneme_once_and_aligned():
    from conlang.phonology.ipa_guide import pronunciation_key
    inv = Inventory.from_ipa("p t k a i u")
    key = pronunciation_key(inv.segments)
    lines = key.splitlines()
    assert len(lines) == 6  # one row per phoneme, no duplicates
    assert all(line.startswith("  /") for line in lines)
    assert "boot" in key  # the /u/ hint


def test_pronunciation_key_dedups_repeated_segments():
    from conlang.phonology.ipa_guide import pronunciation_key
    p = data.consonant("p")
    assert len(pronunciation_key([p, p, p]).splitlines()) == 1


def test_ipa_cli_keys_a_seeded_inventory(capsys):
    args = cli.build_parser().parse_args(["ipa", "--seed", "8"])
    cli.cmd_ipa(args)
    out = capsys.readouterr().out
    assert "Pronunciation key" in out and "Inventory" in out


def test_ipa_cli_all_lists_consonants_and_vowels(capsys):
    args = cli.build_parser().parse_args(["ipa", "--all"])
    cli.cmd_ipa(args)
    out = capsys.readouterr().out
    assert "Consonants:" in out and "Vowels:" in out
    assert "church" in out  # the /t͡ʃ/ hint, an affricate, is present


def test_phonology_cli_includes_a_pronunciation_key_unless_suppressed(capsys):
    base = ["phonology", "--random", "--seed", "5"]
    cli.cmd_phonology(cli.build_parser().parse_args(base))
    assert "Pronunciation key" in capsys.readouterr().out
    cli.cmd_phonology(cli.build_parser().parse_args(base + ["--no-key"]))
    assert "Pronunciation key" not in capsys.readouterr().out


def test_ipa_cli_keys_an_explicit_inventory(capsys):
    args = cli.build_parser().parse_args(["ipa", "--inventory", "p t k a i u"])
    cli.cmd_ipa(args)
    out = capsys.readouterr().out
    assert "Pronunciation key" in out
    assert "boot" in out and "see" in out  # /u/ and /i/ hints
    assert "church" not in out  # /t͡ʃ/ isn't in this inventory


def test_describe_unknown_string_is_graceful():
    from conlang.phonology.ipa_guide import describe
    assert "no pronunciation hint" in describe("ZZZ")  # never raises on an unknown symbol


def test_pronunciation_key_of_nothing_is_empty():
    from conlang.phonology.ipa_guide import pronunciation_key
    assert pronunciation_key([]) == ""
