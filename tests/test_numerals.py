"""Tests for the numeral system: composition, base, and lexicon reuse."""

import random

import pytest

from conlang.phonology.inventory import Inventory
from conlang.phonology.phonotactics import Phonotactics
from conlang.phonology.wordgen import Romanizer
from conlang.lexicon.generator import build_lexicon
from conlang.lexicon.numerals import build_numerals, NumeralSystem, Numeral


def _materials(seed: int):
    rng = random.Random(seed)
    inv = Inventory.random(rng)
    phono = Phonotactics.random(inv, rng)
    romanizer = Romanizer()
    lexicon = build_lexicon(phono, rng, romanizer=romanizer)
    return lexicon, phono, rng, romanizer


# --- Composition --------------------------------------------------------------------
def _decimal_system() -> NumeralSystem:
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    return NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=True, units_before_tens=False, bare_base_for_one=True,
    )


def test_atoms_and_base_compose_decimally():
    s = _decimal_system()
    assert s.number(3).roman == "d3"
    assert s.number(10).roman == "ten"            # bare base for one
    assert s.number(20).roman == "d2 ten"          # multiplier before base
    assert s.number(24).roman == "d2 ten d4"       # tens then units
    assert s.number(100).roman == "hun"
    assert s.number(342).roman == "d3 hun d4 ten d2"


def test_order_switches_change_the_word():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    s = NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=False,  # base before multiplier
        units_before_tens=True,        # units before tens
        bare_base_for_one=False,       # "one ten" not "ten"
    )
    assert s.number(10).roman == "ten d1"          # one ten, base-first
    assert s.number(24).roman == "d4 ten d2"        # units first, base-first tens


def test_out_of_range_raises():
    s = _decimal_system()
    with pytest.raises(ValueError):
        s.number(0)
    with pytest.raises(ValueError):
        s.number(s.max_value + 1)


def test_counting_is_contiguous():
    s = _decimal_system()
    nums = s.counting(15)
    assert [n.value for n in nums] == list(range(1, 16))


# --- Generation ---------------------------------------------------------------------
def test_build_reuses_lexicon_small_numbers():
    lexicon, phono, rng, romanizer = _materials(3)
    system = build_numerals(lexicon, phono, rng, romanizer=romanizer, base=10)
    # atoms 1..5 must match the lexicon's existing number words
    for value, gloss in ((1, "one"), (2, "two"), (5, "five")):
        assert system.atoms[value].roman == lexicon.get(gloss).roman


def test_build_is_reproducible_and_base_is_plausible():
    a = build_numerals(*_materials(8)[:2], random.Random(8))
    b = build_numerals(*_materials(8)[:2], random.Random(8))
    assert a.base == b.base and a.base in (5, 10, 12, 20)
    assert a.number(24).roman == b.number(24).roman


def test_every_number_up_to_a_hundred_has_a_word():
    lexicon, phono, rng, romanizer = _materials(5)
    system = build_numerals(lexicon, phono, rng, romanizer=romanizer)
    for n in range(1, 101):
        num = system.number(n)
        assert num.roman and num.ipa


@pytest.mark.parametrize("base", [5, 10, 12, 20])
def test_composition_is_well_formed_across_bases(base):
    lexicon, phono, rng, romanizer = _materials(4)
    system = build_numerals(lexicon, phono, rng, romanizer=romanizer, base=base)
    assert system.base == base
    top = min(100, system.max_value)
    words = system.counting(top)
    assert [n.value for n in words] == list(range(1, top + 1))
    assert all(n.roman and n.ipa for n in words)
    # the largest representable number is composable too
    assert system.number(system.max_value).roman


def test_base_five_uses_five_as_the_base_word():
    lexicon, phono, rng, romanizer = _materials(3)
    system = build_numerals(lexicon, phono, rng, romanizer=romanizer, base=5)
    assert system.base_word.roman == lexicon.get("five").roman


def test_irregular_teens_override_regular_composition():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    s = NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=True, units_before_tens=False, bare_base_for_one=True,
        irregular={11: Numeral(11, "elv", "elv"), 12: Numeral(12, "twlv", "twlv")},
    )
    assert s.number(11).roman == "elv"     # suppletive, not the regular "ten d1"
    assert s.number(12).roman == "twlv"
    assert s.number(13).roman == "ten d3"  # 13 is not irregular -> regular composition
    assert s.number(111).roman == "hun elv"  # the teen is reused inside a larger number


def test_irregular_teens_work_under_a_vigesimal_base():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 20)}
    s = NumeralSystem(
        base=20, atoms=atoms,
        base_word=Numeral(20, "score", "score"), square_word=Numeral(400, "gross", "gross"),
        multiplier_before_base=True, units_before_tens=False, bare_base_for_one=True,
        irregular={21: Numeral(21, "xa", "xa")},
    )
    assert s.number(21).roman == "xa"              # suppletive
    assert s.number(22).roman == "score d2"        # 22 is regular (only 21 irregular)
    assert s.number(421).roman == "gross xa"       # reused compositionally (400 + 21)


def test_generated_irregular_teens_are_in_range_distinct_and_decimal_or_higher():
    for seed in range(40):
        lexicon, phono, *_ = _materials(seed)
        system = build_numerals(lexicon, phono, random.Random(seed))
        if system.irregular:
            assert system.base >= 10  # never rolled for a small (base-5) system
            others = {system.base_word.ipa, system.square_word.ipa,
                      *(a.ipa for a in system.atoms.values())}
            for value, num in system.irregular.items():
                assert system.base < value < system.base ** 2  # a teen, in range
                assert system.number(value).roman == num.roman  # used directly
                assert num.ipa not in others  # a distinct, suppletive root
            return
    raise AssertionError("no irregular-teen system in 40 seeds (unexpected)")


def test_base_five_never_has_irregular_teens():
    for seed in range(30):
        lexicon, phono, *_ = _materials(seed)
        system = build_numerals(lexicon, phono, random.Random(seed), base=5)
        assert system.irregular == {}  # restricted to base >= 10


# --- Suppletive decades & vigesimal sub-base ----------------------------------------
def test_suppletive_decades_override_regular_composition():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    s = NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=True, units_before_tens=False, bare_base_for_one=True,
        decades={20: Numeral(20, "twty", "twty"), 30: Numeral(30, "thty", "thty")},
    )
    assert s.number(20).roman == "twty"        # suppletive root, not "d2 ten"
    assert s.number(24).roman == "twty d4"      # decade root + unit
    assert s.number(30).roman == "thty"
    assert s.number(40).roman == "d4 ten"       # 40 has no suppletive root -> regular
    assert s.number(10).roman == "ten"          # the base word itself is unchanged
    assert s.number(120).roman == "hun twty"    # the decade root is reused inside a hundred


def test_vigesimal_subbase_groups_decades_in_twenties():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    s = NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=True, units_before_tens=False, bare_base_for_one=True,
        score_word=Numeral(20, "score", "score"),
    )
    assert s.number(20).roman == "score"          # one score
    assert s.number(30).roman == "score ten"       # score + a leftover ten (odd decade)
    assert s.number(80).roman == "d4 score"        # four-twenty  (French quatre-vingts)
    assert s.number(90).roman == "d4 score ten"    # four-twenty-ten (quatre-vingt-dix)
    assert s.number(70).roman == "d3 score ten"    # three-twenty-ten
    assert s.number(84).roman == "d4 score d4"     # 4*20 + 4
    assert s.number(10).roman == "ten"             # sub-ten is still just the base word
    assert s.number(190).roman == "hun d4 score ten"  # scores recurse correctly inside hundreds


def test_vigesimal_subbase_respects_units_before_tens():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    s = NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=True, units_before_tens=True, bare_base_for_one=True,
        score_word=Numeral(20, "score", "score"),
    )
    # units-first: the unit precedes the whole decade block (leftover ten stays with the score)
    assert s.number(95).roman == "d5 d4 score ten"


def test_generated_suppletive_decades_are_distinct_roots():
    lexicon, phono, _, romanizer = _materials(0)  # build materials once; re-roll numerals cheaply
    for r in range(200):
        system = build_numerals(lexicon, phono, random.Random(r), romanizer=romanizer, base=10)
        if system.decades:
            assert system.base in (10, 12)
            assert system.score_word is None  # mutually exclusive with the vigesimal sub-base
            assert set(system.decades) == {t * system.base for t in range(2, system.base)}
            for value, num in system.decades.items():
                assert system.number(value).roman == num.roman  # used directly
            return
    raise AssertionError("no suppletive-decade system in 200 rolls (unexpected)")


def test_generated_vigesimal_subbase_is_base_ten_and_scores():
    lexicon, phono, _, romanizer = _materials(0)
    for r in range(200):
        system = build_numerals(lexicon, phono, random.Random(r), romanizer=romanizer, base=10)
        if system.score_word is not None:
            assert system.base == 10 and not system.decades
            assert system.score_word.roman in system.number(80).roman   # four-twenty, not eight-ten
            assert system.base_word.roman in system.number(90).roman     # +ten for the odd decade
            return
    raise AssertionError("no vigesimal sub-base in 200 rolls (unexpected)")


def test_base_five_and_twenty_keep_regular_decades():
    lexicon, phono, _, romanizer = _materials(0)
    for base in (5, 20):
        for r in range(20):
            system = build_numerals(lexicon, phono, random.Random(r), romanizer=romanizer, base=base)
            assert system.decades == {} and system.score_word is None


def test_bare_base_false_keeps_the_one_in_hundreds():
    atoms = {v: Numeral(v, f"d{v}", f"d{v}") for v in range(1, 10)}
    s = NumeralSystem(
        base=10, atoms=atoms,
        base_word=Numeral(10, "ten", "ten"), square_word=Numeral(100, "hun", "hun"),
        multiplier_before_base=True, units_before_tens=False, bare_base_for_one=False,
    )
    assert s.number(100).roman == "d1 hun"          # "one hundred", not bare "hun"
    assert s.number(105).roman == "d1 hun d5"
