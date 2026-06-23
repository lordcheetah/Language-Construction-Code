"""Tests for the Language aggregate that wires all six stages together."""

import json
import random

from conlang.language import Language
from conlang.lexicon.concepts import CONCEPTS
from conlang.syntax.linearizer import Sentence


def test_generate_is_fully_determined_by_seed():
    a = Language.generate(42)
    b = Language.generate(42)
    assert a.to_dict() == b.to_dict()


def test_determinism_holds_across_many_seeds():
    for seed in range(20):
        assert Language.generate(seed).to_dict() == Language.generate(seed).to_dict()


def test_random_language_records_a_reproducible_seed():
    lang = Language.generate()  # no seed
    assert lang.seed is not None
    again = Language.generate(lang.seed)
    assert again.to_dict() == lang.to_dict()


def test_generate_wires_all_stages():
    lang = Language.generate(7)
    assert lang.inventory.consonants and lang.inventory.vowels
    assert lang.phonotactics.templates
    assert lang.morphology.paradigms
    assert lang.syntax.basic_order
    assert len(lang.lexicon) == len(CONCEPTS)
    assert lang.writing.consonants


def test_compound_order_follows_generated_syntax():
    # The lexicon's compounds must use the same head-directionality the syntax rolled.
    lang = Language.generate(3)
    head_final = not lang.syntax.basic_order.is_vo
    waterfall = lang.lexicon.get("waterfall")
    water, fall = lang.lexicon.get("water").ipa, lang.lexicon.get("fall").ipa
    if waterfall.note in ("water+fall", "fall+water"):  # compound (not a coined fallback)
        expected = water + fall if head_final else fall + water
        assert waterfall.ipa == expected


def test_generated_language_uses_multiple_declensions():
    # Find a seed whose nouns have >1 inflection class and confirm words are spread across
    # them, reproducibly.
    for seed in range(20):
        lang = Language.generate(seed)
        noun_classes = lang.morphology.inflection_classes("noun")
        if len(noun_classes) > 1:
            used = {e.inflection_class for e in lang.lexicon.entries.values()
                    if e.concept.pos == "noun"}
            assert used <= set(noun_classes) and len(used) >= 2
            # the same seed reproduces the same class assignment
            again = Language.generate(seed)
            assert {g: e.inflection_class for g, e in lang.lexicon.entries.items()} == \
                   {g: e.inflection_class for g, e in again.lexicon.entries.items()}
            return
    raise AssertionError("no multi-declension language found in 20 seeds (unexpected)")


def test_make_sentence_runs_the_whole_stack():
    lang = Language.generate(8)
    sent = lang.make_sentence(
        "woman", "see", "bird", subject_definiteness="def", object_definiteness="indef"
    )
    assert isinstance(sent, Sentence)
    assert len(sent.words) >= 3  # subject, verb, object at minimum
    assert sent.text and sent.ipa
    # every surface word should carry a gloss
    assert all(w.gloss for w in sent.words)


def test_make_sentence_intransitive_and_adjective():
    lang = Language.generate(8)
    s1 = lang.make_sentence("child", "run", subject_number="pl", subject_definiteness="def")
    assert len(s1.words) >= 2
    s2 = lang.make_sentence("dog", "eat", "meat", subject_adjective="big")
    # subject (+adjective), verb, object
    assert len(s2.words) >= 4


def test_make_sentence_rejects_unknown_gloss():
    lang = Language.generate(1)
    try:
        lang.make_sentence("woman", "frobnicate")
    except KeyError as exc:
        assert "frobnicate" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unknown verb gloss")


def test_make_sentence_rejects_pos_mismatch():
    lang = Language.generate(1)
    try:
        lang.make_sentence("woman", "stone")  # 'stone' is a noun, not a verb
    except ValueError as exc:
        assert "stone" in str(exc) and "verb" in str(exc)
    else:
        raise AssertionError("expected ValueError when a noun is used as a verb")


def test_make_sentence_works_across_many_seeds():
    # Whatever morphology/alignment a seed rolls, building a clause must not crash.
    for seed in range(12):
        sent = Language.generate(seed).make_sentence(
            "woman", "see", "bird", subject_definiteness="def"
        )
        assert sent.words


def test_evolve_applies_sound_change_to_the_lexicon():
    lang = Language.generate(5)
    evolved = lang.evolve(["[voiceless plosive] > [+voiced] / V_V"])
    assert set(evolved) == set(lang.lexicon.entries)
    # Intervocalic voiceless plosives should have voiced somewhere in the lexicon.
    changed = sum(1 for g, (roman, ipa) in evolved.items() if ipa != lang.lexicon.get(g).ipa)
    assert changed >= 1
    # Compounds are flat segment sequences, so they evolve like any other word.
    assert "waterfall" in evolved


def test_to_dict_is_json_serializable_and_complete():
    lang = Language.generate(11)
    blob = json.dumps(lang.to_dict(), ensure_ascii=False)  # must not raise
    data = json.loads(blob)
    assert data["seed"] == 11 and data["generator_version"] >= 1
    assert set(data) == {
        "seed", "generator_version", "phonology", "morphology", "syntax", "writing", "lexicon"
    }
    assert len(data["lexicon"]) == len(CONCEPTS)
    assert "order" in data["syntax"] and "typology" in data["morphology"]


def test_summary_mentions_each_stage():
    text = Language.generate(2).summary()
    for marker in ("Consonants:", "Syllables:", "Morphology", "Syntax", "Writing system", "Lexicon"):
        assert marker in text
