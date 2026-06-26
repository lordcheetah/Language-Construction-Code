"""Tests for the Language aggregate that wires all six stages together."""

import json
import random

import pytest

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
    assert lang.numerals.base in (5, 10, 12, 20)
    # numerals are part of the deterministic, seed-reproducible language
    assert lang.numerals.number(24).roman == Language.generate(7).numerals.number(24).roman


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


def test_make_compound_coordinates_two_clauses():
    lang = Language.generate(8)
    one = lang.make_sentence("person", "walk")
    compound = lang.make_compound(
        dict(subject="person", verb="walk"),
        dict(subject="person", verb="run"),
        coordinator="and",
    )
    # a compound sentence is strictly longer than either conjunct (it adds the second
    # clause and the medial "and" particle)
    assert len(compound.words) > len(one.words)
    assert any(w.gloss == "AND" for w in compound.words)


def test_make_compound_rejects_bad_input():
    lang = Language.generate(1)
    walk = dict(subject="person", verb="walk")
    with pytest.raises(ValueError):
        lang.make_compound(walk, walk, coordinator="but")  # not a known coordinator
    with pytest.raises(ValueError):
        lang.make_compound(walk)  # a compound needs at least two clauses


def test_make_sentence_ditransitive_adds_a_recipient():
    lang = Language.generate(8)
    mono = lang.make_sentence("woman", "give", "stone")
    ditrans = lang.make_sentence("woman", "give", "stone", recipient="child")
    # the recipient adds exactly one more constituent (the indirect object)
    assert len(ditrans.words) == len(mono.words) + 1
    assert any(w.gloss.startswith("child") for w in ditrans.words)


def test_make_sentence_recipient_requires_a_direct_object():
    lang = Language.generate(1)
    with pytest.raises(ValueError):
        lang.make_sentence("woman", "give", recipient="child")  # no direct object


def test_make_sentence_pro_drop_omits_a_pronoun_subject():
    import dataclasses

    # a language whose verb marks both person and number (rich agreement)
    for seed in range(60):
        lang = Language.generate(seed)
        verb = lang.morphology.paradigms.get("verb")
        marked = {c.name for c in verb.marked} if verb else set()
        if {"person", "number"} <= marked:
            break
    else:
        raise AssertionError("no rich-agreement seed found in range")
    pro = dataclasses.replace(lang, syntax=dataclasses.replace(lang.syntax, pro_drop=True))
    no_pro = dataclasses.replace(lang, syntax=dataclasses.replace(lang.syntax, pro_drop=False))
    # the personal-pronoun subject "I" is dropped only in the pro-drop language
    assert len(pro.make_sentence("I", "run").words) < len(no_pro.make_sentence("I", "run").words)
    # a full lexical subject is never dropped, even under pro-drop
    assert len(pro.make_sentence("person", "run").words) == len(no_pro.make_sentence("person", "run").words)
    # a demonstrative ("this") is not a personal pronoun -> never dropped either
    assert len(pro.make_sentence("this", "run").words) == len(no_pro.make_sentence("this", "run").words)


def test_make_sentence_free_articles_add_determiner_words():
    import dataclasses

    lang = Language.generate(8)
    art = dataclasses.replace(lang, syntax=dataclasses.replace(lang.syntax, articles=True))
    no_art = dataclasses.replace(lang, syntax=dataclasses.replace(lang.syntax, articles=False))
    kw = dict(subject_definiteness="def", object_definiteness="indef")
    with_art = art.make_sentence("woman", "see", "bird", **kw)
    without = no_art.make_sentence("woman", "see", "bird", **kw)
    # the article language adds two determiner words (one per definite/indefinite NP)
    assert len(with_art.words) == len(without.words) + 2
    assert any(w.gloss == "DEF" for w in with_art.words)
    assert any(w.gloss == "INDEF" for w in with_art.words)


def test_make_sentence_differential_object_marking():
    import dataclasses

    from conlang.syntax.parameters import Alignment

    # a nominative-accusative language whose noun actually marks case
    for seed in range(60):
        lang = Language.generate(seed)
        noun = lang.morphology.paradigms.get("noun")
        if (noun and any(c.name == "case" for c in noun.marked)
                and lang.syntax.alignment is Alignment.NOMINATIVE_ACCUSATIVE):
            break
    else:
        raise AssertionError("no nom-acc, case-marking seed found")
    dom = dataclasses.replace(
        lang, syntax=dataclasses.replace(lang.syntax, differential_object_marking=True)
    )
    definite = dom.make_sentence("woman", "see", "bird", object_definiteness="def")
    indefinite = dom.make_sentence("woman", "see", "bird", object_definiteness="indef")
    # the definite object is accusative-marked; the indefinite one is left unmarked
    assert any("bird" in w.gloss and "ACC" in w.gloss for w in definite.words)
    assert not any("bird" in w.gloss and "ACC" in w.gloss for w in indefinite.words)


def test_make_sentence_clusivity():
    # a language whose verb marks clusivity, so "we" distinguishes inclusive from exclusive
    for seed in range(120):
        lang = Language.generate(seed)
        verb = lang.morphology.paradigms.get("verb")
        if verb and any(c.name == "clusivity" for c in verb.marked):
            break
    else:
        raise AssertionError("no clusivity-marking seed found in range")
    incl = lang.make_sentence("we", "run", subject_number="pl", subject_clusivity="inclusive")
    excl = lang.make_sentence("we", "run", subject_number="pl", subject_clusivity="exclusive")
    assert any("INCL" in w.gloss for w in incl.words)
    assert any("EXCL" in w.gloss for w in excl.words)


def test_make_sentence_rejects_unknown_gloss():
    lang = Language.generate(1)
    try:
        lang.make_sentence("woman", "frobnicate")
    except KeyError as exc:
        assert "frobnicate" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unknown verb gloss")


def test_make_sentence_content_question():
    lang = Language.generate(8)
    sent = lang.make_sentence("who", "see", "bird", question="subject")
    # the wh-word is present (its gloss may carry a case tag, e.g. "who.ACC" under ergative)
    assert any(w.gloss.split(".")[0] == "who" for w in sent.words)
    assert len(sent.words) >= 3


def test_make_sentence_rejects_bad_question_and_imperative_question():
    lang = Language.generate(1)
    for bad in (dict(question="subj"), dict(question="object", mood="imperative")):
        try:
            lang.make_sentence("who", "see", "bird", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


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
        "seed", "generator_version", "phonology", "morphology", "syntax", "writing",
        "numerals", "lexicon",
    }
    assert data["numerals"]["base"] in (5, 10, 12, 20)
    assert len(data["lexicon"]) == len(CONCEPTS)
    assert "order" in data["syntax"] and "typology" in data["morphology"]


def test_summary_mentions_each_stage():
    text = Language.generate(2).summary()
    for marker in ("Consonants:", "Syllables:", "Morphology", "Syntax", "Writing system", "Lexicon"):
        assert marker in text
