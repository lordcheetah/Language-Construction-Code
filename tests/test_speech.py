"""Tests for the speech (formant TTS) capstone.

Audio quality can't be unit-tested, but the contract can: output is bounded, deterministic
given a seed, a valid WAV, length tracks the plan, and the acoustics actually differ
between phonemes (distinct vowels / voicing) as the feature mapping intends.
"""

import io
import math
import random
import wave

from conlang.phonology import data
from conlang.speech.phones import Part, plan_for, vowel_formants
from conlang.speech.synth import Voice, Synthesizer, _to_pcm16


def segs(symbols: str):
    return [data.BY_IPA[s] for s in symbols.split()]


def synth(seed: int = 0, **voice_kw) -> Synthesizer:
    return Synthesizer(Voice(**voice_kw), random.Random(seed))


# --- Plan ---------------------------------------------------------------------------
def test_vowel_formants_track_features():
    # Close front /i/ has a low F1 and high F2; open /a/ has a high F1.
    i = vowel_formants(data.vowel("i"))
    a = vowel_formants(data.vowel("a"))
    assert i[0][0] < a[0][0]            # F1(i) < F1(a)  (height)
    assert i[1][0] > vowel_formants(data.vowel("u"))[1][0]  # F2(i) > F2(u) (backness)


def test_rounding_lowers_f2():
    # /y/ is rounded /i/: same height/backness, lower F2.
    assert vowel_formants(data.vowel("y"))[1][0] < vowel_formants(data.vowel("i"))[1][0]


def test_plan_shapes_by_manner():
    assert [p.kind for p in plan_for(data.vowel("a"))] == ["voiced"]
    # voiceless plosive: silent closure then a noise burst
    assert [p.kind for p in plan_for(data.consonant("p"))] == ["silence", "noise"]
    # voiced plosive: a voiced bar instead of silence
    assert plan_for(data.consonant("b"))[0].kind == "voiced"
    assert [p.kind for p in plan_for(data.consonant("s"))] == ["noise"]
    assert [p.kind for p in plan_for(data.consonant("m"))] == ["voiced"]
    # affricate: closure, burst, frication
    assert len(plan_for(data.consonant("t͡ʃ"))) == 3


# --- Synthesis contract -------------------------------------------------------------
def test_samples_are_bounded_and_nonempty():
    out = synth().synthesize(segs("p a t a"))
    assert out
    assert all(-1.0 <= s <= 1.0 for s in out)


def test_synthesis_is_deterministic_with_seed():
    a = synth(7).synthesize(segs("s a m i"))
    b = synth(7).synthesize(segs("s a m i"))
    assert a == b


def test_length_tracks_plan_and_rate():
    sr = 16000
    normal = synth(0, sample_rate=sr, rate=1.0).synthesize(segs("a"))
    fast = synth(0, sample_rate=sr, rate=2.0).synthesize(segs("a"))
    # /a/ is 0.15 s -> ~2400 samples at rate 1; rate 2 roughly halves it.
    assert abs(len(normal) - int(0.15 * sr)) <= 2
    assert abs(len(fast) - len(normal) / 2) <= 2


def test_distinct_vowels_sound_different():
    a = synth(1).synthesize(segs("a"))
    i = synth(1).synthesize(segs("i"))
    assert a != i


def test_voicing_changes_the_signal():
    # /s/ (voiceless) vs /z/ (voiced) share place/manner but differ acoustically.
    s = synth(1).synthesize(segs("s"))
    z = synth(1).synthesize(segs("z"))
    assert s != z


def test_distinct_places_differ():
    # Different places of articulation must produce different signals.
    f = {ipa: synth(2).synthesize(segs(ipa)) for ipa in ("s", "ʃ", "x")}
    assert f["s"] != f["ʃ"] != f["x"] and f["s"] != f["x"]
    p = {ipa: synth(2).synthesize(segs(ipa + " a")) for ipa in ("p", "t", "k")}
    assert p["p"] != p["t"] != p["k"] and p["p"] != p["k"]


def test_cascade_stays_finite_and_bounded():
    # A run of narrow-bandwidth vowels through the 3-stage cascade must not blow up.
    out = synth(0).synthesize(segs("i i i i u u a a"))
    assert all(math.isfinite(s) and -1.0 <= s <= 1.0 for s in out)


def test_determinism_is_per_call_on_one_instance():
    # Reproducible even when reusing the same Synthesizer (it reseeds each call).
    sy = synth(9)
    assert sy.synthesize(segs("s a")) == sy.synthesize(segs("s a"))


def test_pcm16_clamps_out_of_range_samples():
    raw = _to_pcm16([2.0, -2.0, 0.0])
    vals = [int.from_bytes(raw[i:i + 2], "little", signed=True) for i in (0, 2, 4)]
    assert vals == [32767, -32767, 0]


# --- WAV output ---------------------------------------------------------------------
def test_wav_bytes_are_a_valid_wav():
    sy = synth(3, sample_rate=22050)
    samples = sy.synthesize(segs("k a t a"))
    blob = sy.to_wav_bytes(samples)
    with wave.open(io.BytesIO(blob), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 22050
        assert w.getnframes() == len(samples)


def test_silence_input_produces_silence_not_crash():
    # A plan that is only silence (a bare voiceless plosive closure has a burst, so use an
    # empty segment list) returns an empty, valid buffer.
    sy = synth()
    assert sy.synthesize([]) == []
    assert sy.to_wav_bytes([]) is not None
