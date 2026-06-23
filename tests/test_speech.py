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
from conlang.speech.phones import Source, Phone, plan_phone, vowel_formants, consonant_formants
from conlang.speech.synth import Voice, Synthesizer, _to_pcm16


def segs(symbols: str):
    return [data.BY_IPA[s] for s in symbols.split()]


def synth(seed: int = 0, **voice_kw) -> Synthesizer:
    return Synthesizer(Voice(**voice_kw), random.Random(seed))


def _kinds(seg):
    return [s.kind for s in plan_phone(seg).sources]


# --- Plan ---------------------------------------------------------------------------
def test_vowel_formants_track_features():
    # Close front /i/ has a low F1 and high F2; open /a/ has a high F1.
    i = vowel_formants(data.vowel("i"))
    a = vowel_formants(data.vowel("a"))
    assert i[0] < a[0]            # F1(i) < F1(a)  (height)
    assert i[1] > vowel_formants(data.vowel("u"))[1]  # F2(i) > F2(u) (backness)


def test_rounding_lowers_f2():
    # /y/ is rounded /i/: same height/backness, lower F2.
    assert vowel_formants(data.vowel("y"))[1] < vowel_formants(data.vowel("i"))[1]


def test_consonant_f2_locus_tracks_place():
    # The F2 locus is the place cue: bilabial low, alveolar mid, palatal high.
    p = consonant_formants(data.consonant("p"))[1]
    t = consonant_formants(data.consonant("t"))[1]
    k = consonant_formants(data.consonant("k"))[1]
    assert p < t < k or p < k  # bilabial locus is the lowest
    assert p < t


def test_plan_shapes_by_manner():
    assert _kinds(data.vowel("a")) == ["voiced"]
    # voiceless plosive: silent closure then a noise burst
    assert _kinds(data.consonant("p")) == ["silence", "noise"]
    # voiced plosive: a voiced bar instead of silence
    assert plan_phone(data.consonant("b")).sources[0].kind == "voiced"
    assert _kinds(data.consonant("s")) == ["noise"]
    assert _kinds(data.consonant("m")) == ["voiced"]
    # affricate: closure, burst, frication
    assert len(plan_phone(data.consonant("t͡ʃ")).sources) == 3
    # vowels and sonorants are "voiced resonant"; obstruents are not
    assert plan_phone(data.vowel("a")).voiced_resonant
    assert plan_phone(data.consonant("m")).voiced_resonant
    assert not plan_phone(data.consonant("p")).voiced_resonant


def test_formant_transition_into_a_vowel():
    # /b a/: the vowel's F2 should start nearer the bilabial locus and rise toward /a/'s F2,
    # i.e. the track is not flat across the vowel onset (a real transition).
    sy = synth(0, sample_rate=16000)
    f1s, f2s, f3s = sy._formant_tracks(
        [plan_phone(s) for s in segs("b a")],
        sum(max(1, int(s.duration * 16000)) for ph in (plan_phone(x) for x in segs("b a"))
            for s in ph.sources),
    )
    a_f2 = vowel_formants(data.vowel("a"))[1]
    # somewhere in the track F2 sits well below the /a/ steady value (pulled toward the locus)
    assert min(f2s) < a_f2 - 100
    # and it reaches the /a/ steady value by the end
    assert abs(f2s[-1] - a_f2) < 60


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


def test_stable_under_fast_alternating_transitions():
    # Rapid place jumps (bilabial<->velar) drive the time-varying resonators hard.
    out = synth(0).synthesize(segs("g i b a g i b a g i"))
    assert out and all(math.isfinite(s) and -1.0 <= s <= 1.0 for s in out)


def _f2_track(sy, symbols):
    phones = [plan_phone(s) for s in segs(symbols)]
    total = sum(max(1, int(src.duration * sy.voice.sample_rate))
                for ph in phones for src in ph.sources)
    return sy._formant_tracks(phones, total)[1]  # the F2 list


def test_place_cues_the_vowel_transition():
    # The whole point of the rewrite: the vowel's F2 trajectory differs by the consonant's
    # place, because each consonant pulls the transition toward a different locus.
    sy = synth(0)
    f2_b = _f2_track(sy, "b a")  # bilabial: low F2 locus (~800)
    f2_g = _f2_track(sy, "g a")  # velar: high F2 locus (~2000)
    assert f2_b != f2_g
    assert min(f2_b) < min(f2_g) - 200  # the bilabial transition dips much lower


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
