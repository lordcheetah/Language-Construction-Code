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
from conlang.speech.phones import (
    Source, Phone, plan_phone, vowel_formants, consonant_formants, apply_velar_pinch,
    apply_breathy_glottal,
)
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


def _velar_f2(symbols):
    """The F2 locus of the /k/ in a planned, velar-pinched segment string."""
    s = segs(symbols)
    phones = apply_velar_pinch(s, [plan_phone(x) for x in s])
    return next(p.formants[1] for seg, p in zip(s, phones) if seg.ipa == "k")


def test_velar_pinch_tracks_the_adjacent_vowel():
    # The velar locus rises next to a front vowel /i/ and falls next to a back vowel /u/.
    assert _velar_f2("k i") > _velar_f2("k u")
    assert _velar_f2("i k") > _velar_f2("u k")              # works on either side
    # /a..u/ flanking averages the two vowels' F2, landing between the single-vowel cases
    assert _velar_f2("u k") < _velar_f2("a k u") < _velar_f2("a k")


def test_velar_pinch_leaves_other_consonants_and_lone_velars_untouched():
    default_k = consonant_formants(data.consonant("k"))[1]
    # a velar with no adjacent vowel keeps its default locus
    s = segs("k")
    assert apply_velar_pinch(s, [plan_phone(x) for x in s])[0].formants[1] == default_k
    # a non-velar consonant is never pinched
    s = segs("t u")
    pinched = apply_velar_pinch(s, [plan_phone(x) for x in s])
    assert pinched[0].formants == plan_phone(data.consonant("t")).formants


def test_velar_pinch_applies_to_fricatives_but_not_the_nasal():
    # the velar fricative /x/ is an obstruent -> pinched; the velar nasal /ŋ/ is not
    s = segs("x i")
    x_phones = apply_velar_pinch(s, [plan_phone(y) for y in s])
    assert x_phones[0].formants[1] != consonant_formants(data.consonant("x"))[1]
    s = segs("ŋ i")
    ng_phones = apply_velar_pinch(s, [plan_phone(y) for y in s])
    assert ng_phones[0].formants == plan_phone(data.consonant("ŋ")).formants  # untouched


def test_velar_pinch_moves_only_f2_not_f3():
    s = segs("k u")
    p = apply_velar_pinch(s, [plan_phone(x) for x in s])[0]
    default = consonant_formants(data.consonant("k"))
    assert p.formants[0] == default[0] and p.formants[2] == default[2]  # F1, F3 unchanged
    assert p.formants[1] != default[1]                                  # only F2 moved


def test_velarless_word_is_unchanged_by_the_pinch():
    s = segs("p a t i")
    before = [plan_phone(x) for x in s]
    assert apply_velar_pinch(s, list(before)) == before  # byte-identical, a no-op


def test_velar_pinch_bends_the_synthesized_formant_track():
    # Compare the synth's own F2 track for the same word with vs. without the pinch applied,
    # isolating the velar locus shift (the durations, hence the timeline, are identical).
    sy = Synthesizer(Voice())
    s = segs("a k a")
    plain = [plan_phone(x) for x in s]
    pinched = apply_velar_pinch(s, [plan_phone(x) for x in s])
    sr = sy.voice.sample_rate
    total = sum(max(1, int(src.duration * sr)) for ph in pinched for src in ph.sources)
    _, f2_plain, _ = sy._formant_tracks(plain, total)
    _, f2_pinched, _ = sy._formant_tracks(pinched, total)
    assert f2_plain != f2_pinched  # the velar's pinched locus changes the F2 transitions


def test_plan_shapes_by_manner():
    assert _kinds(data.vowel("a")) == ["voiced"]
    # voiceless plosive: silent closure, a noise burst, then aspiration (the VOT breath)
    assert _kinds(data.consonant("p")) == ["silence", "noise", "noise"]
    # voiced plosive: a voiced bar instead of silence, and NO aspiration
    assert plan_phone(data.consonant("b")).sources[0].kind == "voiced"
    assert _kinds(data.consonant("b")) == ["voiced", "noise"]
    assert _kinds(data.consonant("s")) == ["noise"]
    assert _kinds(data.consonant("m")) == ["voiced"]
    # affricate: closure, burst, frication (no separate aspiration — the frication is the release)
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


# --- Aspiration (voiceless stops) and breathy /h/ (anti-click, anti-static) ---------
def test_voiceless_plosive_has_aspiration_after_the_burst():
    # the aspiration source bridges burst -> voicing so the stop isn't a bare click
    src = plan_phone(data.consonant("k")).sources
    assert [s.kind for s in src] == ["silence", "noise", "noise"]
    burst, asp = src[1], src[2]
    assert asp.amp < burst.amp          # aspiration is the softer trailing breath
    assert 0.0 < asp.duration <= 0.05   # short voice-onset puff


def test_voiced_plosive_is_not_aspirated():
    # voiced stops have a voice bar and no aspiration breath
    assert [s.kind for s in plan_phone(data.consonant("g")).sources] == ["voiced", "noise"]


def test_glottal_stop_is_a_bare_closure_no_burst_or_aspiration():
    # /ʔ/ closes the vocal folds: no oral burst, no aspiration — just the closure (silence)
    assert _kinds(data.consonant("ʔ")) == ["silence"]


def test_aspiration_is_place_dependent_velar_more_than_labial():
    # velars carry the longest voice-onset time, labials the shortest
    p_asp = plan_phone(data.consonant("p")).sources[-1].duration
    k_asp = plan_phone(data.consonant("k")).sources[-1].duration
    assert k_asp > p_asp


def test_glottal_fricative_is_a_soft_breath_not_a_sibilant():
    h = plan_phone(data.consonant("h")).sources[0]
    s = plan_phone(data.consonant("s")).sources[0]
    assert h.amp < s.amp  # /h/ is quieter than the sibilant /s/


def test_breathy_glottal_colours_h_with_the_following_vowel():
    # /h/ before /i/ vs before /u/ takes different noise-band centres (the vowel's F2),
    # instead of the same broadband hiss — so it tracks the vowel rather than sounding static.
    def h_band(symbols):
        s = segs(symbols)
        phones = apply_breathy_glottal(s, [plan_phone(x) for x in s])
        return next(p.sources[0].noise_band for seg, p in zip(s, phones) if seg.ipa == "h")
    hi, hu = h_band("h i"), h_band("h u")
    assert hi != hu
    assert hi[0] == vowel_formants(data.vowel("i"))[1]  # centred on /i/'s F2
    assert hi[1] == 700  # narrowed band (was the broadband 3000 that hissed)


def test_breathy_glottal_averages_two_flanking_vowels_and_falls_back():
    from conlang.speech.phones import apply_breathy_glottal as bg
    # intervocalic /h/ blends both flanking vowels' F2 (like the velar pinch)
    s = segs("a h i")
    band = next(p.sources[0].noise_band for seg, p in zip(s, bg(s, [plan_phone(x) for x in s]))
                if seg.ipa == "h")
    expected = (vowel_formants(data.vowel("a"))[1] + vowel_formants(data.vowel("i"))[1]) / 2
    assert band[0] == expected
    # a glottal with no vowel neighbour is left at its default band (a no-op)
    s = segs("h")
    assert bg(s, [plan_phone(x) for x in s]) == [plan_phone(data.consonant("h"))]


# --- Intonation (F0 contour) --------------------------------------------------------
def test_statement_intonation_declines_and_falls():
    sy = Synthesizer(Voice(f0=120, intonation="statement"))
    assert sy._f0_at(0.0) > sy._f0_at(0.5) > sy._f0_at(1.0)  # downward over the utterance
    assert sy._f0_at(0.0) > 120 and sy._f0_at(1.0) < 120     # starts high, ends below base
    # and the excursion is audible, not a near-monotone (guards against a flattened regression)
    assert sy._f0_at(0.0) / sy._f0_at(1.0) > 1.2


def test_question_intonation_rises_at_the_end():
    sy = Synthesizer(Voice(f0=120, intonation="question"))
    assert sy._f0_at(1.0) > sy._f0_at(0.5)   # final rise
    assert sy._f0_at(1.0) > sy._f0_at(0.0)   # ends above where it started


def test_unknown_intonation_falls_back_to_a_statement_contour():
    sy = Synthesizer(Voice(intonation="bogus"))
    stmt = Synthesizer(Voice(intonation="statement"))
    assert sy._f0_at(0.3) == stmt._f0_at(0.3)


def test_intonation_changes_the_audio():
    # the same word spoken as a statement vs a question must differ (pitch track differs)
    a = Synthesizer(Voice(intonation="statement"), random.Random(0)).synthesize(segs("p a t a"))
    b = Synthesizer(Voice(intonation="question"), random.Random(0)).synthesize(segs("p a t a"))
    assert a != b


def test_a_varying_contour_still_synthesizes_a_voiced_buzz():
    # the (non-flat) default statement contour must still produce audible voicing
    out = Synthesizer(Voice()).synthesize(segs("a"))
    assert out and any(abs(x) > 0.1 for x in out)


def test_degenerate_flat_contour_hits_the_span_guard(monkeypatch):
    # a contour whose control points share a position exercises the span<=0 branch (no
    # ZeroDivisionError) and reduces to a fixed-period, constant-pitch train
    from conlang.speech import synth as synth_mod
    monkeypatch.setitem(synth_mod._CONTOURS, "flat", ((0.0, 1.0), (0.0, 1.0)))
    sy = Synthesizer(Voice(f0=120, intonation="flat"))
    assert sy._f0_at(0.0) == 120 and sy._f0_at(0.5) == 120  # constant pitch, no crash
    out = sy.synthesize(segs("a"))
    assert out and any(abs(x) > 0.1 for x in out)


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


# --- Parallel-formant noise bank (fricatives) ---------------------------------------
def test_fricative_uses_a_parallel_formant_bank_not_a_single_band():
    s = plan_phone(data.consonant("s")).sources[0]
    assert s.noise_band is None and len(s.noise_formants) >= 2  # a parallel bank with >=2 peaks
    sh = plan_phone(data.consonant("ʃ")).sources[0]
    assert s.noise_formants != sh.noise_formants  # distinct sibilant spectra


def test_glottal_h_and_stop_burst_stay_single_band():
    # only fricatives (except /h/) get the parallel bank; /h/ and stop bursts keep one band
    h = plan_phone(data.consonant("h")).sources[0]
    burst = plan_phone(data.consonant("k")).sources[1]  # closure, burst, aspiration
    assert h.noise_band is not None and not h.noise_formants
    assert burst.noise_band is not None and not burst.noise_formants


def test_affricate_frication_uses_the_parallel_bank():
    # the frication tail of an affricate shares the fricative spectrum (parallel bank)
    frication = plan_phone(data.consonant("t͡ʃ")).sources[2]  # closure, burst, frication
    assert frication.noise_formants and frication.noise_band is None


def test_parallel_bank_stays_finite_and_bounded():
    out = synth(0).synthesize(segs("a s a ʃ a x a"))
    assert out and all(math.isfinite(v) and -1.0 <= v <= 1.0 for v in out)


def _rms(xs):
    return math.sqrt(sum(x * x for x in xs) / max(1, len(xs)))


def test_vowels_are_audible_next_to_noise_not_drowned_out():
    # Regression: the voiced cascade once rendered ~thousands of times quieter than the noise
    # sources, so a vowel after a stop/fricative vanished under normalization and the word was
    # just static. A vowel's level must be comparable to (here, at least a third of) a
    # fricative's, so it survives in the same word.
    sy = synth(0)
    for v in ("a", "u", "i"):
        vowel = _rms(sy.synthesize(segs(v)))
        fric = _rms(sy.synthesize(segs("s")))
        assert vowel > fric * 0.3, f"/{v}/ ({vowel:.3f}) is too quiet beside /s/ ({fric:.3f})"


def test_bank_fricatives_do_not_drown_the_vowels():
    # The parallel bank must keep every fricative comparable in level to the vowels in the same
    # word — a louder bank place would crush the vowels under normalization (the old "static").
    sy = synth(0)
    for fric in ("s", "ʃ", "x", "f"):
        out = sy.synthesize(segs(f"a {fric} a"))
        n = len(out)
        vowels = _rms(out[: n * 30 // 100] + out[n * 70 // 100:])   # the two /a/ regions
        noise = _rms(out[n * 42 // 100: n * 58 // 100])              # the fricative in the middle
        assert vowels > noise * 0.5, f"/{fric}/ drowns the vowels (v={vowels:.3f} n={noise:.3f})"


def test_a_vowel_after_a_stop_carries_most_of_the_energy():
    # In "ku" the vowel region must dominate the burst+aspiration, not the reverse (the static
    # bug made the noise dominate). Compare the loud tail (vowel) to the head (the /k/).
    out = synth(0).synthesize(segs("k u"))
    head = out[: len(out) // 3]      # the /k/ closure + burst + aspiration
    tail = out[2 * len(out) // 3:]   # well into the /u/ vowel
    assert _rms(tail) > _rms(head)


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
