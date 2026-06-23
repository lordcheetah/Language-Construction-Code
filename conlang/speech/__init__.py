"""Speech: a pure-Python formant synthesizer that gives a conlang a voice.

The first of the capstones. Rather than depend on an external engine, it synthesizes audio
from scratch with only the standard library, driven by the *same* phonological features the
rest of the toolkit uses:

- a vowel's **height** and **backness** set its formant frequencies (F1/F2/F3);
- a consonant's **manner** picks the production model (a buzz for sonorants, shaped noise
  for fricatives, a closure-plus-burst for plosives);
- **voicing** switches the glottal source on or off.

The output is a 16-bit PCM WAV. It is robotic — a simple source/filter formant model, not a
neural voice — but it is real audio, fully offline, deterministic (seed the noise), and the
acoustics follow directly from the phonology. (A higher-fidelity ``espeak-ng`` backend is a
possible future addition.)
"""

from conlang.speech.synth import Voice, Synthesizer
from conlang.speech.phones import plan_for

__all__ = ["Voice", "Synthesizer", "plan_for"]
