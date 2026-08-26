/* ---------- Music theory constants ---------- */

export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export const MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11];
export const MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10];

// Krumhansl-Kessler key profiles, used to correlate a pitch-class
// histogram against every possible tonic/mode to guess the key.
export const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
export const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

export const HARMONY_TYPES = {
  ters: {
    label: 'Ters',
    steps: 2,
    defaultDirection: -1,
    description: 'Den vanligaste stämman i kör- och popsång. Ligger tersavstånd från melodin och växlar naturligt mellan stor och liten ters beroende på var i tonarten melodin befinner sig.',
  },
  kvint: {
    label: 'Kvint',
    steps: 4,
    defaultDirection: -1,
    description: 'Ett öppet, stadigt avstånd som ofta hörs i folkmusik och femklanger. Ger stämman en stram, kraftfull karaktär.',
  },
  sext: {
    label: 'Sext',
    steps: 5,
    defaultDirection: 1,
    description: 'Ett varmt, fylligt avstånd — ett vanligt alternativ till tersen, särskilt i ballader.',
  },
};

export const SOUND_TYPES = {
  sine: { label: 'Ren synt', description: 'En ren sinuston. Enklast att stämma efter, minst "mänsklig".' },
  voice: { label: 'Formantröst', description: 'En syntetisk vokalklang ("oo"), byggd med formantfilter — mer sångbar än en ren ton, men fortfarande syntetisk.' },
  recording: { label: 'Din röst', description: 'Din egen inspelning, kontinuerligt pitchskiftad med formantbevarande till stämmans toner. Mest autentiskt.' },
};

/* ---------- Pure helpers: pitch <-> MIDI <-> scale degree ---------- */

export function freqToMidi(freq) {
  return 69 + 12 * Math.log2(freq / 440);
}

export function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function getScaleIntervals(mode) {
  return mode === 'major' ? MAJOR_INTERVALS : MINOR_INTERVALS;
}

// Semitone offset (from the tonic) of a given absolute diatonic "step".
// Steps count scale degrees, e.g. step 2 is a diatonic third above the tonic,
// step 7 is the tonic one octave up. This is monotonic in step.
export function semitoneForStep(step, intervals) {
  const octave = Math.floor(step / 7);
  const idx = ((step % 7) + 7) % 7;
  return 12 * octave + intervals[idx];
}

export function nearestScaleStep(semitoneOffset, intervals) {
  const approx = Math.round((semitoneOffset * 7) / 12);
  let best = approx;
  let bestDiff = Infinity;
  for (let s = approx - 2; s <= approx + 2; s++) {
    const diff = Math.abs(semitoneForStep(s, intervals) - semitoneOffset);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = s;
    }
  }
  return best;
}

export function midiToScaleStep(midi, tonicPC, mode) {
  const intervals = getScaleIntervals(mode);
  const baseTonicMidi = 60 + tonicPC;
  const semitoneOffset = Math.round(midi) - baseTonicMidi;
  return nearestScaleStep(semitoneOffset, intervals);
}

export function scaleStepToMidi(step, tonicPC, mode) {
  const intervals = getScaleIntervals(mode);
  const baseTonicMidi = 60 + tonicPC;
  return baseTonicMidi + semitoneForStep(step, intervals);
}

export function midiToNoteName(midi) {
  const rounded = Math.round(midi);
  const pc = ((rounded % 12) + 12) % 12;
  const octave = Math.floor(rounded / 12) - 1;
  return `${NOTE_NAMES[pc]}${octave}`;
}

// Re-quantizes every note's `step` against a different key, for a manual
// tonart override after auto-detection guessed wrong (the app's own most
// common cause of an odd-sounding harmony — the notes themselves already
// carry everything needed to redo this without re-running pitch tracking).
// Prefers each note's actual sung pitch (`measuredFreq`, attached by
// attachMeasuredFreq) so the new step reflects what was really sung rather
// than a second, compounding quantization; falls back to re-deriving an
// absolute pitch from the note's existing step in the *old* key for a note
// that has no measured frequency (too little voiced signal in its range).
export function requantizeNotesToKey(notes, oldKey, newKey) {
  return notes.map((n) => {
    const midi = n.measuredFreq ? freqToMidi(n.measuredFreq) : scaleStepToMidi(n.step, oldKey.tonic, oldKey.mode);
    return { ...n, step: midiToScaleStep(midi, newKey.tonic, newKey.mode) };
  });
}
