import { describe, it, expect } from 'vitest';
import { requantizeNotesToKey, freqToMidi } from './musicTheory.js';

describe('requantizeNotesToKey', () => {
  it('re-keys using measuredFreq, independent of the old step', () => {
    // A440 (freqToMidi(440) === 69) is step 5 in C major (a major sixth
    // above the tonic) but exactly the tonic (step 0) of A minor.
    const notes = [{ step: 5, measuredFreq: 440 }];
    const result = requantizeNotesToKey(notes, { tonic: 0, mode: 'major' }, { tonic: 9, mode: 'minor' });
    expect(result[0].step).toBe(0);
  });

  it('falls back to the old step (via the old key) when measuredFreq is missing', () => {
    // Same re-key as above, but via the step-based fallback instead of a
    // measured frequency — should land on the same result since step 5 in
    // C major *is* A440.
    const notes = [{ step: 5, measuredFreq: null }];
    const result = requantizeNotesToKey(notes, { tonic: 0, mode: 'major' }, { tonic: 9, mode: 'minor' });
    expect(result[0].step).toBe(0);
  });

  it('leaves steps unchanged when re-keying to the same key', () => {
    const notes = [
      { step: 0, measuredFreq: 261.63 },
      { step: 4, measuredFreq: 392.0 },
    ];
    const result = requantizeNotesToKey(notes, { tonic: 0, mode: 'major' }, { tonic: 0, mode: 'major' });
    expect(result.map((n) => n.step)).toEqual([0, 4]);
  });

  it('preserves every other field on the note', () => {
    const notes = [{ step: 2, start: 1.2, end: 1.8, measuredFreq: 330 }];
    const result = requantizeNotesToKey(notes, { tonic: 0, mode: 'major' }, { tonic: 2, mode: 'major' });
    expect(result[0].start).toBe(1.2);
    expect(result[0].end).toBe(1.8);
    expect(result[0].measuredFreq).toBe(330);
  });

  it('re-keys D4 (293.66 Hz) from C major to its relative minor (A minor)', () => {
    // D4 is midi 62 — scale degree "re" (step 1) in C major, and step -4
    // (a fourth below the tonic) in A minor. freqToMidi(293.66) rounds to 62.
    expect(freqToMidi(293.66)).toBeCloseTo(62, 0);
    const notes = [{ step: 1, measuredFreq: 293.66 }];
    const result = requantizeNotesToKey(notes, { tonic: 0, mode: 'major' }, { tonic: 9, mode: 'minor' });
    expect(result[0].step).toBe(-4);
  });
});
