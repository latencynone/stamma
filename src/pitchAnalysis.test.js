import { describe, it, expect } from 'vitest';
import { filterOutlierNotes } from './pitchAnalysis.js';

// Reproduces two real patterns found in an actual recording that
// filterTransientArtifacts's local-neighbor check let through: a mic-pop-like
// blip isolated by silence on both sides, and a pair of short notes that are
// each implausible but happen to sit close enough together to look mutually
// "consistent" to a check that only compares immediate neighbors.
describe('filterOutlierNotes', () => {
  it('drops an isolated short blip far from the melody\'s home register', () => {
    const notes = [
      { step: 7, start: 0.875, end: 0.945 }, // isolated artifact, ~14 diatonic steps from home
      { step: -13, start: 1.295, end: 1.435 },
      { step: -11, start: 1.505, end: 1.89 },
      { step: -7, start: 2.625, end: 2.87 },
      { step: -7, start: 2.94, end: 3.85 },
      { step: -7, start: 3.92, end: 4.235 },
    ];
    const result = filterOutlierNotes(notes);
    expect(result.find((n) => n.step === 7)).toBeUndefined();
    expect(result.length).toBe(notes.length - 1);
  });

  it('drops a pair of short notes that are mutually consistent but both far from home', () => {
    const notes = [
      { step: -7, start: 6.335, end: 6.825 },
      { step: -7, start: 6.895, end: 7.175 },
      { step: -7, start: 7.245, end: 7.665 },
      { step: -10, start: 8.715, end: 8.855 },
      { step: -12, start: 9.345, end: 9.45 },
      { step: 2, start: 9.695, end: 9.765 }, // isolated from step -12, but agrees with the next
      { step: 1, start: 9.8, end: 9.87 }, // agrees with previous — evades a pure neighbor-jump check
    ];
    const result = filterOutlierNotes(notes);
    expect(result.find((n) => n.step === 2)).toBeUndefined();
    expect(result.find((n) => n.step === 1)).toBeUndefined();
    expect(result.length).toBe(notes.length - 2);
  });

  it('keeps a real sustained note even if it is far from the home register', () => {
    const notes = [
      { step: -7, start: 0, end: 1 },
      { step: -7, start: 1.1, end: 2 },
      { step: 6, start: 2.1, end: 3.5 }, // long, deliberate leap — not a tracker blip
      { step: -7, start: 3.6, end: 4.5 },
    ];
    const result = filterOutlierNotes(notes);
    expect(result.find((n) => n.step === 6)).toBeDefined();
    expect(result.length).toBe(notes.length);
  });

  it('leaves short lists untouched (not enough context to judge a home register)', () => {
    const notes = [{ step: 7, start: 0, end: 0.1 }, { step: -7, start: 0.2, end: 0.3 }];
    expect(filterOutlierNotes(notes)).toEqual(notes);
  });
});
