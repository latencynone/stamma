import { describe, it, expect } from 'vitest';
import {
  detectOnsetSegments, estimateCoarseOffset, matchOnsetsToMelody,
  buildAlignmentKeyframes, alignOwnTakeToMelody,
} from './takeAlignment.js';
import { synthesizeVoice, toAudioBuffer } from './testUtils.js';

const SR = 44100;

// A simple ascending-then-descending five-note phrase, used as "the melody"
// throughout — real timings, a short gap between each note like a sung
// phrase would have.
function makeNoteList(freqs, { start = 0.3, noteDur = 0.4, gapDur = 0.15 } = {}) {
  const notes = [];
  let t = start;
  freqs.forEach((freq) => {
    notes.push({ start: t, end: t + noteDur, freq });
    t += noteDur + gapDur;
  });
  return notes;
}

describe('detectOnsetSegments', () => {
  it('finds the same number of onsets as notes in a synthesized phrase, close to their real start times', () => {
    const notes = makeNoteList([220, 247, 262, 294, 330]);
    const totalDur = notes[notes.length - 1].end + 0.4;
    const raw = synthesizeVoice(notes, totalDur, SR, { gapNoiseAmp: 0 });
    const segments = detectOnsetSegments(raw, SR);

    expect(segments.length).toBe(notes.length);
    segments.forEach((seg, i) => {
      expect(Math.abs(seg.start - notes[i].start)).toBeLessThan(0.06);
    });
  });

  it('finds nothing in true silence', () => {
    const data = new Float32Array(Math.round(2 * SR));
    expect(detectOnsetSegments(data, SR)).toEqual([]);
  });
});

describe('estimateCoarseOffset', () => {
  it('recovers a known constant delay between two onset sequences', () => {
    const melodyStarts = [0.3, 0.9, 1.6, 2.4, 3.1];
    const delay = 0.42;
    const ownStarts = melodyStarts.map((t) => t + delay);
    // offset is defined as (predicted melody time) - (own time), so a take
    // that started *later* than the melody has a *negative* offset here.
    const offset = estimateCoarseOffset(ownStarts, melodyStarts);
    expect(Math.abs(offset - (-delay))).toBeLessThan(0.03);
  });

  it('is robust to one extra onset in the take that has no melody counterpart', () => {
    const melodyStarts = [0.3, 0.9, 1.6, 2.4, 3.1];
    const delay = -0.15; // own take started *early*
    const ownStarts = [...melodyStarts.map((t) => t + delay), 5.0]; // stray extra onset far away
    const offset = estimateCoarseOffset(ownStarts, melodyStarts);
    expect(Math.abs(offset - (-delay))).toBeLessThan(0.03);
  });
});

describe('matchOnsetsToMelody', () => {
  it('matches every onset one-to-one when the take is just a constant delay away', () => {
    const melodyStarts = [0.3, 0.9, 1.6, 2.4, 3.1];
    const delay = 0.25;
    const ownStarts = melodyStarts.map((t) => t + delay);
    const anchors = matchOnsetsToMelody(ownStarts, melodyStarts);
    expect(anchors.length).toBe(melodyStarts.length);
    anchors.forEach((a, i) => {
      expect(a.melodyTime).toBeCloseTo(melodyStarts[i], 5);
      expect(a.ownTime).toBeCloseTo(ownStarts[i], 5);
    });
  });

  it('stays monotonic and skips a missed melody note rather than mismatching', () => {
    const melodyStarts = [0.3, 0.9, 1.6, 2.4, 3.1];
    // The singer skipped the note at 1.6 entirely — their onsets jump from
    // the equivalent of 0.9 straight to 2.4.
    const ownStarts = [0.3, 0.9, 2.4, 3.1].map((t) => t + 0.1);
    const anchors = matchOnsetsToMelody(ownStarts, melodyStarts);
    // Every matched pair must be strictly increasing in both own and
    // melody time (no crossed/out-of-order matches).
    for (let i = 1; i < anchors.length; i++) {
      expect(anchors[i].ownTime).toBeGreaterThan(anchors[i - 1].ownTime);
      expect(anchors[i].melodyTime).toBeGreaterThan(anchors[i - 1].melodyTime);
    }
    // The skipped note (1.6) should never appear as a matched melodyTime.
    expect(anchors.some((a) => Math.abs(a.melodyTime - 1.6) < 0.01)).toBe(false);
    expect(anchors.length).toBe(4);
  });

  it('finds no anchors when the two sequences share no plausible offset', () => {
    const melodyStarts = [0.3, 0.9, 1.6];
    const ownStarts = [10.0, 14.0, 18.0]; // unrelated, far outside the coarse search range
    expect(matchOnsetsToMelody(ownStarts, melodyStarts)).toEqual([]);
  });
});

describe('buildAlignmentKeyframes', () => {
  it('returns null (meaning "leave untouched") for no anchors', () => {
    expect(buildAlignmentKeyframes([])).toBeNull();
  });

  it('derives a rate that exactly closes the gap between two anchors', () => {
    // Own take's syllable spacing is 1.5x wider than the melody's between
    // these two anchors — should need to play back at 1.5x rate there to
    // still land exactly on the second anchor.
    const anchors = [{ ownTime: 1.0, melodyTime: 1.0 }, { ownTime: 4.0, melodyTime: 3.0 }];
    const keyframes = buildAlignmentKeyframes(anchors);
    const segmentKeyframe = keyframes.find((k) => k.output === 1.0 && k.input === 1.0);
    expect(segmentKeyframe.rate).toBeCloseTo(1.5, 5); // (4.0-1.0)/(3.0-1.0)
  });
});

describe('alignOwnTakeToMelody', () => {
  it('returns the input buffer unchanged when there are no melody notes', async () => {
    const data = new Float32Array(SR);
    const setupCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, data.length, SR);
    const ownBuffer = toAudioBuffer(setupCtx, data, SR);
    const result = await alignOwnTakeToMelody(ownBuffer, [], 3.0);
    expect(result).toBe(ownBuffer);
  });

  it('pulls a late, uniformly-slower take back onto the melody\'s onset times', async () => {
    // Ground truth: melody sung on-time; the "own take" is the *same*
    // phrase, but started 0.3s late and sung 20% slower throughout (as if
    // the singer just naturally dragged) — exactly the kind of drift
    // MTrackAlign-style alignment is for, distinct from a simple constant
    // offset.
    const melodyNotes = makeNoteList([220, 247, 262, 294, 330]);
    const totalDur = melodyNotes[melodyNotes.length - 1].end + 0.6;

    const lateFactor = 1.2;
    const startDelay = 0.3;
    const ownNotes = melodyNotes.map((n) => ({
      start: n.start * lateFactor + startDelay,
      end: n.end * lateFactor + startDelay,
      freq: n.freq * 1.5, // a different harmony pitch than the melody, on purpose
    }));
    const ownTotalDur = ownNotes[ownNotes.length - 1].end + 0.6;
    const raw = synthesizeVoice(ownNotes, ownTotalDur, SR, { gapNoiseAmp: 0 });
    const setupCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, raw.length, SR);
    const ownBuffer = toAudioBuffer(setupCtx, raw, SR);

    const aligned = await alignOwnTakeToMelody(ownBuffer, melodyNotes, totalDur);
    expect(aligned.length).toBe(Math.round(totalDur * SR));

    const alignedSegments = detectOnsetSegments(aligned.getChannelData(0), SR);
    // Every melody onset should now have an aligned-take onset within a
    // fraction of a beat of it — the whole point of the warp.
    melodyNotes.forEach((n) => {
      const nearest = Math.min(...alignedSegments.map((s) => Math.abs(s.start - n.start)));
      expect(nearest).toBeLessThan(0.08);
    });
  });
});
