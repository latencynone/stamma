import { describe, it, expect } from 'vitest';
import { separateHarmonicResidual } from './harmonicSeparation.js';
import { midiToFreq, scaleStepToMidi } from './musicTheory.js';
import { synthesizeVoice, rmsOf } from './testUtils.js';

const SR = 44100;

function makeBuffer(data, sr = SR) {
  const buf = new AudioBuffer({ length: data.length, numberOfChannels: 1, sampleRate: sr });
  buf.copyToChannel(data, 0);
  return buf;
}

// A hand-built signal with known, separated tone/noise segments — unlike
// synthesizeVoice's short consonant-gap noise, this isolates a whole
// noise-only stretch so the harmonic/noise routing itself can be checked
// against ground truth, independent of note timing.
function toneNoiseToneSignal() {
  const totalDur = 2.0;
  const len = Math.round(SR * totalDur);
  const data = new Float32Array(len);
  let seed = 42;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 2 - 1;
  };
  for (let i = 0; i < len; i++) {
    const t = i / SR;
    if (t >= 0.8 && t < 1.2) {
      data[i] = rand() * 0.3;
    } else {
      data[i] = 0.25 * Math.sin(2 * Math.PI * 220 * t)
        + 0.12 * Math.sin(2 * Math.PI * 440 * t)
        + 0.06 * Math.sin(2 * Math.PI * 660 * t);
    }
  }
  return data;
}

describe('separateHarmonicResidual', async () => {
  const keyInfo = { tonic: 0, mode: 'major' };

  it('routes a tonal segment mostly to the harmonic layer and a noise burst mostly to the residual layer', async () => {
    const data = toneNoiseToneSignal();
    const buffer = makeBuffer(data);
    // One note covering the whole duration at the tone's pitch — the noise
    // burst sits *inside* it, exercising the peak-ratio "no real harmonic
    // here" rejection rather than a note-boundary gap.
    const melodyNotes = [{ step: 0, start: 0, end: 2.0, measuredFreq: 220 }];
    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfo);

    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    const toneHarmonicRms = rmsOf(h, Math.round(0.1 * SR), Math.round(0.7 * SR));
    const toneResidualRms = rmsOf(r, Math.round(0.1 * SR), Math.round(0.7 * SR));
    const noiseHarmonicRms = rmsOf(h, Math.round(0.85 * SR), Math.round(1.15 * SR));
    const noiseResidualRms = rmsOf(r, Math.round(0.85 * SR), Math.round(1.15 * SR));

    expect(toneHarmonicRms).toBeGreaterThan(toneResidualRms * 3);
    expect(noiseResidualRms).toBeGreaterThan(noiseHarmonicRms * 2);
  });

  it('reconstructs (harmonic + residual) back to essentially the original signal', async () => {
    const data = toneNoiseToneSignal();
    const buffer = makeBuffer(data);
    const melodyNotes = [{ step: 0, start: 0, end: 2.0, measuredFreq: 220 }];
    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfo);

    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    let sumSqErr = 0;
    let sumSqOrig = 0;
    for (let i = 0; i < data.length; i++) {
      const err = (h[i] + r[i]) - data[i];
      sumSqErr += err * err;
      sumSqOrig += data[i] * data[i];
    }
    const relError = Math.sqrt(sumSqErr / sumSqOrig);
    expect(relError).toBeLessThan(0.01);
  });

  it('still finds the harmonic layer past the last detected note when the singer audibly keeps going (nearest-note fallback)', async () => {
    // Mirrors the exact scenario renderHarmonyOffline's own test suite
    // guards against: note detection only covers 0.3-1.0s, but the source
    // audio keeps singing the same pitch to 2.5s. Without a fallback, every
    // frame past 1.0s has no expected pitch, finds no harmonic peak, and
    // gets routed entirely to the residual layer — i.e. never shifted.
    const keyInfoLocal = { tonic: 0, mode: 'major' };
    const freq = midiToFreq(scaleStepToMidi(0, 0, 'major'));
    const synthNotes = [
      { step: 0, start: 0.3, end: 1.0, freq },
      { step: 0, start: 1.0, end: 2.5, freq }, // undetected continuation
    ];
    const raw = synthesizeVoice(synthNotes, 3.0, SR, { gapNoiseAmp: 0 });
    const buffer = makeBuffer(raw);
    const melodyNotes = [{ step: 0, start: 0.3, end: 1.0 }]; // detector "missed" the rest

    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfoLocal);
    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    const from = Math.round(1.8 * SR);
    const to = Math.round(2.2 * SR);
    expect(rmsOf(h, from, to)).toBeGreaterThan(rmsOf(r, from, to) * 2);
  });

  // The four tests below reproduce the exact gap the 2026-08-23 "audio
  // breaks through" regression fix (see harmonicSeparation.js's module
  // comment on MIN_PEAK_RATIO/MIN_HARMONICS_FOR_CONFIRMATION) closed: every
  // earlier test signal here is a flat, unwavering tone, which real singing
  // never is. A frame this module fails to confirm as harmonic falls
  // through to the residual layer and plays back at the *original*
  // melody's pitch instead of the shifted harmony — audibly "the original
  // voice breaking through" — so these check harmonic-dominance in several
  // separate windows across a note, not just one aggregate measurement,
  // since a regression here typically only shows up during part of a note
  // (a vibrato phase, an amplitude dip), not the whole thing.
  const windowSec = 0.15;
  // Checks harmonic-vs-residual dominance in a series of short windows
  // across [from, totalDur), returning the ones that failed — used instead
  // of one aggregate RMS check so a regression that only breaks part of a
  // note (a vibrato phase, an amplitude dip) is still caught.
  function findFailedWindows(h, r, from, totalDur, dominanceRatio = 1.5) {
    const failed = [];
    // Stops well short of totalDur: the note's own fade-out (10ms, see
    // synthesizeVoice) and this module's documented "last partial hop
    // always routes to residual" tail handling both make the very end of
    // a note genuinely, correctly residual-dominant — not a classification
    // failure worth catching here.
    for (let t = from; t < totalDur - 0.2; t += windowSec) {
      const s0 = Math.round(t * SR);
      const s1 = Math.round(Math.min(t + windowSec, totalDur) * SR);
      const hRms = rmsOf(h, s0, s1);
      const rRms = rmsOf(r, s0, s1);
      if (hRms <= rRms * dominanceRatio) failed.push({ t: t.toFixed(2), hRms, rRms });
    }
    return failed;
  }

  it('stays harmonic-dominant throughout a note with realistic vibrato, not just where the pitch happens to sit still at the note\'s one averaged measuredFreq', async () => {
    const freq = midiToFreq(scaleStepToMidi(2, 0, 'major'));
    const totalDur = 2.0;
    const raw = synthesizeVoice([{ freq, start: 0, end: totalDur }], totalDur, SR, {
      gapNoiseAmp: 0,
      vibratoDepthCents: 55, // wide, realistic vibrato — well inside the 120c f0 search window but far from motionless
      vibratoRateHz: 5.5,
      breathNoiseAmp: 0.05,
      harmonicAmps: [0.11, 1.0, 0.45, 0.22, 0.1], // formant-boosted: fundamental deliberately weak relative to a strong 2nd harmonic, like a real vowel's formant structure
    });
    const buffer = makeBuffer(raw);
    const melodyNotes = [{ step: 2, start: 0, end: totalDur, measuredFreq: freq }];

    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfo);
    const failed = findFailedWindows(harmonicBuffer.getChannelData(0), residualBuffer.getChannelData(0), 0.1, totalDur);
    expect(failed, `windows where the vibrato-swung tone wasn't routed harmonic: ${JSON.stringify(failed)}`).toEqual([]);
  });

  it('stays harmonic-dominant through a quiet dynamic dip, not just at the note\'s loudest moments', async () => {
    const freq = midiToFreq(scaleStepToMidi(4, 0, 'major'));
    const totalDur = 2.0;
    const raw = synthesizeVoice([{ freq, start: 0, end: totalDur }], totalDur, SR, {
      gapNoiseAmp: 0,
      dynamicsDepth: 0.75, // dips to 25% of peak amplitude, periodically
      dynamicsRateHz: 2.2,
      breathNoiseAmp: 0.05,
      harmonicAmps: [0.11, 1.0, 0.45, 0.22, 0.1], // formant-boosted, see the vibrato test above
    });
    const buffer = makeBuffer(raw);
    const melodyNotes = [{ step: 4, start: 0, end: totalDur, measuredFreq: freq }];

    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfo);
    const failed = findFailedWindows(harmonicBuffer.getChannelData(0), residualBuffer.getChannelData(0), 0.1, totalDur);
    expect(failed, `windows where the quiet dip wasn't routed harmonic: ${JSON.stringify(failed)}`).toEqual([]);
  });

  it('stays harmonic-dominant with vibrato and dynamics together — the actual reported regression (a real recording, not a synthetic tone)', async () => {
    const freq = midiToFreq(scaleStepToMidi(1, 0, 'major'));
    const totalDur = 2.5;
    const raw = synthesizeVoice([{ freq, start: 0, end: totalDur }], totalDur, SR, {
      gapNoiseAmp: 0,
      vibratoDepthCents: 45,
      vibratoRateHz: 5.8,
      dynamicsDepth: 0.55,
      dynamicsRateHz: 1.8,
      breathNoiseAmp: 0.05,
      harmonicAmps: [0.11, 1.0, 0.45, 0.22, 0.1], // formant-boosted, see the vibrato test above
    });
    const buffer = makeBuffer(raw);
    const melodyNotes = [{ step: 1, start: 0, end: totalDur, measuredFreq: freq }];

    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfo);
    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);

    const failed = findFailedWindows(h, r, 0.1, totalDur);
    expect(failed, `windows where the real-singing-like tone wasn't routed harmonic: ${JSON.stringify(failed)}`).toEqual([]);

    // A fix that widens confirmation to chase dominance could do so at the
    // cost of corrupting the split itself — reconstruction must still hold.
    let sumSqErr = 0;
    let sumSqOrig = 0;
    for (let i = 0; i < raw.length; i++) {
      const err = (h[i] + r[i]) - raw[i];
      sumSqErr += err * err;
      sumSqOrig += raw[i] * raw[i];
    }
    expect(Math.sqrt(sumSqErr / sumSqOrig)).toBeLessThan(0.01);
  });

  it('leaves genuine silence as (near-)silent in both layers, not shifted-sounding harmonic content', async () => {
    const totalDur = 1.5;
    const data = new Float32Array(Math.round(SR * totalDur)); // all zero — true digital silence
    const buffer = makeBuffer(data);
    const melodyNotes = [{ step: 0, start: 0, end: 1.5, measuredFreq: 220 }];
    const { harmonicBuffer, residualBuffer } = await separateHarmonicResidual(buffer, melodyNotes, keyInfo);

    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    expect(rmsOf(h, 0, data.length)).toBeLessThan(1e-6);
    expect(rmsOf(r, 0, data.length)).toBeLessThan(1e-6);
  });
});
