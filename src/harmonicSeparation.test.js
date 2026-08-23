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

describe('separateHarmonicResidual', () => {
  const keyInfo = { tonic: 0, mode: 'major' };

  it('routes a tonal segment mostly to the harmonic layer and a noise burst mostly to the residual layer', () => {
    const data = toneNoiseToneSignal();
    const buffer = makeBuffer(data);
    // One note covering the whole duration at the tone's pitch — the noise
    // burst sits *inside* it, exercising the peak-ratio "no real harmonic
    // here" rejection rather than a note-boundary gap.
    const melodyNotes = [{ step: 0, start: 0, end: 2.0, measuredFreq: 220 }];
    const { harmonicBuffer, residualBuffer } = separateHarmonicResidual(buffer, melodyNotes, keyInfo);

    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    const toneHarmonicRms = rmsOf(h, Math.round(0.1 * SR), Math.round(0.7 * SR));
    const toneResidualRms = rmsOf(r, Math.round(0.1 * SR), Math.round(0.7 * SR));
    const noiseHarmonicRms = rmsOf(h, Math.round(0.85 * SR), Math.round(1.15 * SR));
    const noiseResidualRms = rmsOf(r, Math.round(0.85 * SR), Math.round(1.15 * SR));

    expect(toneHarmonicRms).toBeGreaterThan(toneResidualRms * 3);
    expect(noiseResidualRms).toBeGreaterThan(noiseHarmonicRms * 2);
  });

  it('reconstructs (harmonic + residual) back to essentially the original signal', () => {
    const data = toneNoiseToneSignal();
    const buffer = makeBuffer(data);
    const melodyNotes = [{ step: 0, start: 0, end: 2.0, measuredFreq: 220 }];
    const { harmonicBuffer, residualBuffer } = separateHarmonicResidual(buffer, melodyNotes, keyInfo);

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

  it('still finds the harmonic layer past the last detected note when the singer audibly keeps going (nearest-note fallback)', () => {
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

    const { harmonicBuffer, residualBuffer } = separateHarmonicResidual(buffer, melodyNotes, keyInfoLocal);
    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    const from = Math.round(1.8 * SR);
    const to = Math.round(2.2 * SR);
    expect(rmsOf(h, from, to)).toBeGreaterThan(rmsOf(r, from, to) * 2);
  });

  it('leaves genuine silence as (near-)silent in both layers, not shifted-sounding harmonic content', () => {
    const totalDur = 1.5;
    const data = new Float32Array(Math.round(SR * totalDur)); // all zero — true digital silence
    const buffer = makeBuffer(data);
    const melodyNotes = [{ step: 0, start: 0, end: 1.5, measuredFreq: 220 }];
    const { harmonicBuffer, residualBuffer } = separateHarmonicResidual(buffer, melodyNotes, keyInfo);

    const h = harmonicBuffer.getChannelData(0);
    const r = residualBuffer.getChannelData(0);
    expect(rmsOf(h, 0, data.length)).toBeLessThan(1e-6);
    expect(rmsOf(r, 0, data.length)).toBeLessThan(1e-6);
  });
});
