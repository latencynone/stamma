import { describe, it, expect } from 'vitest';
import { renderHarmonyOffline } from './harmonyEngine.js';
import { midiToFreq, scaleStepToMidi, HARMONY_TYPES } from './musicTheory.js';
import { autoCorrelate } from './pitchAnalysis.js';
import { synthesizeVoice, toAudioBuffer, rmsOf, maxSampleDelta, dropoutFraction } from './testUtils.js';

const SR = 44100;

// Builds a short melody as ascending diatonic scale steps, each note
// separated by a short (consonant-like) gap, with one deliberately long
// pause inserted partway through — so every test recording exercises both
// the short-gap ramp and the real-pause fade in buildRatioCurve.
function makeMelody(steps, keyInfo, { start = 0.3, noteDur = 0.45, gapDur = 0.07, pauseAfterIndex = 1, pauseDur = 0.9 } = {}) {
  const notes = [];
  let t = start;
  steps.forEach((step, i) => {
    const noteStart = t;
    const noteEnd = t + noteDur;
    notes.push({ step, start: noteStart, end: noteEnd, freq: midiToFreq(scaleStepToMidi(step, keyInfo.tonic, keyInfo.mode)) });
    t = noteEnd + (i === pauseAfterIndex ? pauseDur : gapDur);
  });
  const totalDur = t + 0.4;
  return { melodyNotes: notes.map(({ step, start: s, end: e }) => ({ step, start: s, end: e })), synthNotes: notes, totalDur };
}

async function renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur) {
  const raw = synthesizeVoice(synthNotes, totalDur, SR);
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const setupCtx = new OfflineCtx(1, raw.length, SR);
  const recordedBuffer = toAudioBuffer(setupCtx, raw, SR);

  const outBuffer = await renderHarmonyOffline(recordedBuffer, melodyNotes, harmonyNotes, keyInfo);
  return { inData: recordedBuffer.getChannelData(0), outData: outBuffer.getChannelData(0) };
}

describe('renderHarmonyOffline — sömlöshet', () => {
  const keyInfo = { tonic: 0, mode: 'major' };
  const { melodyNotes, synthNotes, totalDur } = makeMelody([0, 2, 4, 7], keyInfo);
  const harmonyNotes = melodyNotes.map((n) => ({ ...n, hStep: n.step - HARMONY_TYPES.ters.steps }));

  it('lämnar inga tysta hack där originalet har ljud (0% dropout)', async () => {
    const { inData, outData } = await renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur);
    expect(dropoutFraction(inData, outData, SR)).toBe(0);
  });

  it('introducerar inga klick (max sample-till-sample-hopp i samma storleksordning som originalet)', async () => {
    const { inData, outData } = await renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur);
    const deltaIn = maxSampleDelta(inData);
    const deltaOut = maxSampleDelta(outData);
    expect(deltaOut).toBeLessThan(deltaIn * 3);
  });

  it('håller volymen per ton inom 0.7–1.4× originaltonens egen RMS', async () => {
    const { inData, outData } = await renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur);
    synthNotes.forEach((n) => {
      const s0 = Math.round(n.start * SR);
      const s1 = Math.round(n.end * SR);
      const ratio = rmsOf(outData, s0, s1) / rmsOf(inData, s0, s1);
      expect(ratio).toBeGreaterThan(0.7);
      expect(ratio).toBeLessThan(1.4);
    });
  });
});

describe('renderHarmonyOffline — tonhöjdsnoggrannhet', () => {
  const combos = [];
  Object.keys(HARMONY_TYPES).forEach((type) => {
    [-1, 1].forEach((direction) => {
      ['major', 'minor'].forEach((mode) => {
        combos.push({ type, direction, mode });
      });
    });
  });

  it.each(combos)('$type, riktning $direction, $mode: fel < 1% per ton', async ({ type, direction, mode }) => {
    const keyInfo = { tonic: 0, mode };
    const steps = HARMONY_TYPES[type].steps;
    const { melodyNotes, synthNotes, totalDur } = makeMelody([0, 1, 2, 3], keyInfo);
    const harmonyNotes = melodyNotes.map((n) => ({ ...n, hStep: n.step + direction * steps }));

    const { outData } = await renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur);

    melodyNotes.forEach((n, idx) => {
      const expectedFreq = midiToFreq(scaleStepToMidi(harmonyNotes[idx].hStep, keyInfo.tonic, keyInfo.mode));
      const midT = (n.start + n.end) / 2;
      const winLen = 2048;
      const centerSample = Math.round(midT * SR);
      const startSample = Math.max(0, centerSample - winLen / 2);
      const window = outData.slice(startSample, startSample + winLen);
      const { freq } = autoCorrelate(window, SR);
      expect(freq).toBeGreaterThan(0);
      const errPct = (100 * Math.abs(freq - expectedFreq)) / expectedFreq;
      expect(errPct).toBeLessThan(1);
    });
  });
});
