import { describe, it, expect } from 'vitest';
import { renderHarmonyOffline, renderAutotunedMelody } from './harmonyEngine.js';
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

describe('renderHarmonyOffline — energimedveten paus vs. luckor med ljud', () => {
  const keyInfo = { tonic: 0, mode: 'major' };

  it('skiftar fortfarande ljud i en lång lucka som notdetekteringen missade, istället för att spela upp originalet oskiftat', async () => {
    // Only one detected note, but the singer audibly keeps going for 1.5s
    // after it — as if the pitch tracker lost the note without the singer
    // actually going quiet. A real silent gap would (correctly) fall back to
    // "no shift"; here there's real signal, so it must not.
    const melodyNotes = [{ step: 0, start: 0.3, end: 1.0 }];
    const harmonyNotes = [{ step: 0, start: 0.3, end: 1.0, hStep: -2 }];
    const synthNotes = [
      { step: 0, start: 0.3, end: 1.0, freq: midiToFreq(scaleStepToMidi(0, 0, 'major')) },
      { step: 0, start: 1.0, end: 2.5, freq: midiToFreq(scaleStepToMidi(0, 0, 'major')) }, // undetected continuation
    ];
    const totalDur = 3.0;
    const { outData } = await renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur);

    const measure = (centerT) => {
      const winLen = 2048;
      const c = Math.round(centerT * SR);
      const s = Math.max(0, c - winLen / 2);
      return autoCorrelate(outData.slice(s, s + winLen), SR).freq;
    };
    const originalFreq = midiToFreq(scaleStepToMidi(0, 0, 'major'));
    const expectedHarmonyFreq = midiToFreq(scaleStepToMidi(-2, 0, 'major'));

    const measured = measure(2.0);
    expect(measured).toBeGreaterThan(0);
    // It should read as the harmony pitch, not a bare pass-through of the
    // original (which is what "leaking through" looked like in practice).
    expect(Math.abs(measured - expectedHarmonyFreq) / expectedHarmonyFreq).toBeLessThan(0.02);
    expect(Math.abs(measured - originalFreq) / originalFreq).toBeGreaterThan(0.05);
  });

  it('faller fortfarande tillbaka till ingen skiftning i en riktigt tyst paus', async () => {
    const melodyNotes = [
      { step: 0, start: 0.3, end: 1.0 },
      { step: 2, start: 2.5, end: 3.2 }, // genuinely silent 1.5s gap in between
    ];
    const harmonyNotes = melodyNotes.map((n) => ({ ...n, hStep: n.step - 2 }));
    const synthNotes = melodyNotes.map((n) => ({ ...n, freq: midiToFreq(scaleStepToMidi(n.step, keyInfo.tonic, keyInfo.mode)) }));
    const totalDur = 3.6;
    const { inData, outData } = await renderAndMeasure(melodyNotes, harmonyNotes, keyInfo, synthNotes, totalDur);

    const s0 = Math.round(1.3 * SR);
    const s1 = Math.round(2.2 * SR);
    expect(rmsOf(outData, s0, s1)).toBeLessThan(0.01);
    expect(rmsOf(inData, s0, s1)).toBeLessThan(0.001); // sanity: the gap really is silent in the source too
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

describe('renderAutotunedMelody — lätt tonhöjdskorrigering', () => {
  const keyInfo = { tonic: 0, mode: 'major' };

  it('flyttar en falsk ton delvis mot måltonen, inte hela eller ingen väg', async () => {
    const targetFreq = midiToFreq(scaleStepToMidi(0, keyInfo.tonic, keyInfo.mode)); // C4
    const sungFreq = targetFreq * Math.pow(2, 60 / 1200); // 60 cents sharp — audibly off-key
    const melodyNotes = [{ step: 0, start: 0.3, end: 1.3, measuredFreq: sungFreq }];
    const synthNotes = [{ step: 0, start: 0.3, end: 1.3, freq: sungFreq }];
    const totalDur = 1.8;
    const amount = 0.4;

    const raw = synthesizeVoice(synthNotes, totalDur, SR);
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const setupCtx = new OfflineCtx(1, raw.length, SR);
    const recordedBuffer = toAudioBuffer(setupCtx, raw, SR);

    const outBuffer = await renderAutotunedMelody(recordedBuffer, melodyNotes, keyInfo, amount);
    const outData = outBuffer.getChannelData(0);

    const winLen = 2048;
    const centerSample = Math.round(0.8 * SR);
    const { freq } = autoCorrelate(outData.slice(centerSample - winLen / 2, centerSample + winLen / 2), SR);
    expect(freq).toBeGreaterThan(0);

    const expectedFreq = sungFreq * (1 + (targetFreq / sungFreq - 1) * amount);
    const errPct = (100 * Math.abs(freq - expectedFreq)) / expectedFreq;
    expect(errPct).toBeLessThan(1);

    // Sanity: partial correction lands strictly between the sung and target pitch.
    expect(freq).toBeGreaterThan(Math.min(sungFreq, targetFreq));
    expect(freq).toBeLessThan(Math.max(sungFreq, targetFreq));
  });

  it('lämnar en redan korrekt ton praktiskt taget oförändrad', async () => {
    const targetFreq = midiToFreq(scaleStepToMidi(4, keyInfo.tonic, keyInfo.mode));
    const melodyNotes = [{ step: 4, start: 0.3, end: 1.3, measuredFreq: targetFreq }];
    const synthNotes = [{ step: 4, start: 0.3, end: 1.3, freq: targetFreq }];
    const totalDur = 1.8;

    const raw = synthesizeVoice(synthNotes, totalDur, SR);
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const setupCtx = new OfflineCtx(1, raw.length, SR);
    const recordedBuffer = toAudioBuffer(setupCtx, raw, SR);

    const outBuffer = await renderAutotunedMelody(recordedBuffer, melodyNotes, keyInfo);
    const outData = outBuffer.getChannelData(0);
    const winLen = 2048;
    const centerSample = Math.round(0.8 * SR);
    const { freq } = autoCorrelate(outData.slice(centerSample - winLen / 2, centerSample + winLen / 2), SR);
    const errPct = (100 * Math.abs(freq - targetFreq)) / targetFreq;
    expect(errPct).toBeLessThan(1);
  });
});
