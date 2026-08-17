import { MAJOR_PROFILE, MINOR_PROFILE, scaleStepToMidi } from './musicTheory.js';

/* ---------- Pitch detection (autocorrelation, vocal range) ---------- */

export function autoCorrelate(buf, sampleRate) {
  const SIZE = buf.length;
  let rms = 0;
  for (let i = 0; i < SIZE; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / SIZE);
  if (rms < 0.009) return { freq: -1, rms };

  const minLag = Math.floor(sampleRate / 1200); // ~1200 Hz upper bound
  const maxLag = Math.floor(sampleRate / 55); // ~55 Hz lower bound
  let bestLag = -1;
  let bestCorr = 0;
  const c = new Float32Array(maxLag + 2);

  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < SIZE - lag; i++) sum += buf[i] * buf[i + lag];
    c[lag] = sum;
    if (sum > bestCorr) {
      bestCorr = sum;
      bestLag = lag;
    }
  }
  if (bestLag < minLag) return { freq: -1, rms };

  // Parabolic interpolation around the peak for sub-sample accuracy.
  const c0 = c[bestLag - 1] || c[bestLag];
  const c1 = c[bestLag];
  const c2 = c[bestLag + 1] || c[bestLag];
  const denom = c0 - 2 * c1 + c2;
  const shift = denom !== 0 ? 0.5 * (c0 - c2) / denom : 0;
  const refinedLag = bestLag + shift;

  return { freq: sampleRate / refinedLag, rms };
}

/* ---------- Key detection ---------- */

function correlate(hist, profile) {
  const n = 12;
  const meanH = hist.reduce((a, b) => a + b, 0) / n;
  const meanP = profile.reduce((a, b) => a + b, 0) / n;
  let num = 0, denH = 0, denP = 0;
  for (let i = 0; i < n; i++) {
    const dh = hist[i] - meanH;
    const dp = profile[i] - meanP;
    num += dh * dp;
    denH += dh * dh;
    denP += dp * dp;
  }
  if (denH === 0 || denP === 0) return 0;
  return num / Math.sqrt(denH * denP);
}

export function detectKey(hist) {
  let best = { corr: -Infinity, tonic: 0, mode: 'major' };
  for (let tonic = 0; tonic < 12; tonic++) {
    const rotatedMajor = MAJOR_PROFILE.map((_, i) => MAJOR_PROFILE[(i - tonic + 12) % 12]);
    const rotatedMinor = MINOR_PROFILE.map((_, i) => MINOR_PROFILE[(i - tonic + 12) % 12]);
    const cMaj = correlate(hist, rotatedMajor);
    const cMin = correlate(hist, rotatedMinor);
    if (cMaj > best.corr) best = { corr: cMaj, tonic, mode: 'major' };
    if (cMin > best.corr) best = { corr: cMin, tonic, mode: 'minor' };
  }
  return best;
}

/* ---------- Frames -> discrete notes ---------- */

// Bridges short unvoiced dropouts (e.g. a consonant clipping the mic) when
// the pitch is the same right before and right after the gap, so a quick
// syllable doesn't get split into fragments or dropped entirely.
export function fillShortGaps(qFrames, maxGapSec = 0.12) {
  const out = qFrames.map((f) => ({ ...f }));
  let i = 0;
  while (i < out.length) {
    if (out[i].step === null) {
      let j = i;
      while (j < out.length && out[j].step === null) j++;
      const gapStart = i > 0 ? out[i - 1].t : null;
      const gapEnd = j < out.length ? out[j].t : null;
      if (gapStart !== null && gapEnd !== null && out[i - 1].step === out[j].step && gapEnd - gapStart <= maxGapSec) {
        for (let k = i; k < j; k++) out[k].step = out[i - 1].step;
      }
      i = j;
    } else {
      i++;
    }
  }
  return out;
}

export function framesToNotes(qFrames, minDur = 0.07) {
  const notes = [];
  let cur = null;
  qFrames.forEach((f) => {
    if (f.step === null) {
      if (cur) { notes.push(cur); cur = null; }
      return;
    }
    if (cur && cur.step === f.step) {
      cur.end = f.t;
    } else {
      if (cur) notes.push(cur);
      cur = { step: f.step, start: f.t, end: f.t };
    }
  });
  if (cur) notes.push(cur);
  return notes.filter((n) => n.end - n.start >= minDur);
}

// Drops brief notes that sit right next to a stable note but are an
// implausible interval away from it (10+ semitones). These are almost
// always pitch-tracking artifacts from the attack transient of a syllable
// (the tracker briefly locks onto a harmonic before settling), not a note
// the person actually sang — and pitch-shifting them produces an audible
// "chirp".
export function filterTransientArtifacts(notes, tonicPC, mode, maxArtifactDur = 0.15, minJumpSemitones = 10, maxNeighborGap = 0.12) {
  const midiFor = (n) => scaleStepToMidi(n.step, tonicPC, mode);
  return notes.filter((n, i) => {
    if (n.end - n.start > maxArtifactDur) return true;
    const midi = midiFor(n);
    const prev = notes[i - 1];
    const next = notes[i + 1];
    if (prev && n.start - prev.end <= maxNeighborGap && Math.abs(midi - midiFor(prev)) >= minJumpSemitones) return false;
    if (next && next.start - n.end <= maxNeighborGap && Math.abs(midi - midiFor(next)) >= minJumpSemitones) return false;
    return true;
  });
}
