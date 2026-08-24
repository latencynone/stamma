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

// Runs the same window/hop autocorrelation pass used on a live microphone
// stream, but directly against a decoded audio buffer — used for uploaded
// files, which have no live AnalyserNode loop to capture frames from.
export function extractFramesFromBuffer(channelData, sampleRate, { fftSize = 2048, hopSec = 0.035 } = {}) {
  const totalDur = channelData.length / sampleRate;
  const frames = [];
  for (let t = 0; t < totalDur; t += hopSec) {
    const endSample = Math.round(t * sampleRate);
    const startSample = endSample - fftSize;
    const win = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      const si = startSample + i;
      win[i] = si >= 0 && si < channelData.length ? channelData[si] : 0;
    }
    const { freq } = autoCorrelate(win, sampleRate);
    frames.push({ t, freq: freq > 55 && freq < 1200 ? freq : -1 });
  }
  return frames;
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

// Drops short notes that sit in register far from where the singer has
// otherwise been singing, even when they're too isolated (silence on both
// sides) or too mutually-consistent with a neighboring artifact for
// filterTransientArtifacts's local jump check to catch them. A brief
// pitch-tracker misfire — a mic pop before singing starts, an octave/
// subharmonic lock on a breath or sibilant — produces exactly this shape: a
// short note more than an octave from the melody's home register, with nothing
// nearby to compare it against. Long notes are exempt: a real, sustained leap
// of an octave or more is plausible; a 70ms one essentially never is.
export function filterOutlierNotes(notes, maxDeviationSteps = 7, maxOutlierDur = 0.15) {
  if (notes.length < 3) return notes;
  // Duration-weighted median step, in centiseconds of resolution — the
  // melody's "home" register, robust to a handful of short artifacts
  // skewing a simple unweighted average or median.
  const weighted = [];
  notes.forEach((n) => {
    const weight = Math.max(1, Math.round((n.end - n.start) * 100));
    for (let i = 0; i < weight; i++) weighted.push(n.step);
  });
  weighted.sort((a, b) => a - b);
  const homeStep = weighted[Math.floor(weighted.length / 2)];
  return notes.filter((n) => (n.end - n.start) > maxOutlierDur || Math.abs(n.step - homeStep) <= maxDeviationSteps);
}

// Attaches each note's actual (unquantized) sung pitch — the median raw Hz
// reading across its voiced frames — separately from its quantized `step`.
// Autotune needs both: the quantized step says where the note *should* sit,
// `measuredFreq` says where the singer actually put it, and the difference
// between the two is exactly what a light pitch correction should nudge.
export function attachMeasuredFreq(notes, voicedFrames) {
  return notes.map((n) => {
    const inRange = voicedFrames.filter((f) => f.t >= n.start && f.t <= n.end).map((f) => f.freq).sort((a, b) => a - b);
    const measuredFreq = inRange.length ? inRange[Math.floor(inRange.length / 2)] : null;
    return { ...n, measuredFreq };
  });
}

/* ---------- Tempo detection ---------- */

// A simple, honest-about-being-approximate tempo estimate: the gaps
// between consecutive note onsets (inter-onset intervals) cluster around
// the beat duration (or a simple fraction/multiple of it) for anything
// with a steady pulse. Uses the *median* IOI rather than requiring several
// onsets to land in the same narrow histogram bucket — a short recording
// (the app caps at 10s) with only a handful of held notes rarely produces
// two IOIs within 50ms of each other even when sung at a rock-steady
// tempo, since human timing isn't machine-quantized; the old bucket-vote
// approach needed that near-exact repeat to ever return anything, so most
// real (as opposed to metronomically clean) recordings silently got no
// tempo guess at all. A single IOI is still a usable (if rougher) estimate,
// so only one onset gap is required. Folded into a 60–180 BPM range by
// doubling/halving, since a raw IOI can't tell a beat from a half- or
// double-time reading of it.
export function detectTempoBpm(melodyNotes) {
  if (!melodyNotes || melodyNotes.length < 2) return null;
  const onsets = melodyNotes.map((n) => n.start).slice().sort((a, b) => a - b);
  const iois = [];
  for (let i = 1; i < onsets.length; i++) {
    const d = onsets[i] - onsets[i - 1];
    // Upper bound wide enough to admit a breath/phrase pause on a slow
    // ballad (a real beat this long is implausible, but the fold-into-
    // range loop below recovers a sane guess from it anyway) while still
    // excluding within-syllable pitch-tracker blips via the lower bound.
    if (d > 0.15 && d < 4.0) iois.push(d);
  }
  if (iois.length < 1) return null;

  const sorted = iois.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const beatDur = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  if (beatDur <= 0) return null;
  let bpm = 60 / beatDur;
  while (bpm < 60) bpm *= 2;
  while (bpm > 180) bpm /= 2;
  return Math.round(bpm);
}
