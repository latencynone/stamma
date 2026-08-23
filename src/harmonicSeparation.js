import { fft, ifft, hannWindow } from './dsp.js';
import { scaleStepToMidi, midiToFreq } from './musicTheory.js';

/* ---------- Harmonic / residual voice separation ----------
 *
 * A phase-vocoder-style pitch shifter (like the SignalsmithStretch engine
 * this app renders harmonies through) implicitly treats the whole signal as
 * if it had a pitch. A real voice doesn't: it's a *tonal* part (the vocal
 * folds' harmonic buzz — vowels, hums, anything with an actual pitch) plus a
 * *noise* part (breath, aspiration, s/f/h and other unvoiced consonants —
 * genuinely pitchless). Forcing the noise part through a pitch shifter turns
 * it periodic — a real, named phase-vocoder artifact ("robotization") — and
 * is a large part of why a shifted harmony voice can sound metallic even
 * with natural-sounding vowels.
 *
 * This module splits a recording into those two layers via STFT analysis:
 * for each frame, search near the melody's already-known expected pitch (and
 * its harmonics) for real spectral peaks, build a soft per-bin mask from
 * whatever is actually found, and route the frame's spectrum through that
 * mask into a harmonic-only signal and a residual-only signal that sum back
 * to (approximately) the original. Only the harmonic layer then goes through
 * the pitch/formant shift in harmonyEngine.js; the residual is added back
 * unshifted, so consonants and breath keep their natural, non-tonal
 * character instead of being "sung" along with the vowels.
 *
 * Knowing the expected pitch in advance (from the melody's own analysis,
 * rather than blind detection) is what makes a fairly simple peak search
 * viable here — a general-purpose harmonic/noise separator has to solve a
 * much harder blind problem.
 *
 * Reuses the exact STFT/overlap-add scheme (window, hop, windowSum
 * normalization) already proven by the noise-reduction feature
 * (spectralSubtractChannel in App.jsx), just producing two complementary
 * outputs per frame instead of one denoised one.
 */

const FFT_SIZE = 2048;
const HOP = FFT_SIZE / 2;
const NYQUIST_BIN = FFT_SIZE / 2;
// Generous search window around the expected fundamental — wide enough to
// track natural pitch drift/vibrato within a note, narrow enough to stay
// locked onto the right partial rather than a neighboring one.
const F0_SEARCH_CENTS = 100;
// Tighter per-harmonic tolerance: real vibrato/inharmonicity moves a high
// harmonic by more absolute Hz than the fundamental, but by roughly the same
// *cents* amount, so a flat cents window works across the whole series.
const HARMONIC_SEARCH_CENTS = 65;
const MAX_HARMONICS = 20;
// Width (bins on each side of a confirmed peak) stamped into the mask —
// matches roughly one FFT_SIZE=2048 Hann main-lobe width, so a masked-in
// harmonic keeps its real spectral energy rather than just its peak sample.
const PEAK_BIN_WIDTH = 2;
// A candidate peak only counts as a real harmonic (not a noise-floor bump)
// if it reaches at least this fraction of the frame's overall spectral
// peak. Halved for harmonics above the fundamental, which are naturally
// quieter even in a clearly voiced frame.
const MIN_PEAK_RATIO = 0.12;

function centsToBinRange(freqHz, cents, sampleRate) {
  const lo = freqHz * Math.pow(2, -cents / 1200);
  const hi = freqHz * Math.pow(2, cents / 1200);
  const binHz = sampleRate / FFT_SIZE;
  return [Math.floor(lo / binHz), Math.ceil(hi / binHz)];
}

// Parabolic (quadratic) interpolation around a magnitude-spectrum peak bin
// for sub-bin frequency/magnitude refinement — standard peak-picking
// technique, meaningfully more accurate than trusting the raw bin alone at
// this FFT size.
function refinePeak(mag, k) {
  const y0 = mag[k - 1] ?? mag[k];
  const y1 = mag[k];
  const y2 = mag[k + 1] ?? mag[k];
  const denom = y0 - 2 * y1 + y2;
  const rawDelta = denom !== 0 ? (0.5 * (y0 - y2)) / denom : 0;
  const delta = Math.max(-1, Math.min(1, rawDelta));
  return { bin: k + delta, mag: y1 - 0.25 * (y0 - y2) * delta };
}

function findPeakInRange(mag, loBin, hiBin) {
  let bestK = -1;
  let bestMag = -Infinity;
  const lo = Math.max(1, loBin);
  const hi = Math.min(mag.length - 2, hiBin);
  for (let k = lo; k <= hi; k++) {
    if (mag[k] > bestMag) { bestMag = mag[k]; bestK = k; }
  }
  if (bestK < 0) return null;
  return refinePeak(mag, bestK);
}

// Builds a per-bin [0,1] mask for one frame's full-length (conjugate-
// symmetric) spectrum: ~1 near confirmed harmonic partials of
// `expectedF0Hz`, 0 elsewhere. Returns null (→ treat the whole frame as
// residual) when there's no expected pitch (a gap/pause) or no convincing
// fundamental peak turns up (an unvoiced/noisy frame) — in both cases
// there's nothing tonal here to preserve a shift for.
function buildHarmonicMask(mag, sampleRate, expectedF0Hz, scratch) {
  if (!expectedF0Hz) return null;
  const binHz = sampleRate / FFT_SIZE;

  const [f0Lo, f0Hi] = centsToBinRange(expectedF0Hz, F0_SEARCH_CENTS, sampleRate);
  const f0Peak = findPeakInRange(mag, f0Lo, f0Hi);
  if (!f0Peak) return null;

  let overallPeak = 0;
  for (let k = 1; k < NYQUIST_BIN; k++) if (mag[k] > overallPeak) overallPeak = mag[k];
  if (overallPeak <= 0 || f0Peak.mag < overallPeak * MIN_PEAK_RATIO) return null;

  const f0Hz = f0Peak.bin * binHz;
  const mask = scratch;
  mask.fill(0);
  const maxHarmonic = Math.min(MAX_HARMONICS, Math.floor((NYQUIST_BIN * binHz) / f0Hz));

  const stampPeak = (bin) => {
    const center = Math.round(bin);
    for (let d = -PEAK_BIN_WIDTH; d <= PEAK_BIN_WIDTH; d++) {
      const k = center + d;
      if (k < 1 || k >= NYQUIST_BIN) continue;
      const w = 1 - Math.abs(d) / (PEAK_BIN_WIDTH + 1);
      if (w > mask[k]) mask[k] = w;
    }
  };

  stampPeak(f0Peak.bin);
  for (let h = 2; h <= maxHarmonic; h++) {
    const [lo, hi] = centsToBinRange(f0Hz * h, HARMONIC_SEARCH_CENTS, sampleRate);
    const peak = findPeakInRange(mag, lo, hi);
    if (peak && peak.mag >= overallPeak * MIN_PEAK_RATIO * 0.5) stampPeak(peak.bin);
  }

  // Full spectrum is conjugate-symmetric; mirror the mask to match so
  // masking the real/imaginary arrays directly keeps that symmetry (and
  // therefore a real-valued signal after ifft).
  for (let k = 1; k < NYQUIST_BIN; k++) mask[FFT_SIZE - k] = mask[k];
  return mask;
}

// Splits one channel into harmonic-only and residual-only signals via STFT
// / overlap-add, using `f0AtTime(seconds) -> Hz|0` to look up the expected
// pitch per frame.
function splitChannel(channelData, sampleRate, f0AtTime) {
  const window = hannWindow(FFT_SIZE);
  const harmonicOut = new Float32Array(channelData.length);
  const residualOut = new Float32Array(channelData.length);
  const windowSum = new Float32Array(channelData.length);
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  const mag = new Float32Array(NYQUIST_BIN + 1);
  const maskScratch = new Float32Array(FFT_SIZE);
  const reH = new Float32Array(FFT_SIZE);
  const imH = new Float32Array(FFT_SIZE);
  const reR = new Float32Array(FFT_SIZE);
  const imR = new Float32Array(FFT_SIZE);
  let lastCovered = 0;

  for (let start = 0; start + FFT_SIZE <= channelData.length; start += HOP) {
    for (let i = 0; i < FFT_SIZE; i++) {
      re[i] = channelData[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= NYQUIST_BIN; k++) mag[k] = Math.hypot(re[k], im[k]);

    const frameCenterT = (start + FFT_SIZE / 2) / sampleRate;
    const mask = buildHarmonicMask(mag, sampleRate, f0AtTime(frameCenterT), maskScratch);

    for (let k = 0; k < FFT_SIZE; k++) {
      const m = mask ? mask[k] : 0;
      reH[k] = re[k] * m;
      imH[k] = im[k] * m;
      reR[k] = re[k] * (1 - m);
      imR[k] = im[k] * (1 - m);
    }
    ifft(reH, imH);
    ifft(reR, imR);
    for (let i = 0; i < FFT_SIZE; i++) {
      harmonicOut[start + i] += reH[i] * window[i];
      residualOut[start + i] += reR[i] * window[i];
      windowSum[start + i] += window[i] * window[i];
    }
    lastCovered = start + FFT_SIZE;
  }
  for (let i = 0; i < channelData.length; i++) {
    if (windowSum[i] > 1e-6) {
      harmonicOut[i] /= windowSum[i];
      residualOut[i] /= windowSum[i];
    } else {
      harmonicOut[i] = 0;
      residualOut[i] = channelData[i];
    }
  }
  // Same tail handling as spectralSubtractChannel: whatever's shorter than
  // one hop past the last full frame never got a window contribution —
  // carry it as residual (unshifted, i.e. unchanged) rather than dropping it.
  for (let i = lastCovered; i < channelData.length; i++) {
    harmonicOut[i] = 0;
    residualOut[i] = channelData[i];
  }
  return { harmonicOut, residualOut };
}

// melodyNotes -> a t(seconds) -> expectedF0Hz|0 lookup. Prefers each note's
// actually-measured pitch (closer to what's really in the recording than
// its quantized target) and falls back to the quantized target when a
// measurement wasn't available for that note.
//
// A frame that falls outside every note's [start,end) — between two notes,
// or past the last one — still gets the *nearest* note's frequency rather
// than "no expected pitch". The note detector's own boundaries are known to
// be conservative (buildRatioCurveFromSegments already treats an
// energy-backed gap it can't otherwise explain as a continuation, not a
// pause, for exactly this reason): a singer who's still audibly going past
// where note detection gave up is far more likely to still be near that
// note's pitch than to have no expected pitch at all. Giving up here would
// mean that audio finds no harmonic peak, gets routed entirely to the
// residual layer, and plays back unshifted — i.e. the original melody
// leaking through where the harmony should be.
function f0LookupFromNotes(melodyNotes, keyInfo) {
  const freqFor = (note) => note.measuredFreq || midiToFreq(scaleStepToMidi(note.step, keyInfo.tonic, keyInfo.mode));
  return (t) => {
    if (!melodyNotes.length) return 0;
    const inside = melodyNotes.find((n) => t >= n.start && t < n.end);
    if (inside) return freqFor(inside);
    let nearest = melodyNotes[0];
    let nearestDist = Infinity;
    melodyNotes.forEach((n) => {
      const dist = t < n.start ? n.start - t : t - n.end;
      if (dist < nearestDist) { nearestDist = dist; nearest = n; }
    });
    return freqFor(nearest);
  };
}

export function separateHarmonicResidual(recordedBuffer, melodyNotes, keyInfo) {
  const sampleRate = recordedBuffer.sampleRate;
  const channels = recordedBuffer.numberOfChannels;
  const length = recordedBuffer.length;
  const f0AtTime = f0LookupFromNotes(melodyNotes, keyInfo);

  const harmonicBuffer = new AudioBuffer({ length, numberOfChannels: channels, sampleRate });
  const residualBuffer = new AudioBuffer({ length, numberOfChannels: channels, sampleRate });
  for (let c = 0; c < channels; c++) {
    const { harmonicOut, residualOut } = splitChannel(recordedBuffer.getChannelData(c), sampleRate, f0AtTime);
    harmonicBuffer.copyToChannel(harmonicOut, c);
    residualBuffer.copyToChannel(residualOut, c);
  }
  return { harmonicBuffer, residualBuffer };
}

// The split only depends on the source recording + its analyzed notes, not
// on which harmony type/direction/autotune setting is being rendered — so
// it's computed once per recording and shared across every harmony render.
// Keyed by the AudioBuffer object itself: a new recording, upload, or
// normalize/denoise pass always produces a new buffer object, which
// naturally (and automatically, via WeakMap) invalidates the cache without
// any manual bookkeeping in App.jsx.
const splitCache = new WeakMap();

export function getOrComputeHarmonicSplit(recordedBuffer, melodyNotes, keyInfo) {
  const cached = splitCache.get(recordedBuffer);
  if (cached) return cached;
  const split = separateHarmonicResidual(recordedBuffer, melodyNotes, keyInfo);
  splitCache.set(recordedBuffer, split);
  return split;
}
