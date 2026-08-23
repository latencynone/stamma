import { fft, hannWindow } from './dsp.js';
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
 * for each ~46ms frame, search near the melody's already-known expected
 * pitch (and its harmonics) for real spectral peaks, and decide whether the
 * *whole frame* is voiced — routing its entire spectrum to the harmonic
 * signal — or not, routing it entirely to the residual signal. Only the
 * harmonic layer then goes through the pitch/formant shift in
 * harmonyEngine.js; the residual is added back unshifted, so consonants and
 * breath keep their natural, non-tonal character instead of being "sung"
 * along with the vowels.
 *
 * The decision is deliberately per *frame*, not per spectral bin. An
 * earlier version of this module built a narrow per-bin mask — routing only
 * the exact bins right at each confirmed harmonic peak to the harmonic
 * layer, everything else in that same frame to residual. That measurably
 * broke on real recordings: a real voice's energy isn't concentrated at
 * infinitely narrow harmonic lines, it's spread by formant bandwidth,
 * in-frame vibrato, and aspiration mixed in with the tone — all real,
 * legitimately tonal signal that a narrow comb mask doesn't capture. That
 * spread energy was routed to residual and played back unshifted, audible
 * as a quiet, ghostly copy of the original melody's own pitch riding under
 * the shifted harmony ("sounds like there's a second voice quietly in the
 * background"). A frame small enough that speech/singing is overwhelmingly
 * either voiced or unvoiced within it doesn't need — and, on real audio,
 * actively suffers from — that finer-grained split; per-frame is both
 * simpler and correct.
 *
 * Knowing the expected pitch in advance (from the melody's own analysis,
 * rather than blind detection) is what makes a fairly simple peak search
 * viable here — a general-purpose harmonic/noise separator has to solve a
 * much harder blind problem.
 *
 * Reuses the exact STFT/overlap-add scheme (window, hop, windowSum
 * normalization) already proven by the noise-reduction feature
 * (spectralSubtractChannel in App.jsx), just routing each frame's windowed
 * samples to one of two accumulators instead of denoising into one.
 *
 * IMPORTANT failure-direction rule: getting a frame's classification wrong
 * must never be worse than not having this feature at all. A frame this
 * module fails to confirm as voiced gets routed to residual — i.e. played
 * back unshifted — so a genuinely voiced frame that's merely *missed* here
 * sounds like the original melody's own pitch leaking through where the
 * harmony should be. Real singing drifts (vibrato, dynamics) well past a
 * tight search window around a single per-note frequency, so the search
 * tolerances below are deliberately generous, and a frame only needs to
 * confirm *one* other harmonic alongside the fundamental (or a very
 * dominant fundamental on its own) to count as voiced — the goal is to
 * separate out unambiguous noise, not to be a strict voiced/unvoiced
 * classifier.
 */

const FFT_SIZE = 2048;
const HOP = FFT_SIZE / 2;
const NYQUIST_BIN = FFT_SIZE / 2;
// Search window around the expected fundamental — wide enough to absorb
// vibrato and a note-level measuredFreq that's only an average of what the
// singer actually did across the whole note, not a frame-accurate reading.
// Deliberately *not* wider than this: adjacent scale steps in a melody can
// be as little as a semitone (100 cents) apart, and a wider window risks
// locking onto a neighboring note's pitch instead of missing this one —
// which corrupts that stretch far more audibly than routing it to residual
// would have. Real robustness against missed detections comes from the
// confirmation-strictness knobs below, not from casting a wider net here.
const F0_SEARCH_CENTS = 120;
const HARMONIC_SEARCH_CENTS = 90;
// How many harmonics above the fundamental to check for confirmation.
// Deliberately small — not "every harmonic up to Nyquist". Checking many
// independent candidate bins against a modest per-bin threshold makes a
// false "yes, this is voiced" verdict increasingly likely on pure noise by
// chance the more chances it gets (with ~19 candidates up to Nyquist, even
// a modest per-bin false-positive rate compounds to a high chance *some*
// bin clears the bar — this is what let real white noise get misclassified
// as voiced during testing). Restricting to the first few harmonics, which
// are reliably the strongest in real voiced content anyway, keeps the
// confirmation both musically justified and statistically sound.
const HARMONICS_TO_CHECK = 4;
// A candidate peak only counts as a real harmonic (not a noise-floor bump)
// if it stands out from the frame's *average* spectral magnitude by at
// least this factor. Measured against real signals: a clean sung tone's
// fundamental peak sits at roughly 100-300x the frame's average magnitude,
// while broadband noise's strongest bin in any given narrow search window
// — even picking the best of several candidate windows — only reaches
// roughly 1-2x the average (noise energy is spread out, not concentrated).
// This threshold sits with a wide safety margin above the noise ceiling
// and a wide margin below a clean tone, leaving plenty of room for real
// (less pure, formant-shaped) singing in between. Comparing against the
// frame's average is what makes this reliable — comparing against the
// frame's single *global* peak (an earlier version of this check) isn't:
// broadband noise's global peak is itself just another random spike, often
// at some unrelated frequency (observed up in the 15-20kHz range on a pure
// noise test signal), so a narrowband candidate could spuriously look
// "dominant" relative to it purely by chance.
const MIN_PEAK_TO_AVG_RATIO = 4;
// A lone fundamental-range peak could just be a stray noise-floor bump
// that happened to fall in the search window — real voiced content nearly
// always shows at least one other aligned harmonic. Accept it on the
// fundamental alone only when it clears a much higher bar (still far below
// a clean tone's ~100-300x, comfortably above noise's ~1-2x).
const MIN_HARMONICS_FOR_CONFIRMATION = 2;
const DOMINANT_FUNDAMENTAL_TO_AVG_RATIO = 10;
// Number of STFT frames processed per synchronous batch before yielding to
// the event loop — keeps a long recording from blocking the main thread
// (and tripping the browser's "unresponsive page" handling) in one go.
const FRAMES_PER_YIELD = 24;

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

// Decides whether one frame's magnitude spectrum is confidently voiced:
// search near `expectedF0Hz` for a real fundamental peak, then look for at
// least one of its harmonics nearby too (or accept the fundamental alone
// when it's unusually dominant). Returns false when there's no expected
// pitch (a gap/pause) or not enough evidence of real harmonic content —
// the whole frame then gets routed to residual, see splitChannel.
function isVoicedFrame(mag, sampleRate, expectedF0Hz) {
  if (!expectedF0Hz) return false;
  const binHz = sampleRate / FFT_SIZE;

  let sum = 0;
  for (let k = 1; k < NYQUIST_BIN; k++) sum += mag[k];
  const avgMag = sum / (NYQUIST_BIN - 1);
  if (avgMag <= 0) return false;

  const [f0Lo, f0Hi] = centsToBinRange(expectedF0Hz, F0_SEARCH_CENTS, sampleRate);
  const f0Peak = findPeakInRange(mag, f0Lo, f0Hi);
  if (!f0Peak || f0Peak.mag < avgMag * MIN_PEAK_TO_AVG_RATIO) return false;

  const f0Hz = f0Peak.bin * binHz;
  const nyquistHz = NYQUIST_BIN * binHz;

  let confirmed = 1;
  for (let h = 2; h <= HARMONICS_TO_CHECK; h++) {
    const harmonicHz = f0Hz * h;
    if (harmonicHz >= nyquistHz) break;
    const [lo, hi] = centsToBinRange(harmonicHz, HARMONIC_SEARCH_CENTS, sampleRate);
    const peak = findPeakInRange(mag, lo, hi);
    if (peak && peak.mag >= avgMag * MIN_PEAK_TO_AVG_RATIO * 0.5) confirmed++;
  }
  // A single unconfirmed peak isn't enough evidence unless it's dominant —
  // see the module-level note on why this leans toward accepting real
  // voiced content rather than rejecting borderline noise.
  return confirmed >= MIN_HARMONICS_FOR_CONFIRMATION || f0Peak.mag >= avgMag * DOMINANT_FUNDAMENTAL_TO_AVG_RATIO;
}

function yieldToEventLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// Splits one channel into harmonic-only and residual-only signals via STFT
// / overlap-add, using `f0AtTime(seconds) -> Hz|0` to look up the expected
// pitch per frame. Each frame is routed *whole* to one accumulator or the
// other (see the module-level note on why this is per-frame, not per-bin)
// — so there's no need to inverse-FFT a masked spectrum back to the time
// domain; the already-windowed time-domain samples go straight to
// whichever accumulator this frame belongs to.
async function splitChannel(channelData, sampleRate, f0AtTime) {
  const window = hannWindow(FFT_SIZE);
  const harmonicOut = new Float32Array(channelData.length);
  const residualOut = new Float32Array(channelData.length);
  const windowSum = new Float32Array(channelData.length);
  const windowed = new Float32Array(FFT_SIZE);
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  const mag = new Float32Array(NYQUIST_BIN + 1);
  let lastCovered = 0;
  // A lone frame that fails right after a confidently-voiced one is more
  // likely a brief tracking blip (a fast vibrato swing, a momentary dip in
  // level) than a real transition to silence/consonant — holding the
  // voiced verdict for exactly one frame smooths over that without letting
  // a real multi-frame transition (an actual consonant/pause) get stuck.
  let heldVoiced = false;
  let heldUsed = false;
  let frameCount = 0;

  for (let start = 0; start + FFT_SIZE <= channelData.length; start += HOP) {
    for (let i = 0; i < FFT_SIZE; i++) {
      const w = channelData[start + i] * window[i];
      windowed[i] = w;
      re[i] = w;
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= NYQUIST_BIN; k++) mag[k] = Math.hypot(re[k], im[k]);

    const frameCenterT = (start + FFT_SIZE / 2) / sampleRate;
    const voiced = isVoicedFrame(mag, sampleRate, f0AtTime(frameCenterT));

    let effectiveVoiced = voiced;
    if (voiced) {
      heldVoiced = true;
      heldUsed = false;
    } else if (heldVoiced && !heldUsed) {
      effectiveVoiced = true;
      heldUsed = true;
    } else {
      heldVoiced = false;
    }

    const dest = effectiveVoiced ? harmonicOut : residualOut;
    for (let i = 0; i < FFT_SIZE; i++) {
      dest[start + i] += windowed[i] * window[i];
      windowSum[start + i] += window[i] * window[i];
    }
    lastCovered = start + FFT_SIZE;

    frameCount++;
    if (frameCount % FRAMES_PER_YIELD === 0) await yieldToEventLoop();
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
// note's pitch than to have no expected pitch at all.
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

export async function separateHarmonicResidual(recordedBuffer, melodyNotes, keyInfo) {
  const sampleRate = recordedBuffer.sampleRate;
  const channels = recordedBuffer.numberOfChannels;
  const length = recordedBuffer.length;
  const f0AtTime = f0LookupFromNotes(melodyNotes, keyInfo);

  const harmonicBuffer = new AudioBuffer({ length, numberOfChannels: channels, sampleRate });
  const residualBuffer = new AudioBuffer({ length, numberOfChannels: channels, sampleRate });
  for (let c = 0; c < channels; c++) {
    const { harmonicOut, residualOut } = await splitChannel(recordedBuffer.getChannelData(c), sampleRate, f0AtTime);
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
// any manual bookkeeping in App.jsx. Caches the in-flight promise (not just
// the resolved result), so concurrent calls for different harmony types
// share one computation instead of racing to start it multiple times.
const splitCache = new WeakMap();

export function getOrComputeHarmonicSplit(recordedBuffer, melodyNotes, keyInfo) {
  const cached = splitCache.get(recordedBuffer);
  if (cached) return cached;
  const promise = separateHarmonicResidual(recordedBuffer, melodyNotes, keyInfo);
  splitCache.set(recordedBuffer, promise);
  return promise;
}
