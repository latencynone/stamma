import SignalsmithStretch from 'signalsmith-stretch';
import { scaleStepToMidi, midiToFreq } from './musicTheory.js';
import { getOrComputeHarmonicSplit } from './harmonicSeparation.js';

// Short gaps (a consonant, a quick breath) are bridged smoothly into the
// neighboring note's ratio instead of ever dropping to "no shift" — this is
// what used to produce audible seams. Gaps longer than this are only treated
// as a real pause if the original recording is actually quiet there (see
// hasAudibleEnergy) — a long stretch the note detector couldn't confidently
// pin down is not the same thing as silence, and playing it back unshifted
// is exactly as audible a defect as the seams this was built to avoid.
const GAP_BRIDGE_SEC = 0.15;
// How long the curve takes to ease to/from ratio=1.0 around a real pause,
// so even that transition is never a hard step.
const SILENCE_FADE_SEC = 0.04;
// Resolution of the ramp keyframes fed into the pitch-shift schedule.
const KEYFRAME_HOP_SEC = 0.02;
// RMS level above which a window of the original recording counts as
// "has content" rather than silence, and the fraction of windows within a
// gap that must clear it for the whole gap to count as non-silent. Set just
// above the ~0.002 noise floor measured on a real quiet room/mic — high
// enough to not mistake room tone for singing, low enough to still catch
// genuinely soft/breathy vocalizing (which sits around 0.003-0.008 and was
// still slipping past a 0.01 threshold as an unshifted "pause").
const ENERGY_RMS_THRESHOLD = 0.0035;
const ENERGY_VOICED_FRACTION = 0.15;
// How far back from a note's detected start to look for where its audible
// onset actually began, when ramping in from a real pause.
const ONSET_LOOKBACK_SEC = 0.2;
// Default strength of the autotune correction: how far to nudge a note from
// its actual sung pitch toward its quantized target. 1.0 would be a full,
// robotic hard-snap; this is deliberately partial so it still sounds sung.
const DEFAULT_AUTOTUNE_AMOUNT = 0.4;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// A coarse RMS-over-time envelope of the original recording, used to tell a
// real pause apart from a stretch the note detector just couldn't confidently
// pin a pitch to (quiet consonants, breathy or poorly-tracked singing, a
// vocal run that briefly confused the pitch tracker). Energy, not elapsed
// time, is what should decide whether a gap plays back unshifted.
export function computeEnergyEnvelope(channelData, sampleRate, hopSec = 0.02) {
  const hop = Math.max(1, Math.round(hopSec * sampleRate));
  const envelope = [];
  for (let i = 0; i < channelData.length; i += hop) {
    const end = Math.min(channelData.length, i + hop);
    let sum = 0;
    for (let k = i; k < end; k++) sum += channelData[k] * channelData[k];
    envelope.push({ t: i / sampleRate, rms: Math.sqrt(sum / (end - i)) });
  }
  return envelope;
}

// Whether a [fromT, toT) span has real, sustained signal rather than
// silence. Excludes a small margin at each edge so the attack of the note
// that follows (or the tail of the one before) doesn't get misread as
// content inside the gap itself.
function hasAudibleEnergy(envelope, fromT, toT, edgePad = 0.03) {
  const innerFrom = fromT + edgePad;
  const innerTo = toT - edgePad;
  if (innerTo <= innerFrom) return false;
  const inRange = envelope.filter((e) => e.t >= innerFrom && e.t < innerTo);
  if (inRange.length === 0) return false;
  const voiced = inRange.filter((e) => e.rms > ENERGY_RMS_THRESHOLD).length;
  return voiced / inRange.length >= ENERGY_VOICED_FRACTION;
}

// Finds where the audible run leading into `beforeT` actually began, within
// a bounded lookback. Pitch detection only recognizes a note once it's a few
// stable frames in, so a note's nominal start typically lags the real
// acoustic onset by tens of milliseconds — ramping the shift in on a fixed
// window anchored to the detected start can leave a brief stretch of already-
// audible signal playing unshifted right before the ramp begins. Returns
// `beforeT` itself if there's no audible run immediately preceding it.
function findOnsetTime(envelope, beforeT, maxLookback = ONSET_LOOKBACK_SEC) {
  const inRange = envelope.filter((e) => e.t >= beforeT - maxLookback && e.t < beforeT);
  let onset = beforeT;
  for (let i = inRange.length - 1; i >= 0; i--) {
    if (inRange[i].rms > ENERGY_RMS_THRESHOLD) {
      onset = inRange[i].t;
    } else {
      break;
    }
  }
  return onset;
}

// Builds a continuous ratio(t) curve, expressed as keyframes {t, ratio},
// covering the whole recording, from a list of already-decided per-note
// targets ({start, end, ratio}). Inside each note the ratio is held constant;
// short gaps between notes are eased directly from one note's ratio to the
// next; real pauses ease down to ratio=1 (no shift) and back up, so there is
// never an instant jump anywhere in the curve. This is deliberately agnostic
// to *why* a note has the ratio it does — the harmony engine derives it from
// a target interval, the autotune engine derives it from a correction amount,
// and both get the same gap/pause/onset handling for free.
//
// An earlier version of the harmony curve instead followed the singer's raw
// measured pitch within each note (to carry vibrato through more literally),
// using the note's quantized ratio only as a fallback/clamp. Measured on a
// real recording, that made the output *more* jittery than a flat-per-note
// ratio, not less: our ~35ms/2048-sample autocorrelation pitch tracker has
// enough frame-to-frame estimation noise on its own that feeding it straight
// into the shift ratio injected audible roughness the flat ratio doesn't have
// — pitch-shifting the real audio by a constant factor already carries the
// singer's actual vibrato through mechanically, without needing per-frame
// tracking accuracy we don't have at this window size.
export function buildRatioCurveFromSegments(segments, totalDuration, energyEnvelope = []) {
  const keyframes = [{ t: 0, ratio: 1 }];
  const pushRamp = (fromT, toT, fromRatio, toRatio) => {
    const dur = toT - fromT;
    if (dur <= 1e-6) {
      keyframes.push({ t: toT, ratio: toRatio });
      return;
    }
    const steps = Math.max(1, Math.round(dur / KEYFRAME_HOP_SEC));
    for (let s = 1; s <= steps; s++) {
      const f = s / steps;
      const eased = 0.5 - 0.5 * Math.cos(Math.PI * f); // cosine ease: zero slope at both ends
      keyframes.push({ t: fromT + dur * f, ratio: fromRatio + (toRatio - fromRatio) * eased });
    }
  };

  let cursorT = 0;
  let cursorRatio = 1;

  segments.forEach((seg) => {
    const gap = seg.start - cursorT;
    const isRealPause = gap > GAP_BRIDGE_SEC && !hasAudibleEnergy(energyEnvelope, cursorT, seg.start);
    if (isRealPause) {
      const fadeOutEnd = Math.min(cursorT + SILENCE_FADE_SEC, seg.start);
      pushRamp(cursorT, fadeOutEnd, cursorRatio, 1);
      // Ramp in early enough to cover the note's real acoustic onset, not
      // just its nominal (pitch-tracker-recognized, therefore late) start.
      const onset = findOnsetTime(energyEnvelope, seg.start);
      const fadeInStart = Math.max(fadeOutEnd, Math.min(seg.start - SILENCE_FADE_SEC, onset));
      keyframes.push({ t: fadeInStart, ratio: 1 });
      pushRamp(fadeInStart, seg.start, 1, seg.ratio);
    } else {
      pushRamp(cursorT, seg.start, cursorRatio, seg.ratio);
    }
    keyframes.push({ t: seg.end, ratio: seg.ratio });
    cursorT = seg.end;
    cursorRatio = seg.ratio;
  });

  const tailIsRealPause = !hasAudibleEnergy(energyEnvelope, cursorT, totalDuration);
  if (tailIsRealPause) {
    pushRamp(cursorT, Math.min(totalDuration, cursorT + SILENCE_FADE_SEC), cursorRatio, 1);
    keyframes.push({ t: totalDuration, ratio: 1 });
  } else {
    // The singer is still audibly going after the last note we could pin
    // down — hold the last known-good ratio through to the end rather than
    // fading to "no shift", which would otherwise play back a chunk of the
    // original, uncorrected voice.
    keyframes.push({ t: totalDuration, ratio: cursorRatio });
  }

  // Collapse any non-increasing timestamps (can happen at very short notes).
  const cleaned = [];
  keyframes.forEach((k) => {
    if (cleaned.length && k.t <= cleaned[cleaned.length - 1].t) {
      cleaned[cleaned.length - 1] = k;
    } else {
      cleaned.push(k);
    }
  });
  return cleaned;
}

// A harmony voice built by pitch-shifting the *same* take the melody is
// (or, for another harmony type, would be) shifted from is otherwise a
// perfect clone in every way pitch-shifting can't touch: identical timing,
// identical micro-pitch behavior, laser-locked to the target ratio for the
// full length of every note, and — since it's literally the same audio —
// carrying the melody's own vibrato through at the *exact same phase*,
// which two independent singers never do. A live harmony pedal's
// "humanize"/voice-character controls address the pitch side of this with
// a small fixed "this voice tends slightly sharp/flat" detune plus a slow
// (sub-1Hz, well below vibrato rate) drift; on top of that, `vibratoCents`/
// `vibratoHz` add a second, faster (~5-6Hz) oscillation at this voice's own
// phase, so it beats against the melody's carried-through vibrato instead
// of moving in lockstep with it. `formantSemitones` (passed separately to
// SignalsmithStretch, not part of the ratio curve — see renderHarmonyOffline)
// gives the voice its own rough vocal-tract size, since a real other singer
// doesn't share yours. One fixed profile per interval keeps a given type's
// character consistent across renders instead of it just adding random
// per-render noise.
const HARMONY_HUMANIZE_PROFILES = {
  ters: {
    detuneCents: -3, wobbleCents: 4, wobbleHz: 0.22, phase: 0.4,
    vibratoCents: 9, vibratoHz: 5.4, vibratoPhase: 1.2,
    formantSemitones: -1.3,
  },
  kvint: {
    detuneCents: 4, wobbleCents: 3, wobbleHz: 0.31, phase: 2.1,
    vibratoCents: 7, vibratoHz: 5.9, vibratoPhase: 3.6,
    formantSemitones: 1.8,
  },
  sext: {
    detuneCents: -2, wobbleCents: 5, wobbleHz: 0.18, phase: 4.7,
    vibratoCents: 11, vibratoHz: 4.8, vibratoPhase: 0.7,
    formantSemitones: -0.9,
  },
};

// Multiplies every keyframe's ratio by a small detune+wobble+vibrato factor
// (in cents, so it composes with the existing ratio regardless of interval
// size), and — since the source curve holds a single flat ratio for a
// whole sustained note or pause, with no intermediate keyframes — inserts
// extra samples through any gap wider than `maxGapForInsertion` so the
// modulation is actually audible during a held note, not just at its two
// endpoints. The insertion hop is fine enough (20ms, matching the ramp
// keyframe resolution elsewhere in this file) to resolve the ~5-6Hz
// vibrato term smoothly rather than as an aliased/stepped wobble. Ramp/
// onset sections already have keyframes closer together than that gap, so
// their carefully-tuned shape is untouched beyond the same small
// multiplicative factor every other keyframe gets.
function humanizeKeyframes(keyframes, {
  detuneCents = 0, wobbleCents = 0, wobbleHz = 0.25, phase = 0,
  vibratoCents = 0, vibratoHz = 5.5, vibratoPhase = 0,
} = {}) {
  if (!detuneCents && !wobbleCents && !vibratoCents) return keyframes;
  const maxGapForInsertion = 0.15;
  const insertHop = 0.02;
  // OfflineAudioContext.suspend() (see renderWithRatioCurve) rejects a
  // second suspend scheduled at the same sample frame — two keyframes only
  // a fraction of a millisecond apart can round to that same frame. Never
  // insert a point closer than this to a keyframe that's already there
  // (comfortably more than one sample even at a high sample rate, and far
  // below the ~180ms period of the fastest vibrato this modulates).
  const minGap = 0.005;
  const factorAt = (t) => {
    const cents = detuneCents
      + wobbleCents * Math.sin(2 * Math.PI * wobbleHz * t + phase)
      + vibratoCents * Math.sin(2 * Math.PI * vibratoHz * t + vibratoPhase);
    return Math.pow(2, cents / 1200);
  };
  const out = [];
  for (let i = 0; i < keyframes.length; i++) {
    const k = keyframes[i];
    out.push({ t: k.t, ratio: k.ratio * factorAt(k.t) });
    const next = keyframes[i + 1];
    if (next && next.t - k.t > maxGapForInsertion) {
      for (let t = k.t + insertHop; t < next.t - minGap; t += insertHop) {
        const f = (t - k.t) / (next.t - k.t);
        const base = k.ratio + (next.ratio - k.ratio) * f;
        out.push({ t, ratio: base * factorAt(t) });
      }
    }
  }
  // Defensive backstop: collapse any two keyframes left closer than minGap
  // (e.g. two already-close points in the source curve) into one, same
  // idea as the cleanup pass in buildRatioCurveFromSegments.
  const cleaned = [];
  out.forEach((k) => {
    if (cleaned.length && k.t - cleaned[cleaned.length - 1].t < minGap) {
      cleaned[cleaned.length - 1] = k;
    } else {
      cleaned.push(k);
    }
  });
  return cleaned;
}

// Per-note targets for the harmony engine: a fixed interval (harmonyNotes'
// hStep) above/below each melody note's own quantized pitch. `harmonyType`
// (e.g. "ters") selects that voice's humanization character — omit it (as
// the accuracy tests do) to get the exact, unhumanized ratio curve.
export function buildRatioCurve(melodyNotes, harmonyNotes, keyInfo, totalDuration, energyEnvelope = [], harmonyType = null) {
  const segments = melodyNotes.map((mn, i) => {
    const hn = harmonyNotes[i];
    const melodyFreq = midiToFreq(scaleStepToMidi(mn.step, keyInfo.tonic, keyInfo.mode));
    const harmonyFreq = midiToFreq(scaleStepToMidi(hn.hStep, keyInfo.tonic, keyInfo.mode));
    return { start: mn.start, end: mn.end, ratio: harmonyFreq / melodyFreq };
  });
  const curve = buildRatioCurveFromSegments(segments, totalDuration, energyEnvelope);
  const profile = harmonyType && HARMONY_HUMANIZE_PROFILES[harmonyType];
  return profile ? humanizeKeyframes(curve, profile) : curve;
}

// Per-note targets for the autotune engine: nudge each note's *actual* sung
// pitch (measuredFreq) a fraction of the way toward its quantized target,
// rather than following the target itself — a full snap-to-grid would sound
// robotic, and a note whose measured pitch is missing (too little usable
// frame data) is left alone rather than guessed at.
export function buildAutotuneRatioCurve(melodyNotes, keyInfo, totalDuration, energyEnvelope = [], amount = DEFAULT_AUTOTUNE_AMOUNT) {
  const segments = melodyNotes.map((mn) => {
    const targetFreq = midiToFreq(scaleStepToMidi(mn.step, keyInfo.tonic, keyInfo.mode));
    const sourceFreq = mn.measuredFreq || targetFreq;
    const fullRatio = targetFreq / sourceFreq;
    const ratio = 1 + (fullRatio - 1) * amount;
    return { start: mn.start, end: mn.end, ratio };
  });
  return buildRatioCurveFromSegments(segments, totalDuration, energyEnvelope);
}

// Feeds a recording plus a precomputed ratio(t) curve through Signalsmith
// Stretch in one continuous offline pass — no per-note clipping,
// crossfading, or loudness patching, because there are no block boundaries
// for artifacts to hide at. Used by both the harmony and autotune engines.
async function renderWithRatioCurve(recordedBuffer, keyframes, formantBaseHz, formantSemitones = 0) {
  const sr = recordedBuffer.sampleRate;
  const channels = recordedBuffer.numberOfChannels;
  const totalLen = recordedBuffer.length;
  const totalDuration = totalLen / sr;

  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const offlineCtx = new OfflineCtx(channels, totalLen, sr);

  const stretch = await SignalsmithStretch(offlineCtx, {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [channels],
  });
  stretch.connect(offlineCtx.destination);

  // The default analysis block (~150ms+) is tuned for clean sustained-tone
  // quality, but real notes are often shorter than that — a fast ratio ramp
  // landing right on a strong attack transient (a syllable's onset) can make
  // the library's internal state lag behind, producing an audible volume
  // dip/warble right at the attack even though the ratio curve itself is
  // smooth. A shorter block responds fast enough to avoid that; measured
  // against a real recording, it took an attack that dipped to 40-70% of the
  // source's loudness for several consecutive frames down to consistently
  // 85-100%.
  await stretch.configure({ blockMs: 60 });

  const channelData = [];
  for (let c = 0; c < channels; c++) channelData.push(recordedBuffer.getChannelData(c).slice());
  await stretch.addBuffers(channelData);

  const scheduleAt = (k) => stretch.schedule({
    output: k.t,
    outputTime: k.t,
    active: true,
    rate: 1,
    semitones: 12 * Math.log2(k.ratio),
    formantSemitones,
    formantCompensation: true,
    formantBaseHz,
  });

  // schedule() prunes its own history up to the timestamp of the call —
  // it's built for issuing changes as playback approaches them live, not
  // for front-loading a whole automation curve at once (an early call gets
  // wiped out the instant a later one is scheduled). So each keyframe after
  // the first is applied via suspend()/resume(), which pauses the offline
  // render exactly when it reaches that keyframe's time and lets us call
  // schedule() at the moment it's actually meant to take effect.
  await scheduleAt(keyframes[0]);
  keyframes.slice(1).forEach((k) => {
    if (k.t <= 0 || k.t >= totalDuration) return;
    offlineCtx.suspend(k.t).then(() => {
      scheduleAt(k);
      offlineCtx.resume();
    });
  });

  return offlineCtx.startRendering();
}

function formantBaseHzFor(melodyNotes, keyInfo) {
  const melodyFreqs = melodyNotes.map((n) => midiToFreq(scaleStepToMidi(n.step, keyInfo.tonic, keyInfo.mode)));
  return melodyFreqs.length ? median(melodyFreqs) : 200;
}

function sumBuffers(a, b) {
  const out = new AudioBuffer({ length: a.length, numberOfChannels: a.numberOfChannels, sampleRate: a.sampleRate });
  for (let c = 0; c < a.numberOfChannels; c++) {
    const ca = a.getChannelData(c);
    const cb = b.getChannelData(c);
    const sum = new Float32Array(ca.length);
    for (let i = 0; i < ca.length; i++) sum[i] = ca[i] + cb[i];
    out.copyToChannel(sum, c);
  }
  return out;
}

// Renders the harmony as a full-length AudioBuffer: the original recording,
// continuously pitch-shifted to a fixed interval above/below the melody.
//
// Only the harmonic (tonal) layer of the recording is actually fed through
// the pitch/formant shift — see harmonicSeparation.js for why: pitch-
// shifting the residual (breath, consonants — content with no real pitch of
// its own) is what makes a shifted voice sound metallic/robotic. The
// residual is added back afterward unshifted, so it keeps its natural
// character. This runs unconditionally (unlike the vibrato/formant/detune
// "humanize" character, which is opt-in) since it's a pitch-shift quality
// fix, not a voice-character choice.
export async function renderHarmonyOffline(recordedBuffer, melodyNotes, harmonyNotes, keyInfo, harmonyType = null) {
  const totalDuration = recordedBuffer.length / recordedBuffer.sampleRate;
  const energyEnvelope = computeEnergyEnvelope(recordedBuffer.getChannelData(0), recordedBuffer.sampleRate);
  const keyframes = buildRatioCurve(melodyNotes, harmonyNotes, keyInfo, totalDuration, energyEnvelope, harmonyType);
  const profile = harmonyType && HARMONY_HUMANIZE_PROFILES[harmonyType];
  const formantSemitones = profile ? profile.formantSemitones : 0;
  const { harmonicBuffer, residualBuffer } = await getOrComputeHarmonicSplit(recordedBuffer, melodyNotes, keyInfo);
  const shiftedHarmonic = await renderWithRatioCurve(harmonicBuffer, keyframes, formantBaseHzFor(melodyNotes, keyInfo), formantSemitones);
  return sumBuffers(shiftedHarmonic, residualBuffer);
}

// Renders a lightly pitch-corrected copy of the recording itself: each note
// nudged part-way from its actual sung pitch toward its quantized target.
export async function renderAutotunedMelody(recordedBuffer, melodyNotes, keyInfo, amount = DEFAULT_AUTOTUNE_AMOUNT) {
  const totalDuration = recordedBuffer.length / recordedBuffer.sampleRate;
  const energyEnvelope = computeEnergyEnvelope(recordedBuffer.getChannelData(0), recordedBuffer.sampleRate);
  const keyframes = buildAutotuneRatioCurve(melodyNotes, keyInfo, totalDuration, energyEnvelope, amount);
  return renderWithRatioCurve(recordedBuffer, keyframes, formantBaseHzFor(melodyNotes, keyInfo));
}
