import SignalsmithStretch from 'signalsmith-stretch';
import { scaleStepToMidi, midiToFreq } from './musicTheory.js';

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
// covering the whole recording. Inside each melody note the ratio is held
// constant (harmonyFreq/melodyFreq for that note); short gaps between notes
// are eased directly from one note's ratio to the next; real pauses ease
// down to ratio=1 (no shift) and back up, so there is never an instant jump
// anywhere in the curve.
//
// An earlier version of this function instead followed the singer's raw
// measured pitch within each note (to carry vibrato through more literally),
// using the note's quantized ratio only as a fallback/clamp. Measured on a
// real recording, that made the output *more* jittery than this flat-ratio
// version, not less: our ~35ms/2048-sample autocorrelation pitch tracker has
// enough frame-to-frame estimation noise on its own that feeding it straight
// into the shift ratio injected audible roughness the flat-per-note ratio
// doesn't have — pitch-shifting the real audio by a constant factor already
// carries the singer's actual vibrato through mechanically, without needing
// per-frame tracking accuracy we don't have at this window size.
export function buildRatioCurve(melodyNotes, harmonyNotes, keyInfo, totalDuration, energyEnvelope = []) {
  const segments = melodyNotes.map((mn, i) => {
    const hn = harmonyNotes[i];
    const melodyFreq = midiToFreq(scaleStepToMidi(mn.step, keyInfo.tonic, keyInfo.mode));
    const harmonyFreq = midiToFreq(scaleStepToMidi(hn.hStep, keyInfo.tonic, keyInfo.mode));
    return { start: mn.start, end: mn.end, ratio: harmonyFreq / melodyFreq };
  });

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
    // original, un-harmonized voice.
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

// Renders the harmony as a full-length AudioBuffer by feeding the entire
// original recording plus the continuous ratio(t) curve into Signalsmith
// Stretch in one offline pass. There is only ever one, continuous signal
// path — no per-note clipping, crossfading, or loudness patching, because
// there are no block boundaries for artifacts to hide at.
export async function renderHarmonyOffline(recordedBuffer, melodyNotes, harmonyNotes, keyInfo) {
  const sr = recordedBuffer.sampleRate;
  const channels = recordedBuffer.numberOfChannels;
  const totalLen = recordedBuffer.length;
  const totalDuration = totalLen / sr;

  const energyEnvelope = computeEnergyEnvelope(recordedBuffer.getChannelData(0), sr);
  const keyframes = buildRatioCurve(melodyNotes, harmonyNotes, keyInfo, totalDuration, energyEnvelope);

  // Rough singer fundamental, used by the library to analyse/compensate
  // formants so larger intervals don't get a "chipmunk" timbre.
  const melodyFreqs = melodyNotes.map((n) => midiToFreq(scaleStepToMidi(n.step, keyInfo.tonic, keyInfo.mode)));
  const formantBaseHz = melodyFreqs.length ? median(melodyFreqs) : 200;

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
    formantSemitones: 0,
    formantCompensation: true,
    formantBaseHz,
  });

  // schedule() prunes its own history up to the timestamp of the call
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
