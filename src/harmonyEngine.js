import SignalsmithStretch from 'signalsmith-stretch';
import { scaleStepToMidi, midiToFreq } from './musicTheory.js';

// Short gaps (a consonant, a quick breath) are bridged smoothly into the
// neighboring note's ratio instead of ever dropping to "no shift" — this is
// what used to produce audible seams. Only gaps longer than this are treated
// as a real pause between phrases.
const GAP_BRIDGE_SEC = 0.15;
// How long the curve takes to ease to/from ratio=1.0 around a real pause,
// so even that transition is never a hard step.
const SILENCE_FADE_SEC = 0.04;
// Resolution of the ramp keyframes fed into the pitch-shift schedule.
const KEYFRAME_HOP_SEC = 0.02;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
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
export function buildRatioCurve(melodyNotes, harmonyNotes, keyInfo, totalDuration) {
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
    if (gap > GAP_BRIDGE_SEC) {
      const fadeOutEnd = Math.min(cursorT + SILENCE_FADE_SEC, seg.start);
      pushRamp(cursorT, fadeOutEnd, cursorRatio, 1);
      const fadeInStart = Math.max(fadeOutEnd, seg.start - SILENCE_FADE_SEC);
      keyframes.push({ t: fadeInStart, ratio: 1 });
      pushRamp(fadeInStart, seg.start, 1, seg.ratio);
    } else {
      pushRamp(cursorT, seg.start, cursorRatio, seg.ratio);
    }
    keyframes.push({ t: seg.end, ratio: seg.ratio });
    cursorT = seg.end;
    cursorRatio = seg.ratio;
  });

  pushRamp(cursorT, Math.min(totalDuration, cursorT + SILENCE_FADE_SEC), cursorRatio, 1);
  keyframes.push({ t: totalDuration, ratio: 1 });

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

  const keyframes = buildRatioCurve(melodyNotes, harmonyNotes, keyInfo, totalDuration);

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
