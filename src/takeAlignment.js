import SignalsmithStretch from 'signalsmith-stretch';
import { extractFramesFromBuffer } from './pitchAnalysis.js';

// MTrackAlign-style timing alignment for a user's own-recorded ters/kvint/
// sext take: the melody's already-detected note onsets are the fixed
// reference, and the own take is elastically time-warped so its syllables
// land on them. Unlike the harmony engine's pitch shift (which always runs),
// this only ever touches segments it's actually confident matched a melody
// onset to a real onset in the take — an unmatched stretch is left at its
// original (merely constant-shifted, never stretched) timing rather than
// forced into alignment on a guess, since a wrong warp is more jarring than
// a take that's simply a little early/late in one spot.

// How much of a voiced run must be silent before it's treated as a genuine
// gap between syllables rather than a consonant/breath dropout mid-word —
// mirrors pitchAnalysis.js's fillShortGaps default for the same reason (a
// plosive or sibilant briefly drops below the pitch tracker's voicing
// threshold even mid-syllable).
const ONSET_MAX_GAP_SEC = 0.12;
// Drops onset blips shorter than this (a mic pop, a breath the tracker
// briefly read as voiced) that would otherwise show up as a spurious extra
// "syllable" for the matcher to (mis)consider.
const ONSET_MIN_DUR_SEC = 0.08;
// Coarse global-offset search: how far in either direction the take might
// plausibly be shifted relative to the melody (a late/early record-button
// press, not a tempo drift), and the resolution to search it at.
const COARSE_OFFSET_RANGE_SEC = 2.0;
const COARSE_OFFSET_STEP_SEC = 0.01;
// A candidate own-onset/melody-onset pair only contributes to the coarse
// offset's score if they'd land within this of each other — keeps a few
// wildly-mismatched onsets (an extra breath, a missed word) from dragging
// the estimate around.
const COARSE_OFFSET_TOLERANCE_SEC = 0.3;
// Once the coarse offset narrows down roughly where each own-onset should
// land, this is how far a melody onset may sit from that prediction and
// still be accepted as a confident match. Tighter than the coarse
// tolerance above — this is what actually gates "warp this stretch" vs.
// "leave it alone", so it errs toward not touching audio it isn't sure
// about.
const MATCH_WINDOW_SEC = 0.35;

// Frames a decoded channel's raw pitch-tracker output (voiced/unvoiced per
// ~35ms hop, from the same autocorrelation pass the melody analysis uses)
// into onset segments — continuous voiced runs, short gaps bridged, blips
// dropped. Deliberately pitch-agnostic (no scale quantization, no per-note
// pitch grouping): a harmony take sings different notes than the melody by
// design, so only *when* something was sung is useful here, not *what*.
export function detectOnsetSegments(channelData, sampleRate, { minDurSec = ONSET_MIN_DUR_SEC, maxGapSec = ONSET_MAX_GAP_SEC } = {}) {
  const frames = extractFramesFromBuffer(channelData, sampleRate);
  const runs = [];
  let cur = null;
  frames.forEach((f) => {
    if (f.freq > 0) {
      if (cur) cur.end = f.t;
      else cur = { start: f.t, end: f.t };
    } else if (cur) {
      runs.push(cur);
      cur = null;
    }
  });
  if (cur) runs.push(cur);

  const bridged = [];
  runs.forEach((run) => {
    const prev = bridged[bridged.length - 1];
    if (prev && run.start - prev.end <= maxGapSec) {
      prev.end = run.end;
    } else {
      bridged.push({ ...run });
    }
  });

  return bridged.filter((seg) => seg.end - seg.start >= minDurSec);
}

// Scores candidate global offsets by how many own-onsets land near *some*
// melody onset once shifted by that offset, summed rather than requiring
// any single pair to match best — robust to a handful of extra/missing
// onsets on either side, since those simply contribute ~0 rather than
// derailing the whole estimate. `offset` is defined as (predicted melody
// time) − (own take time): adding it to an own-onset predicts where it
// should land on the melody's timeline.
export function estimateCoarseOffset(ownStarts, melodyStarts) {
  if (!ownStarts.length || !melodyStarts.length) return 0;
  let best = { offset: 0, score: -Infinity };
  for (let offset = -COARSE_OFFSET_RANGE_SEC; offset <= COARSE_OFFSET_RANGE_SEC; offset += COARSE_OFFSET_STEP_SEC) {
    let score = 0;
    for (const os of ownStarts) {
      const predicted = os + offset;
      let nearest = Infinity;
      for (const ms of melodyStarts) {
        const d = Math.abs(predicted - ms);
        if (d < nearest) nearest = d;
      }
      if (nearest < COARSE_OFFSET_TOLERANCE_SEC) score += 1 - nearest / COARSE_OFFSET_TOLERANCE_SEC;
    }
    if (score > best.score) best = { offset, score };
  }
  return best.offset;
}

// Finds the highest-confidence *monotonic* (order-preserving) matching
// between own-take onsets and melody onsets — own-onset i can only match a
// melody onset at or after whichever one own-onset i−1 matched, same as a
// real singer can't sing their syllables out of order relative to the
// melody. Standard weighted-alignment DP (bounded reward for a close-enough
// pair, free to skip either side), sized for a ≤10s clip's handful of
// onsets so the O(M×N) table is trivial. Returns anchors sorted by time;
// empty if nothing in either list ever came within MATCH_WINDOW_SEC of the
// coarse-offset prediction.
export function matchOnsetsToMelody(ownStarts, melodyStarts) {
  if (!ownStarts.length || !melodyStarts.length) return [];
  const coarseOffset = estimateCoarseOffset(ownStarts, melodyStarts);

  const M = ownStarts.length;
  const N = melodyStarts.length;
  const dp = Array.from({ length: M + 1 }, () => new Float64Array(N + 1));
  const back = Array.from({ length: M + 1 }, () => new Int8Array(N + 1)); // 0 = match, 1 = skip own, 2 = skip melody

  for (let i = 1; i <= M; i++) {
    for (let j = 1; j <= N; j++) {
      let best = dp[i - 1][j];
      let choice = 1;
      if (dp[i][j - 1] > best) {
        best = dp[i][j - 1];
        choice = 2;
      }
      const predicted = ownStarts[i - 1] + coarseOffset;
      const diff = Math.abs(predicted - melodyStarts[j - 1]);
      if (diff <= MATCH_WINDOW_SEC) {
        const reward = dp[i - 1][j - 1] + (1 - diff / MATCH_WINDOW_SEC);
        if (reward > best) {
          best = reward;
          choice = 0;
        }
      }
      dp[i][j] = best;
      back[i][j] = choice;
    }
  }

  const anchors = [];
  let i = M;
  let j = N;
  while (i > 0 && j > 0) {
    const choice = back[i][j];
    if (choice === 0) {
      anchors.push({ ownTime: ownStarts[i - 1], melodyTime: melodyStarts[j - 1] });
      i--;
      j--;
    } else if (choice === 1) {
      i--;
    } else {
      j--;
    }
  }
  anchors.reverse();
  return anchors;
}

// Bounds on the per-segment stretch rate a pair of anchors is allowed to
// imply — real singing drifts by tens of percent, not multiples. A rate
// outside this range means the anchor pair is more likely a bad match (an
// onset the pitch-agnostic detector picked up from the melody bleeding
// into the mic — see recordOwnTake's echoCancellation note — matched to a
// nearby-but-wrong melody note, say) than genuine tempo drift, and
// SignalsmithStretch audibly struggles at extreme ratios anyway (reported
// as a "dragging"/stammering artifact). Clamping still lands input exactly
// on `next.input` at `next.output` (every keyframe resets both explicitly,
// see renderTimeWarp) — a clamped segment just gets there with a small
// jump instead of a smooth but badly-stretched ramp, which is the less
// audible failure mode of the two.
const MIN_STRETCH_RATE = 0.4;
const MAX_STRETCH_RATE = 2.5;

// Turns confident anchors into SignalsmithStretch schedule keyframes on the
// shared mix timeline (the `output` axis = the melody/recording's own
// timeline, same as every other channel). Before the first anchor and after
// the last, the take plays at its original rate (rate 1) — only constant-
// shifted, never stretched, since there's no second confident point there
// to define a stretch ratio against. Between two anchors, the local rate is
// exactly whatever's needed to make that stretch of the take span the gap
// between them, clamped to a plausible range (see above). Returns null
// (meaning "leave the take untouched") if there are no anchors at all.
export function buildAlignmentKeyframes(anchors) {
  if (!anchors.length) return null;
  const first = anchors[0];
  const keyframes = [{ output: 0, input: Math.max(0, first.ownTime - first.melodyTime), rate: 1 }];
  anchors.forEach((a, idx) => {
    const next = anchors[idx + 1];
    const rate = next
      ? Math.min(MAX_STRETCH_RATE, Math.max(MIN_STRETCH_RATE, (next.ownTime - a.ownTime) / Math.max(1e-6, next.melodyTime - a.melodyTime)))
      : 1;
    keyframes.push({ output: a.melodyTime, input: a.ownTime, rate });
  });
  return keyframes;
}

// Renders the own-take buffer through the keyframe schedule above into a
// buffer exactly `outputDuration` long on the shared mix timeline — same
// suspend()/resume() keyframe-scheduling pattern harmonyEngine.js's
// renderWithRatioCurve uses for its pitch-shift curve, just driving
// SignalsmithStretch's `input`/`rate` fields (time-warp) instead of
// `semitones` (pitch). No pitch or formant shift is ever applied here —
// the take already has the pitch the singer actually sang.
async function renderTimeWarp(ownBuffer, keyframes, outputDuration) {
  const sr = ownBuffer.sampleRate;
  const channels = ownBuffer.numberOfChannels;
  const totalLen = Math.round(outputDuration * sr);

  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const offlineCtx = new OfflineCtx(channels, totalLen, sr);

  const stretch = await SignalsmithStretch(offlineCtx, {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [channels],
  });
  stretch.connect(offlineCtx.destination);
  await stretch.configure({ blockMs: 60 });

  // Past the last keyframe there's no further reset (see
  // buildAlignmentKeyframes: the tail plays at rate 1, unstretched, all
  // the way to `outputDuration`) — so the input position it ends up
  // requesting keeps climbing for as long as output still has left to
  // render. `ownBuffer` is only ever as long as what was actually sung
  // (recorded on a wall-clock timer, not sample-locked to
  // `outputDuration`), so a last anchor that isn't right at the very end
  // of the take needs more input than physically exists — Signalsmith
  // Stretch reading past what addBuffers() gave it is exactly what
  // produced the reported stammering/volume-drop in the last second or
  // two of an aligned take. Padding with real trailing silence (rather
  // than leaving that "one step off the end of the array" case
  // undefined) makes that tail cleanly silent, which is the correct
  // result anyway once the take has run out of actual singing.
  const last = keyframes[keyframes.length - 1];
  const tailReachSec = last.input + Math.max(0, outputDuration - last.output) * (last.rate || 1);
  const paddedLen = Math.max(ownBuffer.length, Math.ceil(tailReachSec * sr) + Math.ceil(sr * 0.25));

  const channelData = [];
  for (let c = 0; c < channels; c++) {
    const src = ownBuffer.getChannelData(c);
    if (paddedLen <= src.length) {
      channelData.push(src.slice());
    } else {
      const padded = new Float32Array(paddedLen);
      padded.set(src);
      channelData.push(padded);
    }
  }
  await stretch.addBuffers(channelData);

  const scheduleAt = (k) => stretch.schedule({
    output: k.output,
    input: k.input,
    active: true,
    rate: k.rate,
  });

  await scheduleAt(keyframes[0]);
  keyframes.slice(1).forEach((k) => {
    if (k.output <= 0 || k.output >= outputDuration) return;
    offlineCtx.suspend(k.output).then(() => {
      scheduleAt(k);
      offlineCtx.resume();
    });
  });

  return offlineCtx.startRendering();
}

// Top-level entry point: aligns a user's own harmony take to the melody's
// timing, returning a new AudioBuffer exactly `outputDuration` long (the
// shared mix/recording timeline). Falls back to returning `ownBuffer`
// itself, completely untouched, whenever there isn't at least one confident
// onset match to anchor a warp to — recorded before any singing happened,
// too sparse a melody, or a take that just doesn't line up with it at all.
export async function alignOwnTakeToMelody(ownBuffer, melodyNotes, outputDuration) {
  if (!melodyNotes || !melodyNotes.length) return ownBuffer;
  const ownStarts = detectOnsetSegments(ownBuffer.getChannelData(0), ownBuffer.sampleRate).map((s) => s.start);
  const melodyStarts = melodyNotes.map((n) => n.start);
  const anchors = matchOnsetsToMelody(ownStarts, melodyStarts);
  const keyframes = buildAlignmentKeyframes(anchors);
  if (!keyframes) return ownBuffer;
  return renderTimeWarp(ownBuffer, keyframes, outputDuration);
}
