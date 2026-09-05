import { useState, useRef, useEffect, useMemo } from 'react';
import {
  NOTE_NAMES,
  HARMONY_TYPES,
  SOUND_TYPES,
  MAJOR_INTERVALS,
  midiToFreq,
  scaleStepToMidi,
  midiToNoteName,
  freqToMidi,
  midiToScaleStep,
  requantizeNotesToKey,
  semitoneForStep,
} from './musicTheory.js';
import SignalsmithStretch from 'signalsmith-stretch';
import {
  autoCorrelate,
  detectKey,
  fillShortGaps,
  framesToNotes,
  filterTransientArtifacts,
  filterOutlierNotes,
  attachMeasuredFreq,
  extractFramesFromBuffer,
  detectTempoBpm,
} from './pitchAnalysis.js';
import { renderHarmonyOffline, renderAutotunedMelody, computeEnergyEnvelope, formantBaseHzFor, HARMONY_HUMANIZE_PROFILES } from './harmonyEngine.js';
import { alignOwnTakeToMelody } from './takeAlignment.js';
import { audioBufferToWavBlob, downloadBlob } from './wav.js';
import { fft, ifft, hannWindow } from './dsp.js';
import { saveProject, listProjects, loadProject, deleteProject, renameProject } from './indexedDb.js';

const HARMONY_KEYS = Object.keys(HARMONY_TYPES);

// One accent color per harmony type, reused consistently between the note
// graph and the mixer channel rows, so the same color always means the same
// interval — needed now that more than one can be on screen/audible at once.
const HARMONY_COLORS = {
  ters: { line: 'var(--teal)', glow: 'rgba(var(--teal-rgb),0.65)' },
  kvint: { line: 'var(--purple)', glow: 'rgba(var(--purple-rgb),0.65)' },
  sext: { line: 'var(--rose)', glow: 'rgba(var(--rose-rgb),0.65)' },
};
const MELODY_COLOR = { line: 'var(--orange)', glow: 'rgba(var(--orange-rgb),0.65)' };

// Canvas 2D can't resolve CSS custom properties itself (unlike DOM/inline
// styles) — a color string handed to ctx.fillStyle/strokeStyle has to
// already be a literal. Every canvas-drawn color in this file is still
// authored as a var(--token) (or an rgba(...) wrapping one), same as every
// other themed color, so draw code runs it through this first: replaces
// each var(--x) with its current live value (picking up whichever theme
// is active) before handing it to canvas. A plain literal with no var()
// in it (rare, but e.g. a fixed white) passes through untouched.
function resolveThemeColor(str) {
  if (typeof str !== 'string' || !str.includes('var(')) return str;
  const styles = getComputedStyle(document.documentElement);
  return str.replace(/var\((--[\w-]+)\)/g, (_, name) => styles.getPropertyValue(name).trim());
}

// Schedules a channel's fade-in/fade-out gain envelope starting from an
// arbitrary point mid-window rather than always from the window's start —
// needed because resuming from a paused/seeked position should reflect
// whatever the fade curve's value already was at that instant, not restart
// the ramp from 0. `elapsedIntoWindow` is how far `playFrom` already is
// past the window's start; the three branches are "still ramping in",
// "already ramping out", and "flat in between" (each also schedules the
// still-upcoming fade-out event, if any).
function scheduleFadeGain(gainParam, now, windowDur, elapsedIntoWindow, fadeInDur, fadeOutDur) {
  const fadeOutStartElapsed = windowDur - fadeOutDur;
  if (fadeInDur > 0 && elapsedIntoWindow < fadeInDur) {
    const startGain = elapsedIntoWindow / fadeInDur;
    gainParam.setValueAtTime(startGain, now);
    gainParam.linearRampToValueAtTime(1, now + (fadeInDur - elapsedIntoWindow));
    if (fadeOutDur > 0) {
      gainParam.setValueAtTime(1, now + Math.max(0, fadeOutStartElapsed - elapsedIntoWindow));
      gainParam.linearRampToValueAtTime(0, now + (windowDur - elapsedIntoWindow));
    }
  } else if (fadeOutDur > 0 && elapsedIntoWindow >= fadeOutStartElapsed) {
    const intoFadeOut = elapsedIntoWindow - fadeOutStartElapsed;
    const startGain = Math.max(0, 1 - intoFadeOut / fadeOutDur);
    gainParam.setValueAtTime(startGain, now);
    gainParam.linearRampToValueAtTime(0, now + Math.max(0.001, fadeOutDur - intoFadeOut));
  } else {
    gainParam.setValueAtTime(1, now);
    if (fadeOutDur > 0) {
      gainParam.setValueAtTime(1, now + Math.max(0, fadeOutStartElapsed - elapsedIntoWindow));
      gainParam.linearRampToValueAtTime(0, now + (windowDur - elapsedIntoWindow));
    }
  }
}

// A synthetic reverb impulse response: exponentially-decaying filtered
// noise, the standard cheap substitute for a recorded impulse when there's
// no audio asset to convolve against. `decaySeconds` controls the tail
// length (see REVERB_LEVELS); the squared falloff keeps the early part
// audible without a harsh, unnaturally long noisy tail.
function generateReverbImpulse(ctx, decaySeconds) {
  const rate = ctx.sampleRate;
  const length = Math.max(1, Math.floor(rate * decaySeconds));
  const impulse = ctx.createBuffer(2, length, rate);
  for (let ch = 0; ch < 2; ch++) {
    const data = impulse.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2.5);
    }
  }
  return impulse;
}

// Every mixer channel connects to this bus's `input` instead of straight to
// `ctx.destination`, so reverb (a mix-wide effect, not per-channel) only
// needs to exist in one place. The dry path is always wired through
// unconditionally — reverb off just means no wet path exists.
function createReverbBus(ctx, reverbOn, reverbLevel) {
  const input = ctx.createGain();
  input.connect(ctx.destination);
  if (!reverbOn || !reverbLevel) return { input };
  const convolver = ctx.createConvolver();
  convolver.buffer = generateReverbImpulse(ctx, reverbLevel.decay);
  const wetGain = ctx.createGain();
  wetGain.gain.value = reverbLevel.wet;
  input.connect(convolver);
  convolver.connect(wetGain);
  wetGain.connect(ctx.destination);
  return { input, convolver, wetGain };
}

/* ---------- Noise reduction (spectral subtraction) ----------
 * A from-scratch, moderate-quality implementation — not a match for a
 * dedicated plugin, but a real spectral noise reduction pass rather than a
 * simple gate: it learns a noise magnitude spectrum from a user-marked
 * (or auto-detected) quiet region, then subtracts that spectrum from every
 * analysis frame across the whole recording via STFT / overlap-add.
 */

// Average magnitude spectrum over [startSec, endSec) — the "what does the
// room/hiss sound like" fingerprint that gets subtracted from every frame
// of the full recording.
function computeNoiseProfile(channelData, sampleRate, startSec, endSec, fftSize, window) {
  const hop = fftSize / 2;
  const startSample = Math.max(0, Math.floor(startSec * sampleRate));
  const endSample = Math.min(channelData.length, Math.floor(endSec * sampleRate));
  const profile = new Float32Array(fftSize / 2 + 1);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  let frameCount = 0;
  for (let start = startSample; start + fftSize <= endSample; start += hop) {
    for (let i = 0; i < fftSize; i++) {
      re[i] = channelData[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= fftSize / 2; k++) profile[k] += Math.hypot(re[k], im[k]);
    frameCount++;
  }
  if (frameCount === 0) {
    // Region shorter than one FFT frame — zero-pad a single frame instead
    // of producing an empty (all-zero, i.e. no-op) profile.
    for (let i = 0; i < fftSize; i++) {
      const idx = startSample + i;
      re[i] = (idx < endSample && idx < channelData.length ? channelData[idx] : 0) * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= fftSize / 2; k++) profile[k] = Math.hypot(re[k], im[k]);
    frameCount = 1;
  }
  for (let k = 0; k < profile.length; k++) profile[k] /= frameCount;
  return profile;
}

// Runs the actual subtraction across the whole channel via STFT / overlap-
// add. `oversubtract` pushes the noise estimate down harder than measured
// (standard spectral-subtraction practice, since a single averaged profile
// under-represents the noise's real peaks); `floorRatio` keeps a small
// fraction of each bin rather than letting it hit zero, which is what
// avoids the "musical noise" (isolated warbling tones) naive spectral
// subtraction is known for.
function spectralSubtractChannel(channelData, noiseProfile, fftSize, window, oversubtract = 1.6, floorRatio = 0.06) {
  const hop = fftSize / 2;
  const out = new Float32Array(channelData.length);
  const windowSum = new Float32Array(channelData.length);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  let lastCovered = 0;

  for (let start = 0; start + fftSize <= channelData.length; start += hop) {
    for (let i = 0; i < fftSize; i++) {
      re[i] = channelData[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= fftSize / 2; k++) {
      const mag = Math.hypot(re[k], im[k]);
      const phase = Math.atan2(im[k], re[k]);
      const floor = mag * floorRatio;
      const newMag = Math.max(floor, mag - noiseProfile[k] * oversubtract);
      re[k] = newMag * Math.cos(phase);
      im[k] = newMag * Math.sin(phase);
      if (k > 0 && k < fftSize / 2) {
        // Real-signal spectrum is conjugate-symmetric — mirror the
        // negative-frequency bin instead of computing it separately.
        re[fftSize - k] = re[k];
        im[fftSize - k] = -im[k];
      }
    }
    ifft(re, im);
    for (let i = 0; i < fftSize; i++) {
      out[start + i] += re[i] * window[i];
      windowSum[start + i] += window[i] * window[i];
    }
    lastCovered = start + fftSize;
  }
  for (let i = 0; i < out.length; i++) {
    out[i] = windowSum[i] > 1e-6 ? out[i] / windowSum[i] : channelData[i];
  }
  // The tail shorter than one hop past the last full frame never got a
  // window contribution at all — carry the original signal through rather
  // than leaving it silent.
  for (let i = lastCovered; i < out.length; i++) out[i] = channelData[i];
  return out;
}

// Finds a plausible noise-only stretch to pre-select: the longest run of
// consecutive low-energy frames (quiet, but not hard digital silence — true
// silence carries no usable noise-floor information), capped at 1.5s so the
// profile stays local instead of averaging over unrelated parts of the
// take. Falls back to the first 0.4s (room tone before singing is common)
// if nothing better stands out.
function detectNoiseRegion(channelData, sampleRate, duration) {
  const envelope = computeEnergyEnvelope(channelData, sampleRate, 0.02);
  if (!envelope.length) return { start: 0, end: Math.min(0.4, duration) };
  const sorted = envelope.map((e) => e.rms).slice().sort((a, b) => a - b);
  const threshold = Math.max(0.0008, sorted[Math.floor(sorted.length * 0.2)] * 1.5);
  let best = null;
  let runStart = null;
  for (let i = 0; i <= envelope.length; i++) {
    const frame = envelope[i];
    const inRun = frame && frame.rms <= threshold && frame.rms > 0.0003;
    if (inRun) {
      if (runStart === null) runStart = frame.t;
    } else if (runStart !== null) {
      const runEnd = frame ? frame.t : duration;
      const len = runEnd - runStart;
      if (len >= 0.4 && (!best || len > best.len)) {
        best = { start: runStart, end: Math.min(runEnd, runStart + 1.5), len };
      }
      runStart = null;
    }
  }
  return best ? { start: best.start, end: best.end } : { start: 0, end: Math.min(0.4, duration) };
}

/* ---------- Tempo detection & metronome ---------- */

// One metronome tick — a short sine blip, accented (higher pitch, a touch
// louder) on the first beat of every 4 so the pulse reads as a bar, not
// just an undifferentiated click.
function playMetronomeClick(ctx, whenTime, accent, gainMultiplier = 1) {
  const osc = ctx.createOscillator();
  osc.type = 'sine';
  osc.frequency.value = accent ? 1500 : 1000;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.0001, whenTime);
  gain.gain.exponentialRampToValueAtTime((accent ? 0.4 : 0.28) * gainMultiplier, whenTime + 0.004);
  gain.gain.exponentialRampToValueAtTime(0.0001, whenTime + 0.06);
  osc.connect(gain).connect(ctx.destination);
  osc.start(whenTime);
  osc.stop(whenTime + 0.08);
}

// Standard "lookahead" scheduler (setInterval polling ahead of playback
// time with sample-accurate start() calls) rather than one setTimeout per
// click — setTimeout alone drifts audibly over even a few bars.
// `startTime`: when the first click should land. Defaults to a fixed
// lookahead from now (the count-in/listen-preview use case); passing the
// same timestamp another scheduled event starts at (e.g. mix playback)
// makes the two sample-accurately simultaneous instead of each just
// starting "soon".
function startMetronomeScheduler(ctx, bpm, startTime, gainMultiplier = 1) {
  const beatDur = 60 / Math.max(20, bpm);
  const lookahead = 0.1;
  const state = { nextClickTime: startTime ?? ctx.currentTime + 0.1, beatCount: 0 };
  state.intervalId = setInterval(() => {
    while (state.nextClickTime < ctx.currentTime + lookahead) {
      playMetronomeClick(ctx, state.nextClickTime, state.beatCount % 4 === 0, gainMultiplier);
      state.nextClickTime += beatDur;
      state.beatCount++;
    }
  }, 25);
  return state;
}

function stopMetronomeScheduler(state) {
  if (state) clearInterval(state.intervalId);
}

// A count-in before recording actually starts — same click sound and
// tempo as the metronome, so it also doubles as a "here's the tempo"
// lead-in. Schedules all COUNT_IN_BEATS clicks up front (sample-accurate,
// same as the scheduler above), and resolves once the last one has
// sounded — i.e. exactly when recording should begin. `onBeat(i)` is
// setTimeout-paced (not sample-accurate) since it only drives a UI digit.
const COUNT_IN_BEATS = 4;
// iOS Safari (worst as a home-screen-installed web app) has been reported
// to audibly duck the page's own audio output once a getUserMedia capture
// with echoCancellation disabled is active alongside it — exactly the
// constraint recordOwnTake requests (see its own comment on why). Can't be
// detected or fixed from script (the OS does it below the Web Audio API),
// so this just plays the monitor melody and count-in/click louder than
// normal specifically during that recording, as a blind compensation —
// needs recalibrating by ear on a real device if it over- or under-shoots.
const OWN_TAKE_MONITOR_BOOST = 1.6;
function playCountIn(ctx, bpm, onBeat, gainMultiplier = 1) {
  const beatDur = 60 / Math.max(20, bpm);
  const start = ctx.currentTime + 0.05;
  for (let i = 0; i < COUNT_IN_BEATS; i++) {
    playMetronomeClick(ctx, start + i * beatDur, i === 0, gainMultiplier);
  }
  return new Promise((resolve) => {
    for (let i = 0; i < COUNT_IN_BEATS; i++) {
      setTimeout(() => onBeat?.(i), i * beatDur * 1000);
    }
    setTimeout(resolve, COUNT_IN_BEATS * beatDur * 1000);
  });
}

/* ---------- Formant "voice" synthesis ---------- */

// Builds a small vowel-ish resonance filter bank ("oo") fed by `source`
// (a sawtooth oscillator). Formant center frequencies stay fixed while the
// oscillator's fundamental changes per note — the same way real vowels work.
function createFormantSum(ctx, source) {
  const formants = [
    { freq: 300, q: 9, gain: 1.0 },
    { freq: 870, q: 11, gain: 0.5 },
    { freq: 2240, q: 13, gain: 0.22 },
  ];
  const sum = ctx.createGain();
  sum.gain.value = 0.8;
  formants.forEach((f) => {
    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = f.freq;
    filter.Q.value = f.q;
    const g = ctx.createGain();
    g.gain.value = f.gain;
    source.connect(filter);
    filter.connect(g);
    g.connect(sum);
  });
  return sum;
}

/* ---------- Pitch-contour visualization (signature element) ---------- */

// `harmonyLayers`: array of { key, notes, color, glow } — one per currently
// active (mixer-enabled) harmony type, drawn simultaneously in its own color.
function PitchCanvas({ melodyNotes, harmonyLayers, keyInfo, keyLabel, duration, playheadTime, onKeyChange, onResetKey, keyIsManual, theme }) {
  const canvasRef = useRef(null);
  const scrollWrapRef = useRef(null);
  const outerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [keyPickerOpen, setKeyPickerOpen] = useState(false);

  // The fullscreen view is driven by our own state (a CSS overlay), not the
  // browser's Fullscreen API — that API is unavailable in a lot of contexts
  // that matter here (in-app browsers, restrictive permissions-policy
  // embeds), where it just rejects. We still request it when possible, for
  // the extra benefit of hiding the browser chrome too, but the visual
  // "fullscreen" result never depends on it succeeding.
  useEffect(() => {
    const handler = () => {
      if (!document.fullscreenElement) setIsFullscreen(false);
    };
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  function toggleFullscreen() {
    if (isFullscreen) {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
      return;
    }
    setZoom(1);
    setIsFullscreen(true);
    if (outerRef.current?.requestFullscreen) {
      outerRef.current.requestFullscreen().catch(() => {});
    }
  }

  const drawRef = useRef(() => {});

  useEffect(() => {
    const canvas = canvasRef.current;
    const scrollWrap = scrollWrapRef.current;
    if (!canvas || !scrollWrap) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const viewportWidth = scrollWrap.clientWidth || 320;
      const width = viewportWidth * zoom;
      const height = scrollWrap.clientHeight || 192;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      ctx.strokeStyle = resolveThemeColor('rgba(var(--ink-rgb),0.06)');
      ctx.lineWidth = 1;
      for (let i = 1; i < duration; i++) {
        const x = (i / duration) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      if (!melodyNotes || melodyNotes.length === 0 || !keyInfo) {
        ctx.fillStyle = resolveThemeColor('rgba(var(--ink-rgb),0.55)');
        ctx.font = '14px "Space Grotesk", sans-serif';
        ctx.fillText('Din tonhöjdskurva ritas upp här efter inspelning', 14, height / 2);
        return;
      }

      const melodyMidis = melodyNotes.map((n) => scaleStepToMidi(n.step, keyInfo.tonic, keyInfo.mode));
      const layers = (harmonyLayers || []).map((layer) => ({
        ...layer,
        midis: layer.notes.map((n) => scaleStepToMidi(n.hStep, keyInfo.tonic, keyInfo.mode)),
      }));
      const all = melodyMidis.concat(layers.flatMap((l) => l.midis));
      const minMidi = Math.min(...all) - 2;
      const maxMidi = Math.max(...all) + 2;
      const topPad = 20;
      const bottomPad = 20;
      const usableHeight = Math.max(1, height - topPad - bottomPad);
      const yFor = (midi) => topPad + usableHeight - ((midi - minMidi) / (maxMidi - minMidi)) * usableHeight;
      const xFor = (t) => (t / duration) * width;

      const drawLine = (notes, midis, color, glow, labelDy) => {
        // Resolved once per call (not per note) — same color/glow for
        // every segment this call draws.
        const resolvedColor = resolveThemeColor(color);
        const resolvedGlow = resolveThemeColor(glow);
        ctx.lineCap = 'round';
        notes.forEach((n, i) => {
          const y = yFor(midis[i]);
          ctx.beginPath();
          ctx.moveTo(xFor(n.start), y);
          ctx.lineTo(xFor(Math.max(n.end, n.start + 0.04)), y);
          ctx.strokeStyle = resolvedColor;
          ctx.lineWidth = 4;
          ctx.shadowColor = resolvedGlow;
          ctx.shadowBlur = 8;
          ctx.stroke();
        });
        ctx.shadowBlur = 0;

        // Note-name labels, centered above (melody) or below (harmony) each
        // segment. Skip a label only if it would actually overlap the
        // previous one (measured, not a fixed width guess) so short notes
        // still get labeled whenever there's room.
        ctx.font = '600 9.5px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = resolvedColor;
        let lastLabelRight = -Infinity;
        notes.forEach((n, i) => {
          const segStart = xFor(n.start);
          const segEnd = xFor(Math.max(n.end, n.start + 0.04));
          const cx = (segStart + segEnd) / 2;
          const label = midiToNoteName(midis[i]);
          const labelWidth = ctx.measureText(label).width;
          const labelLeft = cx - labelWidth / 2 - 2;
          if (labelLeft < lastLabelRight) return; // would overlap the previous label
          const cy = yFor(midis[i]) + labelDy;
          ctx.fillText(label, cx, cy);
          lastLabelRight = cx + labelWidth / 2 + 2;
        });
      };

      drawLine(melodyNotes, melodyMidis, MELODY_COLOR.line, MELODY_COLOR.glow, -10);
      layers.forEach((layer) => {
        drawLine(layer.notes, layer.midis, layer.color, layer.glow, 17);
      });

      if (playheadTime !== null && playheadTime !== undefined) {
        const x = xFor(Math.min(playheadTime, duration));
        // Wide soft highlight band so the position reads at a glance, not
        // just a thin line that can get lost among the note colors.
        const bandGrad = ctx.createLinearGradient(x - 10, 0, x + 10, 0);
        bandGrad.addColorStop(0, resolveThemeColor('rgba(var(--ink-rgb),0)'));
        bandGrad.addColorStop(0.5, resolveThemeColor('rgba(var(--ink-rgb),0.16)'));
        bandGrad.addColorStop(1, resolveThemeColor('rgba(var(--ink-rgb),0)'));
        ctx.fillStyle = bandGrad;
        ctx.fillRect(x - 10, 0, 20, height);

        // The playhead line/marker itself is always the theme's primary
        // "ink" tone (near-white in dark mode, near-black in light) rather
        // than a fixed white — a literal white marker would disappear
        // against a light-mode background.
        const playheadColor = resolveThemeColor('rgba(var(--ink-rgb),1)');
        ctx.strokeStyle = playheadColor;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = resolveThemeColor('rgba(var(--ink-rgb),0.85)');
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.shadowBlur = 0;

        ctx.beginPath();
        ctx.moveTo(x - 5, 0);
        ctx.lineTo(x + 5, 0);
        ctx.lineTo(x, 8);
        ctx.closePath();
        ctx.fillStyle = playheadColor;
        ctx.fill();
      }
    };

    drawRef.current = draw;
    draw();
  }, [melodyNotes, harmonyLayers, keyInfo, duration, playheadTime, zoom, isFullscreen, theme]);

  // A layout change can arrive later than the props/state change that
  // triggered it — most notably entering the CSS-driven fullscreen overlay
  // above, where the wrapper's real size settles a moment after the
  // isFullscreen state flips. Re-measuring and redrawing on any actual size
  // change (rather than trusting the size read at the moment of the state
  // change) avoids ending up with a canvas sized for a stale layout.
  useEffect(() => {
    const scrollWrap = scrollWrapRef.current;
    if (!scrollWrap || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => drawRef.current());
    ro.observe(scrollWrap);
    return () => ro.disconnect();
  }, []);

  return (
    <div ref={outerRef} className={isFullscreen ? 'fixed inset-0 z-50 flex flex-col p-4' : ''} style={isFullscreen ? { backgroundColor: 'var(--bg)' } : undefined}>
      <div className="flex flex-wrap items-center gap-2 mb-2 font-mono-ui text-xs" style={{ color: 'var(--text-muted)' }}>
        <button
          onClick={() => setZoom((z) => Math.max(1, +(z - 0.5).toFixed(1)))}
          className="stamma-btn px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(var(--ink-rgb),0.15)' }}
          aria-label="Zooma ut"
        >
          −
        </button>
        <span className="w-10 text-center">{Math.round(zoom * 100)}%</span>
        <button
          onClick={() => setZoom((z) => Math.min(4, +(z + 0.5).toFixed(1)))}
          className="stamma-btn px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(var(--ink-rgb),0.15)' }}
          aria-label="Zooma in"
        >
          +
        </button>
        {keyLabel && (
          <div className="relative">
            <button
              onClick={() => setKeyPickerOpen((v) => !v)}
              className="stamma-btn px-2.5 py-1 rounded-md flex items-center gap-1"
              style={{ color: 'var(--orange)', border: '1px solid rgba(var(--orange-rgb),0.3)', whiteSpace: 'nowrap' }}
            >
              Tonart: {keyLabel}
              <span style={{ fontSize: 13, display: 'inline-block', transform: keyPickerOpen ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
            </button>
            {keyPickerOpen && (
              <div
                className="absolute z-20 mt-2 rounded-xl p-3"
                style={{ width: 240, backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--ink-rgb),0.12)', boxShadow: '0 8px 24px rgba(0,0,0,0.4)' }}
              >
                <div className="font-mono-ui text-[10px] mb-2" style={{ color: 'var(--text-muted)' }}>
                  Rätta tonarten manuellt
                </div>
                <div className="grid grid-cols-4 gap-1.5">
                  {NOTE_NAMES.map((name, pc) => (
                    <button
                      key={pc}
                      onClick={() => onKeyChange(pc, keyInfo.mode)}
                      className="stamma-btn rounded-md py-1.5 text-xs font-medium"
                      style={{
                        backgroundColor: keyInfo.tonic === pc ? 'rgba(var(--orange-rgb),0.15)' : 'transparent',
                        color: keyInfo.tonic === pc ? 'var(--orange)' : 'var(--text)',
                        border: keyInfo.tonic === pc ? '1px solid rgba(var(--orange-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                      }}
                    >
                      {name}
                    </button>
                  ))}
                </div>
                <div className="flex gap-1.5 mt-1.5">
                  {[['major', 'Dur'], ['minor', 'Moll']].map(([m, label]) => (
                    <button
                      key={m}
                      onClick={() => onKeyChange(keyInfo.tonic, m)}
                      className="stamma-btn flex-1 rounded-md py-1.5 text-xs font-medium"
                      style={{
                        backgroundColor: keyInfo.mode === m ? 'rgba(var(--orange-rgb),0.15)' : 'transparent',
                        color: keyInfo.mode === m ? 'var(--orange)' : 'var(--text)',
                        border: keyInfo.mode === m ? '1px solid rgba(var(--orange-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                      }}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                {keyIsManual && (
                  <button
                    onClick={() => { onResetKey(); setKeyPickerOpen(false); }}
                    className="stamma-btn w-full mt-2.5 rounded-lg py-1.5 font-mono-ui text-[10px]"
                    style={{ color: 'var(--text-muted)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
                  >
                    ↺ Återställ till upptäckt tonart
                  </button>
                )}
              </div>
            )}
          </div>
        )}
        <button
          onClick={toggleFullscreen}
          className="stamma-btn ml-auto px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(var(--ink-rgb),0.15)' }}
        >
          {isFullscreen ? 'Stäng helskärm' : 'Helskärm'}
        </button>
      </div>
      <div
        ref={scrollWrapRef}
        className={isFullscreen ? 'flex-1 overflow-x-auto' : 'w-full h-48 md:h-56 overflow-x-auto rounded-xl'}
      >
        <canvas
          ref={canvasRef}
          className="block rounded-xl"
          role="img"
          aria-label={`Tonhöjdskurva${keyLabel ? ` i ${keyLabel}` : ''} — visar melodins${harmonyLayers.length ? ' och stämmornas' : ''} toner över tid`}
        />
      </div>
    </div>
  );
}

/* ---------- Waveform trim view (Original channel) ---------- */

const TRIM_HANDLE_MIN_GAP = 0.15;
const FADE_MIN_GAP = 0.05;
const FADE_COLOR = 'var(--gold)';

// A waveform of the raw recording with two draggable handles marking the
// [trimStart, trimEnd) window every mixer channel's playback gets clipped
// to (see startMix), plus a pair of Tracktion-Waveform-style fade handles
// — small triangular flags at the top of the clip that you drag inward to
// set fade-in/fade-out length. The diagonal "cut" from the trim edge
// (silence) to the fade's end point (full volume) mirrors that convention,
// adapted for a bipolar (mirrored top/bottom) waveform by drawing the cut
// from the centerline rather than from one corner.
// Not zoomable like PitchCanvas — fullscreen here is purely about giving
// the drag handles more pixels to land precisely on.
const NOISE_COLOR = 'var(--noise-orange)';

function WaveformTrimmer({
  peaks, duration, trimStart, trimEnd, onTrimChange, fadeIn, fadeOut, onFadeChange, playheadTime,
  isPlaying, onPlayPause, onSeek, playDisabled,
  onStop, loopEnabled, onToggleLoop, busy,
  noiseReductionMode, onToggleNoiseReductionMode, noiseSampleStart, noiseSampleEnd, onNoiseSampleChange,
  onApplyNoiseReduction, denoising, denoiseError, noiseReductionApplied,
  metronomeEnabled, onToggleMetronome, theme,
}) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const drawRef = useRef(() => {});

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = wrap.clientWidth || 320;
      const height = wrap.clientHeight || 96;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      const n = peaks.length;
      const midY = height / 2;
      const barW = width / n;
      ctx.fillStyle = resolveThemeColor('rgba(var(--ink-rgb),0.4)');
      for (let i = 0; i < n; i++) {
        const x = (i / n) * width;
        const h = Math.max(1.5, peaks[i] * (height - 10));
        ctx.fillRect(x, midY - h / 2, Math.max(1, barW - 0.5), h);
      }

      const xFor = (t) => (duration > 0 ? (t / duration) * width : 0);
      // Dims a deselected (trimmed-out) region by blending it back toward
      // the page background — so this tracks --bg, not a fixed dark tone,
      // to still read as "fades into the background" in light mode.
      ctx.fillStyle = resolveThemeColor('rgba(var(--bg-rgb),0.72)');
      if (trimStart > 0) ctx.fillRect(0, 0, xFor(trimStart), height);
      if (trimEnd < duration) ctx.fillRect(xFor(trimEnd), 0, width - xFor(trimEnd), height);

      const windowDur = Math.max(0.001, trimEnd - trimStart);
      const fi = Math.max(0, Math.min(fadeIn, windowDur));
      const fo = Math.max(0, Math.min(fadeOut, windowDur - fi));

      if (fi > 0) {
        const x0 = xFor(trimStart);
        const x1 = xFor(trimStart + fi);
        const grad = ctx.createLinearGradient(x0, 0, x1, 0);
        grad.addColorStop(0, resolveThemeColor('rgba(var(--bg-rgb),0.65)'));
        grad.addColorStop(1, resolveThemeColor('rgba(var(--bg-rgb),0)'));
        ctx.fillStyle = grad;
        ctx.fillRect(x0, 0, x1 - x0, height);

        ctx.strokeStyle = resolveThemeColor(FADE_COLOR);
        ctx.lineWidth = 1.75;
        ctx.beginPath();
        ctx.moveTo(x0, midY);
        ctx.lineTo(x1, 0);
        ctx.moveTo(x0, midY);
        ctx.lineTo(x1, height);
        ctx.stroke();
      }
      if (fo > 0) {
        const x1 = xFor(trimEnd);
        const x0 = xFor(trimEnd - fo);
        const grad = ctx.createLinearGradient(x0, 0, x1, 0);
        grad.addColorStop(0, resolveThemeColor('rgba(var(--bg-rgb),0)'));
        grad.addColorStop(1, resolveThemeColor('rgba(var(--bg-rgb),0.65)'));
        ctx.fillStyle = grad;
        ctx.fillRect(x0, 0, x1 - x0, height);

        ctx.strokeStyle = resolveThemeColor(FADE_COLOR);
        ctx.lineWidth = 1.75;
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x1, midY);
        ctx.moveTo(x0, height);
        ctx.lineTo(x1, midY);
        ctx.stroke();
      }

      if (noiseReductionMode) {
        const nx0 = xFor(Math.max(0, noiseSampleStart));
        const nx1 = xFor(Math.min(duration, noiseSampleEnd));
        ctx.fillStyle = resolveThemeColor('rgba(var(--noise-orange-rgb),0.18)');
        ctx.fillRect(nx0, 0, nx1 - nx0, height);
        ctx.strokeStyle = resolveThemeColor(NOISE_COLOR);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(nx0 + 0.75, 0.75, Math.max(0, nx1 - nx0 - 1.5), height - 1.5);
        ctx.setLineDash([]);
      }

      if (playheadTime !== null && playheadTime !== undefined) {
        const x = xFor(Math.min(playheadTime, duration));
        ctx.strokeStyle = resolveThemeColor('rgba(var(--ink-rgb),1)');
        ctx.lineWidth = 2;
        ctx.shadowColor = resolveThemeColor('rgba(var(--ink-rgb),0.85)');
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    };

    drawRef.current = draw;
    draw();
  }, [peaks, duration, trimStart, trimEnd, fadeIn, fadeOut, playheadTime, isFullscreen, noiseReductionMode, noiseSampleStart, noiseSampleEnd, theme]);

  // Same reasoning as PitchCanvas's ResizeObserver: the fullscreen overlay's
  // real size settles a moment after isFullscreen flips, so redraw on
  // actual layout change rather than trusting the size read at that moment.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => drawRef.current());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  function beginDrag(which) {
    return (e) => {
      const handle = e.currentTarget;
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const rect = wrapRef.current.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        const t = frac * duration;
        if (which === 'start') {
          onTrimChange(Math.min(t, trimEnd - TRIM_HANDLE_MIN_GAP), trimEnd);
        } else {
          onTrimChange(trimStart, Math.max(t, trimStart + TRIM_HANDLE_MIN_GAP));
        }
      };
      const up = () => {
        handle.releasePointerCapture(e.pointerId);
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    };
  }

  function beginFadeDrag(which) {
    return (e) => {
      const handle = e.currentTarget;
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const rect = wrapRef.current.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        const t = frac * duration;
        const windowDur = Math.max(0.001, trimEnd - trimStart);
        if (which === 'in') {
          const next = Math.min(Math.max(0, t - trimStart), windowDur - fadeOut - FADE_MIN_GAP);
          onFadeChange(Math.max(0, next), fadeOut);
        } else {
          const next = Math.min(Math.max(0, trimEnd - t), windowDur - fadeIn - FADE_MIN_GAP);
          onFadeChange(fadeIn, Math.max(0, next));
        }
      };
      const up = () => {
        handle.releasePointerCapture(e.pointerId);
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    };
  }

  // Unlike trim/fade, the noise sample region isn't clamped inside the
  // trim window — a quiet stretch worth sampling might sit just outside
  // the part you're actually keeping (room tone before the trimmed-in
  // singing starts, say), so this ranges over the full recording.
  const NOISE_MIN_GAP = 0.1;
  function beginNoiseDrag(which) {
    return (e) => {
      const handle = e.currentTarget;
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const rect = wrapRef.current.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
        const t = frac * duration;
        if (which === 'start') {
          onNoiseSampleChange(Math.min(t, noiseSampleEnd - NOISE_MIN_GAP), noiseSampleEnd);
        } else {
          onNoiseSampleChange(noiseSampleStart, Math.max(t, noiseSampleStart + NOISE_MIN_GAP));
        }
      };
      const up = () => {
        handle.releasePointerCapture(e.pointerId);
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    };
  }

  // Keyboard equivalent of the pointer-drag handlers above, for VoiceOver
  // and keyboard users — the canvas itself has no way to drag with either.
  // ArrowLeft/Right nudge by STEP_SEC (STEP_SEC_LARGE with Shift held);
  // Home/End (trim handles only) jump to that handle's natural extreme.
  // Mirrors each drag handler's own clamping and, for the fade-out handle,
  // its inverted direction (dragging right *shrinks* the fade-out region).
  const STEP_SEC = 0.1;
  const STEP_SEC_LARGE = 1.0;

  function trimKeyDown(which) {
    return (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      e.preventDefault();
      const step = e.shiftKey ? STEP_SEC_LARGE : STEP_SEC;
      if (which === 'start') {
        let next = trimStart;
        if (e.key === 'ArrowLeft') next -= step;
        else if (e.key === 'ArrowRight') next += step;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = trimEnd - TRIM_HANDLE_MIN_GAP;
        onTrimChange(Math.min(Math.max(0, next), trimEnd - TRIM_HANDLE_MIN_GAP), trimEnd);
      } else {
        let next = trimEnd;
        if (e.key === 'ArrowLeft') next -= step;
        else if (e.key === 'ArrowRight') next += step;
        else if (e.key === 'Home') next = trimStart + TRIM_HANDLE_MIN_GAP;
        else if (e.key === 'End') next = duration;
        onTrimChange(trimStart, Math.max(Math.min(duration, next), trimStart + TRIM_HANDLE_MIN_GAP));
      }
    };
  }

  function fadeKeyDown(which) {
    return (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const delta = (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? STEP_SEC_LARGE : STEP_SEC);
      const windowDur = Math.max(0.001, trimEnd - trimStart);
      if (which === 'in') {
        const next = Math.min(Math.max(0, fadeIn + delta), windowDur - fadeOut - FADE_MIN_GAP);
        onFadeChange(Math.max(0, next), fadeOut);
      } else {
        const next = Math.min(Math.max(0, fadeOut - delta), windowDur - fadeIn - FADE_MIN_GAP);
        onFadeChange(fadeIn, Math.max(0, next));
      }
    };
  }

  function noiseKeyDown(which) {
    return (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const delta = (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? STEP_SEC_LARGE : STEP_SEC);
      if (which === 'start') {
        const next = Math.min(Math.max(0, noiseSampleStart + delta), noiseSampleEnd - NOISE_MIN_GAP);
        onNoiseSampleChange(next, noiseSampleEnd);
      } else {
        const next = Math.max(Math.min(duration, noiseSampleEnd + delta), noiseSampleStart + NOISE_MIN_GAP);
        onNoiseSampleChange(noiseSampleStart, next);
      }
    };
  }

  const startPct = duration > 0 ? Math.min(100, (trimStart / duration) * 100) : 0;
  const endPct = duration > 0 ? Math.min(100, (trimEnd / duration) * 100) : 100;
  const windowDurForHandles = Math.max(0.001, trimEnd - trimStart);
  const fadeInClamped = Math.max(0, Math.min(fadeIn, windowDurForHandles));
  const fadeOutClamped = Math.max(0, Math.min(fadeOut, windowDurForHandles - fadeInClamped));
  const fadeInPct = duration > 0 ? Math.min(100, ((trimStart + fadeInClamped) / duration) * 100) : 0;
  const fadeOutPct = duration > 0 ? Math.min(100, ((trimEnd - fadeOutClamped) / duration) * 100) : 100;

  const handleStyle = (pct) => ({
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: `calc(${pct}% - 10px)`,
    width: 20,
    cursor: 'ew-resize',
    touchAction: 'none',
  });

  const fadeHandleStyle = (pct) => ({
    position: 'absolute',
    top: 0,
    left: `calc(${pct}% - 9px)`,
    width: 18,
    height: 18,
    cursor: 'ew-resize',
    touchAction: 'none',
  });

  const fadeFlag = (pointRight) => (
    <div
      style={{
        width: 0,
        height: 0,
        borderTop: `11px solid ${FADE_COLOR}`,
        borderLeft: pointRight ? '2px solid transparent' : '16px solid transparent',
        borderRight: pointRight ? '16px solid transparent' : '2px solid transparent',
        filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.5))',
      }}
    />
  );

  const noiseStartPct = duration > 0 ? Math.min(100, (Math.max(0, noiseSampleStart) / duration) * 100) : 0;
  const noiseEndPct = duration > 0 ? Math.min(100, (Math.min(duration, noiseSampleEnd) / duration) * 100) : 0;

  const noiseHandleStyle = (pct) => ({
    position: 'absolute',
    bottom: 0,
    left: `calc(${pct}% - 9px)`,
    width: 18,
    height: 18,
    cursor: 'ew-resize',
    touchAction: 'none',
  });

  const noiseFlag = (
    <div
      style={{
        width: 0,
        height: 0,
        borderBottom: `11px solid ${NOISE_COLOR}`,
        borderLeft: '9px solid transparent',
        borderRight: '9px solid transparent',
        filter: 'drop-shadow(0 -1px 2px rgba(0,0,0,0.5))',
      }}
    />
  );

  return (
    <div className={isFullscreen ? 'fixed inset-0 z-50 flex flex-col p-4' : ''} style={isFullscreen ? { backgroundColor: 'var(--bg)' } : undefined}>
      <div className="flex items-center justify-between mb-2 font-mono-ui text-xs" style={{ color: 'var(--text-muted)' }}>
        <span>Vågform — beskär &amp; tona in/ut</span>
        <button
          onClick={() => setIsFullscreen((v) => !v)}
          className="stamma-btn px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(var(--ink-rgb),0.15)' }}
        >
          {isFullscreen ? 'Stäng helskärm' : 'Helskärm'}
        </button>
      </div>
      <div
        ref={wrapRef}
        className={isFullscreen ? 'flex-1 relative rounded-xl' : 'relative w-full h-24 rounded-xl'}
      >
        <canvas
          ref={canvasRef}
          className="block rounded-xl absolute inset-0"
          role="img"
          aria-label="Vågform av inspelningen, med hantag för att beskära start och slut och tona in/ut"
        />
        <div
          onPointerDown={beginDrag('start')}
          onKeyDown={trimKeyDown('start')}
          style={handleStyle(startPct)}
          className="stamma-btn"
          role="slider"
          tabIndex={0}
          aria-label="Beskär start"
          aria-valuemin={0}
          aria-valuemax={Math.max(0, trimEnd - TRIM_HANDLE_MIN_GAP)}
          aria-valuenow={Math.round(trimStart * 10) / 10}
          aria-valuetext={`${trimStart.toFixed(1)} sekunder`}
        >
          <div style={{ position: 'absolute', left: 9, top: 0, bottom: 0, width: 2, backgroundColor: 'var(--teal)' }} />
          <div style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 20, height: 34, borderRadius: 7, backgroundColor: 'var(--teal)' }} />
        </div>
        <div
          onPointerDown={beginDrag('end')}
          onKeyDown={trimKeyDown('end')}
          style={handleStyle(endPct)}
          className="stamma-btn"
          role="slider"
          tabIndex={0}
          aria-label="Beskär slut"
          aria-valuemin={Math.min(duration, trimStart + TRIM_HANDLE_MIN_GAP)}
          aria-valuemax={duration}
          aria-valuenow={Math.round(trimEnd * 10) / 10}
          aria-valuetext={`${trimEnd.toFixed(1)} sekunder`}
        >
          <div style={{ position: 'absolute', left: 9, top: 0, bottom: 0, width: 2, backgroundColor: 'var(--rose)' }} />
          <div style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 20, height: 34, borderRadius: 7, backgroundColor: 'var(--rose)' }} />
        </div>
        <div
          onPointerDown={beginFadeDrag('in')}
          onKeyDown={fadeKeyDown('in')}
          style={fadeHandleStyle(fadeInPct)}
          className="stamma-btn"
          title="Tona in"
          role="slider"
          tabIndex={0}
          aria-label="Tona in-längd"
          aria-valuemin={0}
          aria-valuemax={Math.max(0, windowDurForHandles - fadeOutClamped - FADE_MIN_GAP)}
          aria-valuenow={Math.round(fadeInClamped * 10) / 10}
          aria-valuetext={`${fadeInClamped.toFixed(1)} sekunder`}
        >
          {fadeFlag(true)}
        </div>
        <div
          onPointerDown={beginFadeDrag('out')}
          onKeyDown={fadeKeyDown('out')}
          style={fadeHandleStyle(fadeOutPct)}
          className="stamma-btn"
          title="Tona ut"
          role="slider"
          tabIndex={0}
          aria-label="Tona ut-längd"
          aria-valuemin={0}
          aria-valuemax={Math.max(0, windowDurForHandles - fadeInClamped - FADE_MIN_GAP)}
          aria-valuenow={Math.round(fadeOutClamped * 10) / 10}
          aria-valuetext={`${fadeOutClamped.toFixed(1)} sekunder`}
        >
          {fadeFlag(false)}
        </div>
        {isFullscreen && noiseReductionMode && (
          <>
            <div
              onPointerDown={beginNoiseDrag('start')}
              onKeyDown={noiseKeyDown('start')}
              style={noiseHandleStyle(noiseStartPct)}
              className="stamma-btn"
              title="Brusprov start"
              role="slider"
              tabIndex={0}
              aria-label="Brusprov start"
              aria-valuemin={0}
              aria-valuemax={Math.max(0, noiseSampleEnd - NOISE_MIN_GAP)}
              aria-valuenow={Math.round(Math.max(0, noiseSampleStart) * 10) / 10}
              aria-valuetext={`${Math.max(0, noiseSampleStart).toFixed(1)} sekunder`}
            >
              {noiseFlag}
            </div>
            <div
              onPointerDown={beginNoiseDrag('end')}
              onKeyDown={noiseKeyDown('end')}
              style={noiseHandleStyle(noiseEndPct)}
              className="stamma-btn"
              title="Brusprov slut"
              role="slider"
              tabIndex={0}
              aria-label="Brusprov slut"
              aria-valuemin={Math.min(duration, noiseSampleStart + NOISE_MIN_GAP)}
              aria-valuemax={duration}
              aria-valuenow={Math.round(Math.min(duration, noiseSampleEnd) * 10) / 10}
              aria-valuetext={`${Math.min(duration, noiseSampleEnd).toFixed(1)} sekunder`}
            >
              {noiseFlag}
            </div>
          </>
        )}
      </div>
      <div className="flex justify-between mt-1.5 font-mono-ui text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <span style={{ color: 'var(--teal)' }}>{trimStart.toFixed(1)}s</span>
        <span>{(trimEnd - trimStart).toFixed(1)}s vald</span>
        <span style={{ color: 'var(--rose)' }}>{trimEnd.toFixed(1)}s</span>
      </div>
      <div className="flex justify-between mt-1 font-mono-ui text-[10px]" style={{ color: FADE_COLOR }}>
        <span>Tona in: {fadeInClamped.toFixed(1)}s</span>
        <span>Tona ut: {fadeOutClamped.toFixed(1)}s</span>
      </div>

      {/* Fullscreen only, per request — precise placement of the noise
          sample needs the extra room the compact view doesn't have. */}
      {isFullscreen && onToggleNoiseReductionMode && (
        <div className="mt-4 rounded-xl p-3" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: `1px solid ${noiseReductionMode ? `${NOISE_COLOR}55` : 'rgba(var(--ink-rgb),0.1)'}` }}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-sm font-medium">Brusreducering</span>
                {noiseReductionApplied && (
                  <span
                    className="font-mono-ui text-[10px] font-semibold px-1.5 py-0.5 rounded"
                    style={{ backgroundColor: `${NOISE_COLOR}26`, color: NOISE_COLOR, border: `1px solid ${NOISE_COLOR}66` }}
                  >
                    ✓ Aktiv
                  </span>
                )}
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                Markera en del med bara brus — resten av inspelningen renas utifrån den
              </div>
            </div>
            <ToggleSwitch checked={noiseReductionMode} onChange={onToggleNoiseReductionMode} accentColor={NOISE_COLOR} ariaLabel="Brusreducering" />
          </div>
          {noiseReductionMode && (
            <div className="mt-3 pt-3" style={{ borderTop: '1px solid rgba(var(--ink-rgb),0.08)' }}>
              <div className="flex justify-between font-mono-ui text-[10px] mb-2" style={{ color: NOISE_COLOR }}>
                <span>Brusprov: {noiseSampleStart.toFixed(2)}s – {noiseSampleEnd.toFixed(2)}s</span>
                <span>({(noiseSampleEnd - noiseSampleStart).toFixed(2)}s)</span>
              </div>
              <button
                onClick={onApplyNoiseReduction}
                disabled={denoising}
                className="stamma-btn w-full rounded-xl py-2.5 flex items-center justify-center gap-2 font-body font-medium text-sm"
                style={{
                  backgroundColor: denoising ? 'rgba(var(--ink-rgb),0.06)' : NOISE_COLOR,
                  color: denoising ? 'rgba(var(--ink-rgb),0.3)' : 'var(--bg)',
                  cursor: denoising ? 'not-allowed' : 'pointer',
                }}
              >
                {denoising ? 'Bearbetar …' : noiseReductionApplied ? 'Tillämpa igen' : 'Tillämpa brusreducering'}
              </button>
              {noiseReductionApplied && !denoising && (
                <div className="mt-2 flex items-center gap-1.5 font-mono-ui text-[11px]" style={{ color: NOISE_COLOR }}>
                  <span>✓</span>
                  <span>Brusreducering tillämpad på inspelningen</span>
                </div>
              )}
              {denoiseError && (
                <div className="mt-2 text-xs" style={{ color: 'var(--red-softer)' }}>{denoiseError}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Fullscreen only — the normal view's play/pause and seek slider
          live in the parent, right below this component, but that's
          covered by this fixed overlay once isFullscreen is true. */}
      {isFullscreen && onPlayPause && (
        <div className="mt-4">
          <TransportButtons
            loopEnabled={loopEnabled}
            onToggleLoop={onToggleLoop}
            onStop={onStop}
            isPlaying={isPlaying}
            onPlayPause={onPlayPause}
            disabled={playDisabled}
            busy={busy}
            metronomeEnabled={metronomeEnabled}
            onToggleMetronome={onToggleMetronome}
          />
          <div className="flex items-center gap-2 mt-2">
            <span className="font-mono-ui text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>
              {(playheadTime !== null ? playheadTime : trimStart).toFixed(1)}s
            </span>
            <input
              type="range"
              min={trimStart}
              max={trimEnd}
              step="0.01"
              value={playheadTime !== null ? playheadTime : trimStart}
              onChange={(e) => onSeek(parseFloat(e.target.value))}
              className="stamma-fader w-full"
              style={{ accentColor: 'var(--orange)' }}
              aria-label="Spola i vågformen"
            />
            <span className="font-mono-ui text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>{trimEnd.toFixed(1)}s</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main app ---------- */

const DURATION = 10;

// Live-monitor pitch-tracking loop (see liveMonitorPitchLoop): how often
// it samples the mic for a pitch reading, and how many consecutive
// readings of the *same* scale step it needs before actually re-locking
// onto it — same autocorrelation-noise reasoning as the offline engine's
// per-note ratio (see harmonyEngine.js), just applied live: a single
// stray reading right at a note boundary shouldn't flap the shift ratio
// back and forth.
const LIVE_MONITOR_SAMPLE_HOP_SEC = 0.035;
const LIVE_MONITOR_LOCK_FRAMES = 3;
const LIVE_MONITOR_RMS_THRESHOLD = 0.0035; // matches ENERGY_RMS_THRESHOLD in harmonyEngine.js

// Strength stops for autotune once it's switched on — the on/off state
// itself is a separate toggle (autotuneOn), so this is just the "tre
// nivåer" of partial correction strength, expressed as the same 0..1
// amount buildAutotuneRatioCurve already accepted (see harmonyEngine.js).
const AUTOTUNE_LEVELS = [
  { key: 'light', label: 'Lätt', amount: 0.25 },
  { key: 'medium', label: 'Medel', amount: 0.45 },
  { key: 'hard', label: 'Hård', amount: 0.7 },
];

// Strength stops for reverb, same on/off + collapsed strength UI pattern
// as autotune. `decay` is the synthetic impulse response's length
// (seconds); `wet` is how much of the reverberated signal gets mixed
// back in alongside the untouched dry signal.
const REVERB_LEVELS = [
  { key: 'light', label: 'Liten', decay: 0.9, wet: 0.14 },
  { key: 'medium', label: 'Medel', decay: 1.7, wet: 0.24 },
  { key: 'large', label: 'Stor', decay: 3.0, wet: 0.38 },
];

function defaultChannels() {
  return {
    original: { enabled: false, volume: 0.8, pan: 0, solo: false },
    melody: { enabled: true, volume: 0.9, pan: 0, solo: false },
    ters: { enabled: false, volume: 0.8, pan: 0, solo: false, direction: HARMONY_TYPES.ters.defaultDirection },
    kvint: { enabled: false, volume: 0.8, pan: 0, solo: false, direction: HARMONY_TYPES.kvint.defaultDirection },
    sext: { enabled: false, volume: 0.8, pan: 0, solo: false, direction: HARMONY_TYPES.sext.defaultDirection },
  };
}

// A channel is actually audible if it's on, *and* — whenever one or more
// channels are soloed — it's one of the soloed ones. Solo only narrows the
// mix down temporarily; it doesn't change anyone's own enabled/mute state,
// so un-soloing restores exactly what was playing before.
function isChannelAudible(channelsState, key) {
  const ch = channelsState[key];
  if (!ch.enabled) return false;
  const anySolo = Object.values(channelsState).some((c) => c.solo);
  return anySolo ? ch.solo : true;
}

// A separate, linkable page (see the #om hash handling in App) that
// explains what each part of the app actually does — kept plain-language
// and practical rather than technical, matching the app's own tone.
function AboutPage() {
  const sections = [
    {
      id: 'spela-in',
      title: 'Sjung in eller ladda upp',
      body: 'Spela in dig själv (max 10s) direkt i webbläsaren, eller ladda upp en befintlig ljudfil (WAV, MP3, M4A, OGG eller FLAC). Är filen längre än 10 sekunder klipps den av vid gränsen.',
    },
    {
      id: 'metronom',
      title: 'Metronom',
      body: 'Innan du spelar in kan du slå på ett klickspår som hörs medan du sjunger, med en egen takt (BPM) du ställer med +/− eller lyssnar dig fram till i förväg. Har du redan spelat in en gång försöker appen räkna ut takten själv utifrån tonerna — går det inte, går det alltid att sätta den manuellt. Är klickspåret på när du sedan spelar upp inspelningen hörs det också då, i exakt takt med uppspelningen.',
    },
    {
      id: 'live-forhandslyssning',
      title: 'Live-förhandslyssning',
      body: 'En direkt, låg-latens mikrofonlyssning som pitchskiftar din röst i realtid till ters-, kvint- och/eller sextavstånd — valfri kombination, alla samtidigt om du vill — ett smakprov på hur stämmorna skulle låta, utan att behöva spela in och vänta på den vanliga stämmemotorn. Har du redan spelat in en gång följer den tonarten och tonhöjden du faktiskt sjunger, precis som den riktiga stämmemotorn, inklusive samma "Humanisera stämmor"-karaktär (vibrato, formant, tonhöjd) om den är påslagen — så flera stämmor samtidigt faktiskt låter som flera olika röster, inte samma röst tredubblad; innan dess (på startskärmen) används istället ett fast intervall utan karaktär, eftersom ingen tonart är känd än. Spelar du in en egen ters/kvint/sext (rec-knappen på stämmans kanal) medan den är på följer den automatiskt med och spelar bara upp just den stämman under tagningen, så du hör exakt vad du siktar på — dina egna valda stämmor kommer tillbaka när du är klar. Kräver hörlurar, annars hörs rundgång eftersom mikrofonen är öppen samtidigt som ljudet spelas upp. Av som standard — fler samtidiga stämmor kostar mer processorkraft.',
    },
    {
      id: 'tonhojdskurva',
      title: 'Tonhöjdskurvan',
      body: 'Appen lyssnar igenom inspelningen och räknar ut vilken ton som sjungs vid varje ögonblick. Det ritas upp som en kurva du kan zooma i och öppna i helskärm — varje stapel är en enskild ton.',
    },
    {
      id: 'tonart',
      title: 'Tonart',
      body: 'Utifrån vilka toner som förekommer mest gissar appen vilken tonart (dur eller moll) melodin ligger i. Det är den tonarten stämmorna sedan byggs utifrån, så en fel uppfattad tonart är den vanligaste orsaken till att en stämma låter konstig. Går det fel går det att rätta: tryck på "Tonart: …" ovanför tonhöjdskurvan för att välja grundton och dur/moll för hand — stämmorna räknas om direkt.',
    },
    {
      id: 'ljud',
      title: 'Röst: Ren synt / Formantröst / Din röst',
      body: 'Tre sätt att höra melodin och stämmorna på. "Ren synt" är en enkel sinuston, lättast att stämma efter. "Formantröst" är en syntetisk, vokalliknande klang. "Din röst" pitchskiftar din egen inspelning till varje stämmas toner, med bevarad klangfärg — mest verklighetstroget, men kräver att du spelat in (inte laddat upp en synt-fil).',
    },
    {
      id: 'stammor',
      title: 'Stämmor: ters, kvint, sext',
      body: 'Appen bygger tre extra röster som följer din melodi på ett fast musikaliskt avstånd: en ters, en kvint och en sext. Varje stämma kan ligga över eller under melodin (Överstämma/Understämma) — det väljer du per stämma. Under "Din röst" särskiljs andning och konsonanter från själva sångtonen innan tonhöjden skiftas, så bara det som faktiskt har en tonhöjd stäms om — andningen och s/f/h-ljuden spelas tillbaka som de är, vilket gör stämman mindre metallisk.',
    },
    {
      id: 'egen-stamma',
      title: 'Spela in egen stämma',
      body: 'Rec-knappen på en stämmas kanal spelar in dig själv sjungandes just den stämman, med melodin i hörlurarna/högtalaren som stöd samtidigt. Din inspelade stämma dyker upp som en egen kanal ("Egen ters" osv.) och spelas upp precis som du sjöng den, utan pitchskiftning — du kan radera den och spela in igen när du vill. Fäll ut kanalen så visas samma vågform-verktyg som för huvudinspelningen, med egen beskärning och tona in/ut — bra för att putsa bort en falskstart eller ett andningsljud i början utan att spela in om hela tagningen. Har du live-förhandslyssningen påslagen när du trycker rec smalnar den automatiskt av till just den stämma du spelar in, tonarts- och personlighetsanpassad, så du hör målstämman sjungas med dig medan du tar den — och går tillbaka till dina egna valda stämmor när tagningen är klar. (Kräver hörlurar, annars läcker den in i inspelningen.)',
    },
    {
      id: 'autotune',
      title: 'Autotune',
      body: 'Rättar falska toner i själva inspelningen (inte i stämmorna). Av som standard. När den är på väljer du hur mycket rättning som ska ske — Lätt, Medel eller Hård — där Hård drar tonerna närmast helt till rätt tonhöjd.',
    },
    {
      id: 'reverb',
      title: 'Reverb',
      body: 'Lägger på lite rymd på hela mixen (alla kanaler samtidigt, inte en i taget). Av som standard. Storleken — Liten, Medel eller Stor — styr hur lång och påtaglig efterklangen är.',
    },
    {
      id: 'humanisera',
      title: 'Humanisera stämmor',
      body: 'På som standard under "Din röst". Ger ters, kvint och sext var sin egen, konsekventa röstkaraktär istället för att låta som en perfekt klonad kopia av din egen röst: en aning över/under tonhöjden, en långsam svävning, egen vibrato, och en liten formantförskjutning (ungefärlig röstlådestorlek) — som om det faktiskt vore tre olika sångare.',
    },
    {
      id: 'brusreducering',
      title: 'Brusreducering',
      body: 'Markera en del av inspelningen som bara innehåller bakgrundsbrus (t.ex. tystnaden precis innan du börjar sjunga) — resten av inspelningen renas sedan utifrån hur det bruset låter.',
    },
    {
      id: 'normalisera',
      title: 'Normalisera',
      body: 'Höjer inspelningens volym så den högsta toppen når precis under maxnivå, utan att förvränga ljudet. Både brusreducering och normalisering skriver över den inspelning stämmorna byggs från — "Återställ till original" ångrar alla sådana ändringar och går tillbaka till den allra första inspelningen.',
    },
    {
      id: 'vagform',
      title: 'Vågformen: beskärning och fader',
      body: 'Under "Din röst" visas en vågform av originalinspelningen. Dra i de två handtagen i kanterna för att välja vilken del som ska spelas — allt utanför blir mörkare. De gula triangelhandtagen i hörnen styr tona in/tona ut: dra dem inåt för en mjukare start eller avslutning istället för en tvär in/utklippning.',
    },
    {
      id: 'transport',
      title: 'Spela, pausa, loopa',
      body: 'Under vågformen finns tre kontroller i samma stil: en stopp-knapp (stannar och hoppar tillbaka till början), en kombinerad spela/pausa-knapp, och en loop-knapp till höger om den (upprepar den valda delen tills du stänger av den). Pausar du mitt i går det att fortsätta exakt där du var. Reglaget under knapparna spolar fram och tillbaka i den valda delen.',
    },
    {
      id: 'mixer',
      title: 'Mixern',
      body: 'Varje spår — melodi, ters, kvint, sext, och eventuella egna inspelade stämmor — går att slå på eller av var för sig, och alla påslagna spelas samtidigt när du trycker play. Varje spår har en liten play-knapp för att förhandslyssna bara det spåret, en S-knapp för att solo:a (tysta alla andra tillfälligt), och volym/panorering som fälls ut genom att trycka på procentsatsen.',
    },
    {
      id: 'sessioner',
      title: 'Sessioner',
      body: 'Appen sparar automatiskt det du håller på med i webbläsaren, namngivet efter tonart och tidpunkt. Fäll ut "Sessioner" för att se alla sparade tagningar, byta namn på dem, öppna en annan, eller radera. Börjar du en ny inspelning eller laddar upp en ny fil sparas det som en egen, ny session — den gamla ligger kvar orörd i listan.',
    },
    {
      id: 'exportera',
      title: 'Exportera',
      body: 'Ladda ner sången och varje stämma som separata WAV-filer, till exempel för att jobba vidare i ett annat program.',
    },
  ];

  function scrollToSection(e, id) {
    e.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  return (
    <div className="min-h-screen w-full flex justify-center" style={{ backgroundColor: 'var(--bg)', color: 'var(--text)' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
        .font-display { font-family: 'Fraunces', serif; }
        .font-body { font-family: 'Space Grotesk', sans-serif; }
        .font-mono-ui { font-family: 'JetBrains Mono', monospace; }
        .stamma-btn:focus-visible { outline: 2px solid var(--orange); outline-offset: 2px; }
      `}</style>
      <div className="w-full max-w-md px-5 py-8 font-body">
        <a
          href="#"
          className="stamma-btn font-mono-ui text-xs"
          style={{ color: 'var(--teal)' }}
        >
          ← Tillbaka
        </a>
        <h1 className="mt-3 font-display text-3xl font-semibold tracking-tight" style={{ color: 'var(--text)' }}>
          Om Stämifier
        </h1>
        <p className="mt-2 text-base leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          Så fungerar appens olika delar, rent praktiskt.
        </p>

        <nav
          className="mt-5 rounded-xl p-3"
          style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}
          aria-label="Innehåll"
        >
          <div className="font-mono-ui text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
            Innehåll
          </div>
          <ul className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1.5">
            {sections.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  onClick={(e) => scrollToSection(e, s.id)}
                  className="stamma-btn text-sm leading-snug"
                  style={{ color: 'var(--teal)' }}
                >
                  {s.title}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="mt-6 space-y-5">
          {sections.map((s) => (
            <div key={s.id} id={s.id} style={{ scrollMarginTop: 16 }}>
              <h2 className="font-display text-lg font-semibold" style={{ color: 'var(--text)' }}>
                {s.title}
              </h2>
              <p className="mt-1 text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                {s.body}
              </p>
            </div>
          ))}
        </div>

        <a
          href="#"
          className="stamma-btn mt-8 inline-block font-mono-ui text-xs"
          style={{ color: 'var(--teal)' }}
        >
          ← Tillbaka
        </a>
      </div>
    </div>
  );
}

export default function App() {
  // Manual light/dark toggle (not tied to the OS's prefers-color-scheme —
  // see the theme tokens in index.css). Persisted so a reload keeps
  // whatever the user last picked; defaults to dark, the app's original
  // design. Applied as a data-theme attribute on <html> (read by
  // index.css) rather than on the app's own root div, so it also covers
  // the plain white flash a browser can paint before React mounts.
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('stamma-theme') === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    const themeColorMeta = document.querySelector('meta[name="theme-color"]');
    if (themeColorMeta) {
      themeColorMeta.setAttribute('content', theme === 'light' ? '#F5F1E8' : '#10131A');
    }
    try {
      localStorage.setItem('stamma-theme', theme);
    } catch {
      // Private browsing or storage disabled — theme just won't persist
      // across reloads, which is a fine fallback rather than an error.
    }
  }, [theme]);

  const [phase, setPhase] = useState('idle'); // idle | countin | recording | analyzing | ready | error
  const [errorMsg, setErrorMsg] = useState('');
  const [countdown, setCountdown] = useState(DURATION);
  const [countInBeat, setCountInBeat] = useState(0); // 0..COUNT_IN_BEATS-1 during phase 'countin'
  const [keyInfo, setKeyInfo] = useState(null);
  const [melodyNotes, setMelodyNotes] = useState([]);
  const [channels, setChannels] = useState(defaultChannels);
  const [isPlaying, setIsPlaying] = useState(false);
  const [soundType, setSoundType] = useState('recording');
  const [voiceReady, setVoiceReady] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(DURATION);
  const [playheadTime, setPlayheadTime] = useState(null);
  const [previewingKey, setPreviewingKey] = useState(null);
  const [exporting, setExporting] = useState(null);
  const [harmonyRenderingByType, setHarmonyRenderingByType] = useState({});
  const [harmonyRenderErrorsByType, setHarmonyRenderErrorsByType] = useState({});
  const [autotuneOn, setAutotuneOn] = useState(false);
  const [autotuneLevelIndex, setAutotuneLevelIndex] = useState(0);
  const [autotuneStrengthExpanded, setAutotuneStrengthExpanded] = useState(false);
  const [autotuneRendering, setAutotuneRendering] = useState(false);
  const [autotuneRenderError, setAutotuneRenderError] = useState('');
  const [reverbOn, setReverbOn] = useState(false);
  const [reverbLevelIndex, setReverbLevelIndex] = useState(0);
  const [reverbStrengthExpanded, setReverbStrengthExpanded] = useState(false);
  // Per-voice micro-detune + slow pitch drift on the rendered ters/kvint/
  // sext (see HARMONY_HUMANIZE_PROFILES in harmonyEngine.js) — on by
  // default since it's the more natural-sounding option, but toggleable
  // for a perfectly locked-to-interval sound instead.
  const [humanizeOn, setHumanizeOn] = useState(true);
  const [isProcessed, setIsProcessed] = useState(false); // recordedBufferRef differs from the original decode
  const [normalizing, setNormalizing] = useState(false);
  const [normalizeError, setNormalizeError] = useState('');
  const [noiseReductionMode, setNoiseReductionMode] = useState(false);
  const [noiseSampleStart, setNoiseSampleStart] = useState(0);
  const [noiseSampleEnd, setNoiseSampleEnd] = useState(0.5);
  const [denoising, setDenoising] = useState(false);
  const [denoiseError, setDenoiseError] = useState('');
  // True once a noise-reduction pass has actually been applied to the
  // current recording — separate from `denoising` (in-progress) so the UI
  // can show a lasting "applied" confirmation, not just a brief spinner.
  const [noiseReductionApplied, setNoiseReductionApplied] = useState(false);
  const [effectsExpanded, setEffectsExpanded] = useState(false);
  const [ljudExpanded, setLjudExpanded] = useState(false);
  const [mixerExpanded, setMixerExpanded] = useState(false);
  const [noteViewExpanded, setNoteViewExpanded] = useState(false);
  const [exportExpanded, setExportExpanded] = useState(false);
  const [metronomeEnabled, setMetronomeEnabled] = useState(false); // clicks during recording, and along with mix playback
  const [metronomeBpm, setMetronomeBpm] = useState(100);
  const [metronomeListening, setMetronomeListening] = useState(false); // preview click loop, not recording
  const [metronomeExpanded, setMetronomeExpanded] = useState(false);
  const [tempoDetected, setTempoDetected] = useState(false);
  const [micPermission, setMicPermission] = useState('unknown');
  const [introExpanded, setIntroExpanded] = useState(false);
  const [showAbout, setShowAbout] = useState(() => window.location.hash === '#om');
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(null); // null = full recordingDuration
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [waveformPeaks, setWaveformPeaks] = useState(null);
  const [fadeIn, setFadeIn] = useState(0);
  const [fadeOut, setFadeOut] = useState(0);
  const [expandedChannels, setExpandedChannels] = useState({});
  const [recordingOwnTake, setRecordingOwnTake] = useState(null); // { type, countingIn, countInBeat, countdown } | null
  const [ownTakeError, setOwnTakeError] = useState('');
  const [ownTakeAligningByType, setOwnTakeAligningByType] = useState({}); // { ters, kvint, sext } -> bool, see alignOwnTake
  const [ownTakeWaveformPeaksByType, setOwnTakeWaveformPeaksByType] = useState({}); // { ters, kvint, sext } -> Float32Array, for that channel's own WaveformTrimmer
  // The most recently saved project's *metadata* (no audio — see
  // listProjects), offered via a dismissible banner at mount. Never
  // applied automatically; dismissing just hides the banner; the project
  // itself is real, named, saved work, not a throwaway crash-recovery
  // artifact, so dismissing never deletes it — see "Sessioner" below.
  const [restorableSession, setRestorableSession] = useState(null);
  const [restoringSession, setRestoringSession] = useState(false);
  // Every saved project's metadata (id, name, savedAt, settings — no
  // audio), newest first, for the "Sessioner" list.
  const [projects, setProjects] = useState([]);
  const [projectsExpanded, setProjectsExpanded] = useState(false);
  // Live-monitor: a real-time mic -> pitch-shift -> speaker/headphone loop
  // via SignalsmithStretch's live-input mode (same AudioWorklet library the
  // offline harmony engine uses, just fed live audio instead of buffers).
  // Idle/ready-phase only, deliberately separate from the take-recording
  // mic pipeline below (startRecording) rather than run alongside it — two
  // concurrent getUserMedia/AudioContext graphs on one device is exactly
  // the kind of thing mobile browsers choke on (see the concurrent-
  // AudioContext caution in startRecording). Off by default: it needs
  // headphones (mic->speaker while the mic stays open will otherwise
  // howl) and holds an AudioContext + AudioWorklet open the whole time.
  //
  // Two modes, chosen once at startLiveMonitor() time from whether a
  // tonart is already known (never true in the idle-phase render, always
  // true in the ready-phase one — see liveMonitorKeyAware):
  //  - key-aware: a live pitch-tracking loop (liveMonitorPitchLoop) locks
  //    onto whichever scale degree you're currently singing and derives
  //    the harmony ratio the same way the offline engine does — a *flat*
  //    ratio between two scale-quantized frequencies, applied to the raw
  //    live signal. Flat-per-note (not chasing the raw pitch every frame)
  //    is deliberate, for the same reason buildRatioCurveFromSegments'
  //    doc comment gives: re-deriving the ratio from instantaneous pitch
  //    every frame would cancel your own vibrato instead of carrying it
  //    through.
  //  - fixed-interval fallback (liveMonitorSemitonesFor): no melody to
  //    quantize against yet, so a constant major-scale approximation.
  //
  // Any combination of ters/kvint/sext can run at once (liveMonitorTypes)
  // — one SignalsmithStretch node per active voice, all fed from the same
  // mic source and summed into one mix bus (liveMonitorMixRef), same
  // "multiple voices, one mic" shape as the offline Mixer. Each extra
  // simultaneous voice is a full extra AudioWorklet pitch-shift instance
  // though, so this is real, additive CPU cost — worth keeping in mind as
  // the reason this whole feature defaults off.
  const [liveMonitorOn, setLiveMonitorOn] = useState(false);
  const [liveMonitorStarting, setLiveMonitorStarting] = useState(false);
  const [liveMonitorError, setLiveMonitorError] = useState('');
  const [liveMonitorTypes, setLiveMonitorTypes] = useState(() => ({ ters: true, kvint: false, sext: false }));
  const [liveMonitorLatencyMs, setLiveMonitorLatencyMs] = useState(null);
  // While an own-take recording is running (see recordOwnTake) and the
  // live monitor happens to be on, it temporarily narrows to *just* the
  // voice being recorded so you hear that one stämma — tonart- and
  // "Humanisera"-anpassad, off the same pitch tracker as the offline
  // engine — sung back over your shoulder while you record it. This holds
  // the type being followed (or null); liveMonitorRestoreTypesRef holds
  // the user's own selection to put back afterwards.
  const [liveMonitorFollowingOwnTake, setLiveMonitorFollowingOwnTake] = useState(null);
  const liveMonitorFollowingOwnTakeRef = useRef(null);
  const liveMonitorRestoreTypesRef = useRef(null);
  // Serializes the async stop→start restarts so the "narrow to one voice
  // on the shared capture" at record-start and the "restore the user's
  // own session" at record-end can't interleave (a fast start→cancel
  // during the count-in).
  const liveMonitorOpRef = useRef(Promise.resolve());
  // Re-entrancy guard for startLiveMonitor that a queued restart can rely
  // on synchronously (the liveMonitorStarting state lags a render behind).
  const liveMonitorBusyRef = useRef(false);
  const liveMonitorKeyAware = !!keyInfo;
  const liveMonitorStreamRef = useRef(null);
  const liveMonitorCtxRef = useRef(null);
  const liveMonitorSourceRef = useRef(null);
  const liveMonitorMixRef = useRef(null);
  const liveMonitorNodesRef = useRef({}); // { ters?: StretchNode, kvint?: ..., sext?: ... }
  const liveMonitorAnalyserRef = useRef(null);
  const liveMonitorRafRef = useRef(null);
  const liveMonitorLatencySecRef = useRef(0);
  const liveMonitorTypesRef = useRef(liveMonitorTypes);
  // The scale step (see musicTheory.js) the pitch-tracking loop currently
  // believes you're singing, and how many consecutive ~35ms frames the
  // *candidate* next step has read stably — re-locking (and rescheduling
  // new ratios for every active voice) only once that candidate has been
  // stable for LIVE_MONITOR_LOCK_FRAMES frames, so one noisy
  // autocorrelation reading right at a note boundary can't flap the ratio
  // back and forth.
  const liveMonitorLockedStepRef = useRef(null);
  const liveMonitorCandidateRef = useRef(null); // { step, count } | null
  // A single formant anchor for the whole live session, computed once at
  // start from the already-analyzed take (same formula as the offline
  // engine's formantBaseHzFor — the median melody frequency) rather than
  // re-derived every tick, since it shouldn't drift mid-session.
  const liveMonitorFormantBaseHzRef = useRef(0);
  // Per-voice-type "generation" counter, bumped by createLiveMonitorVoice
  // itself at the *start* of every call (before its await). A rapid
  // off->on->off->on double-toggle of the same voice can leave an earlier
  // call's SignalsmithStretch(...) round-trip still in flight when a later
  // one for the same type has already started; without this, the earlier
  // call's stale resolution would see the type is (once again) wanted and
  // register/connect itself too — silently overwriting the newer node's
  // entry in liveMonitorNodesRef while staying connected and playing, an
  // orphaned "ghost" voice stopLiveMonitor() can no longer find to clean
  // up since it's not in the ref map anymore. Comparing the generation
  // captured at call-start against the current one after the await lets a
  // superseded call disconnect itself instead.
  const liveMonitorVoiceGenRef = useRef({});
  const keyInfoRef = useRef(keyInfo);
  const channelsRef = useRef(channels);
  const humanizeOnRef = useRef(humanizeOn);
  const autotuneEnabled = autotuneOn;
  const effectiveTrimEnd = trimEnd === null ? recordingDuration : trimEnd;

  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const framesRef = useRef([]);
  const rafRef = useRef(null);
  const countInCancelledRef = useRef(false);
  const playCtxRef = useRef(null);
  const activeSourcesRef = useRef([]);
  const mixNodesRef = useRef({});
  // Bumped by every stopPlayback() call (including the one every startMix()
  // does at its own top) and captured as `myGeneration` right after; a
  // startMix() call whose generation no longer matches once its (possibly
  // slow, harmony-render-involving) content resolution finishes has been
  // superseded by a newer play/pause/stop/preview press and must not
  // schedule any audio — otherwise its nodes would start playing orphaned,
  // invisible to activeSourcesRef/mixNodesRef (which the newer call already
  // owns), so stopPlayback() could never reach them again and they'd keep
  // sounding — or silently colliding with whatever plays next — until a
  // full page reload.
  const playGenerationRef = useRef(0);
  const meterRefs = useRef({});
  const meterScratchRef = useRef(null);
  const recordedBufferRef = useRef(null);
  // The key auto-detection actually found, kept even after a manual
  // override (see overrideKey) so "Återställ" can always get back to it
  // without re-running analysis.
  const detectedKeyRef = useRef(null);
  // { buffer, blob } — the last WAV encode done for autosave, so a save
  // triggered by something that isn't the audio itself (a mixer volume
  // nudge, a reverb toggle) doesn't re-encode the same several-hundred-KB
  // buffer from scratch every time. See the autosave effect below.
  const autosaveWavCacheRef = useRef(null);
  // Which saved project autosave is currently updating — null means "no
  // project yet, the next autosave creates one". Set on restore/switch,
  // cleared whenever the working recording is replaced wholesale (a new
  // recording/upload, or the explicit reset button) so that doesn't
  // silently overwrite a previous, differently-named project.
  const activeProjectIdRef = useRef(null);
  const activeProjectNameRef = useRef(null);
  // The untouched decode, kept alongside recordedBufferRef (the "current
  // working" buffer everything else reads from) so normalize/noise
  // reduction — both one-shot, destructive operations — have something to
  // revert back to.
  const originalRecordingBufferRef = useRef(null);
  const playTimeoutRef = useRef(null);
  const recordingStartRef = useRef(0);
  const playheadRafRef = useRef(null);
  const finishedRef = useRef(false);
  const harmonyBuffersRef = useRef({});
  const harmonyRenderPromisesRef = useRef({});
  const autotunedBuffersRef = useRef({}); // keyed by autotuneLevelIndex
  const autotuneRenderPromiseRef = useRef(null);
  const fileInputRef = useRef(null);
  const loopEnabledRef = useRef(false);
  const isPlayingRef = useRef(false);
  const ownTakeBuffersRef = useRef({}); // { ters, kvint, sext } -> AudioBuffer | null
  const ownTakeStreamRef = useRef(null);
  const ownTakeRecorderRef = useRef(null);
  const ownTakeChunksRef = useRef([]);
  const ownTakeRafRef = useRef(null);
  const ownTakeStartRef = useRef(0);
  const ownTakeFinishedRef = useRef(false);
  const ownTakeCountInCancelledRef = useRef(false);
  const reverbBusRef = useRef(null);
  const metronomeSchedulerRef = useRef(null);
  // Mirrors the `metronomeOverride` argument a startMix() call was given,
  // so a loop restart (which only has a ref, not the original call's
  // arguments, to work from) can keep clicking at a "Lyssna"-detected tempo
  // instead of falling back to the separate metronomeEnabled setting.
  const metronomeOverrideRef = useRef(null);

  // Read inside the playback-end setTimeout, which was scheduled back when
  // it fired — using state directly there would capture whatever loopEnabled
  // was at schedule time, missing a toggle flipped mid-playback.
  useEffect(() => {
    loopEnabledRef.current = loopEnabled;
  }, [loopEnabled]);
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  // Same staleness problem as loopEnabledRef, for the same reason: a loop
  // restart calls the startMix() closure that existed when the *previous*
  // iteration scheduled its setTimeout, so reading metronomeEnabled/
  // metronomeBpm directly there would silently use whatever they were when
  // that iteration started — a metronome toggle or BPM change made mid-
  // playback wouldn't take effect until the next manual play, which is
  // exactly the "sometimes it clicks, sometimes it doesn't, even though
  // it's on" symptom this was causing.
  const metronomeEnabledRef = useRef(false);
  const metronomeBpmRef = useRef(100);
  useEffect(() => {
    metronomeEnabledRef.current = metronomeEnabled;
  }, [metronomeEnabled]);
  useEffect(() => {
    metronomeBpmRef.current = metronomeBpm;
  }, [metronomeBpm]);

  // Read by the live-monitor pitch-tracking loop (liveMonitorPitchLoop),
  // which runs across many animation frames and would otherwise close
  // over whatever keyInfo/channels were current when it started.
  useEffect(() => { keyInfoRef.current = keyInfo; }, [keyInfo]);
  useEffect(() => { channelsRef.current = channels; }, [channels]);
  useEffect(() => { humanizeOnRef.current = humanizeOn; }, [humanizeOn]);

  // The browser remembers a granted/denied microphone permission on its own
  // (that's an origin-level browser decision, not something a site can
  // configure) — this just lets the UI react to that state instead of
  // firing off a getUserMedia call that we already know will be denied.
  useEffect(() => {
    if (!navigator.permissions?.query) return undefined;
    let status;
    navigator.permissions.query({ name: 'microphone' }).then((s) => {
      status = s;
      setMicPermission(s.state);
      s.onchange = () => setMicPermission(s.state);
    }).catch(() => {});
    return () => { if (status) status.onchange = null; };
  }, []);

  // Live-monitor only renders on the idle/error (no take yet) and ready
  // (take analyzed) screens — stop it the instant phase moves to anything
  // else (a take starts recording, a file's being analyzed, ...), and
  // unconditionally on unmount, so its mic stream/AudioContext never
  // lingers open past the screen it belongs to.
  useEffect(() => {
    const allowed = phase === 'idle' || phase === 'error' || phase === 'ready';
    if (!allowed && liveMonitorCtxRef.current) stopLiveMonitor();
  }, [phase]);
  useEffect(() => () => stopLiveMonitor(), []);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (playheadRafRef.current) cancelAnimationFrame(playheadRafRef.current);
      if (playTimeoutRef.current) clearTimeout(playTimeoutRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') audioCtxRef.current.close().catch(() => {});
      if (playCtxRef.current && playCtxRef.current.state !== 'closed') playCtxRef.current.close().catch(() => {});
      if (ownTakeRafRef.current) cancelAnimationFrame(ownTakeRafRef.current);
      if (ownTakeStreamRef.current) ownTakeStreamRef.current.getTracks().forEach((t) => t.stop());
      if (metronomeSchedulerRef.current) stopMetronomeScheduler(metronomeSchedulerRef.current);
    };
  }, []);

  // The "Om Stämifier" page lives at #om so it's a real, linkable/
  // bookmarkable URL rather than just app state — the back link and the
  // browser's own back button both work by changing the hash.
  useEffect(() => {
    const onHashChange = () => setShowAbout(window.location.hash === '#om');
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // Loads the saved-projects list once at mount — also offers the newest
  // one via the dismissible restore banner, same "never applied silently"
  // rule as before.
  function refreshProjects() {
    listProjects().then(setProjects).catch(() => {});
  }
  useEffect(() => {
    let cancelled = false;
    listProjects().then((list) => {
      if (cancelled) return;
      setProjects(list);
      if (list.length) setRestorableSession(list[0]);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Autosaves the current recording + settings to IndexedDB (debounced)
  // whenever there's a ready take to save. `waveformPeaks` stands in for
  // "the recorded audio itself changed" — it's recomputed by every path
  // that replaces recordedBufferRef.current (new recording/upload,
  // normalize, noise reduction, revert). Skipped while a previous session
  // is still waiting to be restored/dismissed, so it can't be clobbered
  // before the user decides on it. Updates whichever project is currently
  // "active" (activeProjectIdRef) — a fresh recording/upload clears that
  // ref (see resetSourceState), so the first autosave after one creates a
  // brand new project instead of overwriting whatever was open before.
  useEffect(() => {
    if (phase !== 'ready' || !recordedBufferRef.current || restorableSession) return undefined;
    const buffer = recordedBufferRef.current;
    const timer = setTimeout(() => {
      // A save can be triggered by any of this effect's many dependencies
      // (a mixer volume drag, a reverb toggle) — most of which leave the
      // actual recorded audio untouched, so re-encoding it to WAV every
      // time would be pure waste. Only re-encode when the buffer itself
      // (identity, not deep-equal — every real edit path replaces it
      // wholesale, see applyProcessedRecording/revertToOriginalRecording)
      // has actually changed since the last save.
      let blob;
      if (autosaveWavCacheRef.current?.buffer === buffer) {
        blob = autosaveWavCacheRef.current.blob;
      } else {
        blob = audioBufferToWavBlob(buffer);
        autosaveWavCacheRef.current = { buffer, blob };
      }
      if (!activeProjectNameRef.current) {
        const now = new Date();
        const datePart = now.toLocaleDateString('sv-SE', { day: '2-digit', month: '2-digit' });
        const timePart = now.toLocaleTimeString('sv-SE', { hour: '2-digit', minute: '2-digit' });
        const keyPart = keyInfo ? `${NOTE_NAMES[keyInfo.tonic]}-${keyInfo.mode === 'major' ? 'dur' : 'moll'} · ` : '';
        activeProjectNameRef.current = `${keyPart}${datePart} ${timePart}`;
      }
      saveProject({
        id: activeProjectIdRef.current,
        name: activeProjectNameRef.current,
        version: 1,
        savedAt: Date.now(),
        audio: blob,
        soundType,
        autotuneOn,
        autotuneLevelIndex,
        reverbOn,
        reverbLevelIndex,
        humanizeOn,
        fadeIn,
        fadeOut,
        trimStart,
        trimEnd,
        loopEnabled,
        channels,
      }).then((id) => {
        activeProjectIdRef.current = id;
        refreshProjects();
      }).catch(() => {});
    }, 800);
    return () => clearTimeout(timer);
  }, [
    phase, waveformPeaks, restorableSession, soundType, autotuneOn, autotuneLevelIndex,
    reverbOn, reverbLevelIndex, humanizeOn, fadeIn, fadeOut, trimStart, trimEnd,
    loopEnabled, channels, keyInfo,
  ]);

  function harmonyNotesFor(type) {
    if (!melodyNotes.length) return [];
    const steps = HARMONY_TYPES[type].steps;
    const dir = channels[type].direction;
    return melodyNotes.map((n) => ({ ...n, hStep: n.step + dir * steps }));
  }

  function harmonyPlaybackNotesFor(type) {
    if (!keyInfo) return [];
    return harmonyNotesFor(type).map((n) => ({ start: n.start, end: n.end, midi: scaleStepToMidi(n.hStep, keyInfo.tonic, keyInfo.mode) }));
  }

  const melodyPlaybackNotes = useMemo(() => {
    if (!keyInfo) return [];
    return melodyNotes.map((n) => ({ start: n.start, end: n.end, midi: scaleStepToMidi(n.step, keyInfo.tonic, keyInfo.mode) }));
  }, [melodyNotes, keyInfo]);

  // What the note graph should draw: one layer per harmony channel that's
  // currently turned on in the mixer, each in its own color.
  const harmonyLayers = useMemo(() => {
    if (!melodyNotes.length) return [];
    return HARMONY_KEYS.filter((type) => channels[type].enabled).map((type) => ({
      key: type,
      notes: harmonyNotesFor(type),
      color: HARMONY_COLORS[type].line,
      glow: HARMONY_COLORS[type].glow,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [melodyNotes, channels.ters.enabled, channels.ters.direction, channels.kvint.enabled, channels.kvint.direction, channels.sext.enabled, channels.sext.direction]);

  function setChannelVolume(key, volume) {
    setChannels((prev) => ({ ...prev, [key]: { ...prev[key], volume } }));
    const nodes = mixNodesRef.current[key];
    if (nodes?.gainNode) nodes.gainNode.gain.value = volume;
  }

  function setChannelPan(key, pan) {
    setChannels((prev) => ({ ...prev, [key]: { ...prev[key], pan } }));
    const nodes = mixNodesRef.current[key];
    if (nodes?.pannerNode) nodes.pannerNode.pan.value = pan;
  }

  function setChannelDirection(key, direction) {
    setChannels((prev) => {
      const next = { ...prev, [key]: { ...prev[key], direction } };
      if (isPlaying) restartMix(next);
      return next;
    });
  }

  function toggleChannel(key) {
    setChannels((prev) => {
      const next = { ...prev, [key]: { ...prev[key], enabled: !prev[key].enabled } };
      if (isPlaying) restartMix(next);
      return next;
    });
  }

  // Soloing doesn't touch enabled/mute state, just which of the currently-
  // enabled channels actually make it into the mix (see isChannelAudible)
  // — restarting is the only way to apply that live, same as toggling.
  function toggleChannelSolo(key) {
    setChannels((prev) => {
      const next = { ...prev, [key]: { ...prev[key], solo: !prev[key].solo } };
      if (isPlaying) restartMix(next);
      return next;
    });
  }

  // Volume/pan are collapsed by default per channel — purely a display
  // toggle, doesn't touch playback.
  function toggleChannelExpanded(key) {
    setExpandedChannels((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  // Expand/collapse every channel's volume+pan panel at once. Standard
  // expand-all/collapse-all toggle: if any panel is currently collapsed,
  // the button expands everything; only flips to "collapse all" once
  // every panel is already open.
  const mixerChannelKeys = ['melody', ...HARMONY_KEYS];
  const allChannelsExpanded = mixerChannelKeys.every((k) => expandedChannels[k]);
  function toggleAllChannelsExpanded() {
    const next = !allChannelsExpanded;
    setExpandedChannels((prev) => {
      const updated = { ...prev };
      mixerChannelKeys.forEach((k) => { updated[k] = next; });
      return updated;
    });
  }

  // Renders (or returns the cached render of) a pitch-corrected copy of the
  // recording at the currently selected autotune strength. Cached per
  // level index — switching between lätt/medel/hård keeps each one's
  // render around instead of re-rendering on every slider move — and
  // reused as the *source* for harmony rendering too when autotune is on,
  // so the two voices stay in tune with each other, not just with an
  // idealized target neither of them actually sings.
  async function getAutotunedBuffer() {
    if (!recordedBufferRef.current || !melodyNotes.length || !keyInfo) return null;
    const levelIndex = autotuneLevelIndex;
    const level = AUTOTUNE_LEVELS[levelIndex];
    if (!level) return null;
    const cached = autotunedBuffersRef.current[levelIndex];
    if (cached) return cached;
    if (autotuneRenderPromiseRef.current?.levelIndex === levelIndex) {
      return autotuneRenderPromiseRef.current.promise;
    }
    setAutotuneRendering(true);
    setAutotuneRenderError('');
    const promise = renderAutotunedMelody(recordedBufferRef.current, melodyNotes, keyInfo, level.amount)
      .then((buffer) => {
        autotunedBuffersRef.current[levelIndex] = buffer;
        return buffer;
      })
      .catch((err) => {
        setAutotuneRenderError('Kunde inte auto-tuna inspelningen. Testa igen, eller stäng av autotune.');
        throw err;
      })
      .finally(() => {
        setAutotuneRendering(false);
        autotuneRenderPromiseRef.current = null;
      });
    autotuneRenderPromiseRef.current = { levelIndex, promise };
    return promise;
  }

  // The buffer that "Din röst" playback/export and the harmony engine
  // should actually use: the raw recording, or — if autotune is on — the
  // corrected version of it.
  async function getSourceBuffer() {
    if (!autotuneEnabled) return recordedBufferRef.current;
    const buffer = await getAutotunedBuffer();
    return buffer || recordedBufferRef.current;
  }

  // Renders (or returns the cached render of) one harmony type's real,
  // pitch-shifted copy of the singer's own recording (or its autotuned
  // version). Cached per type, keyed by that type's own direction+autotune
  // combination — each of ters/kvint/sext can be active at once, each with
  // its own direction, so each needs its own independent cache slot.
  async function getHarmonyBufferFor(type) {
    if (!recordedBufferRef.current || !melodyNotes.length || !keyInfo) return null;
    const sourceBuffer = await getSourceBuffer();
    if (!sourceBuffer) return null;
    const key = `${channels[type].direction}-${autotuneEnabled ? `at${autotuneLevelIndex}` : 'raw'}-${humanizeOn ? 'hum' : 'plain'}`;
    const cached = harmonyBuffersRef.current[type];
    if (cached && cached.key === key) return cached.buffer;
    if (harmonyRenderPromisesRef.current[type] && cached?.key === `pending-${key}`) {
      return harmonyRenderPromisesRef.current[type];
    }
    harmonyBuffersRef.current[type] = { key: `pending-${key}`, buffer: null };
    setHarmonyRenderingByType((s) => ({ ...s, [type]: true }));
    setHarmonyRenderErrorsByType((s) => ({ ...s, [type]: '' }));
    const hNotes = harmonyNotesFor(type);
    const promise = renderHarmonyOffline(sourceBuffer, melodyNotes, hNotes, keyInfo, humanizeOn ? type : null)
      .then((buffer) => {
        harmonyBuffersRef.current[type] = { key, buffer };
        return buffer;
      })
      .catch((err) => {
        harmonyBuffersRef.current[type] = null;
        setHarmonyRenderErrorsByType((s) => ({ ...s, [type]: `Kunde inte bygga ${HARMONY_TYPES[type].label.toLowerCase()}-stämman.` }));
        throw err;
      })
      .finally(() => {
        setHarmonyRenderingByType((s) => ({ ...s, [type]: false }));
        delete harmonyRenderPromisesRef.current[type];
      });
    harmonyRenderPromisesRef.current[type] = promise;
    return promise;
  }

  // Clears everything tied to the current recording/upload, so a fresh
  // source starts from a clean slate. Shared by a new recording, a new
  // upload, and the explicit reset button.
  function resetSourceState() {
    setKeyInfo(null);
    detectedKeyRef.current = null;
    setMelodyNotes([]);
    setChannels(defaultChannels());
    setSoundType('recording');
    setVoiceReady(false);
    setAutotuneOn(false);
    setAutotuneLevelIndex(0);
    setAutotuneStrengthExpanded(false);
    setExpandedChannels({});
    setTempoDetected(false);
    if (metronomeListening) {
      stopMetronomeScheduler(metronomeSchedulerRef.current);
      metronomeSchedulerRef.current = null;
      setMetronomeListening(false);
    }
    recordedBufferRef.current = null;
    originalRecordingBufferRef.current = null;
    setIsProcessed(false);
    setNormalizing(false);
    setNormalizeError('');
    setNoiseReductionMode(false);
    setNoiseSampleStart(0);
    setNoiseSampleEnd(0.5);
    setDenoising(false);
    setDenoiseError('');
    setNoiseReductionApplied(false);
    harmonyBuffersRef.current = {};
    harmonyRenderPromisesRef.current = {};
    setHarmonyRenderingByType({});
    setHarmonyRenderErrorsByType({});
    autotunedBuffersRef.current = {};
    setAutotuneRenderError('');
    setTrimStart(0);
    setTrimEnd(null);
    setLoopEnabled(false);
    setWaveformPeaks(null);
    setFadeIn(0);
    setFadeOut(0);
    if (recordingOwnTake) {
      ownTakeFinishedRef.current = true; // suppress the in-flight recording's own completion
      if (ownTakeRafRef.current) cancelAnimationFrame(ownTakeRafRef.current);
      if (ownTakeStreamRef.current) ownTakeStreamRef.current.getTracks().forEach((t) => t.stop());
      if (ownTakeRecorderRef.current && ownTakeRecorderRef.current.state !== 'inactive') {
        ownTakeRecorderRef.current.onstop = null;
        ownTakeRecorderRef.current.stop();
      }
      setRecordingOwnTake(null);
    }
    ownTakeBuffersRef.current = {};
    setOwnTakeError('');
    setOwnTakeAligningByType({});
    setOwnTakeWaveformPeaksByType({});
    stopPlayback();
    // The previous take is being *replaced* (new recording/upload, or an
    // explicit reset), not deleted — it's still real, named, saved work,
    // sitting in "Sessioner" for later. Just detach autosave from it so
    // the next save creates a fresh project instead of overwriting it.
    activeProjectIdRef.current = null;
    activeProjectNameRef.current = null;
  }

  // Downsamples a recording into per-column peak amplitudes for the
  // waveform view — computed once per recording rather than redrawing
  // straight from raw sample data on every render/drag.
  function computeWaveformPeaks(buffer, numPeaks = 400) {
    const data = buffer.getChannelData(0);
    const blockSize = Math.max(1, Math.floor(data.length / numPeaks));
    const peaks = new Float32Array(numPeaks);
    for (let i = 0; i < numPeaks; i++) {
      const start = i * blockSize;
      const end = Math.min(data.length, start + blockSize);
      let max = 0;
      for (let j = start; j < end; j++) {
        const v = Math.abs(data[j]);
        if (v > max) max = v;
      }
      peaks[i] = max;
    }
    return peaks;
  }

  // Any processing applied to the recording (normalize, noise reduction)
  // invalidates buffers derived from the old version of it, and needs a
  // redrawn waveform + a stopped mix (playing nodes already reference the
  // old buffer).
  function applyProcessedRecording(newBuffer) {
    recordedBufferRef.current = newBuffer;
    setIsProcessed(true);
    setWaveformPeaks(computeWaveformPeaks(newBuffer));
    harmonyBuffersRef.current = {};
    autotunedBuffersRef.current = {};
    if (isPlaying) stopPlayback();
  }

  function revertToOriginalRecording() {
    if (!originalRecordingBufferRef.current) return;
    recordedBufferRef.current = originalRecordingBufferRef.current;
    setIsProcessed(false);
    setWaveformPeaks(computeWaveformPeaks(originalRecordingBufferRef.current));
    harmonyBuffersRef.current = {};
    autotunedBuffersRef.current = {};
    setNormalizeError('');
    setDenoiseError('');
    setNoiseReductionApplied(false);
    if (isPlaying) stopPlayback();
  }

  // Peak-normalizes `source` to `targetPeak` (just under full scale by
  // default) and returns a new buffer — shared by the main recording's
  // manual "Normalisera" button and the automatic own-take normalization
  // below. Returns null for a silent (or already-empty) source rather than
  // dividing by ~0 into a wall of noise.
  function peakNormalizeBuffer(ctx, source, targetPeak = 0.97, silenceThreshold = 0.0005) {
    let peak = 0;
    for (let ch = 0; ch < source.numberOfChannels; ch++) {
      const data = source.getChannelData(ch);
      for (let i = 0; i < data.length; i++) {
        const abs = Math.abs(data[i]);
        if (abs > peak) peak = abs;
      }
    }
    if (peak <= silenceThreshold) return null;
    const gain = targetPeak / peak;
    const outBuffer = ctx.createBuffer(source.numberOfChannels, source.length, source.sampleRate);
    for (let ch = 0; ch < source.numberOfChannels; ch++) {
      const data = source.getChannelData(ch);
      const out = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) out[i] = data[i] * gain;
      outBuffer.copyToChannel(out, ch);
    }
    return outBuffer;
  }

  // Peak-normalizes the current recording to just under full scale.
  // One-shot and destructive (see revertToOriginalRecording for the way
  // back), matching how "normalize" behaves in a regular audio editor.
  async function normalizeRecording() {
    const source = recordedBufferRef.current;
    if (!source || normalizing) return;
    setNormalizing(true);
    setNormalizeError('');
    try {
      const ctx = await getPlaybackContext();
      const outBuffer = peakNormalizeBuffer(ctx, source);
      if (!outBuffer) {
        setNormalizeError('Inspelningen verkar vara helt tyst — går inte att normalisera.');
        return;
      }
      applyProcessedRecording(outBuffer);
    } catch (err) {
      setNormalizeError('Kunde inte normalisera inspelningen. Testa igen.');
    } finally {
      setNormalizing(false);
    }
  }

  // Turning noise-reduction mode on (from the waveform's fullscreen view)
  // pre-selects a plausible noise sample via detectNoiseRegion — the user
  // can drag the handles from there instead of starting from nothing.
  function enterNoiseReductionMode() {
    const source = recordedBufferRef.current;
    if (source) {
      const region = detectNoiseRegion(source.getChannelData(0), source.sampleRate, source.duration);
      setNoiseSampleStart(region.start);
      setNoiseSampleEnd(region.end);
    }
    setNoiseReductionMode(true);
  }

  // Learns a noise profile from [noiseSampleStart, noiseSampleEnd) and
  // subtracts it from the whole recording via STFT/overlap-add (see
  // spectralSubtractChannel). One-shot and destructive like normalize —
  // always runs against the CURRENT recordedBufferRef, so normalize +
  // denoise compose in whichever order they're applied, and re-running
  // denoise with adjusted handles re-processes from there again.
  async function applyNoiseReduction() {
    const source = recordedBufferRef.current;
    if (!source || denoising) return;
    const start = Math.max(0, Math.min(noiseSampleStart, source.duration));
    const end = Math.max(start + 0.05, Math.min(noiseSampleEnd, source.duration));
    setDenoising(true);
    setDenoiseError('');
    try {
      const fftSize = 2048;
      const window = hannWindow(fftSize);
      const profile = computeNoiseProfile(source.getChannelData(0), source.sampleRate, start, end, fftSize, window);
      const ctx = await getPlaybackContext();
      const outBuffer = ctx.createBuffer(source.numberOfChannels, source.length, source.sampleRate);
      for (let ch = 0; ch < source.numberOfChannels; ch++) {
        const processed = spectralSubtractChannel(source.getChannelData(ch), profile, fftSize, window);
        outBuffer.copyToChannel(processed, ch);
      }
      applyProcessedRecording(outBuffer);
      setNoiseReductionApplied(true);
    } catch (err) {
      setDenoiseError('Kunde inte brusreducera inspelningen. Testa igen, eller välj en annan del som brusprov.');
    } finally {
      setDenoising(false);
    }
  }

  // Fixed semitone shift for a live-monitor interval — the *major-scale*
  // diatonic distance for that harmony type (e.g. ters -> a major third),
  // regardless of an actual tonart: there's no analyzed melody yet to
  // quantize against while the mic is live, so this is a reasonable
  // constant approximation rather than the real per-note diatonic
  // interval the offline harmony engine computes (see buildRatioCurve).
  function liveMonitorSemitonesFor(type) {
    const t = HARMONY_TYPES[type];
    return semitoneForStep(t.steps, MAJOR_INTERVALS) * t.defaultDirection;
  }

  // Schedules the fixed-interval fallback for one voice, using an
  // already-known latency (liveMonitorLatencySecRef) rather than
  // re-awaiting node.latency() — see createLiveMonitorVoice for the one
  // place per session that actually needs the async round-trip.
  function scheduleLiveMonitorFixedInterval(node, ctx, type) {
    node.schedule({
      output: ctx.currentTime + liveMonitorLatencySecRef.current,
      active: true,
      semitones: liveMonitorSemitonesFor(type),
      formantCompensation: true,
      formantBaseHz: 0, // 0 = let the node pitch-track the live voice itself
    });
  }

  // Same cents formula as harmonyEngine.js's humanizeKeyframes (detune +
  // slow wobble + faster vibrato, all in cents so they compose regardless
  // of interval size), just evaluated at a single instant instead of
  // pre-baked into keyframes — see startLiveMonitorPitchLoop for why this
  // has to be called continuously rather than once per note-lock. `t` is
  // the live AudioContext clock, not recording-relative time; that only
  // shifts the wobble/vibrato's phase, which doesn't matter since nothing
  // needs to line up with an offline render here.
  function liveMonitorHumanizeFactor(type, t) {
    const profile = HARMONY_HUMANIZE_PROFILES[type];
    if (!profile) return 1;
    const { detuneCents = 0, wobbleCents = 0, wobbleHz = 0.25, phase = 0, vibratoCents = 0, vibratoHz = 5.5, vibratoPhase = 0 } = profile;
    const cents = detuneCents
      + wobbleCents * Math.sin(2 * Math.PI * wobbleHz * t + phase)
      + vibratoCents * Math.sin(2 * Math.PI * vibratoHz * t + vibratoPhase);
    return Math.pow(2, cents / 1200);
  }

  // Schedules a diatonic ratio for `step` (the scale degree the
  // pitch-tracking loop currently believes you're singing) shifted to
  // `type`'s interval, on `type`'s own voice node — the live-monitor
  // equivalent of one segment in harmonyEngine.js's buildRatioCurve,
  // computed on the fly instead of from a whole pre-analyzed take. Both
  // frequencies come from scaleStepToMidi (not the raw measured pitch),
  // so the *base* ratio is a pure function of which note you're on and
  // stays flat while you hold or vibrato it — applying that flat ratio to
  // the raw live signal is what carries your own vibrato through instead
  // of cancelling it (see the LIVE_MONITOR_* constants' comment).
  // humanizeOnRef then layers this type's fixed detune/wobble/vibrato/
  // formant character on top, same profile the offline engine gives that
  // voice, so e.g. "ters" sounds like the same ters here as in the
  // exported mix — and, singing all three at once, actually sounds like
  // three different voices rather than one voice tripled.
  function scheduleLiveMonitorRatio(node, ctx, step, type) {
    const key = keyInfoRef.current;
    if (!key) return;
    const dir = channelsRef.current[type]?.direction ?? HARMONY_TYPES[type].defaultDirection;
    const hStep = step + dir * HARMONY_TYPES[type].steps;
    const melodyFreq = midiToFreq(scaleStepToMidi(step, key.tonic, key.mode));
    const harmonyFreq = midiToFreq(scaleStepToMidi(hStep, key.tonic, key.mode));
    const humanize = humanizeOnRef.current;
    const ratio = (harmonyFreq / melodyFreq) * (humanize ? liveMonitorHumanizeFactor(type, ctx.currentTime) : 1);
    node.schedule({
      output: ctx.currentTime + liveMonitorLatencySecRef.current,
      active: true,
      semitones: 12 * Math.log2(ratio),
      formantCompensation: true,
      formantSemitones: humanize ? (HARMONY_HUMANIZE_PROFILES[type]?.formantSemitones || 0) : 0,
      formantBaseHz: liveMonitorFormantBaseHzRef.current,
    });
  }

  // Runs for the lifetime of a key-aware live-monitor session: samples
  // the mic every LIVE_MONITOR_SAMPLE_HOP_SEC via the same autocorrelation
  // pitch tracker startRecording uses for the melody, and re-locks onto a
  // new scale step once it's read LIVE_MONITOR_LOCK_FRAMES times in a row
  // (debouncing single noisy frames right at a note boundary). On a lock
  // change (or every tick, while humanize is on — see below), reschedules
  // *every currently active voice* from liveMonitorNodesRef, not just one.
  // Silence/unvoiced frames are simply skipped for locking purposes — the
  // last locked step holds, which is harmless for a live preview (unlike
  // the offline engine, this doesn't need to fade to "no shift" for a
  // real pause). While humanize is on, voices are rescheduled on *every*
  // tick regardless of whether the note changed — the wobble/vibrato
  // terms are time-varying and need continuous updates to be audible
  // while a note is held, the same reason harmonyEngine.js's
  // humanizeKeyframes inserts extra points through a long-held note
  // rather than relying on just the note's two endpoints. Keyed off
  // liveMonitorAnalyserRef (one shared analyser for the whole session,
  // regardless of how many voices are active) rather than any one voice
  // node, so it keeps running as voices are toggled on/off mid-session.
  function startLiveMonitorPitchLoop(ctx, analyser) {
    const buf = new Float32Array(analyser.fftSize);
    let lastSampleTime = -Infinity;
    const tick = () => {
      if (liveMonitorAnalyserRef.current !== analyser) return; // torn down mid-loop
      const now = ctx.currentTime;
      if (now - lastSampleTime >= LIVE_MONITOR_SAMPLE_HOP_SEC) {
        lastSampleTime = now;
        const key = keyInfoRef.current;
        if (key) {
          let justLocked = false;
          analyser.getFloatTimeDomainData(buf);
          const { freq, rms } = autoCorrelate(buf, ctx.sampleRate);
          if (freq > 55 && freq < 1200 && rms > LIVE_MONITOR_RMS_THRESHOLD) {
            const step = midiToScaleStep(freqToMidi(freq), key.tonic, key.mode);
            const candidate = liveMonitorCandidateRef.current;
            liveMonitorCandidateRef.current = candidate && candidate.step === step
              ? { step, count: candidate.count + 1 }
              : { step, count: 1 };
            if (liveMonitorCandidateRef.current.count >= LIVE_MONITOR_LOCK_FRAMES
              && liveMonitorLockedStepRef.current !== step) {
              liveMonitorLockedStepRef.current = step;
              justLocked = true;
            }
          }
          if (liveMonitorLockedStepRef.current != null && (justLocked || humanizeOnRef.current)) {
            const lockedStep = liveMonitorLockedStepRef.current;
            Object.entries(liveMonitorNodesRef.current).forEach(([voiceType, voiceNode]) => {
              scheduleLiveMonitorRatio(voiceNode, ctx, lockedStep, voiceType);
            });
          }
        }
      }
      liveMonitorRafRef.current = requestAnimationFrame(tick);
    };
    liveMonitorRafRef.current = requestAnimationFrame(tick);
  }

  function stopLiveMonitor() {
    setLiveMonitorOn(false);
    setLiveMonitorLatencyMs(null);
    if (liveMonitorRafRef.current) {
      cancelAnimationFrame(liveMonitorRafRef.current);
      liveMonitorRafRef.current = null;
    }
    liveMonitorLockedStepRef.current = null;
    liveMonitorCandidateRef.current = null;
    if (liveMonitorAnalyserRef.current) {
      liveMonitorAnalyserRef.current.disconnect();
      liveMonitorAnalyserRef.current = null;
    }
    Object.values(liveMonitorNodesRef.current).forEach((node) => node.disconnect());
    liveMonitorNodesRef.current = {};
    if (liveMonitorMixRef.current) {
      liveMonitorMixRef.current.disconnect();
      liveMonitorMixRef.current = null;
    }
    liveMonitorSourceRef.current = null;
    if (liveMonitorCtxRef.current && liveMonitorCtxRef.current.state !== 'closed') {
      liveMonitorCtxRef.current.close().catch(() => {});
    }
    liveMonitorCtxRef.current = null;
    if (liveMonitorStreamRef.current) {
      liveMonitorStreamRef.current.getTracks().forEach((t) => t.stop());
      liveMonitorStreamRef.current = null;
    }
  }

  // Creates and wires up one voice's node: source -> node -> the shared
  // mix bus, started, registered in liveMonitorNodesRef under `type`. Does
  // NOT schedule an interval itself — callers decide fixed-interval vs.
  // diatonic depending on whether a tonart/locked note is available yet.
  // SignalsmithStretch(...) is itself an async worklet round-trip, so a
  // session can be torn down (stopLiveMonitor, e.g. because phase moved
  // on mid-start) while this is still in flight — checked here, not just
  // by callers afterward, since callers only learn the node exists once
  // this returns: registering/connecting it regardless would leave a
  // node wired into a mix bus and AudioContext that no longer exist. Also
  // re-checks liveMonitorTypesRef for this specific `type`, not just the
  // ctx: a quick toggle-on-then-off of the *same* voice while this await
  // is in flight leaves ctx untouched (session still running) but the
  // voice no longer wanted — without this check the node would still get
  // wired in and started, leaving an inaudible-in-the-UI "ghost" voice
  // actually playing since liveMonitorTypesRef flips synchronously
  // (before the await) while this only learns about it after.
  async function createLiveMonitorVoice(ctx, source, mix, type) {
    const myGen = (liveMonitorVoiceGenRef.current[type] || 0) + 1;
    liveMonitorVoiceGenRef.current = { ...liveMonitorVoiceGenRef.current, [type]: myGen };
    const node = await SignalsmithStretch(ctx, { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    if (liveMonitorCtxRef.current !== ctx || !liveMonitorTypesRef.current[type] || liveMonitorVoiceGenRef.current[type] !== myGen) {
      node.disconnect();
      return node;
    }
    liveMonitorNodesRef.current = { ...liveMonitorNodesRef.current, [type]: node };
    source.connect(node);
    node.connect(mix);
    node.start();
    return node;
  }

  // Keeps the mix bus from clipping as more simultaneous voices are
  // summed onto it — equal-power-ish headroom (1/sqrt(n)) per active
  // voice, recomputed on the fly as voices are toggled (see
  // toggleLiveMonitorVoice), with a short ramp so a mid-session change
  // isn't an audible step.
  function updateLiveMonitorMixGain(ctx, mix) {
    const activeCount = HARMONY_KEYS.filter((t) => liveMonitorTypesRef.current[t] && liveMonitorNodesRef.current[t]).length;
    mix.gain.setTargetAtTime(activeCount ? 1 / Math.sqrt(activeCount) : 0, ctx.currentTime, 0.02);
  }

  // `externalStream` (a clone of the own-take recording's mic stream — see
  // followOwnTakeWithLiveMonitor) lets the monitor run without opening a
  // second getUserMedia capture of its own: iOS Safari tends to interrupt
  // an existing capture the moment a new one with different constraints
  // starts, which would silently kill whichever of "recording" and
  // "monitoring" started first. Sharing one capture (via .clone(), which
  // doesn't open a new device stream) sidesteps that entirely. We still
  // own the clone and stop it in stopLiveMonitor like any other stream.
  async function startLiveMonitor(externalStream = null) {
    if (liveMonitorBusyRef.current || liveMonitorCtxRef.current) return;
    liveMonitorBusyRef.current = true;
    setLiveMonitorError('');
    if (!externalStream && (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)) {
      setLiveMonitorError('Den här webbläsaren stödjer inte live-förhandslyssning.');
      liveMonitorBusyRef.current = false;
      return;
    }
    if (!externalStream && micPermission === 'denied') {
      setLiveMonitorError('Mikrofonåtkomst är blockerad för den här sidan.');
      liveMonitorBusyRef.current = false;
      return;
    }
    const activeTypes = HARMONY_KEYS.filter((t) => liveMonitorTypesRef.current[t]);
    if (activeTypes.length === 0) {
      setLiveMonitorError('Välj minst en stämma (ters, kvint eller sext) att förhandslyssna.');
      liveMonitorBusyRef.current = false;
      return;
    }
    setLiveMonitorStarting(true);
    try {
      const stream = externalStream || await navigator.mediaDevices.getUserMedia({ audio: true });
      liveMonitorStreamRef.current = stream;
      const AC = window.AudioContext || window.webkitAudioContext;
      const ctx = new AC({ latencyHint: 'interactive' });
      liveMonitorCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      liveMonitorSourceRef.current = source;
      const mix = ctx.createGain();
      liveMonitorMixRef.current = mix;
      mix.connect(ctx.destination);

      // The first voice's latency() is a real async worklet round-trip —
      // awaiting it here (rather than e.g. adding it directly to
      // ctx.currentTime) is load-bearing: a Promise coerced into
      // arithmetic silently stringifies instead of throwing, which turned
      // `output` into "12.3[object Promise]" and made every scheduled
      // change a silent no-op. Every other voice (here or added later via
      // toggleLiveMonitorVoice) reuses this same cached value instead of
      // re-awaiting — they share one AudioContext/worklet config, so the
      // latency is the same for all of them.
      const firstType = activeTypes[0];
      const firstNode = await createLiveMonitorVoice(ctx, source, mix, firstType);
      if (liveMonitorCtxRef.current !== ctx) return; // torn down while awaiting
      liveMonitorLatencySecRef.current = await firstNode.latency();
      if (liveMonitorCtxRef.current !== ctx) return; // torn down while awaiting
      scheduleLiveMonitorFixedInterval(firstNode, ctx, firstType);
      for (const type of activeTypes.slice(1)) {
        const node = await createLiveMonitorVoice(ctx, source, mix, type);
        if (liveMonitorCtxRef.current !== ctx) return; // torn down while awaiting
        scheduleLiveMonitorFixedInterval(node, ctx, type);
      }
      updateLiveMonitorMixGain(ctx, mix);

      // A pitch-tracking analyser tap, separate from any voice's own
      // audio path (source -> node -> mix is untouched) — this is a side
      // branch purely for autoCorrelate to read from.
      if (keyInfoRef.current) {
        liveMonitorFormantBaseHzRef.current = melodyNotes.length ? formantBaseHzFor(melodyNotes, keyInfoRef.current) : 0;
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        liveMonitorAnalyserRef.current = analyser;
        startLiveMonitorPitchLoop(ctx, analyser);
      }
      setLiveMonitorLatencyMs(Math.round(liveMonitorLatencySecRef.current * 1000));
      setLiveMonitorOn(true);
    } catch (err) {
      setLiveMonitorError('Kunde inte starta live-förhandslyssningen. Kontrollera mikrofonbehörigheten.');
      stopLiveMonitor();
    } finally {
      setLiveMonitorStarting(false);
      liveMonitorBusyRef.current = false;
    }
  }

  // Toggles one voice (ters/kvint/sext) on or off. Before the monitor is
  // running, this just updates which voices the next startLiveMonitor()
  // call will use. While it's already running, adds or removes that
  // voice's node live, on the same shared mic source/mix bus, without
  // touching the other active voices or restarting anything.
  async function toggleLiveMonitorVoice(type) {
    const next = !liveMonitorTypesRef.current[type];
    liveMonitorTypesRef.current = { ...liveMonitorTypesRef.current, [type]: next };
    setLiveMonitorTypes(liveMonitorTypesRef.current);
    const ctx = liveMonitorCtxRef.current;
    const mix = liveMonitorMixRef.current;
    if (!liveMonitorOn || !ctx || !mix) return; // not running — selection just takes effect next start
    if (!next) {
      const node = liveMonitorNodesRef.current[type];
      if (node) {
        node.disconnect();
        const rest = { ...liveMonitorNodesRef.current };
        delete rest[type];
        liveMonitorNodesRef.current = rest;
      }
      updateLiveMonitorMixGain(ctx, mix);
      return;
    }
    const node = await createLiveMonitorVoice(ctx, liveMonitorSourceRef.current, mix, type);
    if (liveMonitorNodesRef.current[type] !== node) return; // torn down while awaiting
    updateLiveMonitorMixGain(ctx, mix);
    if (liveMonitorLockedStepRef.current != null && keyInfoRef.current) {
      scheduleLiveMonitorRatio(node, ctx, liveMonitorLockedStepRef.current, type);
    } else {
      scheduleLiveMonitorFixedInterval(node, ctx, type);
    }
  }

  // Points liveMonitorTypesRef (and the state mirror the card renders at)
  // at an exact { ters, kvint, sext } map. Used by the own-take follow to
  // narrow to one voice and to put the user's selection back — a plain
  // assignment because the follow always restarts the monitor right after
  // (a fresh startLiveMonitor reads the ref), rather than diffing voices
  // onto a live session.
  function setLiveMonitorTypeSet(next) {
    liveMonitorTypesRef.current = { ...next };
    setLiveMonitorTypes(liveMonitorTypesRef.current);
  }

  // Serializes stop→start restarts of the monitor so a record-start
  // narrow and a record-end restore can't run concurrently.
  function queueLiveMonitorOp(fn) {
    liveMonitorOpRef.current = liveMonitorOpRef.current.catch(() => {}).then(fn);
    return liveMonitorOpRef.current;
  }

  // Called at own-take record start (see recordOwnTake): if the live
  // monitor is on, remember the user's own voice selection, narrow to
  // just `type`, and restart the monitor onto a *clone of the recording's
  // own mic stream* (see startLiveMonitor's externalStream note) so only
  // one capture is open. No-op when the monitor is off — this is opt-in
  // via that toggle (it needs headphones, same as the monitor itself), it
  // doesn't force the monitor on.
  function followOwnTakeWithLiveMonitor(type, sourceStream) {
    if (!liveMonitorOn || liveMonitorFollowingOwnTakeRef.current) return;
    // Needs its own stoppable copy of the capture (see startLiveMonitor's
    // externalStream note) — bail rather than risk stopLiveMonitor later
    // killing the recording's stream if clone() somehow isn't there.
    if (!sourceStream || typeof sourceStream.clone !== 'function') return;
    const clone = sourceStream.clone();
    liveMonitorRestoreTypesRef.current = { ...liveMonitorTypesRef.current };
    liveMonitorFollowingOwnTakeRef.current = type;
    setLiveMonitorFollowingOwnTake(type);
    setLiveMonitorTypeSet({ ters: type === 'ters', kvint: type === 'kvint', sext: type === 'sext' });
    queueLiveMonitorOp(async () => {
      stopLiveMonitor();
      await startLiveMonitor(clone);
    });
  }

  // Called on every own-take recording exit (via the recordingOwnTake
  // effect below, so count-in cancel / early stop / natural end / a
  // resetSourceState abort all land here) — put the user's own voice
  // selection back and, if the monitor is still running, restart it onto
  // its own fresh capture. If the user turned the monitor off meanwhile
  // the restart is skipped; the restored selection just applies next time
  // they start it.
  function restoreLiveMonitorAfterOwnTake() {
    if (!liveMonitorFollowingOwnTakeRef.current) return;
    const restore = liveMonitorRestoreTypesRef.current;
    liveMonitorFollowingOwnTakeRef.current = null;
    liveMonitorRestoreTypesRef.current = null;
    setLiveMonitorFollowingOwnTake(null);
    if (restore) setLiveMonitorTypeSet(restore);
    queueLiveMonitorOp(async () => {
      const wasRunning = !!liveMonitorCtxRef.current;
      stopLiveMonitor();
      if (wasRunning) await startLiveMonitor();
    });
  }

  // Every own-take recording exit funnels through recordingOwnTake going
  // back to null (natural end, early stop, count-in cancel, or a
  // resetSourceState abort) — the one place to undo followOwnTakeWith-
  // LiveMonitor from. restoreLiveMonitorAfterOwnTake is a no-op unless a
  // follow is actually active, so the null on mount is harmless.
  useEffect(() => {
    if (!recordingOwnTake) restoreLiveMonitorAfterOwnTake();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingOwnTake]);

  async function startRecording() {
    setErrorMsg('');
    resetSourceState();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setPhase('error');
      setErrorMsg('Den här webbläsaren stödjer inte mikrofoninspelning.');
      return;
    }
    if (micPermission === 'denied') {
      setPhase('error');
      setErrorMsg('Mikrofonåtkomst är blockerad för den här sidan. Ändra behörighet i webbläsarens inställningar och försök igen.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // A count-in gets its own click scheduler below if the ongoing
      // metronome is enabled too — stop any "lyssna"-preview loop from the
      // idle screen first either way, so it never keeps clicking
      // (unscheduled, off-tempo-aware) right through the count-in and into
      // the take.
      if (metronomeSchedulerRef.current) {
        stopMetronomeScheduler(metronomeSchedulerRef.current);
        metronomeSchedulerRef.current = null;
      }
      setMetronomeListening(false);

      // A short audible count-in (same click, same tempo as the metronome)
      // so the singer knows exactly when recording actually starts, rather
      // than needing to react the instant they press the button.
      countInCancelledRef.current = false;
      setPhase('countin');
      setCountInBeat(0);
      const clickCtx = await getPlaybackContext();
      await playCountIn(clickCtx, metronomeBpm, setCountInBeat);
      if (countInCancelledRef.current) return;

      // Belt-and-suspenders: a previous recording's context should already
      // be closed by finishRecording, but never leave a live one behind
      // before opening another — see the finally block there for why this
      // matters (mobile browsers cap concurrent AudioContexts).
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close().catch(() => {});
      }
      const AC = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AC();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      if (window.MediaRecorder) {
        chunksRef.current = [];
        try {
          const mr = new MediaRecorder(stream);
          mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
          mediaRecorderRef.current = mr;
          mr.start();
        } catch (e) {
          mediaRecorderRef.current = null;
        }
      }

      framesRef.current = [];
      const startTime = audioCtx.currentTime;
      recordingStartRef.current = startTime;
      finishedRef.current = false;
      let lastSampleTime = -1;
      const buf = new Float32Array(analyser.fftSize);
      setPhase('recording');
      setCountdown(DURATION);

      if (metronomeEnabled) {
        metronomeSchedulerRef.current = startMetronomeScheduler(clickCtx, metronomeBpm);
      }

      const tick = () => {
        const now = audioCtx.currentTime - startTime;
        if (now >= DURATION) {
          finishRecording();
          return;
        }
        setCountdown(Math.max(0, DURATION - now));
        if (now - lastSampleTime >= 0.035) {
          analyser.getFloatTimeDomainData(buf);
          const { freq, rms } = autoCorrelate(buf, audioCtx.sampleRate);
          framesRef.current.push({ t: now, freq: freq > 55 && freq < 1200 ? freq : -1, rms });
          lastSampleTime = now;
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (err) {
      setPhase('error');
      setErrorMsg('Kunde inte komma åt mikrofonen. Kontrollera att appen har mikrofonbehörighet i webbläsaren.');
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    }
  }

  // Bails out of the count-in before the actual mic recording has started
  // — the mic stream was already opened (to fail fast on a denied
  // permission before the count-in plays), so it needs closing here too.
  function cancelCountIn() {
    countInCancelledRef.current = true;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setPhase('idle');
  }

  function finishRecording() {
    if (finishedRef.current) return;
    finishedRef.current = true;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (metronomeSchedulerRef.current) {
      stopMetronomeScheduler(metronomeSchedulerRef.current);
      metronomeSchedulerRef.current = null;
    }

    const audioCtx = audioCtxRef.current;
    const elapsed = audioCtx ? audioCtx.currentTime - recordingStartRef.current : DURATION;
    setRecordingDuration(Math.max(0.3, Math.min(DURATION, elapsed)));

    setPhase('analyzing');
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    discardPlaybackContext();

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: (chunksRef.current[0] && chunksRef.current[0].type) || 'audio/webm' });

        let decodeCtx = null;
        try {
          const arrayBuf = await blob.arrayBuffer();
          decodeCtx = audioCtxRef.current && audioCtxRef.current.state !== 'closed'
            ? audioCtxRef.current
            : new (window.AudioContext || window.webkitAudioContext)();
          const audioBuffer = await decodeCtx.decodeAudioData(arrayBuf);
          recordedBufferRef.current = audioBuffer;
          originalRecordingBufferRef.current = audioBuffer;
          setVoiceReady(true);
          setWaveformPeaks(computeWaveformPeaks(audioBuffer));
        } catch (e) {
          recordedBufferRef.current = null;
          setVoiceReady(false);
        } finally {
          // This context (the live recording one, reused for decode, or a
          // fresh fallback) has no further use — close it instead of
          // leaving it open forever. Every recording used to leak one of
          // these, and mobile browsers cap concurrent AudioContexts, so
          // after a few recordings new contexts (including the playback
          // one) would silently stop producing sound.
          if (decodeCtx && decodeCtx.state !== 'closed') decodeCtx.close().catch(() => {});
          audioCtxRef.current = null;
        }

        runAnalysis();
      };
      mr.stop();
    } else {
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        audioCtxRef.current.close().catch(() => {});
      }
      audioCtxRef.current = null;
      runAnalysis();
    }
  }

  function stopRecordingEarly() {
    if (phase !== 'recording') return;
    finishRecording();
  }

  // Shared by both the live-recording flow and the file-upload flow: turns
  // a list of {t, freq} analysis frames into a key + a filtered note list,
  // or a reason string if there isn't enough to work with.
  function analyzeFrames(frames) {
    const voiced = frames.filter((f) => f.freq > 0);
    if (voiced.length < 15) return { error: 'no-pitch' };

    const hist = new Array(12).fill(0);
    voiced.forEach((f) => {
      const midi = freqToMidi(f.freq);
      const pc = ((Math.round(midi) % 12) + 12) % 12;
      hist[pc] += 1;
    });
    const key = detectKey(hist);

    const qFrames = frames.map((f) => {
      if (f.freq <= 0) return { t: f.t, step: null };
      return { t: f.t, step: midiToScaleStep(freqToMidi(f.freq), key.tonic, key.mode) };
    });
    const bridged = fillShortGaps(qFrames);
    const rawNotes = framesToNotes(bridged);
    const filtered = filterOutlierNotes(filterTransientArtifacts(rawNotes, key.tonic, key.mode));
    if (filtered.length === 0) return { error: 'no-melody' };

    const notes = attachMeasuredFreq(filtered, voiced);
    return { key, notes };
  }

  const ANALYSIS_ERROR_MESSAGES = {
    'no-pitch': 'Vi kunde inte hitta en tydlig tonhöjd i inspelningen. Sjung en enkel, tydlig melodi lite starkare och testa igen.',
    'no-melody': 'Vi kunde inte tolka en tydlig melodislinga. Testa att sjunga lite långsammare och tydligare.',
  };

  // Shared by both the recording and upload paths: guesses a tempo from
  // the note onsets and, if confident enough to return one at all, adopts
  // it as the metronome's tempo. A manual +/- afterward is always
  // available if the guess is off.
  function applyDetectedTempo(notes) {
    const bpm = detectTempoBpm(notes);
    if (bpm) {
      setMetronomeBpm(bpm);
      setTempoDetected(true);
    } else {
      setTempoDetected(false);
    }
  }

  function runAnalysis() {
    const result = analyzeFrames(framesRef.current);
    if (result.error) {
      setPhase('error');
      setErrorMsg(ANALYSIS_ERROR_MESSAGES[result.error]);
      return;
    }
    detectedKeyRef.current = result.key;
    setKeyInfo(result.key);
    setMelodyNotes(result.notes);
    // The count-in and (if the toggle was on) the click running through the
    // whole take already anchored this recording to metronomeBpm — an
    // auto-detect guess right afterward can easily land a beat or two off
    // from that (natural timing variance) and would otherwise silently
    // overwrite the exact tempo just actually sung to. Only guess when no
    // tempo was already committed to, same as the file-upload path (which
    // has no such commitment to preserve).
    if (metronomeEnabledRef.current) {
      setTempoDetected(false);
    } else {
      applyDetectedTempo(result.notes);
    }
    setPhase('ready');
    setNoteViewExpanded(true);
  }

  // Uploaded files are decoded up front (so we have real sample data),
  // trimmed to the same 10s cap as a live recording, then run through the
  // exact same offline frame-extraction + analysis pipeline a recording's
  // live AnalyserNode loop would have produced.
  function trimAudioBuffer(ctx, buffer, maxDur) {
    if (buffer.duration <= maxDur) return buffer;
    const sr = buffer.sampleRate;
    const len = Math.floor(maxDur * sr);
    const trimmed = ctx.createBuffer(buffer.numberOfChannels, len, sr);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      trimmed.copyToChannel(buffer.getChannelData(c).subarray(0, len), c);
    }
    return trimmed;
  }

  // Shared by file-upload and session-restore: takes a fully-decoded
  // buffer, wires it up as the current recording, and runs it through the
  // same pitch-analysis pipeline a fresh recording would go through.
  // Returns whether analysis succeeded (on failure, phase/errorMsg are
  // already set to 'error' and the caller's own message).
  function finalizeLoadedBuffer(buffer) {
    recordedBufferRef.current = buffer;
    originalRecordingBufferRef.current = buffer;
    setVoiceReady(true);
    setRecordingDuration(Math.max(0.3, buffer.duration));
    setWaveformPeaks(computeWaveformPeaks(buffer));

    const frames = extractFramesFromBuffer(buffer.getChannelData(0), buffer.sampleRate);
    const result = analyzeFrames(frames);
    if (result.error) {
      setPhase('error');
      setErrorMsg(
        result.error === 'no-pitch'
          ? 'Vi kunde inte hitta en tydlig tonhöjd i ljudfilen. Prova en fil med en tydlig, enkel melodi.'
          : ANALYSIS_ERROR_MESSAGES[result.error]
      );
      return false;
    }
    detectedKeyRef.current = result.key;
    setKeyInfo(result.key);
    setMelodyNotes(result.notes);
    applyDetectedTempo(result.notes);
    setPhase('ready');
    setNoteViewExpanded(true);
    return true;
  }

  // Manual tonart correction — the app's own most common cause of an odd-
  // sounding harmony is a misheard key (see the "Tonart"-section on the
  // About page), so this lets the singer fix it directly instead of
  // re-recording. Re-quantizes every note against the new key (see
  // requantizeNotesToKey — uses each note's actual measured pitch, not a
  // second pass over the raw audio) and invalidates every buffer built
  // from the old key, exactly like the other edits that replace
  // melodyNotes/keyInfo wholesale.
  function overrideKey(tonic, mode) {
    if (!keyInfo || !melodyNotes.length) return;
    if (tonic === keyInfo.tonic && mode === keyInfo.mode) return;
    const nextKey = { tonic, mode };
    setMelodyNotes((notes) => requantizeNotesToKey(notes, keyInfo, nextKey));
    setKeyInfo(nextKey);
    harmonyBuffersRef.current = {};
    autotunedBuffersRef.current = {};
    if (isPlaying) stopPlayback();
  }

  async function handleFileUpload(file) {
    if (!file) return;
    setErrorMsg('');
    resetSourceState();
    setPhase('analyzing');

    let decodeCtx = null;
    try {
      const arrayBuf = await file.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      decodeCtx = new AC();
      let buffer;
      try {
        buffer = await decodeCtx.decodeAudioData(arrayBuf);
      } catch (e) {
        setPhase('error');
        setErrorMsg('Kunde inte läsa ljudfilen. Kontrollera att det är en vanlig ljudfil (t.ex. WAV, MP3 eller M4A).');
        return;
      }
      buffer = trimAudioBuffer(decodeCtx, buffer, DURATION);
      finalizeLoadedBuffer(buffer);
    } catch (e) {
      setPhase('error');
      setErrorMsg('Kunde inte läsa ljudfilen. Kontrollera att det är en vanlig ljudfil (t.ex. WAV, MP3 eller M4A).');
    } finally {
      // Same leak risk as the recording context: an upload's decode
      // context has no use after this function returns, so it must be
      // closed rather than left open for the rest of the session.
      if (decodeCtx && decodeCtx.state !== 'closed') decodeCtx.close().catch(() => {});
    }
  }

  // Decodes a project's saved audio and re-applies its settings on top of
  // the freshly analyzed recording — shared by the mount-time restore
  // banner and "Öppna" in the Sessioner list. Marks `project` as the one
  // autosave should keep updating from here on.
  async function loadProjectIntoWorkspace(project) {
    let decodeCtx = null;
    try {
      const arrayBuf = await project.audio.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      decodeCtx = new AC();
      const buffer = await decodeCtx.decodeAudioData(arrayBuf);
      const ok = finalizeLoadedBuffer(buffer);
      if (ok) {
        setSoundType(project.soundType ?? 'recording');
        setAutotuneOn(!!project.autotuneOn);
        setAutotuneLevelIndex(project.autotuneLevelIndex ?? 0);
        setReverbOn(!!project.reverbOn);
        setReverbLevelIndex(project.reverbLevelIndex ?? 0);
        setHumanizeOn(project.humanizeOn ?? true);
        setFadeIn(project.fadeIn ?? 0);
        setFadeOut(project.fadeOut ?? 0);
        setTrimStart(project.trimStart ?? 0);
        setTrimEnd(project.trimEnd ?? null);
        setLoopEnabled(!!project.loopEnabled);
        setChannels(project.channels || defaultChannels());
        activeProjectIdRef.current = project.id;
        activeProjectNameRef.current = project.name;
      }
      return ok;
    } catch (e) {
      setPhase('error');
      setErrorMsg('Kunde inte läsa in sessionen.');
      return false;
    } finally {
      if (decodeCtx && decodeCtx.state !== 'closed') decodeCtx.close().catch(() => {});
    }
  }

  // Always consumes (clears) the restore-banner offer, whether it
  // succeeds or not, so the banner can't get stuck.
  async function restoreSession() {
    if (!restorableSession || restoringSession) return;
    setRestoringSession(true);
    try {
      const full = await loadProject(restorableSession.id);
      if (full) await loadProjectIntoWorkspace(full);
    } catch (e) {
      setPhase('error');
      setErrorMsg('Kunde inte återställa förra sessionen.');
    } finally {
      setRestorableSession(null);
      setRestoringSession(false);
    }
  }

  function dismissRestorableSession() {
    // Only hides the mount-time banner for this page load — the project
    // is real, named, saved work, not a throwaway crash-recovery
    // artifact, so dismissing never deletes it (see "Sessioner").
    setRestorableSession(null);
  }

  // "Öppna" in the Sessioner list — replaces the current workspace with a
  // different saved project, same as starting a fresh recording/upload
  // does (resetSourceState), just loading saved content afterward instead
  // of waiting for a new one.
  async function switchToProject(id) {
    if (id === activeProjectIdRef.current) {
      setProjectsExpanded(false);
      return;
    }
    const full = await loadProject(id).catch(() => null);
    if (full) {
      resetSourceState();
      await loadProjectIntoWorkspace(full);
    }
    setProjectsExpanded(false);
  }

  function handleRenameProject(id, name) {
    if (id === activeProjectIdRef.current) activeProjectNameRef.current = name;
    renameProject(id, name).then(refreshProjects).catch(() => {});
  }

  function handleDeleteProject(id) {
    if (id === activeProjectIdRef.current) {
      activeProjectIdRef.current = null;
      activeProjectNameRef.current = null;
    }
    deleteProject(id).then(refreshProjects).catch(() => {});
  }

  function resetAll() {
    resetSourceState();
    setPhase('idle');
    setRecordingDuration(DURATION);
    setErrorMsg('');
  }

  // A single AudioContext, reused for the whole session rather than
  // created fresh per play/stop cycle. Browsers cap how many contexts a
  // page may have open — Safari in particular — and creating-then-closing
  // one on every playback click burns through that budget fast: after
  // enough plays, new contexts stop producing sound at all (silently, no
  // error).
  async function getPlaybackContext() {
    let ctx = playCtxRef.current;
    if (!ctx || ctx.state === 'closed') {
      const AC = window.AudioContext || window.webkitAudioContext;
      ctx = new AC();
      // A context the browser suspends on its own mid-playback (a tab
      // backgrounded long enough to be reclaimed, an OS audio-focus change,
      // a laptop lid closing) otherwise leaves the transport UI stuck
      // showing "Pausa" over silence with nothing to notice or recover —
      // suspend()/resume() is a true pause/continue of Web Audio's own
      // clock (already-scheduled nodes keep their exact place, so resuming
      // needs no position recalculation), so recovery is just calling
      // resume(); only fall back to a clean stopped state if that itself
      // fails. `metronomeSchedulerRef` covers the other three things this
      // context alone ever drives audibly — a "Hitta takten"/"Lyssna"
      // preview, the click during an actual recording, and a mix-synced
      // click (see startMix) — since all three keep it non-null exactly
      // while a click should be sounding. A context that's supposed to be
      // idle suspending on its own is normal, not a problem to react to.
      ctx.onstatechange = () => {
        if (ctx.state === 'suspended' && (isPlayingRef.current || metronomeSchedulerRef.current)) {
          ctx.resume().catch(() => stopPlayback());
        }
      };
      playCtxRef.current = ctx;
    }
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    return ctx;
  }

  // Called right as a getUserMedia recording (main or own-take) stops its
  // stream — see the two call sites. iOS Safari (worst as a home-screen-
  // installed web app, reported symptom: totally silent "Spela mix", no
  // error, right after recording an own take without headphones) can leave
  // an already-open AudioContext's underlying audio session silently
  // broken once a getUserMedia capture that ran alongside it ends, even
  // though the context itself keeps reporting state 'running' — so
  // getPlaybackContext()'s own suspended-check never notices anything is
  // wrong. Discarding the cached context here forces the next
  // getPlaybackContext() call to build a fresh one, which reliably gets a
  // clean session. Cheap and harmless on browsers where this isn't an
  // issue — playback wasn't running during a recording anyway (startMix's
  // own stopPlayback() sees to that), so there's nothing to interrupt.
  function discardPlaybackContext() {
    if (playCtxRef.current && playCtxRef.current.state !== 'closed') {
      playCtxRef.current.close().catch(() => {});
    }
    playCtxRef.current = null;
  }

  // "Lyssna" — preview the click track on its own, independent of any
  // recording, so the tempo can be dialed in and heard before committing
  // to a take. Toggling it off just stops the scheduler.
  async function toggleMetronomeListen() {
    if (metronomeListening) {
      setMetronomeListening(false);
      // Handles both flavors below uniformly: a startMix()-driven melody+
      // click preview (stops the whole mix) and a standalone click-only
      // loop (metronomeSchedulerRef is stopped either way).
      stopPlayback();
      return;
    }
    // Re-detect the tempo from the melody every time "Lyssna" is pressed,
    // not just once right after analysis — so it always offers the app's
    // current best guess, even if a manual BPM edit or an earlier missed
    // detection left a stale/wrong number showing.
    let bpm = metronomeBpm;
    if (melodyNotes.length) {
      const detected = detectTempoBpm(melodyNotes);
      if (detected) {
        bpm = detected;
        setMetronomeBpm(detected);
        setTempoDetected(true);
      }
    }
    setMetronomeListening(true);
    if (melodyNotes.length && recordedBufferRef.current) {
      // Play the melody alongside the click, sample-accurately synced to
      // the same start (see the metronomeOverride param on startMix), so
      // the tempo can be judged directly against the actual singing
      // instead of just an isolated click.
      startMix(channels, ['melody'], undefined, { bpm });
    } else {
      // No melody yet (idle screen, before any recording) — just the
      // click on its own, as before.
      if (metronomeSchedulerRef.current) {
        stopMetronomeScheduler(metronomeSchedulerRef.current);
        metronomeSchedulerRef.current = null;
      }
      const ctx = await getPlaybackContext();
      metronomeSchedulerRef.current = startMetronomeScheduler(ctx, bpm);
    }
  }

  // Shared by the +/- steppers and the tap-to-type BPM field — takes an
  // absolute target rather than a delta.
  function setMetronomeBpmTo(target) {
    const next = Math.max(40, Math.min(240, Math.round(target)));
    setMetronomeBpm(next);
    setTempoDetected(false);
    // The scheduler captured the old tempo at start time — restart it so a
    // change while previewing is actually heard, not applied silently.
    if (metronomeListening && metronomeSchedulerRef.current) {
      const ctx = playCtxRef.current;
      stopMetronomeScheduler(metronomeSchedulerRef.current);
      metronomeSchedulerRef.current = ctx ? startMetronomeScheduler(ctx, next) : null;
    }
  }

  function adjustMetronomeBpm(delta) {
    setMetronomeBpmTo(metronomeBpm + delta);
  }

  // The mixer/idle-screen toggle only flips a stored preference, but if a
  // mix is already playing that change should be heard right away, not
  // wait for the next Play — mirrors the live-restart setMetronomeBpmTo
  // already does for a BPM edit made mid-preview. Skipped while a "Lyssna"
  // preview owns the scheduler (metronomeListening) — that flow manages
  // its own click via the metronomeOverride path in startMix.
  function toggleMetronomeEnabled() {
    setMetronomeEnabled((prev) => {
      const next = !prev;
      if (isPlaying && !metronomeListening) {
        const ctx = playCtxRef.current;
        if (metronomeSchedulerRef.current) {
          stopMetronomeScheduler(metronomeSchedulerRef.current);
          metronomeSchedulerRef.current = null;
        }
        if (next && ctx) {
          metronomeSchedulerRef.current = startMetronomeScheduler(ctx, metronomeBpmRef.current, ctx.currentTime + 0.05);
        }
      }
      return next;
    });
  }

  // `keepPosition`: true for pause (freeze playheadTime so Play resumes
  // from here); false/omitted for a real stop — clears playheadTime, which
  // every "position" reader (the seek slider, startMix's default offset)
  // treats as "start of the trim window", i.e. back to the beginning.
  function stopPlayback({ keepPosition } = {}) {
    // Invalidates any startMix() call still awaiting content resolution
    // (e.g. a first-time harmony render) — without this, that call would
    // still schedule and start audio after the user asked for it to stop.
    // See playGenerationRef.
    playGenerationRef.current += 1;
    if (playTimeoutRef.current) {
      clearTimeout(playTimeoutRef.current);
      playTimeoutRef.current = null;
    }
    if (playheadRafRef.current) {
      cancelAnimationFrame(playheadRafRef.current);
      playheadRafRef.current = null;
    }
    activeSourcesRef.current.forEach((node) => {
      try { node.stop(); } catch (e) { /* already stopped/ended */ }
    });
    activeSourcesRef.current = [];
    // Stops the playback-synced click along with everything else — a
    // "Lyssna" preview loop never reaches here (it's not routed through
    // startMix/stopPlayback), so this only ever cuts off a click that
    // startMix itself started.
    if (metronomeSchedulerRef.current) {
      stopMetronomeScheduler(metronomeSchedulerRef.current);
      metronomeSchedulerRef.current = null;
    }
    Object.values(mixNodesRef.current).forEach((nodes) => {
      try { nodes.gainNode.disconnect(); } catch (e) { /* already disconnected */ }
      try { nodes.fadeGainNode.disconnect(); } catch (e) { /* already disconnected */ }
      try { nodes.pannerNode.disconnect(); } catch (e) { /* already disconnected */ }
    });
    mixNodesRef.current = {};
    if (reverbBusRef.current) {
      try { reverbBusRef.current.input.disconnect(); } catch (e) { /* already disconnected */ }
      try { reverbBusRef.current.convolver?.disconnect(); } catch (e) { /* already disconnected */ }
      try { reverbBusRef.current.wetGain?.disconnect(); } catch (e) { /* already disconnected */ }
      reverbBusRef.current = null;
    }
    Object.values(meterRefs.current).forEach((el) => { if (el) el.style.width = '0%'; });
    setIsPlaying(false);
    if (!keepPosition) setPlayheadTime(null);
    setPreviewingKey(null);
  }

  // `outputNode` is the channel's own gain node — every source (whether a
  // real recording buffer or a synthesized voice) ends up there instead of
  // going straight to the speakers, so the channel's fader and meter apply
  // no matter what kind of source it is.
  // `rangeStart`/`rangeEnd` clip playback to a [start, end) window on the
  // notes' own timeline (seconds) — the trim range — with every scheduled
  // time shifted so the window's start lands at `now`. Notes entirely
  // outside the window are skipped; ones straddling an edge are clipped to
  // it rather than dropped, same as the waveform view's own trim overlay.
  function makeVoice(ctx, outputNode, now, notesArr, voiceSoundType, rangeStart = 0, rangeEnd = Infinity) {
    const osc = ctx.createOscillator();
    osc.type = voiceSoundType === 'voice' ? 'sawtooth' : 'sine';
    const envGain = ctx.createGain();
    envGain.gain.value = 0;
    const outNode = voiceSoundType === 'voice' ? createFormantSum(ctx, osc) : osc;
    outNode.connect(envGain).connect(outputNode);
    osc.start(now);
    let lastEnd = 0;
    notesArr.forEach((n) => {
      const clippedStart = Math.max(n.start, rangeStart);
      const clippedEnd = Math.min(n.end, rangeEnd);
      if (clippedEnd <= clippedStart) return;
      const freq = midiToFreq(n.midi);
      const start = now + (clippedStart - rangeStart);
      const end = Math.max(start + 0.05, now + (clippedEnd - rangeStart));
      const attack = 0.02;
      const release = Math.min(0.06, (end - start) / 3);
      osc.frequency.setValueAtTime(freq, start);
      envGain.gain.setValueAtTime(0, start);
      envGain.gain.linearRampToValueAtTime(1, start + attack);
      envGain.gain.setValueAtTime(1, Math.max(start + attack, end - release));
      envGain.gain.linearRampToValueAtTime(0, end);
      lastEnd = Math.max(lastEnd, end - now);
    });
    osc.stop(now + lastEnd + 0.3);
    return osc;
  }

  function playBufferSource(ctx, outputNode, now, buffer, offset = 0, duration = null) {
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(outputNode);
    if (duration !== null) {
      src.start(now, offset, duration);
    } else {
      src.start(now, offset);
    }
    return src;
  }

  async function renderSynthOffline(notesArr, renderSoundType, sampleRate) {
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const totalDur = (notesArr.length ? notesArr[notesArr.length - 1].end : 0) + 0.4;
    const offlineCtx = new OfflineCtx(1, Math.max(1, Math.ceil(totalDur * sampleRate)), sampleRate);
    makeVoice(offlineCtx, offlineCtx.destination, 0.05, notesArr, renderSoundType);
    return offlineCtx.startRendering();
  }

  const exportSampleRate = () => (recordedBufferRef.current ? recordedBufferRef.current.sampleRate : 44100);

  async function exportSong() {
    if (!recordedBufferRef.current || exporting) return;
    setExporting('song');
    try {
      const blob = audioBufferToWavBlob(recordedBufferRef.current);
      downloadBlob(blob, 'stamma-sang.wav');
    } finally {
      setExporting(null);
    }
  }

  async function exportMelody() {
    if (!melodyPlaybackNotes.length || exporting) return;
    setExporting('melody');
    try {
      if (soundType === 'recording' && recordedBufferRef.current) {
        const sourceBuffer = await getSourceBuffer();
        if (!sourceBuffer) return;
        const blob = audioBufferToWavBlob(sourceBuffer);
        downloadBlob(blob, 'stamma-melodi.wav');
      } else {
        const buffer = await renderSynthOffline(melodyPlaybackNotes, soundType, exportSampleRate());
        downloadBlob(audioBufferToWavBlob(buffer), 'stamma-melodi.wav');
      }
    } finally {
      setExporting(null);
    }
  }

  async function exportHarmonyType(type) {
    if (!melodyNotes.length || exporting) return;
    setExporting(`harmony-${type}`);
    try {
      let buffer;
      if (soundType === 'recording' && recordedBufferRef.current) {
        buffer = await getHarmonyBufferFor(type);
        if (!buffer) return;
      } else {
        buffer = await renderSynthOffline(harmonyPlaybackNotesFor(type), soundType, exportSampleRate());
      }
      const suffix = `${type}-${channels[type].direction === -1 ? 'under' : 'over'}`;
      downloadBlob(audioBufferToWavBlob(buffer), `stamma-stamma-${suffix}.wav`);
    } finally {
      setExporting(null);
    }
  }

  // Renders exactly what the mixer would currently play — every channel
  // that's on (or, if something's soloed, every soloed one) at its own
  // volume/pan, through the same trim window, fades, and reverb bus as live
  // playback — down to one WAV. Built on the same resolveChannelContent +
  // startMixChannelWithContent machinery startMix() uses, so this can't
  // silently drift from what "play" actually sounds like.
  async function exportMixdown() {
    if (exporting) return;
    const activeKeys = Object.keys(channels).filter((k) => isChannelAudible(channels, k));
    if (!activeKeys.length) return;
    setExporting('mixdown');
    try {
      const resolved = await Promise.all(
        activeKeys.map(async (key) => ({ key, content: await resolveChannelContent(key) }))
      );
      const usable = resolved.filter((r) => r.content);
      if (!usable.length) return;

      const rangeStart = Math.max(0, Math.min(trimStart, recordingDuration));
      const rangeEnd = Math.max(rangeStart + 0.05, Math.min(effectiveTrimEnd, recordingDuration));
      const sampleRate = exportSampleRate();
      // Extra tail past the trim window so a release ramp or the reverb's
      // own decay isn't cut off mid-fade the way it would be at exactly
      // rangeEnd.
      const tail = 0.4 + (reverbOn ? REVERB_LEVELS[reverbLevelIndex].decay : 0);
      const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
      const offlineCtx = new OfflineCtx(2, Math.max(1, Math.ceil((rangeEnd - rangeStart + tail) * sampleRate)), sampleRate);

      const reverbBus = createReverbBus(offlineCtx, reverbOn, REVERB_LEVELS[reverbLevelIndex]);
      const now = 0.05;
      usable.forEach(({ key, content }) => {
        // Mirrors startMix's per-channel trim/fade override (an own-take's
        // WaveformTrimmer) — the export should match what actually plays.
        const chState = channels[key];
        const chRangeStart = chState.trimStart != null ? Math.max(rangeStart, Math.min(chState.trimStart, rangeEnd)) : rangeStart;
        const chRangeEnd = chState.trimEnd != null ? Math.max(chRangeStart + 0.05, Math.min(chState.trimEnd, rangeEnd)) : rangeEnd;
        const chFadeIn = chState.fadeIn != null ? chState.fadeIn : fadeIn;
        const chFadeOut = chState.fadeOut != null ? chState.fadeOut : fadeOut;
        const chNow = now + (chRangeStart - rangeStart);
        startMixChannelWithContent(offlineCtx, key, chNow, chState, content, chRangeStart, chRangeEnd, chFadeIn, chFadeOut, chRangeStart, reverbBus.input, false);
      });

      const rendered = await offlineCtx.startRendering();
      downloadBlob(audioBufferToWavBlob(rendered), 'stamma-mix.wav');
    } finally {
      setExporting(null);
    }
  }

  // Resolves what a channel should actually play right now: a ready-made
  // buffer (the real recording, its autotuned version, or a rendered
  // harmony), or a note list for the synthesizer, depending on the current
  // "Ljud" mode. Returns null if the channel has nothing to play yet.
  async function resolveChannelContent(key) {
    if (key === 'original') {
      if (soundType !== 'recording' || !recordedBufferRef.current) return null;
      return { kind: 'buffer', buffer: recordedBufferRef.current };
    }
    if (key === 'melody') {
      if (soundType === 'recording') {
        if (!recordedBufferRef.current) return null;
        const buffer = await getSourceBuffer();
        return buffer ? { kind: 'buffer', buffer } : null;
      }
      return melodyPlaybackNotes.length ? { kind: 'notes', notes: melodyPlaybackNotes } : null;
    }
    // A user's own recorded take of a harmony type (see recordOwnTake) —
    // already at whatever pitch they actually sang, so it's played back
    // as-is with no pitch-shifting or note-derived synthesis.
    if (key.endsWith('Own')) {
      const type = key.slice(0, -3);
      const buffer = ownTakeBuffersRef.current[type];
      return buffer ? { kind: 'buffer', buffer } : null;
    }
    // ters / kvint / sext
    if (soundType === 'recording') {
      if (!recordedBufferRef.current) return null;
      const buffer = await getHarmonyBufferFor(key);
      return buffer ? { kind: 'buffer', buffer } : null;
    }
    const notes = harmonyPlaybackNotesFor(key);
    return notes.length ? { kind: 'notes', notes } : null;
  }

  // Node creation + actually starting a channel is kept synchronous and
  // separate from resolving its content (which can involve an offline
  // render taking a couple hundred ms) — see startMix for why.
  // `rangeStart`/`rangeEnd` are the trim window (seconds, on the
  // recording's own timeline); `playFrom` is where within it playback
  // actually starts (equal to rangeStart unless resuming from a pause/seek).
  // `fadeInDur`/`fadeOutDur` (seconds) drive a separate automation-only
  // gain stage — kept apart from `gainNode` (the user's live volume
  // fader) so a fader drag never collides with a scheduled fade ramp.
  // `destNode` is the reverb bus's input (see createReverbBus) rather than
  // ctx.destination directly — reverb is mix-wide, so every channel feeds
  // the same bus instead of each needing its own send.
  // `trackMeter` is false for an offline render (see exportMixdown) — an
  // offline context's nodes have no business overwriting mixNodesRef, which
  // the live meter loop reads from mid-playback.
  function startMixChannelWithContent(ctx, key, now, channelState, content, rangeStart, rangeEnd, fadeInDur, fadeOutDur, playFrom, destNode, trackMeter = true) {
    const gainNode = ctx.createGain();
    gainNode.gain.value = channelState.volume;

    const fadeGainNode = ctx.createGain();
    const windowDur = Math.max(0.05, rangeEnd - rangeStart);
    const fi = Math.max(0, Math.min(fadeInDur, windowDur));
    const fo = Math.max(0, Math.min(fadeOutDur, windowDur - fi));
    const elapsedIntoWindow = Math.max(0, playFrom - rangeStart);
    scheduleFadeGain(fadeGainNode.gain, now, windowDur, elapsedIntoWindow, fi, fo);
    gainNode.connect(fadeGainNode);

    const analyserNode = ctx.createAnalyser();
    analyserNode.fftSize = 256;
    // Meter reads post-fader/fade, pre-pan — panning splits energy
    // left/right but shouldn't itself move the level reading.
    fadeGainNode.connect(analyserNode);
    const pannerNode = ctx.createStereoPanner();
    pannerNode.pan.value = channelState.pan || 0;
    fadeGainNode.connect(pannerNode);
    pannerNode.connect(destNode);

    let sourceNode;
    if (content.kind === 'buffer') {
      const buf = content.buffer;
      const offset = Math.max(0, Math.min(playFrom, buf.duration));
      const duration = Math.max(0, Math.min(rangeEnd, buf.duration) - offset);
      sourceNode = playBufferSource(ctx, gainNode, now, buf, offset, duration);
    } else {
      sourceNode = makeVoice(ctx, gainNode, now, content.notes, soundType, playFrom, rangeEnd);
    }

    if (trackMeter) mixNodesRef.current[key] = { gainNode, fadeGainNode, analyserNode, pannerNode, sourceNode };
    return sourceNode;
  }

  function updateMeters() {
    const scratch = meterScratchRef.current || (meterScratchRef.current = new Uint8Array(256));
    Object.entries(mixNodesRef.current).forEach(([key, nodes]) => {
      if (!nodes?.analyserNode) return;
      nodes.analyserNode.getByteTimeDomainData(scratch);
      let sumSq = 0;
      for (let i = 0; i < scratch.length; i++) {
        const v = (scratch[i] - 128) / 128;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / scratch.length);
      const level = Math.min(1, rms * 3.2); // headroom so a normal singing level visibly fills the bar
      const el = meterRefs.current[key];
      if (el) el.style.width = `${Math.round(level * 100)}%`;
    });
  }

  // Starts every enabled channel at once, live-mixed through its own
  // fader. Toggling a channel or changing a harmony's direction while
  // already playing just restarts the whole mix from the top rather than
  // trying to splice a new source in mid-playback in sync — simpler and
  // more robust for a few seconds of preview audio.
  //
  // Every channel's content is resolved first, *then* they're all started
  // together at the same timestamp — a channel needing a fresh harmony
  // render (a couple hundred ms) would otherwise still be awaited while the
  // already-cached channels had already started on schedule, so it'd join
  // audibly late and out of sync with them.
  //
  // `keysOverride` lets a single channel's own play button preview just
  // that one track without touching anyone's enabled/solo state — same
  // resolve-then-start machinery, just scoped to one key.
  // `startOffset` (absolute seconds on the recording's timeline) lets Play
  // resume from a paused/seeked position instead of always the window's
  // start — defaults to the window start, which is also what a loop
  // restart and a fresh channel preview want.
  // `metronomeOverride` ({ bpm } | undefined) forces a synced click at that
  // exact tempo regardless of the `metronomeEnabled` toggle — used by
  // "Lyssna" to click alongside the melody at whatever tempo it just
  // detected, without permanently flipping the separate recording/
  // playback-click setting.
  async function startMix(channelsState, keysOverride, startOffset, metronomeOverride, metronomeGainMultiplier = 1) {
    stopPlayback();
    metronomeOverrideRef.current = metronomeOverride || null;
    const myGeneration = playGenerationRef.current;
    const activeKeys = keysOverride || Object.keys(channelsState).filter((k) => isChannelAudible(channelsState, k));
    if (!activeKeys.length) return;

    let ctx;
    let usable;
    try {
      ctx = await getPlaybackContext();
      const resolved = await Promise.all(
        activeKeys.map(async (key) => ({ key, content: await resolveChannelContent(key) }))
      );
      usable = resolved.filter((r) => r.content);
    } catch (err) {
      // A harmony/autotune render or the context itself failing here would
      // otherwise leave nothing scheduled but isPlaying never set — the
      // Play button would look inert rather than actually broken. Nothing
      // was started yet at this point, so there's nothing to clean up
      // beyond the stopPlayback() already run above.
      console.error('Kunde inte starta uppspelning:', err);
      return;
    }
    if (!usable.length) return;
    // A newer startMix() call (another play/pause/preview press while this
    // one was still awaiting content, e.g. a first-time harmony render)
    // has since taken over — starting audio now would create nodes no
    // stopPlayback() can ever reach again. See playGenerationRef.
    if (playGenerationRef.current !== myGeneration) return;

    // The trim window (waveform handles) clips every channel's playback to
    // the same [rangeStart, rangeEnd) span on the recording's timeline.
    const rangeStart = Math.max(0, Math.min(trimStart, recordingDuration));
    const rangeEnd = Math.max(rangeStart + 0.05, Math.min(effectiveTrimEnd, recordingDuration));
    const playFrom = Math.max(rangeStart, Math.min(startOffset ?? rangeStart, rangeEnd - 0.05));

    const reverbBus = createReverbBus(ctx, reverbOn, REVERB_LEVELS[reverbLevelIndex]);
    reverbBusRef.current = reverbBus;

    const now = ctx.currentTime + 0.05;
    // A per-channel try/catch rather than usable.map(...) all-or-nothing —
    // one channel failing to schedule (a malformed buffer, a stale node
    // from a race) shouldn't take the whole mix down with it, and every
    // node that *did* start before a failure is still tracked so
    // stopPlayback() can actually reach it instead of it playing on
    // forever, orphaned, with the transport UI already showing "stopped".
    const started = [];
    usable.forEach(({ key, content }) => {
      try {
        const chState = channelsState[key];
        // An own-take channel's WaveformTrimmer (see MixerChannel's
        // `waveform` slot) can narrow its own window inside the shared
        // one — trimStart/trimEnd/fadeIn/fadeOut null means "use the
        // shared window", same as every other channel.
        const chRangeStart = chState.trimStart != null ? Math.max(rangeStart, Math.min(chState.trimStart, rangeEnd)) : rangeStart;
        const chRangeEnd = chState.trimEnd != null ? Math.max(chRangeStart + 0.05, Math.min(chState.trimEnd, rangeEnd)) : rangeEnd;
        const chFadeIn = chState.fadeIn != null ? chState.fadeIn : fadeIn;
        const chFadeOut = chState.fadeOut != null ? chState.fadeOut : fadeOut;
        // A channel's own trim can start later than the shared playhead
        // (playFrom) — e.g. seeking to the very start while this take is
        // trimmed to skip a false start — so it needs its own delayed
        // start instead of beginning at `now` like every other channel.
        const chPlayFrom = Math.max(playFrom, chRangeStart);
        const chNow = now + Math.max(0, chPlayFrom - playFrom);
        started.push(startMixChannelWithContent(ctx, key, chNow, chState, content, chRangeStart, chRangeEnd, chFadeIn, chFadeOut, chPlayFrom, reverbBus.input));
      } catch (err) {
        console.error(`Kunde inte spela upp kanalen "${key}":`, err);
      }
    });
    if (!started.length) {
      stopPlayback();
      return;
    }

    // The click-track toggle (used for a recording's count-in/click) also
    // doubles as a "click along with playback" aid — started at the exact
    // same `now` the mix itself starts at, on the same context/clock, so
    // it's sample-accurately synced rather than just "roughly around the
    // same time". Any independent "Lyssna" preview scheduler is stopped
    // and superseded here, since they share the one ref/context. Reads the
    // ref, not the closed-over state, so a toggle/BPM change mid-playback
    // is picked up correctly on a loop restart (see metronomeEnabledRef).
    if (metronomeSchedulerRef.current) {
      stopMetronomeScheduler(metronomeSchedulerRef.current);
      metronomeSchedulerRef.current = null;
    }
    if (metronomeOverride) {
      metronomeSchedulerRef.current = startMetronomeScheduler(ctx, metronomeOverride.bpm, now, metronomeGainMultiplier);
    } else if (metronomeEnabledRef.current) {
      metronomeSchedulerRef.current = startMetronomeScheduler(ctx, metronomeBpmRef.current, now, metronomeGainMultiplier);
      if (metronomeListening) setMetronomeListening(false);
    }

    activeSourcesRef.current = started;
    setIsPlaying(true);
    setPreviewingKey(keysOverride && keysOverride.length === 1 ? keysOverride[0] : null);

    const totalDur = rangeEnd - playFrom;
    playTimeoutRef.current = setTimeout(() => {
      // Read via a ref, not the `loopEnabled` closed over at schedule time —
      // this timeout was set up to (totalDur + 0.4)s ago, and a toggle
      // flipped since then would otherwise be missed.
      if (loopEnabledRef.current) {
        // Carries a "Lyssna" click-tempo override through the restart too
        // (see metronomeOverrideRef) — otherwise a looping Lyssna preview
        // would click at the detected tempo for one pass and then silently
        // fall back to the separate metronomeEnabled setting.
        startMix(channelsState, keysOverride, undefined, metronomeOverrideRef.current || undefined, metronomeGainMultiplier);
      } else {
        stopPlayback();
        if (metronomeOverrideRef.current) setMetronomeListening(false);
      }
    }, (totalDur + 0.4) * 1000);

    const tickPlayhead = () => {
      const elapsed = ctx.currentTime - now;
      setPlayheadTime(Math.max(0, playFrom + elapsed));
      updateMeters();
      if (elapsed < totalDur + 0.1) {
        playheadRafRef.current = requestAnimationFrame(tickPlayhead);
      }
    };
    playheadRafRef.current = requestAnimationFrame(tickPlayhead);
  }

  function restartMix(channelsState) {
    startMix(channelsState);
  }

  // A channel's own play button: preview just that track, regardless of
  // its current enabled/solo state in the mixer.
  function previewChannel(key) {
    startMix(channels, [key]);
  }

  // Combined play/pause: pausing freezes playheadTime so the next press
  // resumes from there instead of restarting at the window's start.
  function togglePlayPause() {
    if (isPlaying) {
      stopPlayback({ keepPosition: true });
    } else {
      startMix(channels, undefined, playheadTime !== null ? playheadTime : trimStart);
    }
  }

  // Dragging the waveform's seek slider pauses (if playing) and moves the
  // resume position — the same playheadTime the play/pause button reads.
  function seekTo(t) {
    const clamped = Math.max(trimStart, Math.min(t, effectiveTrimEnd));
    if (isPlaying) stopPlayback({ keepPosition: true });
    setPlayheadTime(clamped);
  }

  // Records the user's own take of a harmony type (ters/kvint/sext),
  // played back as-is — no pitch-shifting, since it's already sung at
  // whatever pitch the singer actually hit. The melody plays back
  // concurrently as a monitor so there's something to harmonize against.
  // Pressing this again (on the original Ters/Kvint/Sext channel, not the
  // resulting "Egen X" one) always re-records and replaces the take.
  async function recordOwnTake(type) {
    if (recordingOwnTake) return;
    setOwnTakeError('');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setOwnTakeError('Den här webbläsaren stödjer inte mikrofoninspelning.');
      return;
    }
    if (micPermission === 'denied') {
      setOwnTakeError('Mikrofonåtkomst är blockerad för den här sidan. Ändra behörighet i webbläsarens inställningar och försök igen.');
      return;
    }

    let stream;
    try {
      // Echo cancellation/noise suppression/AGC are all *on* by default —
      // fine for a solo recording, but this take is captured while the
      // melody plays back through the speakers as a monitor (see
      // startMix(...) below), and the singer is deliberately singing in
      // time and in harmony with it. The browser's AEC then reads much of
      // that overlap as "echo of what we're playing" and aggressively
      // guts it from the mic signal, which sounded exactly like "only
      // records small parts" — the takes were real, just chopped up by
      // the echo canceller. Headphones would sidestep this entirely, but
      // most people won't have them on for a quick take, so ask for the
      // rawest signal the browser will give instead.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (err) {
      setOwnTakeError('Kunde inte komma åt mikrofonen. Kontrollera att appen har mikrofonbehörighet i webbläsaren.');
      return;
    }
    ownTakeStreamRef.current = stream;

    if (isPlaying) stopPlayback();

    // Count-in before the mic actually starts recording, same as the main
    // recording flow — lets the singer come in on time with the melody
    // that's about to play as a monitor.
    ownTakeCountInCancelledRef.current = false;
    setRecordingOwnTake({ type, countingIn: true, countInBeat: 0, countdown: recordingDuration });
    // If the live monitor is on, narrow it to just this voice for the
    // recording so you hear the tonart-/character-anpassad stämma sung
    // back while you record it (restored on any exit via the
    // recordingOwnTake effect). Runs from the count-in on, so the harmony
    // is already there on the first beat you sing.
    followOwnTakeWithLiveMonitor(type, stream);
    try {
      const clickCtx = await getPlaybackContext();
      await playCountIn(clickCtx, metronomeBpm, (i) => {
        setRecordingOwnTake((s) => (s && s.type === type && s.countingIn ? { ...s, countInBeat: i } : s));
      }, OWN_TAKE_MONITOR_BOOST);
    } catch (err) {
      // The mic stream was already opened above — without this, a failure
      // here (the playback context refusing to (re)start, say) would leave
      // it open with nothing to ever close it, and the record button stuck
      // showing "counting in…" forever with no recording actually able to
      // start.
      stream.getTracks().forEach((t) => t.stop());
      ownTakeStreamRef.current = null;
      setOwnTakeError(`Kunde inte spela in din ${HARMONY_TYPES[type].label.toLowerCase()}. Testa igen.`);
      setRecordingOwnTake(null);
      return;
    }
    if (ownTakeCountInCancelledRef.current) return;

    // Boosted melody volume + click gain (see OWN_TAKE_MONITOR_BOOST) —
    // compensating for iOS's own output ducking during this recording,
    // not a real preference, so it's a one-off copy of `channels` rather
    // than a change to the melody channel's actual stored volume.
    const boostedChannels = {
      ...channels,
      melody: { ...channels.melody, volume: channels.melody.volume * OWN_TAKE_MONITOR_BOOST },
    };
    startMix(boostedChannels, ['melody'], undefined, undefined, OWN_TAKE_MONITOR_BOOST);

    ownTakeChunksRef.current = [];
    ownTakeFinishedRef.current = false;
    let mr = null;
    if (window.MediaRecorder) {
      try {
        mr = new MediaRecorder(stream);
        mr.ondataavailable = (e) => { if (e.data.size > 0) ownTakeChunksRef.current.push(e.data); };
        mr.start();
      } catch (e) {
        mr = null;
      }
    }
    ownTakeRecorderRef.current = mr;

    if (!mr) {
      stream.getTracks().forEach((t) => t.stop());
      stopPlayback();
      setOwnTakeError('Den här webbläsaren stödjer inte inspelning.');
      setRecordingOwnTake(null);
      return;
    }

    ownTakeStartRef.current = performance.now();
    setRecordingOwnTake({ type, countingIn: false, countdown: recordingDuration });

    const tick = () => {
      const elapsed = (performance.now() - ownTakeStartRef.current) / 1000;
      const remaining = Math.max(0, recordingDuration - elapsed);
      setRecordingOwnTake((s) => (s && s.type === type ? { ...s, countdown: remaining } : s));
      if (remaining <= 0) {
        finishOwnTakeRecording(type);
        return;
      }
      ownTakeRafRef.current = requestAnimationFrame(tick);
    };
    ownTakeRafRef.current = requestAnimationFrame(tick);
  }

  function stopOwnTakeRecordingEarly() {
    if (!recordingOwnTake) return;
    // Still in the count-in — nothing's actually recording yet, so just
    // bail out of it instead of running the (not-yet-started) recorder
    // teardown in finishOwnTakeRecording.
    if (recordingOwnTake.countingIn) {
      ownTakeCountInCancelledRef.current = true;
      if (ownTakeStreamRef.current) {
        ownTakeStreamRef.current.getTracks().forEach((t) => t.stop());
        ownTakeStreamRef.current = null;
      }
      setRecordingOwnTake(null);
      return;
    }
    finishOwnTakeRecording(recordingOwnTake.type);
  }

  function finishOwnTakeRecording(type) {
    if (ownTakeFinishedRef.current) return;
    ownTakeFinishedRef.current = true;
    if (ownTakeRafRef.current) cancelAnimationFrame(ownTakeRafRef.current);
    stopPlayback(); // stop the monitor
    const stream = ownTakeStreamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    discardPlaybackContext();

    const mr = ownTakeRecorderRef.current;
    if (!mr || mr.state === 'inactive') {
      setRecordingOwnTake(null);
      return;
    }
    mr.onstop = async () => {
      let decodeCtx = null;
      try {
        const blob = new Blob(ownTakeChunksRef.current, { type: (ownTakeChunksRef.current[0] && ownTakeChunksRef.current[0].type) || 'audio/webm' });
        const arrayBuf = await blob.arrayBuffer();
        const AC = window.AudioContext || window.webkitAudioContext;
        decodeCtx = new AC();
        const decoded = await decodeCtx.decodeAudioData(arrayBuf);
        // Own-take capture asks for autoGainControl: false (see the comment
        // on recordOwnTake) so the browser's AGC doesn't fight the melody
        // bleeding in through the mic — but without it, the raw take is
        // captured at whatever level the room/mic happens to produce, often
        // much quieter than the main recording (which keeps AGC on). Peak-
        // normalize it here so a take isn't left sitting at a barely-
        // audible level by default; falls back to the raw decode for a
        // take that's genuinely silent (nothing to normalize against).
        const audioBuffer = peakNormalizeBuffer(decodeCtx, decoded) || decoded;
        ownTakeBuffersRef.current[type] = audioBuffer;
        const ownKey = `${type}Own`;
        setChannels((prev) => ({
          ...prev,
          // trimStart/trimEnd/fadeIn/fadeOut start unset (null) — "use the
          // shared mix window", same as every other channel already does —
          // and only diverge once this channel's own WaveformTrimmer is
          // actually dragged (see startMix's per-channel override).
          [ownKey]: prev[ownKey] || { enabled: true, volume: 0.85, pan: 0, solo: false, trimStart: null, trimEnd: null, fadeIn: null, fadeOut: null },
        }));
        setOwnTakeWaveformPeaksByType((prev) => ({ ...prev, [type]: computeWaveformPeaks(audioBuffer) }));
        alignOwnTake(type, audioBuffer);
      } catch (err) {
        setOwnTakeError(`Kunde inte spela in din ${HARMONY_TYPES[type].label.toLowerCase()}. Testa igen.`);
      } finally {
        if (decodeCtx && decodeCtx.state !== 'closed') decodeCtx.close().catch(() => {});
        setRecordingOwnTake(null);
      }
    };
    mr.stop();
  }

  // Kicked off right after a take is decoded (see mr.onstop above) — MTrack-
  // Align-style timing alignment against the melody's onsets, see
  // takeAlignment.js. Fire-and-forget: the raw take is already enabled and
  // playable while this runs in the background, `ownTakeAligningByType`
  // only drives the "(bygger …)" label on that channel. The identity check
  // before overwriting the ref guards against the take having been deleted
  // or re-recorded (a different buffer now sitting there) before this
  // resolves.
  async function alignOwnTake(type, rawBuffer) {
    setOwnTakeAligningByType((prev) => ({ ...prev, [type]: true }));
    try {
      const aligned = await alignOwnTakeToMelody(rawBuffer, melodyNotes, recordingDuration);
      if (ownTakeBuffersRef.current[type] === rawBuffer) {
        ownTakeBuffersRef.current[type] = aligned;
        if (aligned !== rawBuffer) {
          setOwnTakeWaveformPeaksByType((prev) => ({ ...prev, [type]: computeWaveformPeaks(aligned) }));
        }
      }
    } catch (err) {
      // Alignment is a best-effort enhancement on top of an already-usable
      // take — leave the raw, unaligned recording in place rather than
      // surfacing an error for it.
    } finally {
      setOwnTakeAligningByType((prev) => ({ ...prev, [type]: false }));
    }
  }

  function deleteOwnTake(type) {
    ownTakeBuffersRef.current[type] = null;
    const ownKey = `${type}Own`;
    setChannels((prev) => {
      if (!(ownKey in prev)) return prev;
      const next = { ...prev };
      delete next[ownKey];
      return next;
    });
    setOwnTakeWaveformPeaksByType((prev) => {
      if (!(type in prev)) return prev;
      const next = { ...prev };
      delete next[type];
      return next;
    });
    if (isPlaying) stopPlayback();
  }

  // Own-take trim/fade only ever narrow the *shared* mix window (see
  // startMix) — null means "use the shared window", same as every channel
  // already effectively does before its own WaveformTrimmer is touched.
  function setOwnTakeTrim(type, trimStart, trimEnd) {
    const ownKey = `${type}Own`;
    setChannels((prev) => {
      if (!prev[ownKey]) return prev;
      const next = { ...prev, [ownKey]: { ...prev[ownKey], trimStart, trimEnd } };
      if (isPlaying) restartMix(next);
      return next;
    });
  }

  function setOwnTakeFade(type, fadeIn, fadeOut) {
    const ownKey = `${type}Own`;
    setChannels((prev) => {
      if (!prev[ownKey]) return prev;
      const next = { ...prev, [ownKey]: { ...prev[ownKey], fadeIn, fadeOut } };
      if (isPlaying) restartMix(next);
      return next;
    });
  }

  const keyLabel = keyInfo ? `${NOTE_NAMES[keyInfo.tonic]}-${keyInfo.mode === 'major' ? 'dur' : 'moll'}` : null;
  const keyIsManual = !!(keyInfo && detectedKeyRef.current
    && (keyInfo.tonic !== detectedKeyRef.current.tonic || keyInfo.mode !== detectedKeyRef.current.mode));
  function resetKeyToDetected() {
    if (detectedKeyRef.current) overrideKey(detectedKeyRef.current.tonic, detectedKeyRef.current.mode);
  }
  const anyChannelEnabled = Object.values(channels).some((c) => c.enabled);
  const anyHarmonyBusy = Object.values(harmonyRenderingByType).some(Boolean);

  if (showAbout) {
    return <AboutPage />;
  }

  return (
    <div className="min-h-screen w-full flex justify-center" style={{ backgroundColor: 'var(--bg)', color: 'var(--text)' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
        .font-display { font-family: 'Fraunces', serif; }
        .font-body { font-family: 'Space Grotesk', sans-serif; }
        .font-mono-ui { font-family: 'JetBrains Mono', monospace; }
        @keyframes pulseRec { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .35; transform: scale(0.82); } }
        .rec-dot { animation: pulseRec 1.4s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) { .rec-dot { animation: none; } }
        .stamma-btn:focus-visible { outline: 2px solid var(--orange); outline-offset: 2px; }
        .stamma-fader { -webkit-appearance: none; appearance: none; height: 4px; border-radius: 2px; background: rgba(var(--ink-rgb),0.15); outline: none; }
        .stamma-fader::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; background: var(--text); cursor: pointer; }
        .stamma-fader::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: var(--text); border: none; cursor: pointer; }
        input[type="number"]::-webkit-inner-spin-button, input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
      `}</style>

      <div className="w-full max-w-md px-5 py-8 font-body">
        {restorableSession && phase === 'idle' && (
          <RestoreBanner
            name={restorableSession.name}
            savedAt={restorableSession.savedAt}
            restoring={restoringSession}
            onRestore={restoreSession}
            onDismiss={dismissRestorableSession}
          />
        )}

        {/* Header */}
        <header className="mb-6">
          <div className="flex items-center justify-between gap-2">
            <h1 className="font-display text-4xl font-semibold tracking-tight" style={{ color: 'var(--text)' }}>
              Stämifier
            </h1>
            <button
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              className="stamma-btn shrink-0 rounded-full flex items-center justify-center"
              style={{
                width: 44,
                height: 44,
                color: 'var(--text-muted)',
                border: '1px solid rgba(var(--ink-rgb),0.15)',
              }}
              aria-label={theme === 'dark' ? 'Byt till ljust tema' : 'Byt till mörkt tema'}
              title={theme === 'dark' ? 'Byt till ljust tema' : 'Byt till mörkt tema'}
            >
              {theme === 'dark' ? <SunIcon size={18} /> : <MoonIcon size={18} />}
            </button>
          </div>
          <p className="mt-2 text-base leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            Sjung in eller ladda upp en melodi (max 10s).
            {!introExpanded && (
              <>
                {' '}
                <button
                  onClick={() => setIntroExpanded(true)}
                  className="stamma-btn font-mono-ui text-xs align-middle"
                  style={{ color: 'var(--teal)' }}
                >
                  Läs mer <span style={{ fontSize: 15 }}>▾</span>
                </button>
              </>
            )}
          </p>
          {introExpanded && (
            <p className="mt-1 text-base leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Appen känner av tonarten och bygger stämmor i ters, kvint och sext som du kan mixa och träna in.{' '}
              <button
                onClick={() => setIntroExpanded(false)}
                className="stamma-btn font-mono-ui text-xs align-middle"
                style={{ color: 'var(--teal)' }}
              >
                Visa mindre <span style={{ fontSize: 15 }}>▴</span>
              </button>
            </p>
          )}
        </header>

        {/* Sessioner — saved projects, see indexedDb.js */}
        {projects.length > 0 && phase !== 'recording' && phase !== 'countin' && (
          <div className="rounded-xl p-3 mb-5" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}>
            <button
              onClick={() => setProjectsExpanded((v) => !v)}
              className="stamma-btn w-full flex items-center justify-between"
            >
              <span className="text-sm font-medium">Sessioner ({projects.length})</span>
              <span style={{ display: 'inline-block', fontSize: 15, color: 'var(--text-muted)', transform: projectsExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
            </button>
            {projectsExpanded && (
              <div className="mt-3 space-y-2">
                {projects.map((p) => (
                  <ProjectRow
                    key={p.id}
                    project={p}
                    active={p.id === activeProjectIdRef.current}
                    onOpen={() => switchToProject(p.id)}
                    onRename={(name) => handleRenameProject(p.id, name)}
                    onDelete={() => handleDeleteProject(p.id)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Signature visualization — nothing to show until an analysis has
            actually produced melody notes/a waveform, so this stays hidden
            through idle and error rather than rendering an empty shell. */}
        {phase === 'ready' && (
        <div className="rounded-2xl p-3 mb-5" style={{ backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--ink-rgb),0.08)' }}>
          <button
            onClick={() => setNoteViewExpanded((v) => !v)}
            className="stamma-btn w-full flex items-center justify-between"
          >
            <h2 className="font-display text-lg font-semibold">Tonhöjdskurva &amp; Vågform</h2>
            <span style={{ display: 'inline-block', fontSize: 17, color: 'var(--text-muted)', transform: noteViewExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
          </button>
          {noteViewExpanded && (
            <>
              <div className="mt-2">
                <PitchCanvas
                  melodyNotes={melodyNotes}
                  harmonyLayers={harmonyLayers}
                  keyInfo={keyInfo}
                  keyLabel={keyLabel}
                  duration={recordingDuration}
                  playheadTime={playheadTime}
                  onKeyChange={overrideKey}
                  onResetKey={resetKeyToDetected}
                  keyIsManual={keyIsManual}
                  theme={theme}
                />
              </div>
              {melodyNotes.length > 0 && (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 px-1 font-mono-ui text-sm" style={{ color: 'var(--text-muted)' }}>
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MELODY_COLOR.line }} />
                    melodi
                  </span>
                  {harmonyLayers.map((layer) => (
                    <span key={layer.key} className="flex items-center gap-1.5">
                      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: layer.color }} />
                      {HARMONY_TYPES[layer.key].label.toLowerCase()}
                    </span>
                  ))}
                  {playheadTime !== null && (
                    <span className="ml-auto" style={{ color: 'var(--text)' }}>
                      {playheadTime.toFixed(1)}s / {recordingDuration.toFixed(1)}s
                    </span>
                  )}
                </div>
              )}
              {voiceReady && soundType === 'recording' && waveformPeaks && (
                <div className="mt-3 rounded-2xl p-3" style={{ backgroundColor: 'var(--bg)', border: '1px solid rgba(var(--ink-rgb),0.08)' }}>
                  <WaveformTrimmer
                    peaks={waveformPeaks}
                    duration={recordingDuration}
                    trimStart={trimStart}
                    trimEnd={effectiveTrimEnd}
                    onTrimChange={(start, end) => {
                      setTrimStart(start);
                      setTrimEnd(end);
                      if (isPlaying) stopPlayback();
                    }}
                    fadeIn={fadeIn}
                    fadeOut={fadeOut}
                    onFadeChange={(fi, fo) => {
                      setFadeIn(fi);
                      setFadeOut(fo);
                      if (isPlaying) stopPlayback();
                    }}
                    playheadTime={playheadTime}
                    isPlaying={isPlaying}
                    onPlayPause={togglePlayPause}
                    onSeek={seekTo}
                    playDisabled={!anyChannelEnabled || anyHarmonyBusy || autotuneRendering}
                    onStop={() => stopPlayback()}
                    loopEnabled={loopEnabled}
                    onToggleLoop={() => setLoopEnabled((v) => !v)}
                    busy={anyHarmonyBusy || autotuneRendering}
                    metronomeEnabled={metronomeEnabled}
                    onToggleMetronome={toggleMetronomeEnabled}
                    noiseReductionMode={noiseReductionMode}
                    onToggleNoiseReductionMode={() => (noiseReductionMode ? setNoiseReductionMode(false) : enterNoiseReductionMode())}
                    noiseSampleStart={noiseSampleStart}
                    noiseSampleEnd={noiseSampleEnd}
                    onNoiseSampleChange={(s, e) => { setNoiseSampleStart(s); setNoiseSampleEnd(e); }}
                    onApplyNoiseReduction={applyNoiseReduction}
                    denoising={denoising}
                    denoiseError={denoiseError}
                    noiseReductionApplied={noiseReductionApplied}
                    theme={theme}
                  />

                  <div className="flex items-center gap-2 mt-3">
                    <button
                      onClick={normalizeRecording}
                      disabled={normalizing}
                      className="stamma-btn flex-1 rounded-xl py-2 font-body font-medium text-xs"
                      style={{
                        backgroundColor: 'rgba(var(--ink-rgb),0.06)',
                        color: normalizing ? 'rgba(var(--ink-rgb),0.3)' : 'var(--text)',
                        border: '1px solid rgba(var(--ink-rgb),0.12)',
                        cursor: normalizing ? 'not-allowed' : 'pointer',
                      }}
                    >
                      {normalizing ? 'Normaliserar …' : 'Normalisera'}
                    </button>
                    {isProcessed && (
                      <button
                        onClick={revertToOriginalRecording}
                        className="stamma-btn flex-1 rounded-xl py-2 font-body font-medium text-xs"
                        style={{ backgroundColor: 'rgba(var(--red-rgb),0.1)', color: 'var(--red-softer)', border: '1px solid rgba(var(--red-rgb),0.3)' }}
                      >
                        Återställ till original
                      </button>
                    )}
                  </div>
                  {normalizeError && (
                    <div className="mt-2 text-xs" style={{ color: 'var(--red-softer)' }}>{normalizeError}</div>
                  )}

                  <div className="mt-3">
                    <TransportButtons
                      loopEnabled={loopEnabled}
                      onToggleLoop={() => setLoopEnabled((v) => !v)}
                      onStop={() => stopPlayback()}
                      isPlaying={isPlaying}
                      onPlayPause={togglePlayPause}
                      disabled={!anyChannelEnabled}
                      busy={anyHarmonyBusy || autotuneRendering}
                      metronomeEnabled={metronomeEnabled}
                      onToggleMetronome={toggleMetronomeEnabled}
                    />
                  </div>

                  <div className="flex items-center gap-2 mt-2">
                    <span className="font-mono-ui text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>
                      {(playheadTime !== null ? playheadTime : trimStart).toFixed(1)}s
                    </span>
                    <input
                      type="range"
                      min={trimStart}
                      max={effectiveTrimEnd}
                      step="0.01"
                      value={playheadTime !== null ? playheadTime : trimStart}
                      onChange={(e) => seekTo(parseFloat(e.target.value))}
                      className="stamma-fader w-full"
                      style={{ accentColor: 'var(--orange)' }}
                      aria-label="Spola i vågformen"
                    />
                    <span className="font-mono-ui text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>{effectiveTrimEnd.toFixed(1)}s</span>
                  </div>

                  <div className="mt-3 pt-3 relative" style={{ borderTop: '1px solid rgba(var(--ink-rgb),0.08)' }}>
                    <div className="flex items-center gap-1.5">
                      <MetronomeIcon size={14} />
                      <button
                        onClick={() => setMetronomeExpanded((v) => !v)}
                        className="stamma-btn flex items-center gap-1 font-mono-ui text-xs"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {metronomeBpm} BPM
                        <span style={{ display: 'inline-block', fontSize: 15, transform: metronomeExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                      </button>
                    </div>
                    {metronomeExpanded && (
                      <div
                        className="absolute z-10 mt-2 rounded-xl p-3"
                        style={{ width: 220, backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--ink-rgb),0.12)', boxShadow: '0 8px 24px rgba(0,0,0,0.4)' }}
                      >
                        <div className="flex items-center justify-center gap-4">
                          <button
                            onClick={() => adjustMetronomeBpm(-1)}
                            className="stamma-btn rounded-md flex items-center justify-center"
                            style={{ width: 44, height: 44, backgroundColor: 'rgba(var(--ink-rgb),0.06)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
                            aria-label="Sänk takt"
                          >
                            −
                          </button>
                          <BpmInput
                            value={metronomeBpm}
                            onCommit={setMetronomeBpmTo}
                            className="font-mono-ui text-lg"
                            style={{ minWidth: 56, color: 'var(--text)' }}
                          />
                          <button
                            onClick={() => adjustMetronomeBpm(1)}
                            className="stamma-btn rounded-md flex items-center justify-center"
                            style={{ width: 44, height: 44, backgroundColor: 'rgba(var(--ink-rgb),0.06)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
                            aria-label="Höj takt"
                          >
                            +
                          </button>
                        </div>
                        {tempoDetected && (
                          <div className="mt-1.5 text-center font-mono-ui text-[10px]" style={{ color: 'var(--teal)' }}>
                            Upptäckt från inspelningen
                          </div>
                        )}
                        <button
                          onClick={toggleMetronomeListen}
                          className="stamma-btn w-full mt-3 rounded-lg py-1.5 flex items-center justify-center gap-1.5 font-body font-medium text-xs"
                          style={{
                            backgroundColor: metronomeListening ? 'rgba(var(--teal-rgb),0.15)' : 'rgba(var(--ink-rgb),0.06)',
                            color: metronomeListening ? 'var(--teal)' : 'var(--text)',
                            border: metronomeListening ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                          }}
                        >
                          {metronomeListening ? <PauseIcon size={13} /> : <PlayIcon size={13} />}
                          {metronomeListening ? 'Stoppa' : 'Hitta takten'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
          {phase === 'ready' && !(noteViewExpanded && voiceReady && soundType === 'recording' && waveformPeaks) && (
            <div className="mt-3">
              <TransportButtons
                loopEnabled={loopEnabled}
                onToggleLoop={() => setLoopEnabled((v) => !v)}
                onStop={() => stopPlayback()}
                isPlaying={isPlaying}
                onPlayPause={togglePlayPause}
                disabled={!anyChannelEnabled}
                busy={anyHarmonyBusy || autotuneRendering}
                metronomeEnabled={metronomeEnabled}
                onToggleMetronome={toggleMetronomeEnabled}
              />
            </div>
          )}
        </div>
        )}

        {/* Error banner */}
        {errorMsg && (
          <div className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(var(--red-rgb),0.1)', border: '1px solid rgba(var(--red-rgb),0.3)', color: 'var(--red-softer)' }}>
            {errorMsg}
          </div>
        )}
        {autotuneRenderError && (
          <div className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(var(--red-rgb),0.1)', border: '1px solid rgba(var(--red-rgb),0.3)', color: 'var(--red-softer)' }}>
            {autotuneRenderError}
          </div>
        )}
        {Object.entries(harmonyRenderErrorsByType).map(([type, msg]) => msg && (
          <div key={type} className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(var(--red-rgb),0.1)', border: '1px solid rgba(var(--red-rgb),0.3)', color: 'var(--red-softer)' }}>
            {msg}
          </div>
        ))}
        {ownTakeError && (
          <div className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(var(--red-rgb),0.1)', border: '1px solid rgba(var(--red-rgb),0.3)', color: 'var(--red-softer)' }}>
            {ownTakeError}
          </div>
        )}

        {/* Idle: start button + upload */}
        {(phase === 'idle' || phase === 'error') && (
          <div className="space-y-3">
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}>
              <div className="flex items-center gap-2">
                <MetronomeIcon size={16} />
                <span className="text-sm font-medium">Metronom</span>
                <span className="ml-auto font-mono-ui text-xs" style={{ color: 'var(--text-muted)' }}>Klick vid inspelning</span>
                <ToggleSwitch checked={metronomeEnabled} onChange={() => setMetronomeEnabled((v) => !v)} accentColor="var(--teal)" ariaLabel="Metronom vid inspelning" />
              </div>
              <div className="mt-3 pt-3" style={{ borderTop: '1px solid rgba(var(--ink-rgb),0.08)' }}>
                <button
                  onClick={() => setMetronomeExpanded((v) => !v)}
                  className="stamma-btn w-full flex items-center justify-between font-mono-ui text-xs"
                  style={{ color: 'var(--teal)' }}
                >
                  <span>Takt: {metronomeBpm} BPM</span>
                  <span style={{ display: 'inline-block', fontSize: 15, transform: metronomeExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                </button>
                {metronomeExpanded && (
                  <div className="mt-2.5">
                    <div className="flex items-center justify-center gap-4">
                      <button
                        onClick={() => adjustMetronomeBpm(-1)}
                        className="stamma-btn rounded-md flex items-center justify-center"
                        style={{ width: 44, height: 44, backgroundColor: 'rgba(var(--ink-rgb),0.06)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
                        aria-label="Sänk takt"
                      >
                        −
                      </button>
                      <span className="flex items-center justify-center gap-1" style={{ minWidth: 90 }}>
                        <BpmInput
                          value={metronomeBpm}
                          onCommit={setMetronomeBpmTo}
                          className="font-mono-ui text-xl"
                          style={{ width: 46, color: 'var(--text)' }}
                        />
                        <span className="font-mono-ui text-xl" style={{ color: 'var(--text)' }}>BPM</span>
                      </span>
                      <button
                        onClick={() => adjustMetronomeBpm(1)}
                        className="stamma-btn rounded-md flex items-center justify-center"
                        style={{ width: 44, height: 44, backgroundColor: 'rgba(var(--ink-rgb),0.06)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
                        aria-label="Höj takt"
                      >
                        +
                      </button>
                    </div>
                    <button
                      onClick={toggleMetronomeListen}
                      className="stamma-btn w-full mt-3 rounded-lg py-2 flex items-center justify-center gap-1.5 font-body font-medium text-sm"
                      style={{
                        backgroundColor: metronomeListening ? 'rgba(var(--teal-rgb),0.15)' : 'rgba(var(--ink-rgb),0.06)',
                        color: metronomeListening ? 'var(--teal)' : 'var(--text)',
                        border: metronomeListening ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                      }}
                    >
                      {metronomeListening ? <PauseIcon size={15} /> : <PlayIcon size={15} />}
                      {metronomeListening ? 'Stoppa' : 'Lyssna på takten'}
                    </button>
                  </div>
                )}
              </div>
            </div>
            <LiveMonitorCard
              on={liveMonitorOn}
              starting={liveMonitorStarting}
              error={liveMonitorError}
              types={liveMonitorTypes}
              latencyMs={liveMonitorLatencyMs}
              keyAware={liveMonitorKeyAware}
              keyInfo={keyInfo}
              humanizeOn={humanizeOn}
              onToggle={() => (liveMonitorOn ? stopLiveMonitor() : startLiveMonitor())}
              onToggleType={toggleLiveMonitorVoice}
            />
            <button
              onClick={startRecording}
              className="stamma-btn w-full rounded-2xl py-4 font-body font-medium text-base transition-transform active:scale-[0.98]"
              style={{ backgroundColor: 'var(--orange)', color: 'var(--bg)' }}
            >
              Spela in (max 10s)
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="stamma-btn w-full rounded-xl py-3 font-body font-medium text-sm"
              style={{ backgroundColor: 'rgba(var(--ink-rgb),0.06)', color: 'var(--text)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
            >
              Ladda upp ljudfil
            </button>
          </div>
        )}

        {/* Shared by both the idle upload button and the "ready" phase's
            "Ladda upp ny ljudfil" — always rendered so both can reach it. */}
        <input
          ref={fileInputRef}
          type="file"
          // No `accept` filter: iOS Safari's iCloud Drive picker ANDs a
          // MIME wildcard against listed extensions instead of ORing
          // them, and iCloud often doesn't report MIME metadata for a
          // file — so audio/* fails silently and the file (even a
          // plain .wav) shows up grayed out regardless of extension.
          // handleFileUpload already rejects anything decodeAudioData
          // can't read, with a clear error, so filtering here isn't
          // load-bearing — it was just actively breaking iCloud files.
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            handleFileUpload(file);
          }}
        />

        {/* Count-in — plays before the mic actually starts recording */}
        {phase === 'countin' && (
          <div className="rounded-2xl p-5 flex flex-col items-center" style={{ backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--orange-rgb),0.3)' }}>
            <span className="font-mono-ui text-sm" style={{ color: 'var(--orange)' }}>GÖR DIG REDO</span>
            <span className="font-display text-6xl font-semibold mt-2 tabular-nums" style={{ color: 'var(--text)' }}>
              {COUNT_IN_BEATS - countInBeat}
            </span>
            <button
              onClick={cancelCountIn}
              className="stamma-btn mt-4 rounded-xl py-2 px-5 font-body font-medium text-sm"
              style={{ backgroundColor: 'rgba(var(--ink-rgb),0.06)', color: 'var(--text-muted)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
            >
              Avbryt
            </button>
          </div>
        )}

        {/* Recording */}
        {phase === 'recording' && (
          <div className="rounded-2xl p-5" style={{ backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--red-rgb),0.3)' }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="rec-dot inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: 'var(--red)' }} />
              <span className="font-mono-ui text-sm" style={{ color: 'var(--red)' }}>SPELAR IN</span>
              <span className="ml-auto font-mono-ui text-2xl" style={{ color: 'var(--text)' }}>{Math.ceil(countdown)}s</span>
            </div>
            <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.1)' }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${((DURATION - countdown) / DURATION) * 100}%`, backgroundColor: 'var(--red)', transition: 'width 80ms linear' }}
              />
            </div>
            <p className="mt-3 text-base" style={{ color: 'var(--text-muted)' }}>Sjung en enkel melodislinga, gärna med tydliga toner.</p>
            <button
              onClick={stopRecordingEarly}
              className="stamma-btn w-full mt-4 rounded-xl py-3 font-body font-medium text-base transition-transform active:scale-[0.98]"
              style={{ backgroundColor: 'rgba(var(--red-rgb),0.15)', color: 'var(--red-soft)', border: '1px solid rgba(var(--red-rgb),0.4)' }}
            >
              Stoppa inspelning nu
            </button>
          </div>
        )}

        {/* Analyzing */}
        {phase === 'analyzing' && (
          <div className="rounded-2xl p-5 flex items-center gap-3" style={{ backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--ink-rgb),0.08)' }}>
            <span
              className="inline-block w-5 h-5 rounded-full animate-spin"
              style={{ border: '2px solid rgba(var(--ink-rgb),0.15)', borderTopColor: 'var(--orange)' }}
            />
            <span className="text-base" style={{ color: 'var(--text-muted)' }}>Analyserar tonart och melodi …</span>
          </div>
        )}

        {/* Ready: sound mode, mixer, export */}
        {phase === 'ready' && (
          <div className="space-y-5">
            <LiveMonitorCard
              on={liveMonitorOn}
              starting={liveMonitorStarting}
              error={liveMonitorError}
              types={liveMonitorTypes}
              latencyMs={liveMonitorLatencyMs}
              keyAware={liveMonitorKeyAware}
              keyInfo={keyInfo}
              humanizeOn={humanizeOn}
              followingOwnTake={liveMonitorFollowingOwnTake}
              onToggle={() => (liveMonitorOn ? stopLiveMonitor() : startLiveMonitor())}
              onToggleType={toggleLiveMonitorVoice}
            />
            <div>
              <button
                onClick={() => setLjudExpanded((v) => !v)}
                className="stamma-btn w-full flex items-center justify-between mb-2"
              >
                <h2 className="font-display text-lg font-semibold">Röst</h2>
                <span style={{ display: 'inline-block', fontSize: 17, color: 'var(--text-muted)', transform: ljudExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
              </button>
              {ljudExpanded && (
                <>
                  <div className="grid grid-cols-3 gap-2">
                    {Object.keys(SOUND_TYPES).map((type) => {
                      const active = soundType === type;
                      const disabled = type === 'recording' && !voiceReady;
                      return (
                        <button
                          key={type}
                          onClick={() => { if (!disabled) { setSoundType(type); if (isPlaying) stopPlayback(); } }}
                          disabled={disabled}
                          className="stamma-btn rounded-xl py-3 font-body font-medium text-sm transition-colors"
                          style={{
                            backgroundColor: disabled ? 'rgba(var(--ink-rgb),0.03)' : active ? 'var(--orange)' : 'rgba(var(--ink-rgb),0.06)',
                            color: disabled ? 'rgba(var(--ink-rgb),0.25)' : active ? 'var(--bg)' : 'var(--text)',
                            border: active ? '1px solid var(--orange)' : '1px solid rgba(var(--ink-rgb),0.12)',
                            cursor: disabled ? 'not-allowed' : 'pointer',
                          }}
                        >
                          {SOUND_TYPES[type].label}
                        </button>
                      );
                    })}
                  </div>
                  <p className="mt-3 text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                    {SOUND_TYPES[soundType].description}
                  </p>
                </>
              )}
            </div>

            <div>
              <button
                onClick={() => setEffectsExpanded((v) => !v)}
                className="stamma-btn w-full flex items-center justify-between mb-2"
              >
                <h2 className="font-display text-lg font-semibold">Effekter</h2>
                <span style={{ display: 'inline-block', fontSize: 17, color: 'var(--text-muted)', transform: effectsExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
              </button>
              {effectsExpanded && (
                <div className="space-y-3">
                  {soundType === 'recording' && (
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}>
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium">Autotune</div>
                          <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                            Rättar falska toner i din inspelning{autotuneRendering ? ' (bygger …)' : ''}
                          </div>
                        </div>
                        <ToggleSwitch
                          checked={autotuneOn}
                          onChange={() => { setAutotuneOn((v) => !v); if (isPlaying) stopPlayback(); }}
                          accentColor="var(--teal)"
                          ariaLabel="Autotune"
                        />
                      </div>
                      {autotuneOn && (
                        <div className="mt-3 pt-3" style={{ borderTop: '1px solid rgba(var(--ink-rgb),0.08)' }}>
                          <button
                            onClick={() => setAutotuneStrengthExpanded((v) => !v)}
                            className="stamma-btn w-full flex items-center justify-between font-mono-ui text-xs"
                            style={{ color: 'var(--teal)' }}
                          >
                            <span>Styrka: {AUTOTUNE_LEVELS[autotuneLevelIndex].label}</span>
                            <span style={{ display: 'inline-block', fontSize: 15, transform: autotuneStrengthExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                          </button>
                          {autotuneStrengthExpanded && (
                            <div className="mt-2.5 grid grid-cols-3 gap-1.5">
                              {AUTOTUNE_LEVELS.map((lvl, i) => (
                                <button
                                  key={lvl.key}
                                  onClick={() => { setAutotuneLevelIndex(i); if (isPlaying) stopPlayback(); }}
                                  className="stamma-btn rounded-md py-1.5 text-xs font-medium"
                                  style={{
                                    backgroundColor: autotuneLevelIndex === i ? 'rgba(var(--teal-rgb),0.15)' : 'transparent',
                                    color: autotuneLevelIndex === i ? 'var(--teal)' : 'var(--text-muted)',
                                    border: autotuneLevelIndex === i ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                                  }}
                                >
                                  {lvl.label}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}>
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-medium">Reverb</div>
                        <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                          Lägger på lite rymd på hela mixen
                        </div>
                      </div>
                      <ToggleSwitch
                        checked={reverbOn}
                        onChange={() => { setReverbOn((v) => !v); if (isPlaying) stopPlayback(); }}
                        accentColor="var(--teal)"
                        ariaLabel="Reverb"
                      />
                    </div>
                    {reverbOn && (
                      <div className="mt-3 pt-3" style={{ borderTop: '1px solid rgba(var(--ink-rgb),0.08)' }}>
                        <button
                          onClick={() => setReverbStrengthExpanded((v) => !v)}
                          className="stamma-btn w-full flex items-center justify-between font-mono-ui text-xs"
                          style={{ color: 'var(--teal)' }}
                        >
                          <span>Storlek: {REVERB_LEVELS[reverbLevelIndex].label}</span>
                          <span style={{ display: 'inline-block', fontSize: 15, transform: reverbStrengthExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                        </button>
                        {reverbStrengthExpanded && (
                          <div className="mt-2.5 grid grid-cols-3 gap-1.5">
                            {REVERB_LEVELS.map((lvl, i) => (
                              <button
                                key={lvl.key}
                                onClick={() => { setReverbLevelIndex(i); if (isPlaying) stopPlayback(); }}
                                className="stamma-btn rounded-md py-1.5 text-xs font-medium"
                                style={{
                                  backgroundColor: reverbLevelIndex === i ? 'rgba(var(--teal-rgb),0.15)' : 'transparent',
                                  color: reverbLevelIndex === i ? 'var(--teal)' : 'var(--text-muted)',
                                  border: reverbLevelIndex === i ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                                }}
                              >
                                {lvl.label}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {soundType === 'recording' && (
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}>
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium">Humanisera stämmor</div>
                          <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                            Ger ters/kvint/sext egen vibrato, formant och tonhöjdskaraktär istället för att låta som en klonad kopia av din röst
                          </div>
                        </div>
                        <ToggleSwitch
                          checked={humanizeOn}
                          onChange={() => { setHumanizeOn((v) => !v); if (isPlaying) stopPlayback(); }}
                          accentColor="var(--teal)"
                          ariaLabel="Humanisera stämmor"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div>
              <div className="flex items-center justify-between mb-2 gap-2">
                <button
                  onClick={() => setMixerExpanded((v) => !v)}
                  className="stamma-btn flex-1 flex items-center justify-between min-w-0"
                >
                  <h2 className="font-display text-lg font-semibold">Mixer</h2>
                  <span style={{ display: 'inline-block', fontSize: 17, color: 'var(--text-muted)', transform: mixerExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                </button>
                {isPlaying && (
                  <button onClick={() => stopPlayback()} className="stamma-btn text-xs underline shrink-0" style={{ color: 'var(--text-muted)' }}>
                    Stoppa
                  </button>
                )}
              </div>
              {mixerExpanded && (
                <>
              <p className="text-sm leading-relaxed mb-3" style={{ color: 'var(--text-muted)' }}>
                Slå på de kanaler du vill höra, ställ nivåerna, och tryck play — allt aktiverat spelas samtidigt.
              </p>

              <div className="flex items-center justify-end mb-2 gap-2">
                <button
                  onClick={toggleAllChannelsExpanded}
                  className="stamma-btn flex items-center gap-1 font-mono-ui text-xs shrink-0"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {allChannelsExpanded ? 'Göm alla' : 'Expandera alla'}
                  <span style={{ display: 'inline-block', fontSize: 15, transform: allChannelsExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                </button>
              </div>

              <div className="space-y-2">
                <MixerChannel
                  label={soundType === 'recording' ? (autotuneEnabled ? 'Melodi (autotunad)' : 'Melodi (din röst)') : 'Melodi (ren ton)'}
                  accentColor={MELODY_COLOR.line}
                  enabled={channels.melody.enabled}
                  onToggle={() => toggleChannel('melody')}
                  volume={channels.melody.volume}
                  onVolumeChange={(v) => setChannelVolume('melody', v)}
                  pan={channels.melody.pan}
                  onPanChange={(p) => setChannelPan('melody', p)}
                  solo={channels.melody.solo}
                  onToggleSolo={() => toggleChannelSolo('melody')}
                  meterRef={(el) => { meterRefs.current.melody = el; }}
                  busy={autotuneRendering}
                  previewing={isPlaying && previewingKey === 'melody'}
                  onPreview={() => (isPlaying && previewingKey === 'melody' ? stopPlayback() : previewChannel('melody'))}
                  expanded={!!expandedChannels.melody}
                  onToggleExpanded={() => toggleChannelExpanded('melody')}
                />
                {HARMONY_KEYS.flatMap((type) => {
                  const ownKey = `${type}Own`;
                  const ownChannel = channels[ownKey];
                  return [
                      <MixerChannel
                        key={type}
                        label={HARMONY_TYPES[type].label}
                        accentColor={HARMONY_COLORS[type].line}
                        enabled={channels[type].enabled}
                        onToggle={() => toggleChannel(type)}
                        volume={channels[type].volume}
                        onVolumeChange={(v) => setChannelVolume(type, v)}
                        pan={channels[type].pan}
                        onPanChange={(p) => setChannelPan(type, p)}
                        solo={channels[type].solo}
                        onToggleSolo={() => toggleChannelSolo(type)}
                        meterRef={(el) => { meterRefs.current[type] = el; }}
                        busy={harmonyRenderingByType[type]}
                        direction={channels[type].direction}
                        onSetDirection={(d) => setChannelDirection(type, d)}
                        previewing={isPlaying && previewingKey === type}
                        onPreview={() => (isPlaying && previewingKey === type ? stopPlayback() : previewChannel(type))}
                        expanded={!!expandedChannels[type]}
                        onToggleExpanded={() => toggleChannelExpanded(type)}
                        onRecord={() => recordOwnTake(type)}
                        onStopRecordEarly={stopOwnTakeRecordingEarly}
                        recording={recordingOwnTake?.type === type}
                        recordCountingIn={recordingOwnTake?.type === type && !!recordingOwnTake.countingIn}
                        recordCountInBeat={recordingOwnTake?.type === type ? recordingOwnTake.countInBeat : null}
                        recordCountdown={recordingOwnTake?.type === type ? recordingOwnTake.countdown : null}
                        recordDisabled={!!recordingOwnTake}
                      />,
                      ownChannel && (
                        <MixerChannel
                          key={ownKey}
                          label={`Egen ${HARMONY_TYPES[type].label.toLowerCase()}`}
                          accentColor={HARMONY_COLORS[type].line}
                          enabled={ownChannel.enabled}
                          onToggle={() => toggleChannel(ownKey)}
                          volume={ownChannel.volume}
                          onVolumeChange={(v) => setChannelVolume(ownKey, v)}
                          pan={ownChannel.pan}
                          onPanChange={(p) => setChannelPan(ownKey, p)}
                          solo={ownChannel.solo}
                          onToggleSolo={() => toggleChannelSolo(ownKey)}
                          meterRef={(el) => { meterRefs.current[ownKey] = el; }}
                          busy={ownTakeAligningByType[type]}
                          previewing={isPlaying && previewingKey === ownKey}
                          onPreview={() => (isPlaying && previewingKey === ownKey ? stopPlayback() : previewChannel(ownKey))}
                          expanded={!!expandedChannels[ownKey]}
                          onToggleExpanded={() => toggleChannelExpanded(ownKey)}
                          onDelete={() => deleteOwnTake(type)}
                          waveform={ownTakeWaveformPeaksByType[type] && (
                            <div className="mt-3 rounded-xl p-3" style={{ backgroundColor: 'var(--bg)', border: '1px solid rgba(var(--ink-rgb),0.08)' }}>
                              <WaveformTrimmer
                                peaks={ownTakeWaveformPeaksByType[type]}
                                duration={recordingDuration}
                                trimStart={ownChannel.trimStart ?? trimStart}
                                trimEnd={ownChannel.trimEnd ?? effectiveTrimEnd}
                                onTrimChange={(s, e) => setOwnTakeTrim(type, s, e)}
                                fadeIn={ownChannel.fadeIn ?? fadeIn}
                                fadeOut={ownChannel.fadeOut ?? fadeOut}
                                onFadeChange={(fi, fo) => setOwnTakeFade(type, fi, fo)}
                                playheadTime={playheadTime}
                                isPlaying={isPlaying}
                                onPlayPause={togglePlayPause}
                                onSeek={seekTo}
                                playDisabled={!anyChannelEnabled}
                                onStop={() => stopPlayback()}
                                loopEnabled={loopEnabled}
                                onToggleLoop={() => setLoopEnabled((v) => !v)}
                                busy={ownTakeAligningByType[type]}
                                theme={theme}
                              />
                            </div>
                          )}
                        />
                      ),
                  ].filter(Boolean);
                })}
              </div>

              <div className="mt-4">
                <TransportButtons
                  loopEnabled={loopEnabled}
                  onToggleLoop={() => setLoopEnabled((v) => !v)}
                  onStop={() => stopPlayback()}
                  isPlaying={isPlaying}
                  onPlayPause={togglePlayPause}
                  disabled={!anyChannelEnabled}
                  busy={anyHarmonyBusy || autotuneRendering}
                />
              </div>
                </>
              )}
            </div>

            <button
              onClick={resetAll}
              className="stamma-btn w-full rounded-xl py-3 font-body font-medium text-base transition-transform active:scale-[0.98]"
              style={{ backgroundColor: 'var(--red)', color: 'var(--bg)' }}
            >
              Spela in ny melodi
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="stamma-btn w-full mt-2 rounded-xl py-3 font-body font-medium text-sm"
              style={{ backgroundColor: 'rgba(var(--ink-rgb),0.06)', color: 'var(--text)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
            >
              Ladda upp ny ljudfil
            </button>

            <div>
              <button
                onClick={() => setExportExpanded((v) => !v)}
                className="stamma-btn w-full flex items-center justify-between mb-2"
              >
                <h2 className="font-display text-lg font-semibold">Exportera</h2>
                <span style={{ display: 'inline-block', fontSize: 17, color: 'var(--text-muted)', transform: exportExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
              </button>
              {exportExpanded && (
                <>
                  <p className="text-sm leading-relaxed mb-2" style={{ color: 'var(--text-muted)' }}>
                    Ladda ner hela mixen som den låter nu, eller sång och stämmor var för sig som separata WAV-filer, t.ex. för att jobba vidare i Waveform.
                  </p>
                  <div className="grid grid-cols-1 gap-2">
                    <ExportButton
                      label="Hela mixen"
                      busy={exporting === 'mixdown'}
                      disabled={!anyChannelEnabled || !!exporting || anyHarmonyBusy || autotuneRendering}
                      onClick={exportMixdown}
                    />
                    <ExportButton
                      label="Sång (original)"
                      busy={exporting === 'song'}
                      disabled={!voiceReady || soundType !== 'recording' || !!exporting}
                      onClick={exportSong}
                    />
                    <ExportButton
                      label={
                        soundType === 'recording'
                          ? autotuneEnabled ? 'Melodi (autotunad)' : 'Melodi (samma som originalet)'
                          : 'Melodislinga (ren ton)'
                      }
                      busy={exporting === 'melody' || autotuneRendering}
                      disabled={!melodyNotes.length || !!exporting || autotuneRendering}
                      onClick={exportMelody}
                    />
                    {HARMONY_KEYS.map((type) => (
                      <ExportButton
                        key={type}
                        label={`${HARMONY_TYPES[type].label} (${channels[type].direction === -1 ? 'under' : 'över'})`}
                        busy={exporting === `harmony-${type}` || harmonyRenderingByType[type]}
                        disabled={!melodyNotes.length || !!exporting || harmonyRenderingByType[type]}
                        onClick={() => exportHarmonyType(type)}
                      />
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        <footer className="mt-8 text-center">
          <a
            href="#om"
            className="stamma-btn font-mono-ui text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            Om Stämifier
          </a>
        </footer>
      </div>
    </div>
  );
}

// Loop / Stop-and-rewind / Play-Pause, as one row — reused under the note
// window, under the waveform, and under the last mixer channel, so all
// three places control the exact same mix playback identically.
function TransportButtons({ loopEnabled, onToggleLoop, onStop, isPlaying, onPlayPause, disabled, busy, metronomeEnabled, onToggleMetronome }) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onStop}
        disabled={disabled}
        className="stamma-btn shrink-0 rounded-xl flex items-center justify-center"
        style={{
          width: 44,
          height: 44,
          backgroundColor: 'rgba(var(--ink-rgb),0.06)',
          color: disabled ? 'rgba(var(--ink-rgb),0.25)' : 'var(--text-muted)',
          border: '1px solid rgba(var(--ink-rgb),0.12)',
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
        aria-label="Stoppa och gå till start"
        title="Stoppa och gå till start"
      >
        <StopIcon size={16} />
      </button>
      {onToggleMetronome && (
        <button
          onClick={onToggleMetronome}
          className="stamma-btn shrink-0 rounded-xl flex items-center justify-center"
          style={{
            width: 44,
            height: 44,
            backgroundColor: metronomeEnabled ? 'rgba(var(--teal-rgb),0.15)' : 'rgba(var(--ink-rgb),0.06)',
            color: metronomeEnabled ? 'var(--teal)' : 'var(--text-muted)',
            border: metronomeEnabled ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
          }}
          aria-pressed={metronomeEnabled}
          aria-label="Metronom vid inspelning och uppspelning"
          title="Metronom vid inspelning och uppspelning"
        >
          <MetronomeIcon size={19} />
        </button>
      )}
      <button
        onClick={onPlayPause}
        disabled={disabled || busy}
        className="stamma-btn flex-1 rounded-xl py-3 flex items-center justify-center gap-2 font-body font-medium text-sm transition-transform active:scale-[0.98]"
        style={{
          backgroundColor: 'rgba(var(--ink-rgb),0.06)',
          color: (disabled || busy) ? 'rgba(var(--ink-rgb),0.3)' : 'var(--text-muted)',
          border: '1px solid rgba(var(--ink-rgb),0.12)',
          cursor: (disabled || busy) ? 'not-allowed' : 'pointer',
        }}
      >
        {isPlaying ? <PauseIcon size={19} /> : <PlayIcon size={19} />}
        {isPlaying ? 'Pausa' : busy ? 'Bygger …' : 'Spela mix'}
      </button>
      <button
        onClick={onToggleLoop}
        className="stamma-btn shrink-0 rounded-xl flex items-center justify-center"
        style={{
          width: 44,
          height: 44,
          backgroundColor: loopEnabled ? 'rgba(var(--teal-rgb),0.15)' : 'rgba(var(--ink-rgb),0.06)',
          color: loopEnabled ? 'var(--teal)' : 'var(--text-muted)',
          border: loopEnabled ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
        }}
        aria-pressed={loopEnabled}
        aria-label="Loopa uppspelning"
        title="Loopa uppspelning"
      >
        <LoopIcon size={19} />
      </button>
    </div>
  );
}

// Offers to bring back a session an earlier crash/refresh left behind in
// IndexedDB. Purely a prompt — restoring or dismissing is always an
// explicit click, never automatic.
function RestoreBanner({ name, savedAt, onRestore, onDismiss, restoring }) {
  const label = new Date(savedAt).toLocaleString('sv-SE', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
  return (
    <div
      className="rounded-xl p-4 mb-5 text-sm"
      style={{ backgroundColor: 'rgba(var(--teal-rgb),0.08)', border: '1px solid rgba(var(--teal-rgb),0.3)', color: 'var(--text)' }}
    >
      <p className="leading-relaxed font-medium">Återställ "{name}"?</p>
      <p className="mt-0.5 font-mono-ui text-xs" style={{ color: 'var(--text-muted)' }}>Sparad {label}</p>
      <div className="mt-3 flex gap-3">
        <button
          onClick={onRestore}
          disabled={restoring}
          className="stamma-btn rounded-lg py-2 px-4 font-body font-medium text-sm"
          style={{ backgroundColor: 'var(--teal)', color: 'var(--bg)', opacity: restoring ? 0.7 : 1 }}
        >
          {restoring ? 'Återställer…' : 'Återställ'}
        </button>
        <button
          onClick={onDismiss}
          disabled={restoring}
          className="stamma-btn font-mono-ui text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          Avfärda
        </button>
      </div>
    </div>
  );
}

// One row in the "Sessioner" list — click the name to rename it in place
// (tap-to-type, same pattern as BpmInput), "Öppna" loads it into the
// workspace, the trash button deletes it (native confirm(), a destructive
// action with no undo).
function ProjectRow({ project, active, onOpen, onRename, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(project.name);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const dateLabel = new Date(project.savedAt).toLocaleString('sv-SE', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  });

  function commit() {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== project.name) onRename(trimmed);
    else setDraft(project.name);
  }

  return (
    <div
      className="rounded-lg p-2.5 flex items-center gap-2"
      style={{
        backgroundColor: active ? 'rgba(var(--teal-rgb),0.08)' : 'rgba(var(--ink-rgb),0.03)',
        border: active ? '1px solid rgba(var(--teal-rgb),0.35)' : '1px solid rgba(var(--ink-rgb),0.08)',
      }}
    >
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onFocus={(e) => e.target.select()}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
              else if (e.key === 'Escape') { setDraft(project.name); setEditing(false); }
            }}
            className="w-full bg-transparent text-sm font-medium"
            style={{ color: 'var(--text)', border: 'none', borderBottom: '1px solid rgba(var(--teal-rgb),0.5)', outline: 'none' }}
          />
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="stamma-btn block w-full text-left text-sm font-medium truncate"
            style={{ color: 'var(--text)' }}
            title="Byt namn"
          >
            {project.name}
            {active && <span style={{ color: 'var(--teal)' }}> · aktiv</span>}
          </button>
        )}
        <div className="font-mono-ui text-[10px]" style={{ color: 'var(--text-muted)' }}>{dateLabel}</div>
      </div>
      <button
        onClick={onOpen}
        className="stamma-btn shrink-0 rounded-md px-2.5 py-1.5 font-mono-ui text-xs font-medium"
        style={{ color: 'var(--teal)', border: '1px solid rgba(var(--teal-rgb),0.4)' }}
      >
        Öppna
      </button>
      <button
        onClick={() => setConfirmingDelete(true)}
        className="stamma-btn shrink-0 rounded-md flex items-center justify-center"
        style={{ width: 44, height: 44, color: 'var(--text-muted)', border: '1px solid rgba(var(--ink-rgb),0.15)' }}
        aria-label={`Radera ${project.name}`}
        title={`Radera ${project.name}`}
      >
        <TrashIcon size={13} />
      </button>
      {confirmingDelete && (
        <ConfirmDialog
          title="Radera session?"
          message={`"${project.name}" tas bort permanent. Går inte att ångra.`}
          confirmLabel="Radera"
          onCancel={() => setConfirmingDelete(false)}
          onConfirm={() => { setConfirmingDelete(false); onDelete(); }}
        />
      )}
    </div>
  );
}

// Themed stand-in for window.confirm() — a native browser dialog can't
// pick up the app's own dark/light theme (see the theme toggle) and looks
// jarringly out of place next to it. `destructive` (the default) styles
// the confirm button as a warning (red); pass false for a neutral
// confirmation that isn't undoing/deleting anything.
function ConfirmDialog({ title, message, confirmLabel = 'Radera', cancelLabel = 'Avbryt', destructive = true, onConfirm, onCancel }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onCancel]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-5"
      style={{ backgroundColor: 'rgba(var(--bg-rgb),0.7)' }}
      onClick={onCancel}
    >
      <div
        className="rounded-2xl p-5 w-full"
        style={{ maxWidth: 320, backgroundColor: 'var(--card-bg)', border: '1px solid rgba(var(--ink-rgb),0.12)', boxShadow: '0 12px 32px rgba(0,0,0,0.4)' }}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="confirm-dialog-title" className="font-display text-lg font-semibold" style={{ color: 'var(--text)' }}>
          {title}
        </h2>
        <p id="confirm-dialog-message" className="mt-2 text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          {message}
        </p>
        <div className="mt-4 flex gap-2">
          <button
            onClick={onCancel}
            autoFocus
            className="stamma-btn flex-1 rounded-xl py-2.5 font-body font-medium text-sm"
            style={{ backgroundColor: 'rgba(var(--ink-rgb),0.06)', color: 'var(--text)', border: '1px solid rgba(var(--ink-rgb),0.12)' }}
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className="stamma-btn flex-1 rounded-xl py-2.5 font-body font-medium text-sm"
            style={
              destructive
                ? { backgroundColor: 'rgba(var(--red-rgb),0.15)', color: 'var(--red)', border: '1px solid rgba(var(--red-rgb),0.5)' }
                : { backgroundColor: 'var(--teal)', color: 'var(--bg)', border: 'none' }
            }
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

function ToggleSwitch({ checked, onChange, accentColor = 'var(--teal)', disabled, ariaLabel }) {
  return (
    // The visible track stays the standard 40x24 switch size — the button
    // itself is padded out to Apple's 44x44pt minimum tap target (HIG),
    // centered around that track, so the hit area grows without changing
    // how the control looks.
    <button
      onClick={onChange}
      disabled={disabled}
      className="stamma-btn shrink-0 flex items-center justify-center"
      style={{
        width: 44,
        height: 44,
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
    >
      <span
        style={{
          width: 40,
          height: 24,
          borderRadius: 12,
          backgroundColor: checked ? accentColor : 'rgba(var(--ink-rgb),0.15)',
          position: 'relative',
          transition: 'background-color 150ms ease',
        }}
      >
        <span
          style={{
            position: 'absolute',
            top: 2.5,
            left: checked ? 18 : 2.5,
            width: 19,
            height: 19,
            borderRadius: 10,
            backgroundColor: 'var(--bg)',
            transition: 'left 150ms ease',
          }}
        />
      </span>
    </button>
  );
}

// A tap-to-type BPM number, alongside the +/- steppers — keeps its own
// draft text while focused (so a mid-edit value like "1" isn't clamped/
// clobbered by a re-render before the user finishes typing) and only
// commits (clamped, via onCommit) on blur or Enter. Esc reverts.
function BpmInput({ value, onCommit, className, style }) {
  const [draft, setDraft] = useState(String(value));
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setDraft(String(value));
  }, [value, editing]);

  const commit = () => {
    const n = parseInt(draft, 10);
    if (!Number.isNaN(n)) onCommit(n);
    setEditing(false);
  };

  return (
    <input
      type="number"
      inputMode="numeric"
      value={draft}
      onFocus={(e) => { setEditing(true); e.target.select(); }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.currentTarget.blur();
        } else if (e.key === 'Escape') {
          setDraft(String(value));
          setEditing(false);
          e.currentTarget.blur();
        }
      }}
      aria-label="Takt i BPM"
      className={`stamma-btn ${className || ''}`}
      style={{
        background: 'transparent',
        border: 'none',
        outline: 'none',
        textAlign: 'center',
        font: 'inherit',
        color: 'inherit',
        appearance: 'textfield',
        MozAppearance: 'textfield',
        ...style,
      }}
    />
  );
}

const SOLO_COLOR = 'var(--gold)';

function MixerChannel({
  label, accentColor, enabled, onToggle, volume, onVolumeChange, meterRef, busy,
  direction, onSetDirection, solo, onToggleSolo, pan, onPanChange, previewing, onPreview,
  expanded, onToggleExpanded,
  onRecord, onStopRecordEarly, recording, recordCountingIn, recordCountInBeat, recordCountdown, recordDisabled, onDelete,
  waveform,
}) {
  const panLabel = Math.abs(pan) < 0.04 ? 'C' : pan < 0 ? `L${Math.round(-pan * 100)}` : `R${Math.round(pan * 100)}`;
  return (
    <div
      className="rounded-xl p-3"
      style={{
        backgroundColor: 'rgba(var(--ink-rgb),0.04)',
        border: solo ? `1px solid ${SOLO_COLOR}88` : enabled ? `1px solid ${accentColor}55` : '1px solid rgba(var(--ink-rgb),0.1)',
      }}
    >
      <div className="flex items-center gap-2">
        <div className="flex flex-col items-center gap-1">
          <ToggleSwitch checked={enabled} onChange={onToggle} accentColor={accentColor} ariaLabel={`Aktivera ${label}`} />
          {onDelete && (
            // Outer button is a full 44x44pt tap target (Apple HIG minimum);
            // the visible chip inside stays the original small size — see
            // the same pattern on ToggleSwitch above.
            <button
              onClick={onDelete}
              className="stamma-btn shrink-0 flex items-center justify-center"
              style={{ width: 44, height: 44 }}
              aria-label={`Radera ${label.toLowerCase()}`}
              title={`Radera ${label.toLowerCase()}`}
            >
              <span
                className="rounded-full flex items-center justify-center"
                style={{
                  width: 20,
                  height: 20,
                  backgroundColor: 'transparent',
                  color: 'var(--text-muted)',
                  border: '1px solid rgba(var(--ink-rgb),0.15)',
                }}
              >
                <TrashIcon size={11} />
              </span>
            </button>
          )}
        </div>
        <button
          onClick={onToggleSolo}
          className="stamma-btn shrink-0 flex items-center justify-center"
          style={{ width: 44, height: 44 }}
          aria-pressed={solo}
          aria-label="Solo"
          title="Solo"
        >
          <span
            className="rounded-md font-mono-ui text-xs font-semibold flex items-center justify-center"
            style={{
              width: 24,
              height: 24,
              backgroundColor: solo ? SOLO_COLOR : 'transparent',
              color: solo ? 'var(--bg)' : 'var(--text-muted)',
              border: solo ? `1px solid ${SOLO_COLOR}` : '1px solid rgba(var(--ink-rgb),0.15)',
            }}
          >
            S
          </span>
        </button>
        <span className="flex-1 text-sm font-medium truncate">
          {label}
          {busy ? <span style={{ color: 'var(--text-muted)' }}> (bygger …)</span> : null}
        </span>
        <button
          onClick={onPreview}
          disabled={busy}
          className="stamma-btn shrink-0 flex items-center justify-center"
          style={{ width: 44, height: 44, cursor: busy ? 'not-allowed' : 'pointer' }}
          aria-label={previewing ? 'Pausa' : 'Spela upp bara den här kanalen'}
          title={previewing ? 'Pausa' : 'Spela upp bara den här kanalen'}
        >
          <span
            className="rounded-md flex items-center justify-center"
            style={{
              width: 24,
              height: 24,
              backgroundColor: previewing ? `${accentColor}26` : 'transparent',
              color: busy ? 'rgba(var(--ink-rgb),0.25)' : accentColor,
              border: `1px solid ${previewing ? `${accentColor}66` : 'rgba(var(--ink-rgb),0.15)'}`,
            }}
          >
            {previewing ? <PauseIcon size={12} /> : <PlayIcon size={12} />}
          </span>
        </button>
      </div>

      {direction !== undefined && (
        <div className="flex items-center gap-2 mt-1.5">
          {onRecord && (
            <button
              onClick={recording ? onStopRecordEarly : onRecord}
              disabled={recordDisabled && !recording}
              className="stamma-btn shrink-0 flex items-center justify-center"
              style={{
                width: 44,
                height: 44,
                opacity: (recordDisabled && !recording) ? 0.4 : 1,
                cursor: (recordDisabled && !recording) ? 'not-allowed' : 'pointer',
              }}
              aria-label={recording ? 'Stoppa inspelningen' : `Spela in egen ${label.toLowerCase()}`}
              title={recording ? 'Stoppa inspelningen' : `Spela in egen ${label.toLowerCase()}`}
            >
              <span
                className="rounded-md flex items-center justify-center"
                style={{
                  width: 40,
                  height: 28,
                  backgroundColor: recordCountingIn ? 'rgba(var(--orange-rgb),0.15)' : recording ? 'var(--red)' : 'rgba(var(--red-rgb),0.12)',
                  color: recordCountingIn ? 'var(--orange)' : recording ? 'var(--bg)' : 'var(--red)',
                  border: `1px solid ${recordCountingIn ? 'rgba(var(--orange-rgb),0.6)' : 'rgba(var(--red-rgb),0.5)'}`,
                }}
              >
                {recordCountingIn ? (
                  <span className="font-mono-ui" style={{ fontSize: 11, fontWeight: 700 }}>
                    {COUNT_IN_BEATS - (recordCountInBeat || 0)}
                  </span>
                ) : recording ? (
                  <span className="rec-dot font-mono-ui" style={{ fontSize: 9, fontWeight: 700 }}>
                    {Math.ceil(recordCountdown)}
                  </span>
                ) : (
                  <RecordIcon size={12} />
                )}
              </span>
            </button>
          )}
          {[{ v: -1, label: 'Understämma' }, { v: 1, label: 'Överstämma' }].map((opt) => (
            <button
              key={opt.v}
              onClick={() => onSetDirection(opt.v)}
              className="stamma-btn flex-1 rounded-md text-[10px] font-medium leading-tight"
              style={{
                height: 44,
                backgroundColor: direction === opt.v ? `${accentColor}26` : 'transparent',
                color: direction === opt.v ? accentColor : 'var(--text-muted)',
                border: direction === opt.v ? `1px solid ${accentColor}66` : '1px solid rgba(var(--ink-rgb),0.12)',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 mt-2.5 ml-[64px]">
        <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.08)' }}>
          <div ref={meterRef} style={{ width: '0%', height: '100%', backgroundColor: accentColor, transition: 'width 60ms linear' }} />
        </div>
        <button
          onClick={onToggleExpanded}
          className="stamma-btn shrink-0 flex items-center gap-1 font-mono-ui text-[11px]"
          style={{ color: enabled ? accentColor : 'var(--text-muted)' }}
          aria-expanded={expanded}
          aria-label="Visa volym och panorering"
        >
          {Math.round(volume * 100)}%
          <span style={{ display: 'inline-block', fontSize: 15, transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
        </button>
      </div>

      {expanded && (
        <>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
            className="stamma-fader w-full mt-2 ml-0"
            style={{ accentColor }}
          />

          <div className="flex items-center gap-2 mt-1.5">
            <span className="font-mono-ui text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>L</span>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={pan}
              onChange={(e) => onPanChange(parseFloat(e.target.value))}
              className="stamma-fader w-full"
              style={{ accentColor, height: 3 }}
              aria-label="Panorering"
            />
            <span className="font-mono-ui text-[10px] shrink-0" style={{ color: 'var(--text-muted)' }}>R</span>
            <span className="font-mono-ui text-[10px] shrink-0 w-6 text-right" style={{ color: 'var(--text-muted)' }}>{panLabel}</span>
          </div>
          {waveform}
        </>
      )}
    </div>
  );
}

function DownloadIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3.5v11.5" />
      <path d="M7.5 11l4.5 4.5L16.5 11" />
      <path d="M4.5 18.5h15" />
    </svg>
  );
}

function SunIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2.5M12 19v2.5M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2.5 12H5M19 12h2.5M4.2 19.8 6 18M18 6l1.8-1.8" />
    </svg>
  );
}

function MoonIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.4 14.7A8.5 8.5 0 1 1 9.3 3.6a7 7 0 0 0 11.1 11.1Z" />
    </svg>
  );
}

function PlayIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M7.5 4.8v14.4c0 .77.85 1.24 1.5.83l11.3-7.2c.62-.4.62-1.3 0-1.7L9 3.97c-.65-.4-1.5.07-1.5.84Z" />
    </svg>
  );
}

function PauseIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="4.5" width="4.2" height="15" rx="1.2" />
      <rect x="13.8" y="4.5" width="4.2" height="15" rx="1.2" />
    </svg>
  );
}

function StopIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <rect x="5" y="5" width="14" height="14" rx="2.5" />
    </svg>
  );
}

function LoopIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12a8 8 0 0 1 8-8h6" />
      <path d="M15 1l3 3-3 3" />
      <path d="M20 12a8 8 0 0 1-8 8H6" />
      <path d="M9 23l-3-3 3-3" />
    </svg>
  );
}

function RecordIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <circle cx="12" cy="12" r="9" />
    </svg>
  );
}

function TrashIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 7h16" />
      <path d="M9 7V4.5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1V7" />
      <path d="M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
}

function MetronomeIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 21h8" />
      <path d="M7 21L11 4h2l4 17" />
      <path d="M9.5 13l5-3.5" />
      <circle cx="12" cy="7" r="1" fill="currentColor" stroke="none" />
    </svg>
  );
}

function HeadphonesIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 13v-1a8 8 0 0 1 16 0v1" />
      <rect x="2.5" y="13" width="5" height="7" rx="2" />
      <rect x="16.5" y="13" width="5" height="7" rx="2" />
    </svg>
  );
}

// Idle/ready-phase card for the live pitch-shift monitor (see
// startLiveMonitor et al.) — same card in both phases, just with
// `keyAware` reflecting whether a tonart is established yet (only true
// once a take's been analyzed, i.e. never in the idle-phase render).
// `types` is a { ters, kvint, sext } bool map — any combination can be
// active at once (see toggleLiveMonitorVoice), so the buttons below are
// independent toggles, not an exclusive choice.
function LiveMonitorCard({ on, starting, error, types, latencyMs, keyAware, keyInfo, humanizeOn, followingOwnTake, onToggle, onToggleType }) {
  const keyLabel = keyAware && keyInfo ? `${NOTE_NAMES[keyInfo.tonic]}-${keyInfo.mode === 'major' ? 'dur' : 'moll'}` : null;
  const anyActive = HARMONY_KEYS.some((t) => types[t]);
  const following = !!followingOwnTake;
  return (
    <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(var(--ink-rgb),0.04)', border: '1px solid rgba(var(--ink-rgb),0.1)' }}>
      <div className="flex items-center gap-2">
        <HeadphonesIcon size={16} />
        <span className="text-sm font-medium">Live-förhandslyssning</span>
        <span className="ml-auto font-mono-ui text-xs" style={{ color: 'var(--text-muted)' }}>
          {starting ? 'Startar …' : following ? 'Följer inspelningen' : on && latencyMs != null ? `${latencyMs} ms` : 'Kräver hörlurar'}
        </span>
        <ToggleSwitch checked={on || following} onChange={onToggle} disabled={starting || following} accentColor="var(--teal)" ariaLabel="Live-förhandslyssning" />
      </div>
      {(on || following) && (
        <div className="mt-3 pt-3" style={{ borderTop: '1px solid rgba(var(--ink-rgb),0.08)' }}>
          <div className="flex gap-2">
            {HARMONY_KEYS.map((t) => (
              <button
                key={t}
                onClick={() => { if (!following) onToggleType(t); }}
                disabled={following}
                aria-pressed={!!types[t]}
                className="stamma-btn flex-1 rounded-lg py-2 font-body font-medium text-sm"
                style={{
                  backgroundColor: types[t] ? 'rgba(var(--teal-rgb),0.15)' : 'rgba(var(--ink-rgb),0.06)',
                  color: types[t] ? 'var(--teal)' : 'var(--text)',
                  border: types[t] ? '1px solid rgba(var(--teal-rgb),0.5)' : '1px solid rgba(var(--ink-rgb),0.12)',
                  opacity: following && !types[t] ? 0.4 : 1,
                  cursor: following ? 'default' : 'pointer',
                }}
              >
                {HARMONY_TYPES[t].label}
              </button>
            ))}
          </div>
          <p className="mt-2 font-mono-ui text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {following
              ? `Följer din inspelning av egen ${HARMONY_TYPES[followingOwnTake].label.toLowerCase()} — övriga stämmor pausade tills tagningen är klar.`
              : !anyActive
                ? 'Ingen stämma vald — tyst tills du väljer en.'
                : keyLabel
                  ? `Tonartsanpassad (${keyLabel})${humanizeOn ? ', humaniserad röstkaraktär' : ''} — följer tonhöjden du sjunger.`
                  : 'Fast intervall (ingen tonart känd ännu).'}
          </p>
        </div>
      )}
      {error && <p className="mt-2 text-xs" style={{ color: 'var(--red-soft)' }}>{error}</p>}
    </div>
  );
}

function ExportButton({ label, onClick, disabled, busy }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="stamma-btn w-full rounded-xl py-3 px-4 flex items-center justify-between text-sm font-medium transition-colors"
      style={{
        backgroundColor: disabled ? 'rgba(var(--ink-rgb),0.03)' : 'rgba(var(--teal-rgb),0.1)',
        color: disabled ? 'rgba(var(--ink-rgb),0.25)' : 'var(--teal)',
        border: '1px solid rgba(var(--ink-rgb),0.1)',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <span>{label}</span>
      {busy ? (
        <span
          className="inline-block w-4 h-4 rounded-full animate-spin shrink-0"
          style={{ border: '2px solid rgba(var(--teal-rgb),0.25)', borderTopColor: 'currentColor' }}
        />
      ) : (
        <DownloadIcon />
      )}
    </button>
  );
}
