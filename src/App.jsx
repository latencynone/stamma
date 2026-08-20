import { useState, useRef, useEffect, useMemo } from 'react';
import {
  NOTE_NAMES,
  HARMONY_TYPES,
  SOUND_TYPES,
  midiToFreq,
  scaleStepToMidi,
  midiToNoteName,
  freqToMidi,
  midiToScaleStep,
} from './musicTheory.js';
import {
  autoCorrelate,
  detectKey,
  fillShortGaps,
  framesToNotes,
  filterTransientArtifacts,
  filterOutlierNotes,
  attachMeasuredFreq,
  extractFramesFromBuffer,
} from './pitchAnalysis.js';
import { renderHarmonyOffline, renderAutotunedMelody } from './harmonyEngine.js';
import { audioBufferToWavBlob, downloadBlob } from './wav.js';

const HARMONY_KEYS = Object.keys(HARMONY_TYPES);

// One accent color per harmony type, reused consistently between the note
// graph and the mixer channel rows, so the same color always means the same
// interval — needed now that more than one can be on screen/audible at once.
const HARMONY_COLORS = {
  ters: { line: '#55D6C0', glow: 'rgba(85,214,192,0.65)' },
  kvint: { line: '#A78BFA', glow: 'rgba(167,139,250,0.65)' },
  sext: { line: '#FB7185', glow: 'rgba(251,113,133,0.65)' },
};
const MELODY_COLOR = { line: '#FFB454', glow: 'rgba(255,180,84,0.65)' };

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
function PitchCanvas({ melodyNotes, harmonyLayers, keyInfo, duration, playheadTime }) {
  const canvasRef = useRef(null);
  const scrollWrapRef = useRef(null);
  const outerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);

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

      ctx.strokeStyle = 'rgba(241,237,228,0.06)';
      ctx.lineWidth = 1;
      for (let i = 1; i < duration; i++) {
        const x = (i / duration) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      if (!melodyNotes || melodyNotes.length === 0 || !keyInfo) {
        ctx.fillStyle = 'rgba(241,237,228,0.55)';
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
        ctx.lineCap = 'round';
        notes.forEach((n, i) => {
          const y = yFor(midis[i]);
          ctx.beginPath();
          ctx.moveTo(xFor(n.start), y);
          ctx.lineTo(xFor(Math.max(n.end, n.start + 0.04)), y);
          ctx.strokeStyle = color;
          ctx.lineWidth = 4;
          ctx.shadowColor = glow;
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
        ctx.fillStyle = color;
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
        bandGrad.addColorStop(0, 'rgba(241,237,228,0)');
        bandGrad.addColorStop(0.5, 'rgba(241,237,228,0.16)');
        bandGrad.addColorStop(1, 'rgba(241,237,228,0)');
        ctx.fillStyle = bandGrad;
        ctx.fillRect(x - 10, 0, 20, height);

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = 'rgba(255,255,255,0.85)';
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
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();
      }
    };

    drawRef.current = draw;
    draw();
  }, [melodyNotes, harmonyLayers, keyInfo, duration, playheadTime, zoom, isFullscreen]);

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
    <div ref={outerRef} className={isFullscreen ? 'fixed inset-0 z-50 flex flex-col p-4' : ''} style={isFullscreen ? { backgroundColor: '#10131A' } : undefined}>
      <div className="flex items-center gap-2 mb-2 font-mono-ui text-xs" style={{ color: '#C7CBDA' }}>
        <button
          onClick={() => setZoom((z) => Math.max(1, +(z - 0.5).toFixed(1)))}
          className="stamma-btn px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(241,237,228,0.15)' }}
          aria-label="Zooma ut"
        >
          −
        </button>
        <span className="w-10 text-center">{Math.round(zoom * 100)}%</span>
        <button
          onClick={() => setZoom((z) => Math.min(4, +(z + 0.5).toFixed(1)))}
          className="stamma-btn px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(241,237,228,0.15)' }}
          aria-label="Zooma in"
        >
          +
        </button>
        <button
          onClick={toggleFullscreen}
          className="stamma-btn ml-auto px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(241,237,228,0.15)' }}
        >
          {isFullscreen ? 'Stäng helskärm' : 'Helskärm'}
        </button>
      </div>
      <div
        ref={scrollWrapRef}
        className={isFullscreen ? 'flex-1 overflow-x-auto' : 'w-full h-48 md:h-56 overflow-x-auto rounded-xl'}
      >
        <canvas ref={canvasRef} className="block rounded-xl" />
      </div>
    </div>
  );
}

/* ---------- Waveform trim view (Original channel) ---------- */

const TRIM_HANDLE_MIN_GAP = 0.15;
const FADE_MIN_GAP = 0.05;
const FADE_COLOR = '#FFD84D';

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
function WaveformTrimmer({ peaks, duration, trimStart, trimEnd, onTrimChange, fadeIn, fadeOut, onFadeChange, playheadTime }) {
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
      ctx.fillStyle = 'rgba(241,237,228,0.4)';
      for (let i = 0; i < n; i++) {
        const x = (i / n) * width;
        const h = Math.max(1.5, peaks[i] * (height - 10));
        ctx.fillRect(x, midY - h / 2, Math.max(1, barW - 0.5), h);
      }

      const xFor = (t) => (duration > 0 ? (t / duration) * width : 0);
      ctx.fillStyle = 'rgba(16,19,26,0.72)';
      if (trimStart > 0) ctx.fillRect(0, 0, xFor(trimStart), height);
      if (trimEnd < duration) ctx.fillRect(xFor(trimEnd), 0, width - xFor(trimEnd), height);

      const windowDur = Math.max(0.001, trimEnd - trimStart);
      const fi = Math.max(0, Math.min(fadeIn, windowDur));
      const fo = Math.max(0, Math.min(fadeOut, windowDur - fi));

      if (fi > 0) {
        const x0 = xFor(trimStart);
        const x1 = xFor(trimStart + fi);
        const grad = ctx.createLinearGradient(x0, 0, x1, 0);
        grad.addColorStop(0, 'rgba(16,19,26,0.65)');
        grad.addColorStop(1, 'rgba(16,19,26,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(x0, 0, x1 - x0, height);

        ctx.strokeStyle = FADE_COLOR;
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
        grad.addColorStop(0, 'rgba(16,19,26,0)');
        grad.addColorStop(1, 'rgba(16,19,26,0.65)');
        ctx.fillStyle = grad;
        ctx.fillRect(x0, 0, x1 - x0, height);

        ctx.strokeStyle = FADE_COLOR;
        ctx.lineWidth = 1.75;
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x1, midY);
        ctx.moveTo(x0, height);
        ctx.lineTo(x1, midY);
        ctx.stroke();
      }

      if (playheadTime !== null && playheadTime !== undefined) {
        const x = xFor(Math.min(playheadTime, duration));
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(255,255,255,0.85)';
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
  }, [peaks, duration, trimStart, trimEnd, fadeIn, fadeOut, playheadTime, isFullscreen]);

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

  return (
    <div className={isFullscreen ? 'fixed inset-0 z-50 flex flex-col p-4' : ''} style={isFullscreen ? { backgroundColor: '#10131A' } : undefined}>
      <div className="flex items-center justify-between mb-2 font-mono-ui text-xs" style={{ color: '#C7CBDA' }}>
        <span>Vågform — beskär &amp; tona in/ut</span>
        <button
          onClick={() => setIsFullscreen((v) => !v)}
          className="stamma-btn px-2.5 py-1 rounded-md"
          style={{ border: '1px solid rgba(241,237,228,0.15)' }}
        >
          {isFullscreen ? 'Stäng helskärm' : 'Helskärm'}
        </button>
      </div>
      <div
        ref={wrapRef}
        className={isFullscreen ? 'flex-1 relative rounded-xl' : 'relative w-full h-24 rounded-xl'}
      >
        <canvas ref={canvasRef} className="block rounded-xl absolute inset-0" />
        <div onPointerDown={beginDrag('start')} style={handleStyle(startPct)}>
          <div style={{ position: 'absolute', left: 9, top: 0, bottom: 0, width: 2, backgroundColor: '#55D6C0' }} />
          <div style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 20, height: 34, borderRadius: 7, backgroundColor: '#55D6C0' }} />
        </div>
        <div onPointerDown={beginDrag('end')} style={handleStyle(endPct)}>
          <div style={{ position: 'absolute', left: 9, top: 0, bottom: 0, width: 2, backgroundColor: '#FB7185' }} />
          <div style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 20, height: 34, borderRadius: 7, backgroundColor: '#FB7185' }} />
        </div>
        <div onPointerDown={beginFadeDrag('in')} style={fadeHandleStyle(fadeInPct)} title="Tona in">
          {fadeFlag(true)}
        </div>
        <div onPointerDown={beginFadeDrag('out')} style={fadeHandleStyle(fadeOutPct)} title="Tona ut">
          {fadeFlag(false)}
        </div>
      </div>
      <div className="flex justify-between mt-1.5 font-mono-ui text-[10px]" style={{ color: '#C7CBDA' }}>
        <span style={{ color: '#55D6C0' }}>{trimStart.toFixed(1)}s</span>
        <span>{(trimEnd - trimStart).toFixed(1)}s vald</span>
        <span style={{ color: '#FB7185' }}>{trimEnd.toFixed(1)}s</span>
      </div>
      <div className="flex justify-between mt-1 font-mono-ui text-[10px]" style={{ color: FADE_COLOR }}>
        <span>Tona in: {fadeInClamped.toFixed(1)}s</span>
        <span>Tona ut: {fadeOutClamped.toFixed(1)}s</span>
      </div>
    </div>
  );
}

/* ---------- Main app ---------- */

const DURATION = 10;

// Strength stops for autotune once it's switched on — the on/off state
// itself is a separate toggle (autotuneOn), so this is just the "tre
// nivåer" of partial correction strength, expressed as the same 0..1
// amount buildAutotuneRatioCurve already accepted (see harmonyEngine.js).
const AUTOTUNE_LEVELS = [
  { key: 'light', label: 'Lätt', amount: 0.25 },
  { key: 'medium', label: 'Medel', amount: 0.45 },
  { key: 'hard', label: 'Hård', amount: 0.7 },
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
      title: 'Sjung in eller ladda upp',
      body: 'Spela in dig själv (max 10 sekunder) direkt i webbläsaren, eller ladda upp en befintlig ljudfil (WAV, MP3, M4A, OGG eller FLAC). Är filen längre än 10 sekunder klipps den av vid gränsen.',
    },
    {
      title: 'Tonhöjdskurvan',
      body: 'Appen lyssnar igenom inspelningen och räknar ut vilken ton som sjungs vid varje ögonblick. Det ritas upp som en kurva du kan zooma i och öppna i helskärm — varje stapel är en enskild ton.',
    },
    {
      title: 'Uppfattad tonart',
      body: 'Utifrån vilka toner som förekommer mest gissar appen vilken tonart (dur eller moll) melodin ligger i. Det är den tonarten stämmorna sedan byggs utifrån, så en fel uppfattad tonart är den vanligaste orsaken till att en stämma låter konstig.',
    },
    {
      title: 'Ljud: Ren synt / Formantröst / Din röst',
      body: 'Tre sätt att höra melodin och stämmorna på. "Ren synt" är en enkel sinuston, lättast att stämma efter. "Formantröst" är en syntetisk, vokalliknande klang. "Din röst" pitchskiftar din egen inspelning till varje stämmas toner, med bevarad klangfärg — mest verklighetstroget, men kräver att du spelat in (inte laddat upp en synt-fil).',
    },
    {
      title: 'Stämmor: ters, kvint, sext',
      body: 'Appen bygger tre extra röster som följer din melodi på ett fast musikaliskt avstånd: en ters, en kvint och en sext. Varje stämma kan ligga över eller under melodin (Överstämma/Understämma) — det väljer du per stämma.',
    },
    {
      title: 'Autotune',
      body: 'Rättar lätt falska toner i själva inspelningen (inte i stämmorna). Av som standard. När den är på väljer du hur mycket rättning som ska ske — Lätt, Medel eller Hård — där Hård drar tonerna närmast helt till rätt tonhöjd.',
    },
    {
      title: 'Vågformen: beskärning och fader',
      body: 'Under "Din röst" visas en vågform av originalinspelningen. Dra i de två handtagen i kanterna för att välja vilken del som ska spelas — allt utanför blir mörkare. De gula triangelhandtagen i hörnen styr tona in/tona ut: dra dem inåt för en mjukare start eller avslutning istället för en tvär in/utklippning.',
    },
    {
      title: 'Spela, pausa, loopa',
      body: 'Under vågformen finns tre kontroller: en loop-knapp (upprepar den valda delen tills du stänger av den), en stopp-knapp (stannar och hoppar tillbaka till början) och en kombinerad spela/pausa-knapp. Pausar du mitt i går det att fortsätta exakt där du var. Reglaget under knapparna spolar fram och tillbaka i den valda delen.',
    },
    {
      title: 'Mixern',
      body: 'Varje spår — melodi, ters, kvint, sext — går att slå på eller av var för sig, och alla påslagna spelas samtidigt när du trycker play. Varje spår har en liten play-knapp för att förhandslyssna bara det spåret, en S-knapp för att solo:a (tysta alla andra tillfälligt), och volym/panorering som fälls ut genom att trycka på procentsatsen.',
    },
    {
      title: 'Exportera',
      body: 'Ladda ner sången och varje stämma som separata WAV-filer, till exempel för att jobba vidare i ett annat program.',
    },
  ];

  return (
    <div className="min-h-screen w-full flex justify-center" style={{ backgroundColor: '#10131A', color: '#F1EDE4' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
        .font-display { font-family: 'Fraunces', serif; }
        .font-body { font-family: 'Space Grotesk', sans-serif; }
        .font-mono-ui { font-family: 'JetBrains Mono', monospace; }
        .stamma-btn:focus-visible { outline: 2px solid #FFB454; outline-offset: 2px; }
      `}</style>
      <div className="w-full max-w-md px-5 py-8 font-body">
        <a
          href="#"
          className="stamma-btn font-mono-ui text-xs"
          style={{ color: '#55D6C0' }}
        >
          ← Tillbaka
        </a>
        <h1 className="mt-3 font-display text-3xl font-semibold tracking-tight" style={{ color: '#F1EDE4' }}>
          Om Stämifier
        </h1>
        <p className="mt-2 text-base leading-relaxed" style={{ color: '#C7CBDA' }}>
          Så fungerar appens olika delar, rent praktiskt.
        </p>

        <div className="mt-6 space-y-5">
          {sections.map((s) => (
            <div key={s.title}>
              <h2 className="font-display text-lg font-semibold" style={{ color: '#F1EDE4' }}>
                {s.title}
              </h2>
              <p className="mt-1 text-sm leading-relaxed" style={{ color: '#C7CBDA' }}>
                {s.body}
              </p>
            </div>
          ))}
        </div>

        <a
          href="#"
          className="stamma-btn mt-8 inline-block font-mono-ui text-xs"
          style={{ color: '#55D6C0' }}
        >
          ← Tillbaka
        </a>
      </div>
    </div>
  );
}

export default function App() {
  const [phase, setPhase] = useState('idle'); // idle | recording | analyzing | ready | error
  const [errorMsg, setErrorMsg] = useState('');
  const [countdown, setCountdown] = useState(DURATION);
  const [keyInfo, setKeyInfo] = useState(null);
  const [melodyNotes, setMelodyNotes] = useState([]);
  const [channels, setChannels] = useState(defaultChannels);
  const [isPlaying, setIsPlaying] = useState(false);
  const [soundType, setSoundType] = useState('sine');
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
  const autotuneEnabled = autotuneOn;
  const effectiveTrimEnd = trimEnd === null ? recordingDuration : trimEnd;

  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const framesRef = useRef([]);
  const rafRef = useRef(null);
  const playCtxRef = useRef(null);
  const activeSourcesRef = useRef([]);
  const mixNodesRef = useRef({});
  const meterRefs = useRef({});
  const meterScratchRef = useRef(null);
  const recordedBufferRef = useRef(null);
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

  // Read inside the playback-end setTimeout, which was scheduled back when
  // it fired — using state directly there would capture whatever loopEnabled
  // was at schedule time, missing a toggle flipped mid-playback.
  useEffect(() => {
    loopEnabledRef.current = loopEnabled;
  }, [loopEnabled]);

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

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (playheadRafRef.current) cancelAnimationFrame(playheadRafRef.current);
      if (playTimeoutRef.current) clearTimeout(playTimeoutRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') audioCtxRef.current.close().catch(() => {});
      if (playCtxRef.current && playCtxRef.current.state !== 'closed') playCtxRef.current.close().catch(() => {});
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
    const key = `${channels[type].direction}-${autotuneEnabled ? `at${autotuneLevelIndex}` : 'raw'}`;
    const cached = harmonyBuffersRef.current[type];
    if (cached && cached.key === key) return cached.buffer;
    if (harmonyRenderPromisesRef.current[type] && cached?.key === `pending-${key}`) {
      return harmonyRenderPromisesRef.current[type];
    }
    harmonyBuffersRef.current[type] = { key: `pending-${key}`, buffer: null };
    setHarmonyRenderingByType((s) => ({ ...s, [type]: true }));
    setHarmonyRenderErrorsByType((s) => ({ ...s, [type]: '' }));
    const hNotes = harmonyNotesFor(type);
    const promise = renderHarmonyOffline(sourceBuffer, melodyNotes, hNotes, keyInfo)
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
    setMelodyNotes([]);
    setChannels(defaultChannels());
    setSoundType('sine');
    setVoiceReady(false);
    setAutotuneOn(false);
    setAutotuneLevelIndex(0);
    setAutotuneStrengthExpanded(false);
    setExpandedChannels({});
    recordedBufferRef.current = null;
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
    stopPlayback();
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
    }
  }

  function finishRecording() {
    if (finishedRef.current) return;
    finishedRef.current = true;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const audioCtx = audioCtxRef.current;
    const elapsed = audioCtx ? audioCtx.currentTime - recordingStartRef.current : DURATION;
    setRecordingDuration(Math.max(0.3, Math.min(DURATION, elapsed)));

    setPhase('analyzing');
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());

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

  function runAnalysis() {
    const result = analyzeFrames(framesRef.current);
    if (result.error) {
      setPhase('error');
      setErrorMsg(ANALYSIS_ERROR_MESSAGES[result.error]);
      return;
    }
    setKeyInfo(result.key);
    setMelodyNotes(result.notes);
    setPhase('ready');
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

      recordedBufferRef.current = buffer;
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
        return;
      }
      setKeyInfo(result.key);
      setMelodyNotes(result.notes);
      setPhase('ready');
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
      playCtxRef.current = ctx;
    }
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    return ctx;
  }

  // `keepPosition`: true for pause (freeze playheadTime so Play resumes
  // from here); false/omitted for a real stop — clears playheadTime, which
  // every "position" reader (the seek slider, startMix's default offset)
  // treats as "start of the trim window", i.e. back to the beginning.
  function stopPlayback({ keepPosition } = {}) {
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
    Object.values(mixNodesRef.current).forEach((nodes) => {
      try { nodes.gainNode.disconnect(); } catch (e) { /* already disconnected */ }
      try { nodes.fadeGainNode.disconnect(); } catch (e) { /* already disconnected */ }
      try { nodes.pannerNode.disconnect(); } catch (e) { /* already disconnected */ }
    });
    mixNodesRef.current = {};
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
  function startMixChannelWithContent(ctx, key, now, channelState, content, rangeStart, rangeEnd, fadeInDur, fadeOutDur, playFrom) {
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
    pannerNode.connect(ctx.destination);

    let sourceNode;
    if (content.kind === 'buffer') {
      const buf = content.buffer;
      const offset = Math.max(0, Math.min(playFrom, buf.duration));
      const duration = Math.max(0, Math.min(rangeEnd, buf.duration) - offset);
      sourceNode = playBufferSource(ctx, gainNode, now, buf, offset, duration);
    } else {
      sourceNode = makeVoice(ctx, gainNode, now, content.notes, soundType, playFrom, rangeEnd);
    }

    mixNodesRef.current[key] = { gainNode, fadeGainNode, analyserNode, pannerNode, sourceNode };
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
  async function startMix(channelsState, keysOverride, startOffset) {
    stopPlayback();
    const activeKeys = keysOverride || Object.keys(channelsState).filter((k) => isChannelAudible(channelsState, k));
    if (!activeKeys.length) return;

    const ctx = await getPlaybackContext();
    const resolved = await Promise.all(
      activeKeys.map(async (key) => ({ key, content: await resolveChannelContent(key) }))
    );
    const usable = resolved.filter((r) => r.content);
    if (!usable.length) return;

    // The trim window (waveform handles) clips every channel's playback to
    // the same [rangeStart, rangeEnd) span on the recording's timeline.
    const rangeStart = Math.max(0, Math.min(trimStart, recordingDuration));
    const rangeEnd = Math.max(rangeStart + 0.05, Math.min(effectiveTrimEnd, recordingDuration));
    const playFrom = Math.max(rangeStart, Math.min(startOffset ?? rangeStart, rangeEnd - 0.05));

    const now = ctx.currentTime + 0.05;
    const started = usable.map(({ key, content }) => startMixChannelWithContent(ctx, key, now, channelsState[key], content, rangeStart, rangeEnd, fadeIn, fadeOut, playFrom));

    activeSourcesRef.current = started;
    setIsPlaying(true);
    setPreviewingKey(keysOverride && keysOverride.length === 1 ? keysOverride[0] : null);

    const totalDur = rangeEnd - playFrom;
    playTimeoutRef.current = setTimeout(() => {
      // Read via a ref, not the `loopEnabled` closed over at schedule time —
      // this timeout was set up to (totalDur + 0.4)s ago, and a toggle
      // flipped since then would otherwise be missed.
      if (loopEnabledRef.current) {
        startMix(channelsState, keysOverride);
      } else {
        stopPlayback();
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

  const keyLabel = keyInfo ? `${NOTE_NAMES[keyInfo.tonic]}-${keyInfo.mode === 'major' ? 'dur' : 'moll'}` : null;
  const anyChannelEnabled = Object.values(channels).some((c) => c.enabled);
  const anyHarmonyBusy = Object.values(harmonyRenderingByType).some(Boolean);

  if (showAbout) {
    return <AboutPage />;
  }

  return (
    <div className="min-h-screen w-full flex justify-center" style={{ backgroundColor: '#10131A', color: '#F1EDE4' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
        .font-display { font-family: 'Fraunces', serif; }
        .font-body { font-family: 'Space Grotesk', sans-serif; }
        .font-mono-ui { font-family: 'JetBrains Mono', monospace; }
        @keyframes pulseRec { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .35; transform: scale(0.82); } }
        .rec-dot { animation: pulseRec 1.4s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) { .rec-dot { animation: none; } }
        .stamma-btn:focus-visible { outline: 2px solid #FFB454; outline-offset: 2px; }
        .stamma-fader { -webkit-appearance: none; appearance: none; height: 4px; border-radius: 2px; background: rgba(241,237,228,0.15); outline: none; }
        .stamma-fader::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #F1EDE4; cursor: pointer; }
        .stamma-fader::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: #F1EDE4; border: none; cursor: pointer; }
      `}</style>

      <div className="w-full max-w-md px-5 py-8 font-body">
        {/* Header */}
        <header className="mb-6">
          <h1 className="font-display text-4xl font-semibold tracking-tight" style={{ color: '#F1EDE4' }}>
            Stämifier
          </h1>
          <p className="mt-2 text-base leading-relaxed" style={{ color: '#C7CBDA' }}>
            Sjung in eller ladda upp en melodi (max 10s).
            {!introExpanded && (
              <>
                {' '}
                <button
                  onClick={() => setIntroExpanded(true)}
                  className="stamma-btn font-mono-ui text-xs align-middle"
                  style={{ color: '#55D6C0' }}
                >
                  Läs mer ▾
                </button>
              </>
            )}
          </p>
          {introExpanded && (
            <p className="mt-1 text-base leading-relaxed" style={{ color: '#C7CBDA' }}>
              Appen känner av tonarten och bygger stämmor i ters, kvint och sext som du kan mixa och träna in.{' '}
              <button
                onClick={() => setIntroExpanded(false)}
                className="stamma-btn font-mono-ui text-xs align-middle"
                style={{ color: '#55D6C0' }}
              >
                Visa mindre ▴
              </button>
            </p>
          )}
        </header>

        {/* Signature visualization */}
        <div className="rounded-2xl p-3 mb-5" style={{ backgroundColor: '#171B26', border: '1px solid rgba(241,237,228,0.08)' }}>
          <PitchCanvas melodyNotes={melodyNotes} harmonyLayers={harmonyLayers} keyInfo={keyInfo} duration={recordingDuration} playheadTime={playheadTime} />
          {melodyNotes.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 px-1 font-mono-ui text-sm" style={{ color: '#C7CBDA' }}>
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
                <span className="ml-auto" style={{ color: '#F1EDE4' }}>
                  {playheadTime.toFixed(1)}s / {recordingDuration.toFixed(1)}s
                </span>
              )}
            </div>
          )}
        </div>

        {/* Error banner */}
        {errorMsg && (
          <div className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(255,107,107,0.1)', border: '1px solid rgba(255,107,107,0.3)', color: '#FFB4B4' }}>
            {errorMsg}
          </div>
        )}
        {autotuneRenderError && (
          <div className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(255,107,107,0.1)', border: '1px solid rgba(255,107,107,0.3)', color: '#FFB4B4' }}>
            {autotuneRenderError}
          </div>
        )}
        {Object.entries(harmonyRenderErrorsByType).map(([type, msg]) => msg && (
          <div key={type} className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(255,107,107,0.1)', border: '1px solid rgba(255,107,107,0.3)', color: '#FFB4B4' }}>
            {msg}
          </div>
        ))}

        {/* Idle: start button + upload */}
        {(phase === 'idle' || phase === 'error') && (
          <div className="space-y-3">
            <button
              onClick={startRecording}
              className="stamma-btn w-full rounded-2xl py-4 font-body font-medium text-base transition-transform active:scale-[0.98]"
              style={{ backgroundColor: '#FFB454', color: '#10131A' }}
            >
              Spela in (max 10 sekunder)
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="stamma-btn w-full rounded-xl py-3 font-body font-medium text-sm"
              style={{ backgroundColor: 'rgba(241,237,228,0.06)', color: '#F1EDE4', border: '1px solid rgba(241,237,228,0.12)' }}
            >
              Ladda upp ljudfil
            </button>
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
          </div>
        )}

        {/* Recording */}
        {phase === 'recording' && (
          <div className="rounded-2xl p-5" style={{ backgroundColor: '#171B26', border: '1px solid rgba(255,107,107,0.3)' }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="rec-dot inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#FF6B6B' }} />
              <span className="font-mono-ui text-sm" style={{ color: '#FF6B6B' }}>SPELAR IN</span>
              <span className="ml-auto font-mono-ui text-2xl" style={{ color: '#F1EDE4' }}>{Math.ceil(countdown)}s</span>
            </div>
            <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(241,237,228,0.1)' }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${((DURATION - countdown) / DURATION) * 100}%`, backgroundColor: '#FF6B6B', transition: 'width 80ms linear' }}
              />
            </div>
            <p className="mt-3 text-base" style={{ color: '#C7CBDA' }}>Sjung en enkel melodislinga, gärna med tydliga toner.</p>
            <button
              onClick={stopRecordingEarly}
              className="stamma-btn w-full mt-4 rounded-xl py-3 font-body font-medium text-base transition-transform active:scale-[0.98]"
              style={{ backgroundColor: 'rgba(255,107,107,0.15)', color: '#FF9B9B', border: '1px solid rgba(255,107,107,0.4)' }}
            >
              Stoppa inspelning nu
            </button>
          </div>
        )}

        {/* Analyzing */}
        {phase === 'analyzing' && (
          <div className="rounded-2xl p-5 flex items-center gap-3" style={{ backgroundColor: '#171B26', border: '1px solid rgba(241,237,228,0.08)' }}>
            <span
              className="inline-block w-5 h-5 rounded-full animate-spin"
              style={{ border: '2px solid rgba(241,237,228,0.15)', borderTopColor: '#FFB454' }}
            />
            <span className="text-base" style={{ color: '#C7CBDA' }}>Analyserar tonart och melodi …</span>
          </div>
        )}

        {/* Ready: key readout, sound mode, mixer, export */}
        {phase === 'ready' && (
          <div className="space-y-5">
            <div className="rounded-2xl p-4 flex items-center justify-between" style={{ backgroundColor: '#171B26', border: '1px solid rgba(241,237,228,0.08)' }}>
              <span className="text-base" style={{ color: '#C7CBDA' }}>Uppfattad tonart</span>
              <span className="font-mono-ui text-lg" style={{ color: '#FFB454' }}>{keyLabel}</span>
            </div>

            <div>
              <h2 className="font-display text-lg font-semibold mb-2">Ljud</h2>
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
                        backgroundColor: disabled ? 'rgba(241,237,228,0.03)' : active ? '#FFB454' : 'rgba(241,237,228,0.06)',
                        color: disabled ? 'rgba(241,237,228,0.25)' : active ? '#10131A' : '#F1EDE4',
                        border: active ? '1px solid #FFB454' : '1px solid rgba(241,237,228,0.12)',
                        cursor: disabled ? 'not-allowed' : 'pointer',
                      }}
                    >
                      {SOUND_TYPES[type].label}
                    </button>
                  );
                })}
              </div>
              <p className="mt-3 text-sm leading-relaxed" style={{ color: '#C7CBDA' }}>
                {SOUND_TYPES[soundType].description}
              </p>

              {soundType === 'recording' && (
                <div className="mt-3 rounded-xl p-3" style={{ backgroundColor: 'rgba(241,237,228,0.04)', border: '1px solid rgba(241,237,228,0.1)' }}>
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium">Autotune</div>
                      <div className="text-xs mt-0.5" style={{ color: '#C7CBDA' }}>
                        Rättar falska toner i din inspelning{autotuneRendering ? ' (bygger …)' : ''}
                      </div>
                    </div>
                    <ToggleSwitch
                      checked={autotuneOn}
                      onChange={() => { setAutotuneOn((v) => !v); if (isPlaying) stopPlayback(); }}
                      accentColor="#55D6C0"
                    />
                  </div>
                  {autotuneOn && (
                    <div className="mt-3 pt-3" style={{ borderTop: '1px solid rgba(241,237,228,0.08)' }}>
                      <button
                        onClick={() => setAutotuneStrengthExpanded((v) => !v)}
                        className="stamma-btn w-full flex items-center justify-between font-mono-ui text-xs"
                        style={{ color: '#55D6C0' }}
                      >
                        <span>Styrka: {AUTOTUNE_LEVELS[autotuneLevelIndex].label}</span>
                        <span style={{ display: 'inline-block', transform: autotuneStrengthExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                      </button>
                      {autotuneStrengthExpanded && (
                        <div className="mt-2.5 grid grid-cols-3 gap-1.5">
                          {AUTOTUNE_LEVELS.map((lvl, i) => (
                            <button
                              key={lvl.key}
                              onClick={() => { setAutotuneLevelIndex(i); if (isPlaying) stopPlayback(); }}
                              className="stamma-btn rounded-md py-1.5 text-xs font-medium"
                              style={{
                                backgroundColor: autotuneLevelIndex === i ? 'rgba(85,214,192,0.15)' : 'transparent',
                                color: autotuneLevelIndex === i ? '#55D6C0' : '#C7CBDA',
                                border: autotuneLevelIndex === i ? '1px solid rgba(85,214,192,0.5)' : '1px solid rgba(241,237,228,0.12)',
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
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <h2 className="font-display text-lg font-semibold">Mixer</h2>
                <div className="flex items-center gap-3">
                  {isPlaying && (
                    <button onClick={() => stopPlayback()} className="stamma-btn text-xs underline" style={{ color: '#C7CBDA' }}>
                      Stoppa
                    </button>
                  )}
                  <button
                    onClick={toggleAllChannelsExpanded}
                    className="stamma-btn flex items-center gap-1 font-mono-ui text-xs"
                    style={{ color: '#C7CBDA' }}
                  >
                    {allChannelsExpanded ? 'Göm alla' : 'Expandera alla'}
                    <span style={{ display: 'inline-block', transform: allChannelsExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
                  </button>
                </div>
              </div>
              <p className="text-sm leading-relaxed mb-3" style={{ color: '#C7CBDA' }}>
                Slå på de kanaler du vill höra, ställ nivåerna, och tryck play — allt aktiverat spelas samtidigt.
              </p>

              {voiceReady && soundType === 'recording' && waveformPeaks && (
                <div className="mb-3 rounded-2xl p-3" style={{ backgroundColor: '#171B26', border: '1px solid rgba(241,237,228,0.08)' }}>
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
                  />

                  <div className="flex items-center gap-2 mt-3">
                    <button
                      onClick={() => setLoopEnabled((v) => !v)}
                      className="stamma-btn shrink-0 rounded-xl flex items-center justify-center"
                      style={{
                        width: 40,
                        height: 40,
                        backgroundColor: loopEnabled ? 'rgba(85,214,192,0.15)' : 'rgba(241,237,228,0.06)',
                        color: loopEnabled ? '#55D6C0' : '#C7CBDA',
                        border: loopEnabled ? '1px solid rgba(85,214,192,0.5)' : '1px solid rgba(241,237,228,0.12)',
                      }}
                      aria-pressed={loopEnabled}
                      aria-label="Loopa uppspelning"
                      title="Loopa uppspelning"
                    >
                      <LoopIcon size={18} />
                    </button>
                    <button
                      onClick={() => stopPlayback()}
                      disabled={!anyChannelEnabled}
                      className="stamma-btn shrink-0 rounded-xl flex items-center justify-center"
                      style={{
                        width: 40,
                        height: 40,
                        backgroundColor: 'rgba(241,237,228,0.06)',
                        color: anyChannelEnabled ? '#C7CBDA' : 'rgba(241,237,228,0.25)',
                        border: '1px solid rgba(241,237,228,0.12)',
                        cursor: anyChannelEnabled ? 'pointer' : 'not-allowed',
                      }}
                      aria-label="Stoppa och gå till start"
                      title="Stoppa och gå till start"
                    >
                      <StopIcon size={15} />
                    </button>
                    <button
                      onClick={togglePlayPause}
                      disabled={!anyChannelEnabled || anyHarmonyBusy || autotuneRendering}
                      className="stamma-btn flex-1 rounded-xl py-2.5 flex items-center justify-center gap-2 font-body font-medium text-sm"
                      style={{
                        backgroundColor: (!anyChannelEnabled || anyHarmonyBusy || autotuneRendering) ? 'rgba(241,237,228,0.06)' : '#FFB454',
                        color: (!anyChannelEnabled || anyHarmonyBusy || autotuneRendering) ? 'rgba(241,237,228,0.3)' : '#10131A',
                        cursor: (!anyChannelEnabled || anyHarmonyBusy || autotuneRendering) ? 'not-allowed' : 'pointer',
                      }}
                    >
                      {isPlaying ? <PauseIcon size={18} /> : <PlayIcon size={18} />}
                      {isPlaying ? 'Pausa' : (anyHarmonyBusy || autotuneRendering) ? 'Bygger …' : 'Spela mix'}
                    </button>
                  </div>

                  <div className="flex items-center gap-2 mt-2">
                    <span className="font-mono-ui text-[10px] shrink-0" style={{ color: '#C7CBDA' }}>
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
                      style={{ accentColor: '#FFB454' }}
                      aria-label="Spola i vågformen"
                    />
                    <span className="font-mono-ui text-[10px] shrink-0" style={{ color: '#C7CBDA' }}>{effectiveTrimEnd.toFixed(1)}s</span>
                  </div>
                </div>
              )}

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
                {HARMONY_KEYS.map((type) => (
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
                  />
                ))}
              </div>

              <div className="flex items-center gap-2 mt-4">
                <button
                  onClick={() => setLoopEnabled((v) => !v)}
                  className="stamma-btn shrink-0 rounded-2xl flex items-center justify-center"
                  style={{
                    width: 52,
                    height: 52,
                    backgroundColor: loopEnabled ? 'rgba(85,214,192,0.15)' : 'rgba(241,237,228,0.06)',
                    color: loopEnabled ? '#55D6C0' : '#C7CBDA',
                    border: loopEnabled ? '1px solid rgba(85,214,192,0.5)' : '1px solid rgba(241,237,228,0.12)',
                  }}
                  aria-pressed={loopEnabled}
                  aria-label="Loopa uppspelning"
                  title="Loopa uppspelning"
                >
                  <LoopIcon size={22} />
                </button>
                <button
                  onClick={() => (isPlaying ? stopPlayback() : startMix(channels))}
                  disabled={!anyChannelEnabled || anyHarmonyBusy || autotuneRendering}
                  className="stamma-btn flex-1 rounded-2xl py-4 font-body font-medium text-base transition-transform active:scale-[0.98] flex items-center justify-center gap-2"
                  style={{
                    backgroundColor: (!anyChannelEnabled || anyHarmonyBusy || autotuneRendering) ? 'rgba(241,237,228,0.06)' : '#FFB454',
                    color: (!anyChannelEnabled || anyHarmonyBusy || autotuneRendering) ? 'rgba(241,237,228,0.3)' : '#10131A',
                    cursor: (!anyChannelEnabled || anyHarmonyBusy || autotuneRendering) ? 'not-allowed' : 'pointer',
                  }}
                >
                  {isPlaying ? <PauseIcon size={22} /> : <PlayIcon size={22} />}
                  {isPlaying ? 'Stoppa mixen' : (anyHarmonyBusy || autotuneRendering) ? 'Bygger …' : 'Spela mixen'}
                </button>
              </div>
            </div>

            <div>
              <h2 className="font-display text-lg font-semibold mb-2">Exportera</h2>
              <p className="text-sm leading-relaxed mb-2" style={{ color: '#C7CBDA' }}>
                Ladda ner sång och stämmor som separata WAV-filer, t.ex. för att jobba vidare i Waveform.
              </p>
              <div className="grid grid-cols-1 gap-2">
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
            </div>

            <button
              onClick={resetAll}
              className="stamma-btn w-full rounded-xl py-3 font-body font-medium text-base transition-transform active:scale-[0.98]"
              style={{ backgroundColor: '#FF6B6B', color: '#10131A' }}
            >
              Spela in igen
            </button>
          </div>
        )}

        <footer className="mt-8 text-center">
          <a
            href="#om"
            className="stamma-btn font-mono-ui text-xs"
            style={{ color: '#C7CBDA' }}
          >
            Om Stämifier
          </a>
        </footer>
      </div>
    </div>
  );
}

function ToggleSwitch({ checked, onChange, accentColor = '#55D6C0', disabled }) {
  return (
    <button
      onClick={onChange}
      disabled={disabled}
      className="stamma-btn shrink-0"
      style={{
        width: 40,
        height: 24,
        borderRadius: 12,
        backgroundColor: checked ? accentColor : 'rgba(241,237,228,0.15)',
        position: 'relative',
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
      aria-pressed={checked}
    >
      <span
        style={{
          position: 'absolute',
          top: 2.5,
          left: checked ? 18 : 2.5,
          width: 19,
          height: 19,
          borderRadius: 10,
          backgroundColor: '#10131A',
          transition: 'left 150ms ease',
        }}
      />
    </button>
  );
}

const SOLO_COLOR = '#FFD84D';

function MixerChannel({
  label, accentColor, enabled, onToggle, volume, onVolumeChange, meterRef, busy,
  direction, onSetDirection, solo, onToggleSolo, pan, onPanChange, previewing, onPreview,
  expanded, onToggleExpanded,
}) {
  const panLabel = Math.abs(pan) < 0.04 ? 'C' : pan < 0 ? `L${Math.round(-pan * 100)}` : `R${Math.round(pan * 100)}`;
  return (
    <div
      className="rounded-xl p-3"
      style={{
        backgroundColor: 'rgba(241,237,228,0.04)',
        border: solo ? `1px solid ${SOLO_COLOR}88` : enabled ? `1px solid ${accentColor}55` : '1px solid rgba(241,237,228,0.1)',
      }}
    >
      <div className="flex items-center gap-2">
        <ToggleSwitch checked={enabled} onChange={onToggle} accentColor={accentColor} />
        <button
          onClick={onToggleSolo}
          className="stamma-btn shrink-0 rounded-md font-mono-ui text-xs font-semibold"
          style={{
            width: 24,
            height: 24,
            backgroundColor: solo ? SOLO_COLOR : 'transparent',
            color: solo ? '#10131A' : '#C7CBDA',
            border: solo ? `1px solid ${SOLO_COLOR}` : '1px solid rgba(241,237,228,0.15)',
          }}
          aria-pressed={solo}
          aria-label="Solo"
          title="Solo"
        >
          S
        </button>
        <span className="flex-1 text-sm font-medium truncate">
          {label}
          {busy ? <span style={{ color: '#C7CBDA' }}> (bygger …)</span> : null}
        </span>
        <button
          onClick={onPreview}
          disabled={busy}
          className="stamma-btn shrink-0 rounded-md flex items-center justify-center"
          style={{
            width: 24,
            height: 24,
            backgroundColor: previewing ? `${accentColor}26` : 'transparent',
            color: busy ? 'rgba(241,237,228,0.25)' : accentColor,
            border: `1px solid ${previewing ? `${accentColor}66` : 'rgba(241,237,228,0.15)'}`,
            cursor: busy ? 'not-allowed' : 'pointer',
          }}
          aria-label={previewing ? 'Pausa' : 'Spela upp bara den här kanalen'}
          title={previewing ? 'Pausa' : 'Spela upp bara den här kanalen'}
        >
          {previewing ? <PauseIcon size={12} /> : <PlayIcon size={12} />}
        </button>
      </div>

      {direction !== undefined && (
        <div className="flex gap-1 mt-1.5 ml-[64px]">
          {[{ v: -1, label: 'Understämma' }, { v: 1, label: 'Överstämma' }].map((opt) => (
            <button
              key={opt.v}
              onClick={() => onSetDirection(opt.v)}
              className="stamma-btn flex-1 rounded-md py-0.5 text-[10px] font-medium leading-tight"
              style={{
                backgroundColor: direction === opt.v ? `${accentColor}26` : 'transparent',
                color: direction === opt.v ? accentColor : '#C7CBDA',
                border: direction === opt.v ? `1px solid ${accentColor}66` : '1px solid rgba(241,237,228,0.12)',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 mt-2.5 ml-[64px]">
        <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(241,237,228,0.08)' }}>
          <div ref={meterRef} style={{ width: '0%', height: '100%', backgroundColor: accentColor, transition: 'width 60ms linear' }} />
        </div>
        <button
          onClick={onToggleExpanded}
          className="stamma-btn shrink-0 flex items-center gap-1 font-mono-ui text-[11px]"
          style={{ color: enabled ? accentColor : '#C7CBDA' }}
          aria-expanded={expanded}
          aria-label="Visa volym och panorering"
        >
          {Math.round(volume * 100)}%
          <span style={{ display: 'inline-block', transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 150ms ease' }}>▾</span>
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
            <span className="font-mono-ui text-[10px] shrink-0" style={{ color: '#C7CBDA' }}>L</span>
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
            <span className="font-mono-ui text-[10px] shrink-0" style={{ color: '#C7CBDA' }}>R</span>
            <span className="font-mono-ui text-[10px] shrink-0 w-6 text-right" style={{ color: '#C7CBDA' }}>{panLabel}</span>
          </div>
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

function ExportButton({ label, onClick, disabled, busy }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="stamma-btn w-full rounded-xl py-3 px-4 flex items-center justify-between text-sm font-medium transition-colors"
      style={{
        backgroundColor: disabled ? 'rgba(241,237,228,0.03)' : 'rgba(85,214,192,0.1)',
        color: disabled ? 'rgba(241,237,228,0.25)' : '#55D6C0',
        border: '1px solid rgba(241,237,228,0.1)',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <span>{label}</span>
      {busy ? (
        <span
          className="inline-block w-4 h-4 rounded-full animate-spin shrink-0"
          style={{ border: '2px solid rgba(85,214,192,0.25)', borderTopColor: 'currentColor' }}
        />
      ) : (
        <DownloadIcon />
      )}
    </button>
  );
}
