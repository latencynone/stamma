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

/* ---------- Main app ---------- */

const DURATION = 10;

function defaultChannels() {
  return {
    original: { enabled: false, volume: 0.8 },
    melody: { enabled: true, volume: 0.9 },
    ters: { enabled: false, volume: 0.8, direction: HARMONY_TYPES.ters.defaultDirection },
    kvint: { enabled: false, volume: 0.8, direction: HARMONY_TYPES.kvint.defaultDirection },
    sext: { enabled: false, volume: 0.8, direction: HARMONY_TYPES.sext.defaultDirection },
  };
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
  const [exporting, setExporting] = useState(null);
  const [harmonyRenderingByType, setHarmonyRenderingByType] = useState({});
  const [harmonyRenderErrorsByType, setHarmonyRenderErrorsByType] = useState({});
  const [autotuneEnabled, setAutotuneEnabled] = useState(false);
  const [autotuneRendering, setAutotuneRendering] = useState(false);
  const [autotuneRenderError, setAutotuneRenderError] = useState('');
  const [micPermission, setMicPermission] = useState('unknown');

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
  const autotunedBufferRef = useRef(null);
  const autotuneRenderPromiseRef = useRef(null);
  const fileInputRef = useRef(null);

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

  function setChannelDirection(key, direction) {
    setChannels((prev) => ({ ...prev, [key]: { ...prev[key], direction } }));
    if (isPlaying) restartMix({ ...channels, [key]: { ...channels[key], direction } });
  }

  function toggleChannel(key) {
    const next = { ...channels, [key]: { ...channels[key], enabled: !channels[key].enabled } };
    setChannels(next);
    if (isPlaying) restartMix(next);
  }

  // Renders (or returns the cached render of) a lightly pitch-corrected copy
  // of the recording. Cached per-recording (there's only ever one amount),
  // and reused as the *source* for harmony rendering too when autotune is on
  // — so the two voices stay in tune with each other, not just with an
  // idealized target neither of them actually sings.
  async function getAutotunedBuffer() {
    if (!recordedBufferRef.current || !melodyNotes.length || !keyInfo) return null;
    if (autotunedBufferRef.current) return autotunedBufferRef.current;
    if (autotuneRenderPromiseRef.current) return autotuneRenderPromiseRef.current;
    setAutotuneRendering(true);
    setAutotuneRenderError('');
    const promise = renderAutotunedMelody(recordedBufferRef.current, melodyNotes, keyInfo)
      .then((buffer) => {
        autotunedBufferRef.current = buffer;
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
    autotuneRenderPromiseRef.current = promise;
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
    const key = `${channels[type].direction}-${autotuneEnabled ? 'at' : 'raw'}`;
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
    setAutotuneEnabled(false);
    recordedBufferRef.current = null;
    harmonyBuffersRef.current = {};
    harmonyRenderPromisesRef.current = {};
    setHarmonyRenderingByType({});
    setHarmonyRenderErrorsByType({});
    autotunedBufferRef.current = null;
    setAutotuneRenderError('');
    stopPlayback();
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

        try {
          const arrayBuf = await blob.arrayBuffer();
          const decodeCtx = audioCtxRef.current && audioCtxRef.current.state !== 'closed'
            ? audioCtxRef.current
            : new (window.AudioContext || window.webkitAudioContext)();
          const audioBuffer = await decodeCtx.decodeAudioData(arrayBuf);
          recordedBufferRef.current = audioBuffer;
          setVoiceReady(true);
        } catch (e) {
          recordedBufferRef.current = null;
          setVoiceReady(false);
        }

        runAnalysis();
      };
      mr.stop();
    } else {
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

    try {
      const arrayBuf = await file.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      const decodeCtx = new AC();
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

  function stopPlayback() {
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
    });
    mixNodesRef.current = {};
    Object.values(meterRefs.current).forEach((el) => { if (el) el.style.width = '0%'; });
    setIsPlaying(false);
    setPlayheadTime(null);
  }

  // `outputNode` is the channel's own gain node — every source (whether a
  // real recording buffer or a synthesized voice) ends up there instead of
  // going straight to the speakers, so the channel's fader and meter apply
  // no matter what kind of source it is.
  function makeVoice(ctx, outputNode, now, notesArr, voiceSoundType) {
    const osc = ctx.createOscillator();
    osc.type = voiceSoundType === 'voice' ? 'sawtooth' : 'sine';
    const envGain = ctx.createGain();
    envGain.gain.value = 0;
    const outNode = voiceSoundType === 'voice' ? createFormantSum(ctx, osc) : osc;
    outNode.connect(envGain).connect(outputNode);
    osc.start(now);
    notesArr.forEach((n) => {
      const freq = midiToFreq(n.midi);
      const start = now + n.start;
      const end = Math.max(start + 0.05, now + n.end);
      const attack = 0.02;
      const release = Math.min(0.06, (end - start) / 3);
      osc.frequency.setValueAtTime(freq, start);
      envGain.gain.setValueAtTime(0, start);
      envGain.gain.linearRampToValueAtTime(1, start + attack);
      envGain.gain.setValueAtTime(1, Math.max(start + attack, end - release));
      envGain.gain.linearRampToValueAtTime(0, end);
    });
    const lastEnd = notesArr.length ? notesArr[notesArr.length - 1].end : 0;
    osc.stop(now + lastEnd + 0.3);
    return osc;
  }

  function playBufferSource(ctx, outputNode, now, buffer) {
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(outputNode);
    src.start(now);
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
  function startMixChannelWithContent(ctx, key, now, channelState, content) {
    const gainNode = ctx.createGain();
    gainNode.gain.value = channelState.volume;
    const analyserNode = ctx.createAnalyser();
    analyserNode.fftSize = 256;
    gainNode.connect(analyserNode);
    gainNode.connect(ctx.destination);

    const sourceNode = content.kind === 'buffer'
      ? playBufferSource(ctx, gainNode, now, content.buffer)
      : makeVoice(ctx, gainNode, now, content.notes, soundType);

    mixNodesRef.current[key] = { gainNode, analyserNode, sourceNode };
    return sourceNode;
  }

  function computeMixTotalDuration() {
    if (soundType === 'recording' && recordedBufferRef.current) return recordedBufferRef.current.duration;
    let maxEnd = melodyPlaybackNotes.length ? melodyPlaybackNotes[melodyPlaybackNotes.length - 1].end : 0;
    HARMONY_KEYS.forEach((type) => {
      const notes = harmonyPlaybackNotesFor(type);
      if (notes.length) maxEnd = Math.max(maxEnd, notes[notes.length - 1].end);
    });
    return maxEnd + 0.4;
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
  async function startMix(channelsState) {
    stopPlayback();
    const activeKeys = Object.keys(channelsState).filter((k) => channelsState[k].enabled);
    if (!activeKeys.length) return;

    const ctx = await getPlaybackContext();
    const resolved = await Promise.all(
      activeKeys.map(async (key) => ({ key, content: await resolveChannelContent(key) }))
    );
    const usable = resolved.filter((r) => r.content);
    if (!usable.length) return;

    const now = ctx.currentTime + 0.05;
    const started = usable.map(({ key, content }) => startMixChannelWithContent(ctx, key, now, channelsState[key], content));

    activeSourcesRef.current = started;
    setIsPlaying(true);

    const totalDur = computeMixTotalDuration();
    playTimeoutRef.current = setTimeout(() => {
      stopPlayback();
    }, (totalDur + 0.4) * 1000);

    const tickPlayhead = () => {
      const elapsed = ctx.currentTime - now;
      setPlayheadTime(Math.max(0, elapsed));
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

  const keyLabel = keyInfo ? `${NOTE_NAMES[keyInfo.tonic]}-${keyInfo.mode === 'major' ? 'dur' : 'moll'}` : null;
  const anyChannelEnabled = Object.values(channels).some((c) => c.enabled);
  const anyHarmonyBusy = Object.values(harmonyRenderingByType).some(Boolean);

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
            Sjung in en melodi (max 10 sekunder), eller ladda upp en ljudfil. Appen känner av tonarten och bygger stämmor i ters, kvint och sext som du kan mixa och träna in.
          </p>
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
              accept="audio/*"
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
                <div className="mt-3 flex items-center justify-between gap-3 rounded-xl p-3" style={{ backgroundColor: 'rgba(241,237,228,0.04)', border: '1px solid rgba(241,237,228,0.1)' }}>
                  <div>
                    <div className="text-sm font-medium">Autotune</div>
                    <div className="text-xs mt-0.5" style={{ color: '#C7CBDA' }}>
                      Rättar lätt falska toner i din inspelning{autotuneRendering ? ' (bygger …)' : ''}
                    </div>
                  </div>
                  <ToggleSwitch
                    checked={autotuneEnabled}
                    onChange={() => { setAutotuneEnabled((v) => !v); if (isPlaying) stopPlayback(); }}
                    accentColor="#55D6C0"
                  />
                </div>
              )}
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <h2 className="font-display text-lg font-semibold">Mixer</h2>
                {isPlaying && (
                  <button onClick={stopPlayback} className="stamma-btn text-xs underline" style={{ color: '#C7CBDA' }}>
                    Stoppa
                  </button>
                )}
              </div>
              <p className="text-sm leading-relaxed mb-3" style={{ color: '#C7CBDA' }}>
                Slå på de kanaler du vill höra, ställ nivåerna, och tryck play — allt aktiverat spelas samtidigt.
              </p>

              <div className="space-y-2">
                {voiceReady && soundType === 'recording' && (
                  <MixerChannel
                    label="Original (rå inspelning)"
                    accentColor="#F1EDE4"
                    enabled={channels.original.enabled}
                    onToggle={() => toggleChannel('original')}
                    volume={channels.original.volume}
                    onVolumeChange={(v) => setChannelVolume('original', v)}
                    meterRef={(el) => { meterRefs.current.original = el; }}
                  />
                )}
                <MixerChannel
                  label={soundType === 'recording' ? (autotuneEnabled ? 'Melodi (autotunad)' : 'Melodi (din röst)') : 'Melodi (ren ton)'}
                  accentColor={MELODY_COLOR.line}
                  enabled={channels.melody.enabled}
                  onToggle={() => toggleChannel('melody')}
                  volume={channels.melody.volume}
                  onVolumeChange={(v) => setChannelVolume('melody', v)}
                  meterRef={(el) => { meterRefs.current.melody = el; }}
                  busy={autotuneRendering}
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
                    meterRef={(el) => { meterRefs.current[type] = el; }}
                    busy={harmonyRenderingByType[type]}
                    direction={channels[type].direction}
                    onSetDirection={(d) => setChannelDirection(type, d)}
                  />
                ))}
              </div>

              <button
                onClick={() => (isPlaying ? stopPlayback() : startMix(channels))}
                disabled={!anyChannelEnabled || anyHarmonyBusy || autotuneRendering}
                className="stamma-btn w-full mt-4 rounded-2xl py-4 font-body font-medium text-base transition-transform active:scale-[0.98] flex items-center justify-center gap-2"
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

function MixerChannel({ label, accentColor, enabled, onToggle, volume, onVolumeChange, meterRef, busy, direction, onSetDirection }) {
  return (
    <div
      className="rounded-xl p-3"
      style={{
        backgroundColor: 'rgba(241,237,228,0.04)',
        border: enabled ? `1px solid ${accentColor}55` : '1px solid rgba(241,237,228,0.1)',
      }}
    >
      <div className="flex items-center gap-3">
        <ToggleSwitch checked={enabled} onChange={onToggle} accentColor={accentColor} />
        <span className="flex-1 text-sm font-medium truncate">
          {label}
          {busy ? <span style={{ color: '#C7CBDA' }}> (bygger …)</span> : null}
        </span>
        <span className="font-mono-ui text-xs shrink-0" style={{ color: enabled ? accentColor : '#C7CBDA' }}>
          {Math.round(volume * 100)}%
        </span>
      </div>

      {direction !== undefined && (
        <div className="flex gap-1.5 mt-2 ml-[52px]">
          {[{ v: -1, label: 'Under' }, { v: 1, label: 'Över' }].map((opt) => (
            <button
              key={opt.v}
              onClick={() => onSetDirection(opt.v)}
              className="stamma-btn flex-1 rounded-md py-1 text-xs font-medium"
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

      <div className="mt-2.5 ml-[52px] h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgba(241,237,228,0.08)' }}>
        <div ref={meterRef} style={{ width: '0%', height: '100%', backgroundColor: accentColor, transition: 'width 60ms linear' }} />
      </div>

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
