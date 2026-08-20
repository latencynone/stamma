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
} from './pitchAnalysis.js';
import { renderHarmonyOffline } from './harmonyEngine.js';
import { audioBufferToWavBlob, downloadBlob } from './wav.js';

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

function PitchCanvas({ melodyNotes, harmonyNotes, keyInfo, duration, playheadTime }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 320;
    const height = canvas.clientHeight || 160;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
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
    const harmonyMidis = harmonyNotes && harmonyNotes.length
      ? harmonyNotes.map((n) => scaleStepToMidi(n.hStep, keyInfo.tonic, keyInfo.mode))
      : [];
    const all = melodyMidis.concat(harmonyMidis);
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

    drawLine(melodyNotes, melodyMidis, '#FFB454', 'rgba(255,180,84,0.65)', -10);
    if (harmonyNotes && harmonyNotes.length) {
      drawLine(harmonyNotes, harmonyMidis, '#55D6C0', 'rgba(85,214,192,0.65)', 17);
    }

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
  }, [melodyNotes, harmonyNotes, keyInfo, duration, playheadTime]);

  return <canvas ref={canvasRef} className="w-full h-48 md:h-56 block rounded-xl" />;
}

/* ---------- Main app ---------- */

const DURATION = 10;

export default function App() {
  const [phase, setPhase] = useState('idle'); // idle | recording | analyzing | ready | error
  const [errorMsg, setErrorMsg] = useState('');
  const [countdown, setCountdown] = useState(DURATION);
  const [keyInfo, setKeyInfo] = useState(null);
  const [melodyNotes, setMelodyNotes] = useState([]);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [harmonyType, setHarmonyType] = useState(null);
  const [direction, setDirection] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [nowPlaying, setNowPlaying] = useState(null);
  const [soundType, setSoundType] = useState('sine');
  const [voiceReady, setVoiceReady] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(DURATION);
  const [playheadTime, setPlayheadTime] = useState(null);
  const [exporting, setExporting] = useState(null);
  const [harmonyRendering, setHarmonyRendering] = useState(false);
  const [harmonyRenderError, setHarmonyRenderError] = useState('');

  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const framesRef = useRef([]);
  const rafRef = useRef(null);
  const audioElRef = useRef(null);
  const playCtxRef = useRef(null);
  const recordedUrlRef = useRef(null);
  const recordedBufferRef = useRef(null);
  const playTimeoutRef = useRef(null);
  const recordingStartRef = useRef(0);
  const playheadRafRef = useRef(null);
  const finishedRef = useRef(false);
  const harmonyBufferRef = useRef(null);
  const harmonyBufferKeyRef = useRef(null);
  const harmonyRenderPromiseRef = useRef(null);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (playheadRafRef.current) cancelAnimationFrame(playheadRafRef.current);
      if (playTimeoutRef.current) clearTimeout(playTimeoutRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') audioCtxRef.current.close().catch(() => {});
      if (playCtxRef.current && playCtxRef.current.state !== 'closed') playCtxRef.current.close().catch(() => {});
      if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current);
    };
  }, []);

  const harmonyNotes = useMemo(() => {
    if (!harmonyType || !melodyNotes.length) return [];
    const steps = HARMONY_TYPES[harmonyType].steps;
    return melodyNotes.map((n) => ({ ...n, hStep: n.step + direction * steps }));
  }, [harmonyType, direction, melodyNotes]);

  const melodyPlaybackNotes = useMemo(() => {
    if (!keyInfo) return [];
    return melodyNotes.map((n) => ({ start: n.start, end: n.end, midi: scaleStepToMidi(n.step, keyInfo.tonic, keyInfo.mode) }));
  }, [melodyNotes, keyInfo]);

  const harmonyPlaybackNotes = useMemo(() => {
    if (!keyInfo || !harmonyNotes.length) return [];
    return harmonyNotes.map((n) => ({ start: n.start, end: n.end, midi: scaleStepToMidi(n.hStep, keyInfo.tonic, keyInfo.mode) }));
  }, [harmonyNotes, keyInfo]);

  function selectHarmony(type) {
    setHarmonyType(type);
    setDirection(HARMONY_TYPES[type].defaultDirection);
  }

  // Renders (or returns the cached render of) the harmony as a real,
  // pitch-shifted copy of the singer's own recording. Cached per
  // harmonyType+direction combination since it's expensive to recompute on
  // every play/export and neither the recording nor the melody notes change
  // without a full re-record.
  async function getHarmonyBuffer() {
    if (!recordedBufferRef.current || !harmonyType || !harmonyNotes.length || !keyInfo) return null;
    const key = `${harmonyType}-${direction}`;
    if (harmonyBufferKeyRef.current === key && harmonyBufferRef.current) {
      return harmonyBufferRef.current;
    }
    if (harmonyRenderPromiseRef.current && harmonyBufferKeyRef.current === `pending-${key}`) {
      return harmonyRenderPromiseRef.current;
    }
    harmonyBufferKeyRef.current = `pending-${key}`;
    setHarmonyRendering(true);
    setHarmonyRenderError('');
    const promise = renderHarmonyOffline(recordedBufferRef.current, melodyNotes, harmonyNotes, keyInfo)
      .then((buffer) => {
        harmonyBufferRef.current = buffer;
        harmonyBufferKeyRef.current = key;
        return buffer;
      })
      .catch((err) => {
        harmonyBufferKeyRef.current = null;
        setHarmonyRenderError('Kunde inte bygga stämman av din inspelning. Testa igen, eller välj ett syntetiskt ljud istället.');
        throw err;
      })
      .finally(() => {
        setHarmonyRendering(false);
        harmonyRenderPromiseRef.current = null;
      });
    harmonyRenderPromiseRef.current = promise;
    return promise;
  }

  async function startRecording() {
    setErrorMsg('');
    setKeyInfo(null);
    setMelodyNotes([]);
    setHarmonyType(null);
    setSoundType('sine');
    setVoiceReady(false);
    recordedBufferRef.current = null;
    harmonyBufferRef.current = null;
    harmonyBufferKeyRef.current = null;
    setHarmonyRenderError('');
    stopPlayback();
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
      setRecordedUrl(null);
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setPhase('error');
      setErrorMsg('Den här webbläsaren stödjer inte mikrofoninspelning.');
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
        const url = URL.createObjectURL(blob);
        recordedUrlRef.current = url;
        setRecordedUrl(url);

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

  function runAnalysis() {
    const frames = framesRef.current;
    const voiced = frames.filter((f) => f.freq > 0);
    if (voiced.length < 15) {
      setPhase('error');
      setErrorMsg('Vi kunde inte hitta en tydlig tonhöjd i inspelningen. Sjung en enkel, tydlig melodi lite starkare och testa igen.');
      return;
    }

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
    const notes = filterOutlierNotes(filterTransientArtifacts(rawNotes, key.tonic, key.mode));

    if (notes.length === 0) {
      setPhase('error');
      setErrorMsg('Vi kunde inte tolka en tydlig melodislinga. Testa att sjunga lite långsammare och tydligare.');
      return;
    }

    setKeyInfo(key);
    setMelodyNotes(notes);
    setPhase('ready');
  }

  function resetAll() {
    stopPlayback();
    setPhase('idle');
    setKeyInfo(null);
    setMelodyNotes([]);
    setHarmonyType(null);
    setSoundType('sine');
    setVoiceReady(false);
    recordedBufferRef.current = null;
    harmonyBufferRef.current = null;
    harmonyBufferKeyRef.current = null;
    setHarmonyRenderError('');
    setRecordingDuration(DURATION);
    setErrorMsg('');
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
      setRecordedUrl(null);
    }
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
    if (playCtxRef.current) {
      playCtxRef.current.close().catch(() => {});
      playCtxRef.current = null;
    }
    if (audioElRef.current) audioElRef.current.pause();
    setIsPlaying(false);
    setNowPlaying(null);
    setPlayheadTime(null);
  }

  function makeVoice(ctx, now, notesArr, gainLevel, voiceSoundType) {
    const osc = ctx.createOscillator();
    osc.type = voiceSoundType === 'voice' ? 'sawtooth' : 'sine';
    const gain = ctx.createGain();
    gain.gain.value = 0;
    const outNode = voiceSoundType === 'voice' ? createFormantSum(ctx, osc) : osc;
    outNode.connect(gain).connect(ctx.destination);
    osc.start(now);
    notesArr.forEach((n) => {
      const freq = midiToFreq(n.midi);
      const start = now + n.start;
      const end = Math.max(start + 0.05, now + n.end);
      const attack = 0.02;
      const release = Math.min(0.06, (end - start) / 3);
      osc.frequency.setValueAtTime(freq, start);
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(gainLevel, start + attack);
      gain.gain.setValueAtTime(gainLevel, Math.max(start + attack, end - release));
      gain.gain.linearRampToValueAtTime(0, end);
    });
    const lastEnd = notesArr.length ? notesArr[notesArr.length - 1].end : 0;
    osc.stop(now + lastEnd + 0.3);
    return osc;
  }

  function playBufferSource(ctx, now, buffer, gainLevel) {
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    const g = ctx.createGain();
    g.gain.value = gainLevel;
    src.connect(g).connect(ctx.destination);
    src.start(now);
    return src;
  }

  async function renderSynthOffline(notesArr, renderSoundType, sampleRate) {
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const totalDur = (notesArr.length ? notesArr[notesArr.length - 1].end : 0) + 0.4;
    const offlineCtx = new OfflineCtx(1, Math.max(1, Math.ceil(totalDur * sampleRate)), sampleRate);
    makeVoice(offlineCtx, 0.05, notesArr, 0.6, renderSoundType);
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
        const blob = audioBufferToWavBlob(recordedBufferRef.current);
        downloadBlob(blob, 'stamma-melodi.wav');
      } else {
        const buffer = await renderSynthOffline(melodyPlaybackNotes, soundType, exportSampleRate());
        downloadBlob(audioBufferToWavBlob(buffer), 'stamma-melodi.wav');
      }
    } finally {
      setExporting(null);
    }
  }

  async function exportHarmony() {
    if (!harmonyType || !harmonyPlaybackNotes.length || exporting) return;
    setExporting('harmony');
    try {
      let buffer;
      if (soundType === 'recording' && recordedBufferRef.current) {
        buffer = await getHarmonyBuffer();
        if (!buffer) return;
      } else {
        buffer = await renderSynthOffline(harmonyPlaybackNotes, soundType, exportSampleRate());
      }
      const suffix = `${harmonyType}-${direction === -1 ? 'under' : 'over'}`;
      downloadBlob(audioBufferToWavBlob(buffer), `stamma-stamma-${suffix}.wav`);
    } finally {
      setExporting(null);
    }
  }

  async function playSynth(which) {
    // Pre-render (or fetch the cached) real-voice harmony before opening the
    // playback context, so the AudioContext clock only starts once the
    // buffer is actually ready to schedule.
    let hBuf = null;
    if (soundType === 'recording' && recordedBufferRef.current && (which === 'harmony' || which === 'both')) {
      hBuf = await getHarmonyBuffer();
      if (!hBuf) return;
    }

    stopPlayback();
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    playCtxRef.current = ctx;
    const now = ctx.currentTime + 0.05;
    setIsPlaying(true);
    setNowPlaying(which);

    let totalDur = Math.max(
      melodyPlaybackNotes.length ? melodyPlaybackNotes[melodyPlaybackNotes.length - 1].end : 0,
      harmonyPlaybackNotes.length ? harmonyPlaybackNotes[harmonyPlaybackNotes.length - 1].end : 0
    );

    if (soundType === 'recording' && recordedBufferRef.current) {
      const recBuf = recordedBufferRef.current;
      totalDur = recBuf.duration;
      if (which === 'melody') {
        playBufferSource(ctx, now, recBuf, 0.9);
      } else if (which === 'harmony') {
        playBufferSource(ctx, now, hBuf, 0.9);
      } else if (which === 'both') {
        playBufferSource(ctx, now, recBuf, 0.7);
        playBufferSource(ctx, now, hBuf, 0.7);
      }
    } else {
      if (which === 'melody') {
        makeVoice(ctx, now, melodyPlaybackNotes, 0.22, soundType);
      } else if (which === 'harmony') {
        makeVoice(ctx, now, harmonyPlaybackNotes, 0.22, soundType);
      } else if (which === 'both') {
        makeVoice(ctx, now, melodyPlaybackNotes, 0.16, soundType);
        makeVoice(ctx, now, harmonyPlaybackNotes, 0.16, soundType);
      }
    }

    playTimeoutRef.current = setTimeout(() => {
      setIsPlaying(false);
      setNowPlaying(null);
      setPlayheadTime(null);
      playTimeoutRef.current = null;
    }, (totalDur + 0.4) * 1000);

    const tickPlayhead = () => {
      const elapsed = ctx.currentTime - now;
      setPlayheadTime(Math.max(0, elapsed));
      if (elapsed < totalDur + 0.1) {
        playheadRafRef.current = requestAnimationFrame(tickPlayhead);
      }
    };
    playheadRafRef.current = requestAnimationFrame(tickPlayhead);
  }

  function playOriginal() {
    stopPlayback();
    if (audioElRef.current) {
      audioElRef.current.currentTime = 0;
      audioElRef.current.play();
      setIsPlaying(true);
      setNowPlaying('original');

      const tickPlayhead = () => {
        const el = audioElRef.current;
        if (!el) return;
        setPlayheadTime(el.currentTime);
        if (!el.paused && !el.ended) {
          playheadRafRef.current = requestAnimationFrame(tickPlayhead);
        } else {
          setPlayheadTime(null);
        }
      };
      playheadRafRef.current = requestAnimationFrame(tickPlayhead);
    }
  }

  const keyLabel = keyInfo ? `${NOTE_NAMES[keyInfo.tonic]}-${keyInfo.mode === 'major' ? 'dur' : 'moll'}` : null;

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
      `}</style>

      <div className="w-full max-w-md px-5 py-8 font-body">
        {/* Header */}
        <header className="mb-6">
          <h1 className="font-display text-4xl font-semibold tracking-tight" style={{ color: '#F1EDE4' }}>
            Stämma
          </h1>
          <p className="mt-2 text-base leading-relaxed" style={{ color: '#C7CBDA' }}>
            Sjung en melodi i tio sekunder. Appen känner av tonarten och bygger en stämma i ters, kvint eller sext som du kan träna in.
          </p>
        </header>

        {/* Signature visualization */}
        <div className="rounded-2xl p-3 mb-5" style={{ backgroundColor: '#171B26', border: '1px solid rgba(241,237,228,0.08)' }}>
          <PitchCanvas melodyNotes={melodyNotes} harmonyNotes={harmonyNotes} keyInfo={keyInfo} duration={recordingDuration} playheadTime={playheadTime} />
          {melodyNotes.length > 0 && (
            <div className="flex items-center gap-4 mt-2 px-1 font-mono-ui text-sm" style={{ color: '#C7CBDA' }}>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#FFB454' }} />
                melodi
              </span>
              {harmonyType && (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#55D6C0' }} />
                  stämma
                </span>
              )}
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
        {harmonyRenderError && (
          <div className="rounded-xl p-4 mb-5 text-sm" style={{ backgroundColor: 'rgba(255,107,107,0.1)', border: '1px solid rgba(255,107,107,0.3)', color: '#FFB4B4' }}>
            {harmonyRenderError}
          </div>
        )}

        {/* Idle: start button */}
        {(phase === 'idle' || phase === 'error') && (
          <button
            onClick={startRecording}
            className="stamma-btn w-full rounded-2xl py-4 font-body font-medium text-base transition-transform active:scale-[0.98]"
            style={{ backgroundColor: '#FFB454', color: '#10131A' }}
          >
            Spela in (max 10 sekunder)
          </button>
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

        {/* Ready: key readout, harmony selection, playback */}
        {phase === 'ready' && (
          <div className="space-y-5">
            <div className="rounded-2xl p-4 flex items-center justify-between" style={{ backgroundColor: '#171B26', border: '1px solid rgba(241,237,228,0.08)' }}>
              <span className="text-base" style={{ color: '#C7CBDA' }}>Uppfattad tonart</span>
              <span className="font-mono-ui text-lg" style={{ color: '#FFB454' }}>{keyLabel}</span>
            </div>

            <div>
              <h2 className="font-display text-lg font-semibold mb-2">Välj stämma</h2>
              <div className="grid grid-cols-3 gap-2">
                {Object.keys(HARMONY_TYPES).map((type) => {
                  const active = harmonyType === type;
                  return (
                    <button
                      key={type}
                      onClick={() => selectHarmony(type)}
                      className="stamma-btn rounded-xl py-3 font-body font-medium text-sm transition-colors"
                      style={{
                        backgroundColor: active ? '#55D6C0' : 'rgba(241,237,228,0.06)',
                        color: active ? '#10131A' : '#F1EDE4',
                        border: active ? '1px solid #55D6C0' : '1px solid rgba(241,237,228,0.12)',
                      }}
                    >
                      {HARMONY_TYPES[type].label}
                    </button>
                  );
                })}
              </div>

              {harmonyType && (
                <>
                  <p className="mt-3 text-sm leading-relaxed" style={{ color: '#C7CBDA' }}>
                    {HARMONY_TYPES[harmonyType].description}
                  </p>
                  <div className="mt-3 flex gap-2">
                    {[{ v: -1, label: 'Under melodin' }, { v: 1, label: 'Över melodin' }].map((opt) => (
                      <button
                        key={opt.v}
                        onClick={() => setDirection(opt.v)}
                        className="stamma-btn flex-1 rounded-lg py-2 text-xs font-medium"
                        style={{
                          backgroundColor: direction === opt.v ? 'rgba(85,214,192,0.15)' : 'transparent',
                          color: direction === opt.v ? '#55D6C0' : '#C7CBDA',
                          border: direction === opt.v ? '1px solid rgba(85,214,192,0.4)' : '1px solid rgba(241,237,228,0.12)',
                        }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </>
              )}
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
                      onClick={() => !disabled && setSoundType(type)}
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
            </div>

            <div>
              <h2 className="font-display text-lg font-semibold mb-2">Lyssna</h2>
              <div className="grid grid-cols-1 gap-2">
                {recordedUrl && (
                  <PlaybackButton
                    label="Original (din inspelning)"
                    active={isPlaying && nowPlaying === 'original'}
                    onClick={playOriginal}
                  />
                )}
                <PlaybackButton
                  label={soundType === 'recording' ? 'Din inspelade melodi' : 'Melodislinga (ren ton)'}
                  active={isPlaying && nowPlaying === 'melody'}
                  onClick={() => playSynth('melody')}
                />
                <PlaybackButton
                  label={soundType === 'recording' ? (harmonyRendering ? 'Stämma (bygger …)' : 'Stämma (din röst, formantbevarande)') : 'Stämma'}
                  primary
                  disabled={!harmonyType || harmonyRendering}
                  active={isPlaying && nowPlaying === 'harmony'}
                  onClick={() => playSynth('harmony')}
                />
                <PlaybackButton
                  label={soundType === 'recording' ? (harmonyRendering ? 'Melodi + stämma (bygger …)' : 'Melodi + stämma (din röst)') : 'Melodi + stämma tillsammans'}
                  disabled={!harmonyType || harmonyRendering}
                  active={isPlaying && nowPlaying === 'both'}
                  onClick={() => playSynth('both')}
                />
              </div>
              {isPlaying && (
                <button onClick={stopPlayback} className="stamma-btn mt-2 text-xs underline" style={{ color: '#C7CBDA' }}>
                  Stoppa uppspelning
                </button>
              )}
            </div>

            <div>
              <h2 className="font-display text-lg font-semibold mb-2">Exportera</h2>
              <p className="text-sm leading-relaxed mb-2" style={{ color: '#C7CBDA' }}>
                Ladda ner sång och stämma som separata WAV-filer, t.ex. för att jobba vidare i Waveform.
              </p>
              <div className="grid grid-cols-1 gap-2">
                <ExportButton
                  label="Sång (original)"
                  busy={exporting === 'song'}
                  disabled={!recordedUrl || !!exporting}
                  onClick={exportSong}
                />
                <ExportButton
                  label={soundType === 'recording' ? 'Melodi (samma som originalet)' : 'Melodislinga (ren ton)'}
                  busy={exporting === 'melody'}
                  disabled={!melodyNotes.length || !!exporting}
                  onClick={exportMelody}
                />
                <ExportButton
                  label="Stämma"
                  busy={exporting === 'harmony' || harmonyRendering}
                  disabled={!harmonyType || !!exporting || harmonyRendering}
                  onClick={exportHarmony}
                />
              </div>
            </div>

            <button
              onClick={resetAll}
              className="stamma-btn w-full rounded-xl py-3 text-sm font-medium"
              style={{ backgroundColor: 'transparent', color: '#C7CBDA', border: '1px solid rgba(241,237,228,0.12)' }}
            >
              Spela in igen
            </button>

            {recordedUrl && <audio ref={audioElRef} src={recordedUrl} onEnded={() => { setIsPlaying(false); setNowPlaying(null); setPlayheadTime(null); }} className="hidden" />}
          </div>
        )}
      </div>
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
        backgroundColor: disabled ? 'rgba(241,237,228,0.03)' : 'rgba(85,214,192,0.1)',
        color: disabled ? 'rgba(241,237,228,0.25)' : '#55D6C0',
        border: '1px solid rgba(241,237,228,0.1)',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <span>{label}</span>
      <span className="font-mono-ui text-xs">{busy ? '…' : '⬇'}</span>
    </button>
  );
}

function PlaybackButton({ label, onClick, disabled, active, primary }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="stamma-btn w-full rounded-xl py-3 px-4 flex items-center justify-between text-sm font-medium transition-colors"
      style={{
        backgroundColor: disabled ? 'rgba(241,237,228,0.03)' : primary ? 'rgba(255,180,84,0.12)' : 'rgba(241,237,228,0.06)',
        color: disabled ? 'rgba(241,237,228,0.25)' : primary ? '#FFB454' : '#F1EDE4',
        border: active ? '1px solid #FFB454' : '1px solid rgba(241,237,228,0.1)',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <span>{label}</span>
      <span className="font-mono-ui text-xs">{active ? '❚❚' : '▶'}</span>
    </button>
  );
}
