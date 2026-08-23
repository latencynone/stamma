// Synthesizes a voice-like recording from a note list: a handful of
// harmonics per note (roughly voice-shaped) with short fades at each note's
// edges, plus optional broadband "consonant" noise filling short gaps
// between notes. Used to drive the harmony engine with something structurally
// like a sung phrase, without needing an actual recording.
//
// `vibratoDepthCents`/`vibratoRateHz` and `dynamicsDepth`/`dynamicsRateHz`
// (all 0 by default, i.e. the plain steady tone earlier tests rely on) add
// real-singing-like pitch and amplitude wobble within each note — a flat,
// unwavering tone is exactly what real recordings never are, and is why a
// harmonic-separation regression (see harmonicSeparation.test.js) can pass
// against this synthesizer's default output yet still fail on an actual
// voice. `n.freq` remains each note's *nominal* pitch — the vibrato swings
// the actual instantaneous frequency around it, matching how a real note's
// one measured/quantized frequency is only ever an average of what was
// actually sung.
//
// `harmonicAmps` overrides the default monotonically-decreasing 1/h profile
// — a real voice's formants routinely make a mid harmonic louder than the
// fundamental itself (e.g. [0.5, 1, 0.7, 0.4, 0.2]), unlike this
// synthesizer's default falling series.
export function synthesizeVoice(notes, totalDur, sr, { gapNoiseAmp = 0.03, vibratoDepthCents = 0, vibratoRateHz = 5.5, dynamicsDepth = 0, dynamicsRateHz = 2.2, breathNoiseAmp = 0, harmonicAmps = [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5] } = {}) {
  const totalLen = Math.round(totalDur * sr);
  const buf = new Float32Array(totalLen);
  const fadeSamples = Math.round(0.01 * sr);

  notes.forEach((n) => {
    const s0 = Math.round(n.start * sr);
    const s1 = Math.round(n.end * sr);
    // Phase-modulation vibrato: instantaneous frequency is f0 + modIndex *
    // vibratoRateHz * cos(...), which peaks at f0 * 2^(depthCents/1200) —
    // an exact closed form, so no per-sample integration/drift error.
    const freqRatio = Math.pow(2, vibratoDepthCents / 1200) - 1;
    const modIndex = (n.freq * freqRatio) / vibratoRateHz;
    for (let i = s0; i < s1 && i < totalLen; i++) {
      const t = (i - s0) / sr;
      let env = 1;
      if (i - s0 < fadeSamples) env = (i - s0) / fadeSamples;
      if (s1 - i < fadeSamples) env = Math.min(env, (s1 - i) / fadeSamples);
      if (dynamicsDepth > 0) {
        env *= 1 - dynamicsDepth * (0.5 + 0.5 * Math.sin(2 * Math.PI * dynamicsRateHz * t));
      }
      const vibratoPhase = modIndex * Math.sin(2 * Math.PI * vibratoRateHz * t);
      let v = 0;
      harmonicAmps.forEach((amp, idx) => { const h = idx + 1; v += amp * Math.sin(2 * Math.PI * n.freq * h * t + h * vibratoPhase); });
      buf[i] += 0.15 * env * v;
      // Continuous breath/aspiration noise under the tone — real voiced
      // singing is never a pure harmonic series the way this synthesizer's
      // base tone is; some broadband noise rides along under the pitch the
      // whole time, not just in the gaps gapNoiseAmp fills between notes.
      if (breathNoiseAmp > 0) buf[i] += breathNoiseAmp * env * (Math.random() * 2 - 1);
    }
  });

  if (gapNoiseAmp > 0) {
    for (let i = 0; i < notes.length - 1; i++) {
      const gapStart = notes[i].end;
      const gapEnd = notes[i + 1].start;
      if (gapEnd - gapStart <= 0 || gapEnd - gapStart > 0.15) continue; // only short (consonant-like) gaps
      const s0 = Math.round(gapStart * sr);
      const s1 = Math.round(gapEnd * sr);
      for (let k = s0; k < s1 && k < totalLen; k++) buf[k] += gapNoiseAmp * (Math.random() * 2 - 1);
    }
  }

  return buf;
}

export function toAudioBuffer(ctx, data, sr) {
  const buffer = ctx.createBuffer(1, data.length, sr);
  buffer.copyToChannel(data, 0);
  return buffer;
}

export function rmsOf(data, from, to) {
  from = Math.max(0, from);
  to = Math.min(data.length, to);
  if (to <= from) return 0;
  let s = 0;
  for (let i = from; i < to; i++) s += data[i] * data[i];
  return Math.sqrt(s / (to - from));
}

export function maxSampleDelta(data) {
  let max = 0;
  for (let i = 1; i < data.length; i++) {
    const d = Math.abs(data[i] - data[i - 1]);
    if (d > max) max = d;
  }
  return max;
}

// Fraction (0-1) of 20ms windows where the input has audible signal but the
// output has gone (near-)silent — the "dropout" failure mode from the old
// block-based shifter.
export function dropoutFraction(inData, outData, sr, { winSec = 0.02, rmsThresh = 0.005, silenceRatio = 0.15 } = {}) {
  const winSamples = Math.round(winSec * sr);
  let dropouts = 0;
  let voiced = 0;
  for (let i = 0; i + winSamples <= inData.length; i += winSamples) {
    const rmsIn = rmsOf(inData, i, i + winSamples);
    if (rmsIn > rmsThresh) {
      voiced++;
      const rmsOut = rmsOf(outData, i, i + winSamples);
      if (rmsOut < rmsThresh * silenceRatio) dropouts++;
    }
  }
  return voiced ? dropouts / voiced : 0;
}
