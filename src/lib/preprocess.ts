/** Preprocessing for instrumentiq_model.onnx.
 *
 * Matches the Colab training pipeline:
 * 16 kHz mono, 4 seconds, n_fft 1024, hop 256, 64 Slaney mel bands,
 * librosa.power_to_db(ref=1.0, top_db=80), clip to [-80, 20], then (db + 80) / 100.
 * The graph input is `spectrogram` with shape [batch, 1, 64, 251].
 */

export const SAMPLE_RATE = 16_000;
export const CLIP_SECONDS = 4;
export const CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS;
export const N_FFT = 1024;
export const HOP_LENGTH = 256;
export const N_MELS = 64;
export const N_FRAMES = 1 + Math.floor(CLIP_SAMPLES / HOP_LENGTH);
const POWER_TO_DB_AMIN = 1e-10;
const POWER_TO_DB_TOP_DB = 80;
const DB_CLIP_MIN = -80;
const DB_CLIP_MAX = 20;
export const CLASSES = ["krar", "guitar", "piano"] as const;

export type InstrumentClass = (typeof CLASSES)[number];

const F_MIN = 0;
const F_MAX = SAMPLE_RATE / 2;

let melWeights: Float64Array[] | null = null;
let hannWindow: Float64Array | null = null;

function hzToMel(hz: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27;
  if (hz >= minLogHz) {
    return minLogMel + Math.log(hz / minLogHz) / logstep;
  }
  return hz / fSp;
}

function melToHz(mel: number): number {
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27;
  if (mel >= minLogMel) {
    return minLogHz * Math.exp(logstep * (mel - minLogMel));
  }
  return fSp * mel;
}

function getMelWeights(): Float64Array[] {
  if (melWeights) return melWeights;

  const nFreqs = N_FFT / 2 + 1;
  const fftFreqs = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) {
    fftFreqs[i] = (i * SAMPLE_RATE) / N_FFT;
  }

  const nPoints = N_MELS + 2;
  const melPoints = new Float64Array(nPoints);
  const melMin = hzToMel(F_MIN);
  const melMax = hzToMel(F_MAX);
  for (let i = 0; i < nPoints; i++) {
    const mel = melMin + ((melMax - melMin) * i) / (nPoints - 1);
    melPoints[i] = melToHz(mel);
  }

  const fdiff = new Float64Array(nPoints - 1);
  for (let i = 0; i < fdiff.length; i++) {
    fdiff[i] = melPoints[i + 1] - melPoints[i];
  }

  const weights: Float64Array[] = [];
  for (let band = 0; band < N_MELS; band++) {
    const row = new Float64Array(nFreqs);
    for (let bin = 0; bin < nFreqs; bin++) {
      const lower = (fftFreqs[bin] - melPoints[band]) / fdiff[band];
      const upper = (melPoints[band + 2] - fftFreqs[bin]) / fdiff[band + 1];
      row[bin] = Math.max(0, Math.min(lower, upper));
    }
    const enorm = 2 / (melPoints[band + 2] - melPoints[band]);
    for (let bin = 0; bin < nFreqs; bin++) row[bin] *= enorm;
    weights.push(row);
  }

  melWeights = weights;
  return weights;
}

function getHannWindow(): Float64Array {
  if (hannWindow) return hannWindow;
  const window = new Float64Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) {
    window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  }
  hannWindow = window;
  return window;
}

function fftRadix2(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; (j & bit) !== 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const realSwap = real[i];
      real[i] = real[j];
      real[j] = realSwap;
      const imagSwap = imag[i];
      imag[i] = imag[j];
      imag[j] = imagSwap;
    }
  }

  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const wLenReal = Math.cos(angle);
    const wLenImag = Math.sin(angle);
    for (let start = 0; start < n; start += length) {
      let wReal = 1;
      let wImag = 0;
      const half = length >> 1;
      for (let j = 0; j < half; j++) {
        const evenReal = real[start + j];
        const evenImag = imag[start + j];
        const oddIndex = start + j + half;
        const oddReal = real[oddIndex] * wReal - imag[oddIndex] * wImag;
        const oddImag = real[oddIndex] * wImag + imag[oddIndex] * wReal;
        real[start + j] = evenReal + oddReal;
        imag[start + j] = evenImag + oddImag;
        real[oddIndex] = evenReal - oddReal;
        imag[oddIndex] = evenImag - oddImag;
        const nextReal = wReal * wLenReal - wImag * wLenImag;
        wImag = wReal * wLenImag + wImag * wLenReal;
        wReal = nextReal;
      }
    }
  }
}

export function sliceClip(samples: Float32Array, startSample: number): Float32Array {
  const clip = new Float32Array(CLIP_SAMPLES);
  if (startSample >= samples.length) return clip;
  const start = Math.max(0, startSample);
  const count = Math.min(CLIP_SAMPLES, samples.length - start);
  clip.set(samples.subarray(start, start + count));
  return clip;
}

/** Training features laid out as [1, 1, 64, 251], one mel band per row. */
export function extractFeatures(clip: Float32Array): Float32Array {
  if (clip.length !== CLIP_SAMPLES) {
    throw new Error(`Expected ${CLIP_SAMPLES} samples, received ${clip.length}.`);
  }

  const window = getHannWindow();
  const weights = getMelWeights();
  const nFreqs = N_FFT / 2 + 1;
  const padded = new Float64Array(N_FFT + CLIP_SAMPLES);
  for (let i = 0; i < CLIP_SAMPLES; i++) padded[N_FFT / 2 + i] = clip[i];

  const real = new Float64Array(N_FFT);
  const imag = new Float64Array(N_FFT);
  const power = new Float64Array(nFreqs);
  const mel = new Float64Array(N_MELS * N_FRAMES);

  for (let frame = 0; frame < N_FRAMES; frame++) {
    const offset = frame * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      real[i] = padded[offset + i] * window[i];
      imag[i] = 0;
    }
    fftRadix2(real, imag);
    for (let bin = 0; bin < nFreqs; bin++) {
      power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
    }
    for (let band = 0; band < N_MELS; band++) {
      const filter = weights[band];
      let sum = 0;
      for (let bin = 0; bin < nFreqs; bin++) sum += filter[bin] * power[bin];
      mel[band * N_FRAMES + frame] = sum;
    }
  }

  const refDb = 10 * Math.log10(Math.max(POWER_TO_DB_AMIN, 1));
  let peak = Number.NEGATIVE_INFINITY;
  const db = new Float64Array(mel.length);
  for (let i = 0; i < mel.length; i++) {
    const value = 10 * Math.log10(Math.max(POWER_TO_DB_AMIN, mel[i])) - refDb;
    db[i] = value;
    if (value > peak) peak = value;
  }

  const floor = peak - POWER_TO_DB_TOP_DB;
  const output = new Float32Array(db.length);
  for (let i = 0; i < db.length; i++) {
    const clipped = Math.min(DB_CLIP_MAX, Math.max(DB_CLIP_MIN, Math.max(db[i], floor)));
    output[i] = (clipped + 80) / 100;
  }
  return output;
}

export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const exps = Array.from(logits, (value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}
