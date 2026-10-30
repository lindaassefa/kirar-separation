import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { N_FFT, N_FRAMES, N_MELS, extractFeatures } from "../src/lib/preprocess.ts";

function readFloat32(filePath: string): Float32Array {
  const bytes = readFileSync(filePath);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

const fixtureDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const audio = readFloat32(path.join(fixtureDir, "krar_0_4_audio.f32"));
const expected = readFloat32(path.join(fixtureDir, "krar_0_4_features.f32"));
const actual = extractFeatures(audio);

if (actual.length !== N_MELS * N_FRAMES || expected.length !== actual.length) {
  throw new Error(
    `Feature length mismatch. js=${actual.length} python=${expected.length} expected=${N_MELS * N_FRAMES}`,
  );
}

let maxDiff = 0;
let maxIndex = 0;
for (let i = 0; i < actual.length; i++) {
  const diff = Math.abs(actual[i] - expected[i]);
  if (diff > maxDiff) {
    maxDiff = diff;
    maxIndex = i;
  }
}

const band = Math.floor(maxIndex / N_FRAMES);
const frame = maxIndex % N_FRAMES;
console.log(
  `n_fft=${N_FFT} frames=${N_FRAMES} bands=${N_MELS} maxAbsDiff=${maxDiff.toExponential(3)} at mel=${band} frame=${frame}`,
);
if (maxDiff > 1e-4) {
  throw new Error(`Features drifted from the Colab librosa pipeline by ${maxDiff}`);
}
