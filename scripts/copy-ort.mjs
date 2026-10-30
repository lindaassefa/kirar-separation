import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "node_modules", "onnxruntime-web", "dist");
const output = path.join(root, "public", "ort");

const files = [
  "ort.wasm.min.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
];

await mkdir(output, { recursive: true });
for (const file of files) {
  await copyFile(path.join(dist, file), path.join(output, file));
}
