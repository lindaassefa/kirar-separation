import {
  CLASSES,
  CLIP_SAMPLES,
  N_FRAMES,
  N_MELS,
  type InstrumentClass,
  extractFeatures,
  sliceClip,
  softmax,
} from "@/lib/preprocess";

const MODEL_URL = "/models/instrumentiq_model.onnx";
const ORT_URL = "/ort/ort.wasm.min.mjs";

type OrtTensor = { data: Float32Array };

type OrtModule = {
  env: {
    wasm: {
      wasmPaths?: string;
      numThreads?: number;
    };
  };
  Tensor: new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown;
  InferenceSession: {
    create: (
      path: string,
      options?: { executionProviders?: string[] },
    ) => Promise<{
      run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>>;
    }>;
  };
};

export type Prediction = {
  label: InstrumentClass;
  probabilities: Record<InstrumentClass, number>;
  padded: boolean;
};

let runtimePromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<Awaited<ReturnType<OrtModule["InferenceSession"]["create"]>>> | null =
  null;

async function loadRuntime(): Promise<OrtModule> {
  runtimePromise ??= importOrt();
  return runtimePromise;
}

async function importOrt(): Promise<OrtModule> {
  const ort = (await import(
    /* webpackIgnore: true */
    /* turbopackIgnore: true */
    ORT_URL
  )) as OrtModule;
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = 1;
  return ort;
}

async function loadSession() {
  if (!sessionPromise) {
    sessionPromise = loadRuntime()
      .then((ort) =>
        ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
        }),
      )
      .catch((error: unknown) => {
        sessionPromise = null;
        runtimePromise = null;
        throw error;
      });
  }
  return sessionPromise;
}

export async function classifySamples(
  samples: Float32Array,
  startSample: number,
): Promise<Prediction> {
  const remaining = Math.max(0, samples.length - Math.max(0, startSample));
  const padded = remaining < CLIP_SAMPLES;
  const clip = sliceClip(samples, startSample);
  const spectrogram = extractFeatures(clip);
  const ort = await loadRuntime();
  const session = await loadSession();
  const tensor = new ort.Tensor("float32", spectrogram, [1, 1, N_MELS, N_FRAMES]);
  const outputs = await session.run({ spectrogram: tensor });
  const logits = outputs.logits?.data;
  if (!logits || logits.length !== CLASSES.length) {
    throw new Error("The model did not return three class logits.");
  }

  const probabilities = softmax(logits);
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }

  return {
    label: CLASSES[bestIndex],
    probabilities: {
      krar: probabilities[0],
      guitar: probabilities[1],
      piano: probabilities[2],
    },
    padded,
  };
}
