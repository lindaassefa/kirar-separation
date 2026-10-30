import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const GENERATION = 3;

export function usesModalSeparation(): boolean {
  return Boolean(process.env.MODAL_TOKEN_ID && process.env.MODAL_TOKEN_SECRET);
}

export function modalWorkerInstalled(): boolean {
  return existsSync(
    path.join(/*turbopackIgnore: true*/ process.cwd(), "separator", ".modal-venv", "bin", "python"),
  );
}

export class VocalWorkerError extends Error {
  readonly cancelled: boolean;

  constructor(message: string, cancelled = false) {
    super(message);
    this.name = cancelled ? "CancelledSeparation" : "VocalWorkerError";
    this.cancelled = cancelled;
  }
}

type SeparationMode = "stems" | "vocals";

type VocalJob = {
  id: string;
  input: string;
  outputDir: string;
  mode: SeparationMode;
  onStart?: () => Promise<void>;
  resolve: () => void;
  reject: (error: Error) => void;
};

type WorkerState = {
  generation: number;
  backend: "modal" | "cpu" | null;
  child: ChildProcessWithoutNullStreams | null;
  ready: boolean;
  buffer: string;
  stderr: string;
  queue: VocalJob[];
  active: VocalJob | null;
};

function pythonBin(): string {
  if (usesModalSeparation()) {
    return path.join(
      /*turbopackIgnore: true*/ process.cwd(),
      "separator",
      ".modal-venv",
      "bin",
      "python",
    );
  }
  return path.join(
    /*turbopackIgnore: true*/ process.cwd(),
    "separator",
    ".venv",
    "bin",
    "python",
  );
}

function workerArgs(): string[] {
  // Modal credentials select the private T4 worker. Without them, local
  // development stays on CPU. A configured GPU call is never retried on CPU.
  if (usesModalSeparation()) return ["-u", "-m", "separator.modal_client", "--serve"];
  return ["-u", "-m", "separator.worker", "--serve"];
}

function workerState(): WorkerState {
  const holder = globalThis as typeof globalThis & { __kirarVocalWorker?: WorkerState };
  const current = holder.__kirarVocalWorker;
  if (current && current.generation !== GENERATION) {
    current.child?.kill();
    const error = new VocalWorkerError("Vocal separation failed.");
    current.active?.reject(error);
    for (const job of current.queue) job.reject(error);
    holder.__kirarVocalWorker = undefined;
  }
  if (!holder.__kirarVocalWorker) {
    holder.__kirarVocalWorker = {
      generation: GENERATION,
      backend: null,
      child: null,
      ready: false,
      buffer: "",
      stderr: "",
      queue: [],
      active: null,
    };
  }
  return holder.__kirarVocalWorker;
}

function failActive(state: WorkerState, error: Error): void {
  const active = state.active;
  state.active = null;
  state.ready = false;
  state.child = null;
  active?.reject(error);
  const queued = state.queue;
  state.queue = [];
  for (const job of queued) job.reject(error);
}

function handleLine(state: WorkerState, line: string): void {
  let message: { event?: string; id?: string; error?: string };
  try {
    message = JSON.parse(line) as { event?: string; id?: string; error?: string };
  } catch {
    return;
  }
  if (message.event === "ready") {
    state.ready = true;
    pump(state);
    return;
  }
  if (!state.active || message.id !== state.active.id) return;
  const active = state.active;
  state.active = null;
  if (message.event === "done") active.resolve();
  else if (message.event === "cancelled") active.reject(new VocalWorkerError("cancelled", true));
  else active.reject(new VocalWorkerError(message.error || "Vocal separation failed."));
  pump(state);
}

function readStdout(state: WorkerState, chunk: Buffer): void {
  state.buffer += chunk.toString();
  let newline = state.buffer.indexOf("\n");
  while (newline >= 0) {
    const line = state.buffer.slice(0, newline).trim();
    state.buffer = state.buffer.slice(newline + 1);
    if (line) handleLine(state, line);
    newline = state.buffer.indexOf("\n");
  }
}

function ensureWorker(state: WorkerState): void {
  const backend = usesModalSeparation() ? "modal" : "cpu";
  if (state.child && state.child.exitCode === null && state.backend === backend) return;
  if (state.child && state.child.exitCode === null) state.child.kill();
  const child = spawn(pythonBin(), workerArgs(), {
    cwd: process.cwd(),
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: ["pipe", "pipe", "pipe"],
  });
  state.child = child;
  state.backend = backend;
  state.ready = false;
  state.buffer = "";
  state.stderr = "";
  child.stdout.on("data", (chunk: Buffer) => readStdout(state, chunk));
  child.stderr.on("data", (chunk: Buffer) => {
    state.stderr = (state.stderr + chunk.toString()).slice(-500);
  });
  child.on("error", (cause) => {
    failActive(state, cause instanceof Error ? cause : new VocalWorkerError("Vocal separation failed."));
  });
  child.on("close", (code) => {
    if (state.child !== child) return;
    const detail = state.stderr.trim();
    failActive(state, new VocalWorkerError(detail || `Vocal separation failed (${code ?? "unknown"}).`));
  });
}

function pump(state: WorkerState): void {
  if (!state.child || state.child.exitCode !== null) ensureWorker(state);
  if (!state.ready || state.active || state.queue.length === 0 || !state.child?.stdin) return;
  const next = state.queue.shift();
  if (!next) return;
  state.active = next;
  void Promise.resolve(next.onStart?.())
    .catch(() => undefined)
    .finally(() => {
      if (state.active !== next || !state.child?.stdin) return;
      state.child.stdin.write(
        `${JSON.stringify({
          id: next.id,
          input: next.input,
          output: next.outputDir,
          mode: next.mode,
          progress: path.join(next.outputDir, "progress.json"),
        })}\n`,
      );
    });
}

export function isVocalBusy(): boolean {
  const state = workerState();
  return state.active !== null || state.queue.length > 0;
}

export function enqueueVocalJob(job: {
  id: string;
  input: string;
  outputDir: string;
  mode: "stems" | "vocals";
  onStart?: () => Promise<void>;
}): Promise<void> {
  const state = workerState();
  ensureWorker(state);
  return new Promise((resolve, reject) => {
    state.queue.push({ ...job, resolve, reject });
    pump(state);
  });
}

export function cancelVocalJob(id: string): void {
  const state = workerState();
  const queued = state.queue.find((job) => job.id === id);
  if (queued) {
    state.queue = state.queue.filter((job) => job.id !== id);
    queued.reject(new VocalWorkerError("cancelled", true));
    return;
  }
  if (state.active?.id === id) {
    state.child?.stdin.write(`${JSON.stringify({ cmd: "cancel", id })}\n`);
  }
}
