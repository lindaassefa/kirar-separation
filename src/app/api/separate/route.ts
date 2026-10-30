import { randomUUID } from "node:crypto";
import { createReadStream, writeFileSync } from "node:fs";
import { mkdir, open, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Readable } from "node:stream";

import type { NextRequest } from "next/server";

import {
  cancelVocalJob,
  enqueueVocalJob,
  isVocalBusy,
  modalWorkerInstalled,
  usesModalSeparation,
  VocalWorkerError,
} from "@/lib/vocal-worker";

const ROOT = path.join(tmpdir(), "kirar-separation");
const STEM_FILES = {
  stems: ["vocals", "drums", "bass", "other"],
  vocals: ["vocals", "instrumental"],
} as const;
type SeparationMode = keyof typeof STEM_FILES;
const ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_BYTES = 120 * 1024 * 1024;
const JOB_MAX_AGE_MS = 2 * 60 * 60 * 1000;
const STEM_NOTE = "Vocals, drums, bass, and other are ready. A separate krar stem was not created.";
const VOCAL_NOTE = "Vocals and an instrumental are ready. This mode can treat krar as vocals. A separate krar stem was not created.";
const AUDIO_SUFFIXES = new Set([".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".aif", ".aiff", ".mp4"]);
const RATE_WINDOW_MS = 10 * 60 * 1000;
const RATE_LIMIT = 6;
const SPEND_CEILING_USD = 5;

function uploadRate(): number[] {
  const holder = globalThis as typeof globalThis & { __kirarUploadRate?: number[] };
  if (!holder.__kirarUploadRate) holder.__kirarUploadRate = [];
  return holder.__kirarUploadRate;
}

function allowUpload(): boolean {
  const now = Date.now();
  const stamps = uploadRate();
  const recent = stamps.filter((stamp) => now - stamp < RATE_WINDOW_MS);
  stamps.length = 0;
  stamps.push(...recent);
  if (stamps.length >= RATE_LIMIT) return false;
  stamps.push(now);
  return true;
}

function configuredSpendLimit(): number | null {
  const raw = process.env.MODAL_SPEND_LIMIT_USD;
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0 || value > SPEND_CEILING_USD) return null;
  return value;
}

function separationMode(value: FormDataEntryValue | null): SeparationMode {
  return value === "vocals" ? "vocals" : "stems";
}

function noteFor(mode: SeparationMode): string {
  return mode === "vocals" ? VOCAL_NOTE : STEM_NOTE;
}

function audioSuffix(name: string): string {
  const suffix = path.extname(name).toLowerCase();
  return AUDIO_SUFFIXES.has(suffix) ? suffix : ".audio";
}

function jobDirectory(id: string): string | null {
  if (!ID_PATTERN.test(id)) return null;
  const directory = path.resolve(ROOT, id);
  if (!directory.startsWith(`${path.resolve(ROOT)}${path.sep}`)) return null;
  return directory;
}

function publicError(message: string): string {
  const text = message.replace(/\s+/g, " ").trim();
  const modal = usesModalSeparation();
  if (/could not read|afconvert|could not open|invalid data|not a supported/i.test(text)) {
    return "Could not read that audio file.";
  }
  if (/under 10 minutes/i.test(text)) return "Choose a file under 10 minutes.";
  if (/under 120 mb/i.test(text)) return "Choose a file under 120 MB.";
  if (/spend limit/i.test(text)) return "Separation is paused because the spend limit was reached.";
  if (/too many uploads/i.test(text)) return "Too many uploads. Wait a few minutes.";
  if (/vocal model is missing/i.test(text)) return "The vocal model is not installed.";
  if (/stem model is not installed/i.test(text)) return "The stem model is not installed.";
  if (modal) return "Vocal separation is unavailable.";
  if (!text || text.length > 180) return "Vocal separation failed.";
  return text;
}

async function assertWav(file: string): Promise<void> {
  const info = await stat(file);
  if (info.size <= 44) throw new Error("The separator did not write audio.");
  const handle = await open(file, "r");
  try {
    const header = Buffer.alloc(12);
    await handle.read(header, 0, 12, 0);
    if (header.toString("ascii", 0, 4) !== "RIFF" || header.toString("ascii", 8, 12) !== "WAVE") {
      throw new Error("The separator did not write audio.");
    }
  } finally {
    await handle.close();
  }
}

async function sweepOldJobs(): Promise<void> {
  let entries;
  try {
    entries = await readdir(ROOT, { withFileTypes: true });
  } catch {
    return;
  }
  const now = Date.now();
  await Promise.all(
    entries.map(async (entry) => {
      if (!entry.isDirectory() || !ID_PATTERN.test(entry.name)) return;
      const directory = path.join(ROOT, entry.name);
      try {
        const info = await stat(directory);
        if (now - info.mtimeMs > JOB_MAX_AGE_MS) await rm(directory, { recursive: true, force: true });
      } catch {
        // Another request may already have removed it.
      }
    }),
  );
}

const jobsInFlight = new Set<Promise<void>>();

async function readJson(file: string): Promise<Record<string, unknown> | null> {
  try {
    return JSON.parse(await readFile(file, "utf8")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

async function writeStatus(directory: string, status: Record<string, unknown>): Promise<void> {
  await writeFile(path.join(directory, "status.json"), JSON.stringify(status));
}

async function jobStatus(directory: string): Promise<Record<string, unknown>> {
  const status = await readJson(path.join(directory, "status.json"));
  const progress = await readJson(path.join(directory, "progress.json"));
  const chunksDone = numberOrNull(progress?.chunksDone ?? status?.chunksDone);
  const chunksTotal = numberOrNull(progress?.chunksTotal ?? status?.chunksTotal);
  const body: Record<string, unknown> = {
    status: typeof status?.status === "string" ? status.status : "running",
    chunksDone,
    chunksTotal,
    startedAt: numberOrNull(status?.startedAt),
  };
  if (typeof status?.error === "string") body.error = status.error;
  if (numberOrNull(status?.receivedAt) !== null) body.receivedAt = numberOrNull(status?.receivedAt);
  if (numberOrNull(status?.readyAt) !== null) body.readyAt = numberOrNull(status?.readyAt);
  if (status?.timings && typeof status.timings === "object") body.timings = status.timings;
  const mode: SeparationMode = status?.mode === "vocals" ? "vocals" : "stems";
  body.mode = mode;
  if (body.status === "done") {
    const ready = new Set<string>(STEM_FILES[mode]);
    body.outputs = {
      vocals: ready.has("vocals"),
      drums: ready.has("drums"),
      bass: ready.has("bass"),
      other: ready.has("other"),
      instrumental: ready.has("instrumental"),
      krar: false,
    };
    body.note = noteFor(mode);
  }
  return body;
}

function finishJob(id: string, directory: string, input: string, mode: SeparationMode): void {
  const files = STEM_FILES[mode];
  let task: Promise<void> = Promise.resolve();
  task = enqueueVocalJob({
    id,
    input,
    outputDir: directory,
    mode,
    onStart: async () => {
      const existing = await readJson(path.join(directory, "status.json"));
      if (existing?.status === "running" && numberOrNull(existing.startedAt) !== null) return;
      await writeStatus(directory, {
        status: "running",
        mode,
        chunksDone: 0,
        chunksTotal: null,
        startedAt: Date.now(),
        receivedAt: numberOrNull(existing?.receivedAt) ?? Date.now(),
      });
    },
  })
    .then(async () => {
      for (const stem of files) await assertWav(path.join(directory, `${stem}.wav`));
      await rm(input, { force: true });
      const progress = await readJson(path.join(directory, "progress.json"));
      const status = await readJson(path.join(directory, "status.json"));
      const timings = await readJson(path.join(directory, "timings.json"));
      await writeStatus(directory, {
        status: "done",
        mode,
        chunksDone: numberOrNull(progress?.chunksDone),
        chunksTotal: numberOrNull(progress?.chunksTotal),
        startedAt: numberOrNull(status?.startedAt) ?? Date.now(),
        receivedAt: numberOrNull(status?.receivedAt),
        readyAt: Date.now(),
        timings,
      });
    })
    .catch(async (cause: unknown) => {
      if (cause instanceof VocalWorkerError && cause.cancelled) return;
      const message = cause instanceof Error ? cause.message : "Vocal separation failed.";
      await writeStatus(directory, {
        status: "error",
        mode,
        error: publicError(message),
        chunksDone: null,
        chunksTotal: null,
        startedAt: null,
      }).catch(() => undefined);
      await rm(input, { force: true }).catch(() => undefined);
      await Promise.all(
        ["vocals", "drums", "bass", "other", "instrumental"].map((stem) =>
          rm(path.join(directory, `${stem}.wav`), { force: true }).catch(() => undefined),
        ),
      );
    })
    .finally(() => {
      jobsInFlight.delete(task);
    });
  jobsInFlight.add(task);
}

export async function POST(request: NextRequest) {
  if (!allowUpload()) {
    return Response.json({ error: "Too many uploads. Wait a few minutes." }, { status: 429 });
  }
  await sweepOldJobs();
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return Response.json({ error: "Choose an audio file." }, { status: 400 });
  }
  const audio = form.get("audio");
  if (!(audio instanceof File) || audio.size === 0) {
    return Response.json({ error: "Choose an audio file." }, { status: 400 });
  }
  if (audio.size > MAX_BYTES) {
    return Response.json({ error: "Choose a file under 120 MB." }, { status: 400 });
  }
  if (usesModalSeparation() && (!modalWorkerInstalled() || configuredSpendLimit() === null)) {
    return Response.json({ error: "Vocal separation is unavailable." }, { status: 503 });
  }
  const originalName = audio.name || "";
  if (originalName.includes("\0") || originalName.length > 240) {
    return Response.json({ error: "Could not read that audio file." }, { status: 400 });
  }

  const id = randomUUID();
  const directory = jobDirectory(id);
  if (!directory) return Response.json({ error: "Vocal separation failed." }, { status: 500 });
  const input = path.resolve(directory, `input${audioSuffix(originalName)}`);
  if (!input.startsWith(`${directory}${path.sep}`)) {
    return Response.json({ error: "Could not read that audio file." }, { status: 400 });
  }
  const payload = Buffer.from(await audio.arrayBuffer());
  if (payload.length === 0 || payload.length > MAX_BYTES) {
    return Response.json({ error: "Choose a file under 120 MB." }, { status: 400 });
  }
  await mkdir(directory, { recursive: true });
  const handle = await open(input, "w", 0o600);
  try {
    await handle.writeFile(payload);
  } finally {
    await handle.close();
  }
  const written = await stat(input);
  if (!written.isFile() || written.size !== payload.length) {
    await rm(directory, { recursive: true, force: true });
    return Response.json({ error: "Could not read that audio file." }, { status: 400 });
  }
  const mode = separationMode(form.get("mode"));
  const behind = isVocalBusy();
  writeFileSync(
    path.join(directory, "status.json"),
    JSON.stringify({
      status: behind ? "queued" : "running",
      mode,
      chunksDone: 0,
      chunksTotal: null,
      startedAt: behind ? null : Date.now(),
      receivedAt: Date.now(),
    }),
  );
  finishJob(id, directory, input, mode);

  return Response.json({
    id,
    status: behind ? "queued" : "running",
    mode,
    note: noteFor(mode),
  });
}

export async function GET(request: NextRequest) {
  const id = request.nextUrl.searchParams.get("id") ?? "";
  const stem = request.nextUrl.searchParams.get("stem");
  const directory = jobDirectory(id);
  if (!directory) {
    return Response.json({ error: "That stem was not created." }, { status: 404 });
  }
  if (!stem) return Response.json(await jobStatus(directory));
  const stored = await readJson(path.join(directory, "status.json"));
  const mode: SeparationMode = stored?.mode === "vocals" ? "vocals" : "stems";
  if (!new Set<string>(STEM_FILES[mode]).has(stem)) {
    return Response.json({ error: "That stem was not created." }, { status: 404 });
  }
  const file = path.join(directory, `${stem}.wav`);
  try {
    await assertWav(file);
  } catch {
    return Response.json({ error: "That stem was not created." }, { status: 404 });
  }
  const stream = Readable.toWeb(createReadStream(file)) as ReadableStream;
  return new Response(stream, {
    headers: {
      "Content-Type": "audio/wav",
      "Content-Disposition": `attachment; filename="${stem}.wav"`,
    },
  });
}

export async function DELETE(request: NextRequest) {
  const id = request.nextUrl.searchParams.get("id") ?? "";
  const directory = jobDirectory(id);
  if (!directory) return Response.json({ error: "That job was not found." }, { status: 404 });
  cancelVocalJob(id);
  await rm(directory, { recursive: true, force: true });
  return Response.json({ deleted: true });
}
