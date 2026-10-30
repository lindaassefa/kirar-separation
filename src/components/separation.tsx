"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

const MAX_SECONDS = 10 * 60;
const MAX_BYTES = 120 * 1024 * 1024;
const WAVE_BINS = 280;

const STEM_OUTPUTS = [
  { id: "vocals", title: "Vocals", file: "vocals.wav", tone: "text-lilac", detail: "Singing" },
  { id: "drums", title: "Drums", file: "drums.wav", tone: "text-lilac", detail: "Drums" },
  { id: "bass", title: "Bass", file: "bass.wav", tone: "text-lilac", detail: "Bass" },
  { id: "other", title: "Other", file: "other.wav", tone: "text-accent", detail: "Remaining instruments" },
] as const;

const VOCAL_OUTPUTS = [
  { id: "vocals", title: "Vocals", file: "vocals.wav", tone: "text-lilac", detail: "Vocal estimate" },
  { id: "instrumental", title: "Instrumental", file: "instrumental.wav", tone: "text-accent", detail: "Mix minus vocals" },
] as const;

type SeparationMode = "stems" | "vocals";
type OutputId = (typeof STEM_OUTPUTS)[number]["id"] | (typeof VOCAL_OUTPUTS)[number]["id"];

type Envelope = { min: number; max: number };

type LoadedSong = {
  name: string;
  url: string;
  file: File;
  samples: Float32Array;
  duration: number;
};

type StemAudio = {
  url: string;
  samples: Float32Array;
  duration: number;
};

type SeparationProgress = {
  queued: boolean;
  done: number | null;
  total: number | null;
  started: number | null;
};

function formatSeparationStatus(mode: SeparationMode, progress: SeparationProgress): string {
  const label = mode === "vocals" ? "Separating vocals" : "Separating stems";
  if (progress.queued) return "Waiting for the previous separation to finish.";
  const elapsedSeconds = progress.started ? Math.max(0, (Date.now() - progress.started) / 1000) : 0;
  const elapsed = formatClock(elapsedSeconds);
  if (progress.done === null || progress.total === null || progress.total < 1) {
    return `${label}… ${elapsed}`;
  }
  const position = `${progress.done} of ${progress.total}`;
  if (progress.done < 1 || progress.started === null || progress.done >= progress.total) {
    return `${label}… ${position} · ${elapsed} elapsed`;
  }
  const remaining = (elapsedSeconds * (progress.total - progress.done)) / progress.done;
  return `${label}… ${position} · ${elapsed} elapsed · about ${formatClock(remaining)} left`;
}

function formatClock(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const whole = Math.floor(safe % 60);
  return `${minutes}:${whole.toString().padStart(2, "0")}`;
}

function buildEnvelope(samples: Float32Array, bins: number): Envelope[] {
  const envelope = Array.from({ length: bins }, () => ({ min: 0, max: 0 }));
  if (samples.length === 0) return envelope;
  const step = samples.length / bins;
  let loudest = 1e-6;
  for (let bin = 0; bin < bins; bin++) {
    const start = Math.floor(bin * step);
    const end = Math.max(start + 1, Math.min(samples.length, Math.floor((bin + 1) * step)));
    let min = 0;
    let max = 0;
    for (let index = start; index < end; index++) {
      const sample = samples[index];
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    envelope[bin] = { min, max };
    loudest = Math.max(loudest, Math.abs(min), Math.abs(max));
  }
  return envelope.map((point) => ({ min: point.min / loudest, max: point.max / loudest }));
}

function waveformPath(envelope: Envelope[]): string {
  if (envelope.length === 0) return "";
  const top = envelope.map((point, index) => `${index},${50 - point.max * 46}`);
  const bottom = [...envelope]
    .reverse()
    .map((point, index) => `${envelope.length - 1 - index},${50 - point.min * 46}`);
  return `M ${top.join(" L ")} L ${bottom.join(" L ")} Z`;
}

function Waveform({
  envelope,
  playhead,
  accent,
  label,
}: {
  envelope: Envelope[];
  playhead?: number;
  accent: "lime" | "lilac" | "muted";
  label: string;
}) {
  const gradientId = useId().replaceAll(":", "");
  const stroke = accent === "lime" ? "var(--accent)" : accent === "lilac" ? "var(--lilac)" : "var(--line)";
  const empty = envelope.every((point) => point.min === 0 && point.max === 0);

  return (
    <div className="relative h-28 bg-paper">
      <svg viewBox={`0 0 ${Math.max(envelope.length - 1, 1)} 100`} preserveAspectRatio="none" className="h-full w-full" aria-hidden="true">
        <line x1="0" x2={Math.max(envelope.length - 1, 1)} y1="50" y2="50" stroke="var(--line)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        {empty ? null : (
          <>
            <path d={waveformPath(envelope)} fill={`url(#${gradientId})`} />
            <path d={waveformPath(envelope)} fill="none" stroke={stroke} strokeWidth="1.25" vectorEffect="non-scaling-stroke" />
            <defs>
              <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={stroke} stopOpacity="0.28" />
                <stop offset="50%" stopColor={stroke} stopOpacity="0.08" />
                <stop offset="100%" stopColor={stroke} stopOpacity="0.28" />
              </linearGradient>
            </defs>
          </>
        )}
      </svg>
      {playhead !== undefined ? (
        <div className="pointer-events-none absolute inset-y-0 w-px bg-accent" style={{ left: `${playhead * 100}%` }} />
      ) : null}
      <span className="sr-only">{label}</span>
    </div>
  );
}

function postAudio(
  file: File,
  mode: SeparationMode,
  onProgress: (ratio: number) => void,
  signal: AbortSignal,
): Promise<{
  ok: boolean;
  status: number;
  body: {
    id?: string;
    note?: string;
    error?: string;
    status?: string;
    outputs?: Partial<Record<OutputId, boolean>>;
    chunksDone?: number | null;
    chunksTotal?: number | null;
    startedAt?: number | null;
  };
}> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("audio", file, file.name);
    form.append("mode", mode);
    let lastPercent = -1;
    xhr.open("POST", "/api/separate");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || event.total === 0) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      if (percent === lastPercent) return;
      lastPercent = percent;
      onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText) as {
          id?: string;
          note?: string;
          error?: string;
          status?: string;
          outputs?: Partial<Record<OutputId, boolean>>;
          chunksDone?: number | null;
          chunksTotal?: number | null;
          startedAt?: number | null;
        };
        resolve({ ok: xhr.status >= 200 && xhr.status < 300, status: xhr.status, body });
      } catch {
        reject(new Error("Separation failed. No files were created."));
      }
    };
    xhr.onerror = () => reject(new Error("Could not reach the separator. No audio was created."));
    xhr.onabort = () => reject(new DOMException("The upload was cancelled.", "AbortError"));
    signal.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(form);
  });
}

async function decodeForPreview(file: File): Promise<{ samples: Float32Array; duration: number }> {
  const decodeContext = new AudioContext();
  try {
    const encoded = await file.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(encoded.slice(0));
    if (decoded.duration > MAX_SECONDS) {
      throw new Error("Choose a file of ten minutes or less.");
    }
    const offline = new OfflineAudioContext(1, decoded.length, decoded.sampleRate);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    return {
      samples: new Float32Array(rendered.getChannelData(0)),
      duration: rendered.length / rendered.sampleRate,
    };
  } finally {
    await decodeContext.close();
  }
}

function StemPlayer({ audio }: { audio?: StemAudio }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => {
          const element = ref.current;
          if (!element) return;
          if (element.paused) void element.play();
          else element.pause();
        }}
        disabled={!audio}
        className="border border-line px-4 py-2 text-sm transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:text-muted disabled:hover:border-line"
        aria-pressed={playing}
      >
        {playing ? "Pause" : "Play"}
      </button>
      {audio ? (
        <audio
          ref={ref}
          className="sr-only"
          src={audio.url}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />
      ) : null}
    </>
  );
}

export default function Separation() {
  const [song, setSong] = useState<LoadedSong | null>(null);
  const [mode, setMode] = useState<SeparationMode>("stems");
  const [stems, setStems] = useState<Partial<Record<OutputId, StemAudio>>>({});
  const [status, setStatus] = useState("Choose an audio file.");
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const requestId = useRef(0);
  const separationAbort = useRef<AbortController | null>(null);

  const envelope = useMemo(
    () => (song ? buildEnvelope(song.samples, WAVE_BINS) : buildEnvelope(new Float32Array(), WAVE_BINS)),
    [song],
  );
  const emptyEnvelope = useMemo(() => buildEnvelope(new Float32Array(), WAVE_BINS), []);

  useEffect(() => {
    return () => {
      if (song?.url) URL.revokeObjectURL(song.url);
    };
  }, [song]);

  useEffect(() => {
    return () => {
      for (const stem of Object.values(stems)) {
        if (stem) URL.revokeObjectURL(stem.url);
      }
    };
  }, [stems]);

  async function onFile(file: File | null) {
    if (!file) return;
    const id = ++requestId.current;
    separationAbort.current?.abort();
    setError(null);
    setNote(null);
    setChecking(false);
    setPlaying(false);
    setCurrentTime(0);
    setStems((current) => {
      for (const stem of Object.values(current)) {
        if (stem) URL.revokeObjectURL(stem.url);
      }
      return {};
    });
    if (file.size > MAX_BYTES) {
      setError("Choose a file under 120 MB.");
      setStatus("Choose a WAV, MP3, M4A, or FLAC.");
      return;
    }
    setStatus("Reading audio…");
    try {
      const decoded = await decodeForPreview(file);
      if (requestId.current !== id) return;
      setSong((current) => {
        if (current?.url) URL.revokeObjectURL(current.url);
        return {
          name: file.name,
          url: URL.createObjectURL(file),
          file,
          samples: decoded.samples,
          duration: decoded.duration,
        };
      });
      setStatus("Source loaded. Separation has not run.");
    } catch (cause) {
      if (requestId.current !== id) return;
      const message = cause instanceof Error ? cause.message : "";
      setError(/decode|encoding/i.test(message) || !message ? "Could not read that file." : message);
      setStatus("Choose a WAV, MP3, M4A, or FLAC.");
    }
  }

  function clearStems() {
    setStems((current) => {
      for (const stem of Object.values(current)) {
        if (stem) URL.revokeObjectURL(stem.url);
      }
      return {};
    });
  }

  function chooseMode(next: SeparationMode) {
    if (checking || next === mode) return;
    setMode(next);
    setNote(null);
    clearStems();
  }

  async function requestSeparation() {
    if (!song || checking) return;
    const requestedMode = mode;
    const outputs = requestedMode === "vocals" ? VOCAL_OUTPUTS : STEM_OUTPUTS;
    const id = requestId.current;
    const controller = new AbortController();
    separationAbort.current?.abort();
    separationAbort.current = controller;
    const phase = { current: "upload" as "upload" | "separate" | "preview" };
    const progress: SeparationProgress = { queued: false, done: null, total: null, started: null };
    const timer = window.setInterval(() => {
      if (requestId.current !== id || phase.current !== "separate") return;
      setStatus(formatSeparationStatus(requestedMode, progress));
    }, 1000);
    setChecking(true);
    setError(null);
    setNote(null);
    clearStems();
    setStatus("Uploading… 0%");
    let separating = false;
    try {
      const response = await postAudio(
        song.file,
        requestedMode,
        (ratio) => {
          if (requestId.current !== id) return;
          setStatus(`Uploading… ${Math.round(ratio * 100)}%`);
        },
        controller.signal,
      );
      if (requestId.current !== id) {
        if (response.body.id) void fetch(`/api/separate?id=${response.body.id}`, { method: "DELETE" }).catch(() => undefined);
        return;
      }
      separating = true;
      phase.current = "separate";
      let body = response.body;
      if (!response.ok || !body.id) {
        setError(body.error ?? "Separation failed. No files were created.");
        setStatus("Separation did not run.");
        return;
      }
      const jobId = body.id;
      const forgetJob = () => {
        void fetch(`/api/separate?id=${jobId}`, { method: "DELETE" }).catch(() => undefined);
      };
      controller.signal.addEventListener("abort", forgetJob, { once: true });
      if (body.status === "running" || body.status === "queued" || !body.outputs) {
        progress.queued = body.status === "queued";
        progress.started = typeof body.startedAt === "number" ? body.startedAt : Date.now();
        setStatus(formatSeparationStatus(requestedMode, progress));
        while (!body.outputs) {
          const poll = await fetch(`/api/separate?id=${jobId}`, { signal: controller.signal });
          const polled = (await poll.json()) as typeof body;
          if (!poll.ok) {
            setError(polled.error ?? "Separation did not finish. No audio was created.");
            setStatus("Separation did not run.");
            return;
          }
          body = polled;
          if (body.status === "error") {
            setError(body.error ?? "Separation failed. No files were created.");
            setStatus("Separation did not run.");
            return;
          }
          progress.queued = body.status === "queued";
          progress.done = typeof body.chunksDone === "number" ? body.chunksDone : null;
          progress.total = typeof body.chunksTotal === "number" ? body.chunksTotal : null;
          progress.started = typeof body.startedAt === "number" ? body.startedAt : progress.started;
          if (requestId.current !== id) return;
          setStatus(formatSeparationStatus(requestedMode, progress));
          if (body.outputs) break;
          await new Promise<void>((resolve, reject) => {
            const wait = window.setTimeout(resolve, 400);
            controller.signal.addEventListener(
              "abort",
              () => {
                window.clearTimeout(wait);
                reject(new DOMException("The upload was cancelled.", "AbortError"));
              },
              { once: true },
            );
          });
        }
      }
      if (!body.outputs) {
        setError(body.error ?? "Separation failed. No files were created.");
        setStatus("Separation did not run.");
        return;
      }
      phase.current = "preview";
      setStatus("Preparing previews…");
      const next: Partial<Record<OutputId, StemAudio>> = {};
      for (const output of outputs) {
        if (!body.outputs[output.id]) continue;
        const stemResponse = await fetch(`/api/separate?id=${jobId}&stem=${output.id}`, { signal: controller.signal });
        if (!stemResponse.ok) {
          setError("A stem file was missing. Nothing was downloaded.");
          setStatus("Separation did not run.");
          return;
        }
        const blob = await stemResponse.blob();
        if (blob.size <= 44) {
          setError("A stem file was empty. Nothing was downloaded.");
          setStatus("Separation did not run.");
          return;
        }
        const decoded = await decodeForPreview(new File([blob], output.file, { type: "audio/wav" }));
        if (decoded.samples.length === 0) {
          setError("A stem file was empty. Nothing was downloaded.");
          setStatus("Separation did not run.");
          return;
        }
        next[output.id] = { url: URL.createObjectURL(blob), samples: decoded.samples, duration: decoded.duration };
      }
      if (requestId.current !== id) {
        for (const stem of Object.values(next)) {
          if (stem) URL.revokeObjectURL(stem.url);
        }
        return;
      }
      controller.signal.removeEventListener("abort", forgetJob);
      void fetch(`/api/separate?id=${jobId}`, { method: "DELETE" }).catch(() => undefined);
      setStems(next);
      setNote(
        body.note ??
          (requestedMode === "vocals"
            ? "Vocals and an instrumental are ready. This mode can treat krar as vocals. A separate krar stem was not created."
            : "Vocals, drums, bass, and other are ready. A separate krar stem was not created."),
      );
      setStatus(requestedMode === "vocals" ? "Vocals and instrumental are ready." : "Four stems are ready.");
    } catch (cause) {
      if (requestId.current !== id) return;
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(
        separating
          ? "Could not finish separation. No audio was created."
          : cause instanceof Error
            ? cause.message
            : "Could not reach the separator. No audio was created.",
      );
      setStatus("Separation did not run.");
    } finally {
      window.clearInterval(timer);
      if (requestId.current === id) setChecking(false);
    }
  }

  function togglePlayback() {
    const element = audioRef.current;
    if (!element) return;
    if (element.paused) {
      void element.play();
    } else {
      element.pause();
    }
  }

  const playhead = song && song.duration > 0 ? currentTime / song.duration : 0;

  return (
    <div className="flex flex-col gap-6">
      <section
        className={`border bg-paper transition-colors ${dragging ? "border-accent" : "border-line"}`}
        onDragEnter={(event) => {
          event.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          dragDepth.current = 0;
          setDragging(false);
          void onFile(event.dataTransfer.files?.[0] ?? null);
        }}
      >
        <div className="flex flex-col gap-4 border-b border-line px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <p className="text-xs tracking-wide text-muted uppercase">Source</p>
            <p className="truncate text-sm" data-testid="file-name">
              {song ? song.name : "No file"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <p className="text-sm text-muted tabular-nums">{song ? formatClock(song.duration) : "0:00"}</p>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="bg-accent px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-accent-deep"
            >
              Choose audio
            </button>
            <input
              ref={fileRef}
              id="song-file"
              data-testid="song-file"
              className="sr-only"
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"
              onChange={(event) => {
                void onFile(event.target.files?.[0] ?? null);
                event.target.value = "";
              }}
            />
          </div>
        </div>

        <div className="px-4 py-4">
          <Waveform
            envelope={envelope}
            playhead={song ? playhead : undefined}
            accent="lime"
            label={song ? "Waveform of the source recording" : "No source waveform"}
          />
        </div>

        <div className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-4">
          <button
            type="button"
            onClick={togglePlayback}
            disabled={!song}
            className="border border-line px-4 py-2 text-sm transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:text-muted disabled:hover:border-line"
            aria-pressed={playing}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <span className="text-sm text-muted tabular-nums">
            {formatClock(currentTime)} / {song ? formatClock(song.duration) : "0:00"}
          </span>
          <label className="sr-only" htmlFor="playback-position">
            Playback position
          </label>
          <input
            id="playback-position"
            className="playback-range"
            type="range"
            min={0}
            max={song?.duration ?? 0}
            step={0.01}
            disabled={!song}
            value={song ? Math.min(currentTime, song.duration) : 0}
            onChange={(event) => {
              const next = Number(event.target.value);
              setCurrentTime(next);
              if (audioRef.current) audioRef.current.currentTime = next;
            }}
          />
          <div className="flex border border-line" role="group" aria-label="Separation mode">
            <button
              type="button"
              data-testid="mode-stems"
              aria-pressed={mode === "stems"}
              disabled={checking}
              onClick={() => chooseMode("stems")}
              className={`px-4 py-2 text-sm transition-colors disabled:cursor-not-allowed ${mode === "stems" ? "bg-accent text-ink" : "text-muted hover:text-accent"}`}
            >
              Four stems
            </button>
            <button
              type="button"
              data-testid="mode-vocals"
              aria-pressed={mode === "vocals"}
              disabled={checking}
              onClick={() => chooseMode("vocals")}
              className={`border-l border-line px-4 py-2 text-sm transition-colors disabled:cursor-not-allowed ${mode === "vocals" ? "bg-accent text-ink" : "text-muted hover:text-accent"}`}
            >
              Vocals and instrumental
            </button>
          </div>
          <button
            type="button"
            data-testid="separate"
            onClick={() => {
              void requestSeparation();
            }}
            disabled={!song || checking}
            className="border border-accent px-4 py-2 text-sm text-accent transition-colors hover:bg-accent hover:text-ink disabled:cursor-not-allowed disabled:border-line disabled:text-muted disabled:hover:bg-transparent"
          >
            {checking ? "Processing…" : "Separate"}
          </button>
          {song ? (
            <audio
              ref={audioRef}
              className="sr-only"
              src={song.url}
              preload="metadata"
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
            />
          ) : null}
        </div>
      </section>

      <section aria-live="polite" className="flex flex-col gap-3">
        <div className="h-px bg-line">
          {checking ? <div className="listen-rule h-px bg-accent" /> : null}
        </div>
        <p className="text-sm text-muted" data-testid="separation-status">
          {status}
        </p>
        {error ? (
          <p className="border border-line bg-paper px-4 py-3 text-sm leading-6 text-lilac" data-testid="separation-error">
            {error}
          </p>
        ) : null}
        {note ? <p className="text-sm text-muted">{note}</p> : null}
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        {(mode === "vocals" ? VOCAL_OUTPUTS : STEM_OUTPUTS).map((output) => {
          const stem = stems[output.id];
          return (
            <section key={output.id} className="flex flex-col border border-line bg-paper">
              <div className="flex items-baseline justify-between gap-4 border-b border-line px-4 py-3">
                <div>
                  <h2 className={`font-display text-2xl font-semibold tracking-tight ${output.tone}`}>{output.title}</h2>
                  <p className="text-xs text-muted">{output.detail}</p>
                </div>
                <p className="text-xs text-muted">{output.file}</p>
              </div>
              <div className="px-4 py-4">
                <Waveform
                  envelope={stem ? buildEnvelope(stem.samples, WAVE_BINS) : emptyEnvelope}
                  accent={output.tone === "text-accent" ? "lime" : "lilac"}
                  label={stem ? `${output.title} waveform` : `${output.title} waveform unavailable`}
                />
                <p className="pt-3 text-sm text-muted" data-testid={`preview-${output.id}`}>
                  {stem ? formatClock(stem.duration) : "No audio"}
                </p>
              </div>
              <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3">
                <StemPlayer audio={stem} />
                {stem ? (
                  <a
                    href={stem.url}
                    download={output.file}
                    data-testid={`download-${output.id}`}
                    className="border border-line px-4 py-2 text-sm transition-colors hover:border-accent hover:text-accent"
                  >
                    Download WAV
                  </a>
                ) : (
                  <button
                    type="button"
                    disabled
                    data-testid={`download-${output.id}`}
                    className="cursor-not-allowed border border-line px-4 py-2 text-sm text-muted"
                  >
                    Download WAV
                  </button>
                )}
              </div>
            </section>
          );
        })}
      </div>

      <section className="flex flex-col border border-lilac/40 bg-paper">
        <div className="flex items-baseline justify-between gap-4 border-b border-line px-4 py-3">
          <div>
            <p className="text-xs tracking-wide text-lilac uppercase">Experimental</p>
            <h2 className="font-display text-2xl font-semibold tracking-tight text-lilac">Krar</h2>
          </div>
          <p className="text-xs text-muted">krar.wav</p>
        </div>
        <div className="px-4 py-4">
          <p className="text-sm leading-6 text-muted" data-testid="preview-krar">
            A separate krar stem is not available. Other is the remaining instruments, and the instrumental is the mix minus the vocal estimate. Neither is isolated krar.
          </p>
        </div>
        <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3">
          <button
            type="button"
            disabled
            className="cursor-not-allowed border border-line px-4 py-2 text-sm text-muted"
          >
            Play
          </button>
          <button
            type="button"
            disabled
            data-testid="download-krar"
            className="cursor-not-allowed border border-line px-4 py-2 text-sm text-muted"
          >
            Download WAV
          </button>
        </div>
      </section>
    </div>
  );
}
