"use client";

// Earlier instrument classifier, kept for reference. The product page does not use it.
// instrumentiq_model.onnx names krar, guitar, or piano. It cannot write separated audio.

import { useEffect, useMemo, useRef, useState } from "react";
import { classifySamples, type Prediction } from "@/lib/inference";
import { CLIP_SAMPLES, CLIP_SECONDS, SAMPLE_RATE } from "@/lib/preprocess";

const MAX_SECONDS = 10 * 60;
const WAVE_BINS = 180;

const INSTRUMENTS = [
  { id: "krar", label: "Krar" },
  { id: "guitar", label: "Guitar" },
  { id: "piano", label: "Piano" },
] as const;

type LoadedAudio = {
  name: string;
  url: string;
  samples: Float32Array;
  duration: number;
};

function formatSeconds(seconds: number): string {
  return seconds.toFixed(2);
}

function formatClock(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const whole = Math.floor(safe % 60);
  return `${minutes}:${whole.toString().padStart(2, "0")}`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function buildPeaks(samples: Float32Array, bins: number): number[] {
  if (samples.length === 0) return Array.from({ length: bins }, () => 0);
  const peaks = new Array<number>(bins).fill(0);
  const step = samples.length / bins;
  for (let bin = 0; bin < bins; bin++) {
    const start = Math.floor(bin * step);
    const end = Math.min(samples.length, Math.floor((bin + 1) * step));
    let peak = 0;
    for (let i = start; i < end; i++) peak = Math.max(peak, Math.abs(samples[i]));
    peaks[bin] = peak;
  }
  const loudest = Math.max(...peaks, 1e-6);
  return peaks.map((peak) => peak / loudest);
}

async function decodeTo16kMono(file: File): Promise<{ samples: Float32Array; duration: number }> {
  const decodeContext = new AudioContext();
  try {
    const encoded = await file.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(encoded.slice(0));
    if (decoded.duration > MAX_SECONDS) {
      throw new Error("Please choose a recording of ten minutes or less.");
    }
    const frameCount = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE));
    const offline = new OfflineAudioContext(1, frameCount, SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    return {
      samples: new Float32Array(rendered.getChannelData(0)),
      duration: rendered.length / SAMPLE_RATE,
    };
  } finally {
    await decodeContext.close();
  }
}

export default function Studio() {
  const [audio, setAudio] = useState<LoadedAudio | null>(null);
  const [startSeconds, setStartSeconds] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [status, setStatus] = useState("Drop a recording to begin.");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const requestId = useRef(0);
  const dragDepth = useRef(0);

  const peaks = useMemo(
    () => (audio ? buildPeaks(audio.samples, WAVE_BINS) : []),
    [audio],
  );

  useEffect(() => {
    return () => {
      if (audio?.url) URL.revokeObjectURL(audio.url);
    };
  }, [audio]);

  useEffect(() => {
    if (!audio) return;
    const id = ++requestId.current;
    const timer = window.setTimeout(() => {
      void (async () => {
        setBusy(true);
        setError(null);
        setStatus("Listening to the excerpt…");
        try {
          const startSample = Math.round(startSeconds * SAMPLE_RATE);
          const next = await classifySamples(audio.samples, startSample);
          if (requestId.current !== id) return;
          setPrediction(next);
          const end = Math.min(audio.duration, startSeconds + CLIP_SECONDS);
          setStatus(`Heard ${formatSeconds(startSeconds)}–${formatSeconds(end)} s.`);
          if (audioRef.current && Math.abs(audioRef.current.currentTime - startSeconds) > 0.05) {
            audioRef.current.currentTime = startSeconds;
          }
        } catch (cause) {
          if (requestId.current !== id) return;
          setPrediction(null);
          setError(cause instanceof Error ? cause.message : "Recognition did not finish.");
          setStatus("The excerpt could not be heard.");
        } finally {
          if (requestId.current === id) setBusy(false);
        }
      })();
    }, 180);

    return () => window.clearTimeout(timer);
  }, [audio, startSeconds]);

  async function onFile(file: File | null) {
    if (!file) return;
    const id = ++requestId.current;
    setBusy(true);
    setError(null);
    setPrediction(null);
    setPlaying(false);
    setCurrentTime(0);
    setStatus("Reading the recording…");
    try {
      const decoded = await decodeTo16kMono(file);
      if (requestId.current !== id) return;
      setAudio((current) => {
        if (current?.url) URL.revokeObjectURL(current.url);
        return {
          name: file.name,
          url: URL.createObjectURL(file),
          samples: decoded.samples,
          duration: decoded.duration,
        };
      });
      setStartSeconds(0);
    } catch (cause) {
      if (requestId.current !== id) return;
      setError(cause instanceof Error ? cause.message : "That file could not be read.");
      setStatus("Choose a WAV, MP3, M4A, or FLAC the browser can play.");
    } finally {
      if (requestId.current === id) setBusy(false);
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

  const maxStart = audio ? Math.max(0, audio.duration - CLIP_SECONDS) : 0;
  const excerptIsShort = audio ? audio.duration < CLIP_SECONDS : false;
  const excerptEnd = audio ? Math.min(audio.duration, startSeconds + CLIP_SECONDS) : 0;
  const windowStart = audio && audio.duration > 0 ? startSeconds / audio.duration : 0;
  const windowWidth =
    audio && audio.duration > 0 ? Math.min(CLIP_SECONDS, audio.duration) / audio.duration : 0;
  const playhead = audio && audio.duration > 0 ? currentTime / audio.duration : 0;
  const predictionLabel = prediction
    ? INSTRUMENTS.find((item) => item.id === prediction.label)?.label
    : null;

  return (
    <div className="grid items-start gap-16 lg:grid-cols-12 lg:gap-10">
      <div className="flex flex-col gap-10 lg:col-span-7">
        <div
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
          className={`border px-5 py-8 transition-colors sm:px-7 sm:py-10 ${
            dragging ? "border-accent bg-paper" : "border-line bg-transparent hover:border-accent"
          }`}
        >
          <label htmlFor="audio-file" className="flex cursor-pointer flex-col gap-3">
            <span className="font-serif text-3xl leading-tight font-light sm:text-4xl">
              {audio ? "Drop another recording" : "Drop a recording"}
            </span>
            <span className="max-w-md text-sm leading-6 text-muted">
              Or choose a file. WAV, MP3, M4A, or FLAC. The audio stays on this device.
            </span>
            <input
              id="audio-file"
              data-testid="audio-file"
              className="sr-only"
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"
              onChange={(event) => {
                void onFile(event.target.files?.[0] ?? null);
                event.target.value = "";
              }}
            />
          </label>
        </div>

        {audio ? (
          <div className="flex flex-col gap-8">
            <div className="flex items-baseline justify-between gap-4">
              <p className="truncate text-sm text-foreground" data-testid="file-name">
                {audio.name}
              </p>
              <p className="shrink-0 text-sm text-muted tabular-nums">{formatClock(audio.duration)}</p>
            </div>

            <figure className="flex flex-col gap-3">
              <figcaption className="sr-only">
                Waveform of the recording. The shaded span is the four-second excerpt used for
                instrument recognition.
              </figcaption>
              <div className="relative h-28 border-y border-line">
                <svg
                  viewBox={`0 0 ${WAVE_BINS} 100`}
                  preserveAspectRatio="none"
                  className="h-full w-full text-foreground/80"
                  aria-hidden="true"
                >
                  {peaks.map((peak, index) => {
                    const height = Math.max(2, peak * 78);
                    const y = (100 - height) / 2;
                    return (
                      <line
                        key={index}
                        x1={index + 0.5}
                        x2={index + 0.5}
                        y1={y}
                        y2={y + height}
                        stroke="currentColor"
                        strokeWidth="0.72"
                      />
                    );
                  })}
                </svg>
                <div
                  className="pointer-events-none absolute inset-y-0 bg-accent/10"
                  style={{
                    left: `${windowStart * 100}%`,
                    width: `${windowWidth * 100}%`,
                  }}
                />
                <div
                  className="pointer-events-none absolute inset-y-0 w-px bg-accent"
                  style={{ left: `${playhead * 100}%` }}
                />
              </div>
            </figure>

            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  onClick={togglePlayback}
                  className="border border-accent bg-accent px-4 py-2 text-sm text-paper transition-colors hover:bg-accent-deep"
                  aria-pressed={playing}
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <span className="text-sm text-muted tabular-nums">
                  {formatClock(currentTime)} / {formatClock(audio.duration)}
                </span>
              </div>
              <label className="sr-only" htmlFor="playback-position">
                Playback position
              </label>
              <input
                id="playback-position"
                className="playback-range"
                type="range"
                min={0}
                max={audio.duration}
                step={0.01}
                value={Math.min(currentTime, audio.duration)}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  setCurrentTime(next);
                  if (audioRef.current) audioRef.current.currentTime = next;
                }}
              />
              <audio
                ref={audioRef}
                className="sr-only"
                src={audio.url}
                preload="metadata"
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
              />
            </div>

            {maxStart > 0 ? (
              <label className="flex flex-col gap-3 text-sm" htmlFor="excerpt-slider">
                <span className="flex items-baseline justify-between gap-4">
                  <span>Four-second excerpt</span>
                  <span className="text-muted tabular-nums" data-testid="excerpt-start">
                    {formatSeconds(startSeconds)}–{formatSeconds(excerptEnd)} s
                  </span>
                </span>
                <input
                  id="excerpt-slider"
                  data-testid="excerpt-slider"
                  className="excerpt-range"
                  type="range"
                  min={0}
                  max={maxStart}
                  step={0.05}
                  value={Math.min(startSeconds, maxStart)}
                  aria-valuetext={`${formatSeconds(startSeconds)} to ${formatSeconds(excerptEnd)} seconds`}
                  onChange={(event) => setStartSeconds(Number(event.target.value))}
                />
              </label>
            ) : null}
            {excerptIsShort ? (
              <p className="text-sm leading-6 text-muted">
                This recording is shorter than four seconds, so the rest of the excerpt is silence.
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      <section className="flex flex-col gap-8 lg:col-span-5 lg:mt-28" aria-live="polite">
        <div className="flex flex-col gap-3">
          <h2 className="text-sm tracking-wide text-accent">Instrument recognition</h2>
          <p className="font-serif text-5xl leading-none font-light" data-testid="prediction">
            {predictionLabel ?? "—"}
          </p>
          <div className="h-px w-full bg-line" aria-hidden="true">
            {busy ? <div className="listen-rule h-px w-full bg-accent" /> : null}
          </div>
          <p className="text-sm leading-6 text-muted" data-testid="status">
            {busy ? "Listening…" : status}
          </p>
        </div>

        {error ? (
          <p
            className="border border-accent/40 bg-paper px-4 py-3 text-sm leading-6 text-accent-deep"
            data-testid="error"
          >
            {error}
          </p>
        ) : null}

        <ul className="flex flex-col gap-5">
          {INSTRUMENTS.map((item) => {
            const probability = prediction?.probabilities[item.id] ?? 0;
            const width = prediction ? Math.max(0, Math.min(100, probability * 100)) : 0;
            return (
              <li key={item.id} className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-3">
                <span className="text-sm">{item.label}</span>
                <span className="h-px bg-track">
                  <span
                    className="block h-1.5 -translate-y-[3px] bg-accent transition-[width] duration-500 ease-out motion-reduce:transition-none"
                    style={{ width: `${width}%` }}
                  />
                </span>
                <span
                  className="text-right text-sm text-muted tabular-nums"
                  data-testid={`score-${item.id}`}
                >
                  {prediction ? formatPercent(probability) : "—"}
                </span>
              </li>
            );
          })}
        </ul>

        <p className="max-w-sm text-sm leading-6 text-muted">
          Piano currently uses acoustic keyboard-family training examples, so that result means an
          acoustic keyboard sound rather than a particular electronic keyboard.
        </p>
        {prediction?.padded ? (
          <p className="text-sm text-muted" data-testid="padded">
            The excerpt was padded with silence to {CLIP_SAMPLES.toLocaleString()} samples.
          </p>
        ) : null}
      </section>
    </div>
  );
}
