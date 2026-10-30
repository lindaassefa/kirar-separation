"use client";

import { useEffect, useRef, useState } from "react";

import {
  CHOICES,
  EXCERPTS,
  TRACKS,
  type ChoiceId,
  type ExcerptId,
  type ExcerptJudgment,
  type ListeningNotes,
  type TrackId,
  emptyNotes,
} from "@/lib/listen-catalog";

function formatClock(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const whole = Math.floor(safe % 60);
  return `${minutes}:${whole.toString().padStart(2, "0")}`;
}

function ExcerptPlayer({
  excerptId,
  title,
  detail,
  judgment,
  onJudgment,
  claimPlayback,
  releasePlayback,
  registerStop,
}: {
  excerptId: ExcerptId;
  title: string;
  detail: string;
  judgment: ExcerptJudgment;
  onJudgment: (excerpt: ExcerptId, patch: Partial<ExcerptJudgment>) => void;
  claimPlayback: (excerpt: ExcerptId) => void;
  releasePlayback: (excerpt: ExcerptId) => void;
  registerStop: (excerpt: ExcerptId, stop: () => void) => void;
}) {
  const buffersRef = useRef<Partial<Record<TrackId, AudioBuffer>>>({});
  const contextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const tokenRef = useRef(0);
  const selectedRef = useRef<TrackId>("original");
  const playingRef = useRef(false);
  const loopRef = useRef(true);
  const offsetRef = useRef(0);
  const startedAtRef = useRef(0);
  const frameRef = useRef(0);

  const [selected, setSelected] = useState<TrackId>("original");
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.6);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const aliveRef = useRef(true);
  const pauseRef = useRef<() => void>(() => {});
  const volumeRef = useRef(volume);

  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  useEffect(() => {
    loopRef.current = loop;
  }, [loop]);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      cancelAnimationFrame(frameRef.current);
      try {
        sourceRef.current?.stop();
      } catch {
        // The source may already have ended.
      }
    };
  }, []);

  useEffect(() => {
    const audio = new AudioContext();
    contextRef.current = audio;
    const gain = audio.createGain();
    gain.gain.value = volumeRef.current;
    gain.connect(audio.destination);
    gainRef.current = gain;
    let cancelled = false;

    async function load() {
      const buffers = await Promise.all(
        TRACKS.map(async (track) => {
          const response = await fetch(`/api/listen/audio?excerpt=${excerptId}&track=${track.id}`);
          if (!response.ok) throw new Error("A recording is missing.");
          const bytes = await response.arrayBuffer();
          const decoded = await audio.decodeAudioData(bytes.slice(0));
          return [track.id, decoded] as const;
        }),
      );
      if (cancelled) return;
      buffersRef.current = Object.fromEntries(buffers);
      const first = buffers[0]?.[1];
      setDuration(first ? first.duration : 0);
      setReady(true);
    }

    load().catch(() => {
      if (!cancelled) setError("A recording could not be loaded.");
    });

    return () => {
      cancelled = true;
      gain.disconnect();
      contextRef.current = null;
      void audio.close();
    };
  }, [excerptId]);

  useEffect(() => {
    volumeRef.current = volume;
    if (gainRef.current) gainRef.current.gain.value = volume;
  }, [volume]);

  function currentOffset(): number {
    const audio = contextRef.current;
    const buffer = buffersRef.current[selectedRef.current];
    if (!audio || !buffer || !playingRef.current) return offsetRef.current;
    const elapsed = audio.currentTime - startedAtRef.current;
    return Math.min(buffer.duration, offsetRef.current + elapsed);
  }

  function stopSource() {
    tokenRef.current += 1;
    const source = sourceRef.current;
    sourceRef.current = null;
    if (!source) return;
    source.onended = null;
    try {
      source.stop();
    } catch {
      // The source may already have ended.
    }
  }

  function pauseAt(offset: number) {
    stopSource();
    playingRef.current = false;
    offsetRef.current = offset;
    setPlaying(false);
    setTime(offset);
    cancelAnimationFrame(frameRef.current);
    releasePlayback(excerptId);
  }

  useEffect(() => {
    pauseRef.current = () => {
      if (playingRef.current) pauseAt(currentOffset());
    };
    registerStop(excerptId, () => pauseRef.current());
  });

  function watch() {
    const tick = () => {
      if (!playingRef.current) return;
      setTime(currentOffset());
      frameRef.current = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(frameRef.current);
    frameRef.current = requestAnimationFrame(tick);
  }

  function startAt(offset: number, track: TrackId) {
    const audio = contextRef.current;
    const gain = gainRef.current;
    const buffer = buffersRef.current[track];
    if (!audio || !gain || !buffer) return;
    stopSource();
    const source = audio.createBufferSource();
    source.buffer = buffer;
    source.connect(gain);
    const token = tokenRef.current + 1;
    tokenRef.current = token;
    const start = Math.min(Math.max(0, offset), Math.max(0, buffer.duration - 0.05));
    source.onended = () => {
      if (!aliveRef.current || tokenRef.current !== token) return;
      if (loopRef.current) {
        startAt(0, selectedRef.current);
        return;
      }
      playingRef.current = false;
      offsetRef.current = buffer.duration;
      setPlaying(false);
      setTime(buffer.duration);
      releasePlayback(excerptId);
    };
    source.start(0, start);
    sourceRef.current = source;
    startedAtRef.current = audio.currentTime;
    offsetRef.current = start;
    playingRef.current = true;
    setPlaying(true);
    setTime(start);
    watch();
  }

  async function toggle() {
    const audio = contextRef.current;
    if (!audio || !ready) return;
    if (playingRef.current) {
      pauseAt(currentOffset());
      return;
    }
    if (audio.state === "suspended") await audio.resume();
    claimPlayback(excerptId);
    const buffer = buffersRef.current[selectedRef.current];
    const offset = buffer && offsetRef.current >= buffer.duration - 0.05 ? 0 : offsetRef.current;
    startAt(offset, selectedRef.current);
  }

  function chooseTrack(track: TrackId) {
    const offset = currentOffset();
    setSelected(track);
    selectedRef.current = track;
    if (playingRef.current) startAt(offset, track);
    else setTime(offset);
  }

  function seek(next: number) {
    offsetRef.current = next;
    setTime(next);
    if (playingRef.current) startAt(next, selectedRef.current);
  }

  function onKeyDown(key: string) {
    const index = Number(key) - 1;
    const track = TRACKS[index];
    if (!track) return;
    chooseTrack(track.id);
  }

  return (
    <section
      className="flex flex-col border border-line bg-paper"
      data-testid={`listen-${excerptId}`}
      onKeyDown={(event) => onKeyDown(event.key)}
    >
      <div className="flex items-baseline justify-between gap-4 border-b border-line px-4 py-3">
        <h2 className="font-display text-2xl font-semibold tracking-tight">{title}</h2>
        <p className="text-xs text-muted">{detail}</p>
      </div>

      <div className="flex flex-col gap-4 px-4 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void toggle()}
            disabled={!ready || Boolean(error)}
            className="border border-line px-4 py-2 text-sm transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:text-muted"
            aria-pressed={playing}
            data-testid={`play-${excerptId}`}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <span className="text-sm text-muted tabular-nums" data-testid={`time-${excerptId}`}>
            {formatClock(time)} / {formatClock(duration)}
          </span>
          <label className="sr-only" htmlFor={`seek-${excerptId}`}>
            Position
          </label>
          <input
            id={`seek-${excerptId}`}
            className="playback-range"
            type="range"
            min={0}
            max={duration || 0}
            step={0.01}
            value={Math.min(time, duration || 0)}
            disabled={!ready}
            onChange={(event) => seek(Number(event.target.value))}
          />
          <label className="flex items-center gap-2 text-sm text-muted">
            <input type="checkbox" checked={loop} onChange={(event) => setLoop(event.target.checked)} />
            Loop
          </label>
          <label className="flex items-center gap-2 text-sm text-muted">
            Volume
            <input
              className="listen-volume"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              aria-label={`${title} volume`}
              onChange={(event) => setVolume(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="flex flex-wrap gap-2" role="group" aria-label={`${title} sources`}>
          {TRACKS.map((track, index) => {
            const active = selected === track.id;
            return (
              <button
                key={track.id}
                type="button"
                onClick={() => chooseTrack(track.id)}
                disabled={!ready}
                aria-pressed={active}
                data-testid={`track-${excerptId}-${track.id}`}
                className={`border px-3 py-2 text-sm transition-colors disabled:cursor-not-allowed disabled:text-muted ${
                  active ? "border-accent text-accent" : "border-line hover:border-lilac hover:text-lilac"
                }`}
              >
                {index + 1} {track.label}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-muted">Keys 1–5 switch source while this excerpt is focused.</p>
        {error ? <p className="text-sm text-lilac">{error}</p> : null}
        {!ready && !error ? <p className="text-sm text-muted">Loading the five recordings…</p> : null}
      </div>

      <div className="grid gap-4 border-t border-line px-4 py-4 sm:grid-cols-2">
        <Judgment
          excerptId={excerptId}
          question="preserves-krar"
          legend="Which preserves the krar better?"
          value={judgment.preservesKrar}
          onChange={(value) => onJudgment(excerptId, { preservesKrar: value })}
        />
        <Judgment
          excerptId={excerptId}
          question="removes-singing"
          legend="Which removes the singing better?"
          value={judgment.removesSinging}
          onChange={(value) => onJudgment(excerptId, { removesSinging: value })}
        />
      </div>
      <label className="flex flex-col gap-2 border-t border-line px-4 py-4 text-sm">
        Note
        <textarea
          className="min-h-20 border border-line bg-ink px-3 py-2 text-sm text-foreground"
          value={judgment.note}
          maxLength={2000}
          data-testid={`note-${excerptId}`}
          onChange={(event) => onJudgment(excerptId, { note: event.target.value })}
        />
      </label>
    </section>
  );
}

function Judgment({
  excerptId,
  question,
  legend,
  value,
  onChange,
}: {
  excerptId: ExcerptId;
  question: string;
  legend: string;
  value: ChoiceId | null;
  onChange: (value: ChoiceId) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="text-sm">{legend}</legend>
      <div className="flex flex-col gap-2">
        {CHOICES.map((choice) => (
          <label key={choice.id} className="flex items-center gap-2 text-sm text-muted">
            <input
              type="radio"
              name={`${excerptId}-${question}`}
              value={choice.id}
              checked={value === choice.id}
              data-testid={`${question}-${excerptId}-${choice.id}`}
              onChange={() => onChange(choice.id)}
            />
            <span className={value === choice.id ? "text-foreground" : undefined}>{choice.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export default function ListenComparison() {
  const [notes, setNotes] = useState<ListeningNotes>(emptyNotes);
  const [status, setStatus] = useState("Notes stay on this computer, next to the recordings.");
  const activeRef = useRef<ExcerptId | null>(null);
  const stopsRef = useRef<Partial<Record<ExcerptId, () => void>>>({});
  const notesRef = useRef(notes);
  const dirtyRef = useRef(false);
  const saveQueue = useRef(Promise.resolve());

  useEffect(() => {
    let cancelled = false;
    fetch("/api/listen/notes")
      .then(async (response) => {
        if (!response.ok) throw new Error("notes");
        return (await response.json()) as ListeningNotes;
      })
      .then((loaded) => {
        if (cancelled || dirtyRef.current) return;
        notesRef.current = loaded;
        setNotes(loaded);
      })
      .catch(() => {
        if (!cancelled) setStatus("Saved notes could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function claimPlayback(excerpt: ExcerptId) {
    const previous = activeRef.current;
    activeRef.current = excerpt;
    if (previous && previous !== excerpt) stopsRef.current[previous]?.();
  }

  function releasePlayback(excerpt: ExcerptId) {
    if (activeRef.current === excerpt) activeRef.current = null;
  }

  function registerStop(excerpt: ExcerptId, stop: () => void) {
    stopsRef.current[excerpt] = stop;
  }

  function onJudgment(excerpt: ExcerptId, patch: Partial<ExcerptJudgment>) {
    dirtyRef.current = true;
    const next = { ...notesRef.current[excerpt], ...patch };
    const all = { ...notesRef.current, [excerpt]: next };
    notesRef.current = all;
    setNotes(all);
    setStatus("Saving…");
    saveQueue.current = saveQueue.current
      .then(async () => {
        const response = await fetch("/api/listen/notes", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ excerpt, ...next }),
        });
        if (!response.ok) throw new Error("save");
        setStatus("Saved.");
      })
      .catch(() => {
        setStatus("The note could not be saved.");
      });
  }

  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-muted" data-testid="listen-status" aria-live="polite">
        {status}
      </p>
      {EXCERPTS.map((excerpt) => (
        <ExcerptPlayer
          key={excerpt.id}
          excerptId={excerpt.id}
          title={excerpt.title}
          detail={excerpt.detail}
          judgment={notes[excerpt.id]}
          onJudgment={onJudgment}
          claimPlayback={claimPlayback}
          releasePlayback={releasePlayback}
          registerStop={registerStop}
        />
      ))}
    </div>
  );
}
