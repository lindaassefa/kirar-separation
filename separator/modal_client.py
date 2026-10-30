"""Server-side caller for the private T4 worker.

The Next.js process starts this with MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
It never reads those values from a shell profile on its own, and it never
prints them. Run it only when the GPU backend is explicitly configured.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from separator.modal_names import APP_NAME, CLASS_NAME, PROGRESS_DICT

UPLOAD_SECONDS = 180
WAIT_SECONDS = 360
# 300s function cap plus the 180s warm tail, at the measured T4 all-in rate, rounded up.
WORST_CASE_USD = 0.15
CODE_CEILING_USD = 5.0


def _require_tokens() -> None:
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        raise SystemExit("Vocal separation is unavailable.")


def _safe_error(cause: BaseException) -> str:
    text = " ".join(str(cause).split())
    lowered = text.lower()
    if "modal_token" in lowered or "ak-" in text or "as-" in text:
        return "Vocal separation is unavailable."
    if any(phrase in lowered for phrase in ("not found", "lookup failed", "unauthorized", "unauthenticated")):
        return "Vocal separation is unavailable."
    if "spend limit" in lowered:
        return "Separation is paused because the spend limit was reached."
    if "could not read" in lowered:
        return "Could not read that audio file."
    if "under 10 minutes" in lowered:
        return "Choose a file under 10 minutes."
    if "under 120" in lowered:
        return "Choose a file under 120 MB."
    if not text or len(text) > 180:
        return "Vocal separation is unavailable."
    return text


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


def _write_progress(path: Path, done: int, total: int) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"chunksDone": done, "chunksTotal": total}))
    temporary.replace(path)


def _refuse_if_over_budget() -> None:
    """Stop before spawn when this month's bill plus one worst-case call exceeds the cap."""
    raw_limit = os.environ.get("MODAL_SPEND_LIMIT_USD", "")
    try:
        limit = float(raw_limit)
    except ValueError:
        raise RuntimeError("Vocal separation is unavailable.") from None
    if not 0 < limit <= CODE_CEILING_USD:
        raise RuntimeError("Vocal separation is unavailable.")
    modal_bin = Path(sys.executable).with_name("modal")
    try:
        raw = subprocess.check_output(
            [str(modal_bin), "billing", "summary", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        metered = float(json.loads(raw)["metered_cost"])
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RuntimeError("Vocal separation is unavailable.") from None
    if metered + WORST_CASE_USD > limit:
        raise RuntimeError("Separation is paused because the spend limit was reached.")


def _run_job(message: dict[str, object], incoming: queue.Queue[dict[str, object]]) -> None:
    import modal

    _refuse_if_over_budget()
    job_id = str(message.get("id", ""))
    source = Path(str(message["input"]))
    output_dir = Path(str(message["output"]))
    progress_path = Path(str(message["progress"]))
    audio = source.read_bytes()
    if len(audio) > 120 * 1024 * 1024:
        raise RuntimeError("Choose a file under 120 MB.")
    suffix = source.suffix.lower() or ".audio"
    mode = str(message.get("mode") or "stems")
    if mode not in {"stems", "vocals"}:
        raise RuntimeError("Vocal separation is unavailable.")
    progress = modal.Dict.from_name(PROGRESS_DICT, create_if_missing=True)
    remote = modal.Cls.from_name(APP_NAME, CLASS_NAME)()
    worker = remote.separate_stems if mode == "stems" else remote.separate
    started = time.perf_counter()

    def alarm(_signum: int, _frame: object) -> None:
        raise TimeoutError("upload")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(UPLOAD_SECONDS)
    try:
        call = worker.spawn(job_id, audio, suffix)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    accepted = time.perf_counter()
    first_progress: float | None = None
    deadline = accepted + WAIT_SECONDS
    while True:
        try:
            note = incoming.get_nowait()
        except queue.Empty:
            note = None
        if note and note.get("cmd") == "cancel" and str(note.get("id", "")) == job_id:
            try:
                progress[f"cancel:{job_id}"] = True
            except Exception:
                pass
            call.cancel(terminate_containers=False)
            raise InterruptedError("cancelled")
        if time.perf_counter() > deadline:
            call.cancel(terminate_containers=False)
            raise TimeoutError("wait")
        try:
            result = call.get(timeout=0.4)
            break
        except TimeoutError:
            snapshot = progress.get(job_id, None)
            if isinstance(snapshot, dict) and "chunksDone" in snapshot and "chunksTotal" in snapshot:
                if first_progress is None:
                    first_progress = time.perf_counter()
                _write_progress(progress_path, int(snapshot["chunksDone"]), int(snapshot["chunksTotal"]))
    if not isinstance(result, dict):
        raise RuntimeError("Vocal separation is unavailable.")
    names = ("vocals", "drums", "bass", "other") if mode == "stems" else ("vocals", "instrumental")
    wavs: dict[str, bytes] = {}
    for name in names:
        payload = result.get(name)
        if not isinstance(payload, (bytes, bytearray)) or bytes(payload[:4]) != b"RIFF":
            raise RuntimeError("The separator did not write audio.")
        wavs[name] = bytes(payload)
    providers = result.get("timings", {})
    if not isinstance(providers, dict):
        raise RuntimeError("Vocal separation is unavailable.")
    if mode == "stems":
        if providers.get("device") != "cuda":
            raise RuntimeError("Vocal separation is unavailable.")
    else:
        provider_list = providers.get("providers")
        if not isinstance(provider_list, list) or not provider_list or provider_list[0] != "CUDAExecutionProvider":
            raise RuntimeError("Vocal separation is unavailable.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in wavs.items():
        (output_dir / f"{name}.wav").write_bytes(payload)
    timings = dict(providers)
    timings["acceptS"] = accepted - started
    timings["startupS"] = (first_progress or time.perf_counter()) - accepted
    timings["backend"] = "cuda"
    temporary = output_dir / "timings.tmp"
    temporary.write_text(json.dumps(timings))
    temporary.replace(output_dir / "timings.json")
    for key in (job_id, f"cancel:{job_id}"):
        try:
            progress.pop(key)
        except Exception:
            pass


def serve() -> None:
    _require_tokens()
    incoming: queue.Queue[dict[str, object]] = queue.Queue()

    def read_stdin() -> None:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                incoming.put(message)

    threading.Thread(target=read_stdin, name="kirar-modal-stdin", daemon=True).start()
    _emit({"event": "ready"})
    while True:
        message = incoming.get()
        if message.get("cmd") == "cancel":
            continue
        job_id = str(message.get("id", ""))
        try:
            _run_job(message, incoming)
        except InterruptedError:
            _emit({"event": "cancelled", "id": job_id})
            continue
        except Exception as cause:
            message = _safe_error(cause)
            _emit({"event": "error", "id": job_id, "error": message})
            if isinstance(cause, TimeoutError):
                print(message, file=sys.stderr, flush=True)
                raise SystemExit(1) from cause
            continue
        _emit({"event": "done", "id": job_id})


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    raise SystemExit("Run with --serve from the Next.js server.")


if __name__ == "__main__":
    main()
