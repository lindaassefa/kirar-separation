"""Local separation worker.

`mode=stems` runs pretrained htdemucs and writes vocals, drums, bass, and other.
`mode=vocals` keeps the UVR vocal/instrumental pipeline. Neither mode writes krar.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

from separator.vocals import SeparationCancelled, request_cancel, separate_file


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


def _read_stdin(incoming: queue.Queue[dict[str, object]]) -> None:
    import sys

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


def serve() -> None:
    incoming: queue.Queue[dict[str, object]] = queue.Queue()
    threading.Thread(target=_read_stdin, args=(incoming,), name="kirar-worker-stdin", daemon=True).start()
    _emit({"event": "ready"})
    while True:
        message = incoming.get()
        if message.get("cmd") == "cancel":
            request_cancel(str(message.get("id", "")))
            continue
        job_id = str(message.get("id", ""))
        mode = str(message.get("mode") or "stems")
        try:
            source = Path(str(message["input"]))
            output = Path(str(message["output"]))
            progress = Path(str(message["progress"]))
            if mode == "vocals":
                separate_file(source, output, progress_path=progress, job_id=job_id)
            elif mode == "stems":
                from separator.stems import separate_stems

                separate_stems(source, output, progress_path=progress, job_id=job_id, device="cpu")
            else:
                raise RuntimeError("Vocal separation failed.")
        except SeparationCancelled:
            _emit({"event": "cancelled", "id": job_id})
            continue
        except Exception as cause:
            _emit({"event": "error", "id": job_id, "error": str(cause)})
            continue
        _emit({"event": "done", "id": job_id})


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    raise SystemExit("Run with --serve from the Next.js server.")


if __name__ == "__main__":
    main()
