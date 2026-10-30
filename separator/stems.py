"""Four-stem separation with pretrained Hybrid Transformer Demucs.

The checkpoint is the official htdemucs model. It writes vocals, drums, bass,
and other. It does not write a krar stem, and the other stem is not krar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from separator.vocals import (
    MAX_SECONDS,
    SAMPLE_RATE,
    SeparationCancelled,
    is_cancelled,
    load_mix,
    write_progress,
    write_stereo,
)

STEM_NAMES = ("vocals", "drums", "bass", "other")
MODEL_NAME = "htdemucs"
# One forward pass per segment. shifts=1 repeats that pass after a random offset.
OVERLAP = 0.25

_model = None


def load_stem_model(device: str = "cpu"):
    """Load htdemucs once. A later call can move the same module to another device."""
    global _model
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The stem model could not use the GPU.")
    if _model is None:
        try:
            from demucs.pretrained import get_model
        except ImportError as cause:
            raise RuntimeError("The stem model is not installed.") from cause
        _model = get_model(MODEL_NAME)
        _model.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    _model.to(device)
    return _model


def _model_segment(model):
    """The checkpoint's own segment. HTDemucs cannot run a longer one."""
    models = getattr(model, "models", None)
    chosen = models[0] if models else model
    segment = chosen.segment
    if float(segment) <= 0:
        raise RuntimeError("The separator did not write audio.")
    return segment


def _separate_segments(model, mixture: torch.Tensor, device: str, on_step) -> tuple[torch.Tensor, int]:
    """Same overlap blend as demucs.apply, with one progress step per segment."""
    from demucs.apply import TensorChunk, apply_model

    _batch, channels, length = mixture.shape
    segment = _model_segment(model)
    segment_length = int(model.samplerate * segment)
    stride = max(1, int((1 - OVERLAP) * segment_length))
    offsets = list(range(0, length, stride))
    total = len(offsets)
    weight = torch.cat(
        [
            torch.arange(1, segment_length // 2 + 1),
            torch.arange(segment_length - segment_length // 2, 0, -1),
        ]
    )
    weight = weight / weight.max()
    estimates = torch.zeros(1, len(model.sources), channels, length)
    sum_weight = torch.zeros(length)
    for index, offset in enumerate(offsets):
        on_step(index, total)
        chunk = TensorChunk(mixture, offset, segment_length)
        with torch.inference_mode():
            chunk_out = apply_model(
                model,
                chunk,
                shifts=0,
                split=False,
                overlap=OVERLAP,
                progress=False,
                device=device,
                num_workers=0,
                segment=segment,
            )
        chunk_length = int(chunk_out.shape[-1])
        chunk_cpu = chunk_out.detach().to("cpu")
        estimates[..., offset : offset + chunk_length] += weight[:chunk_length] * chunk_cpu
        sum_weight[offset : offset + chunk_length] += weight[:chunk_length]
        del chunk_out, chunk_cpu
    if float(sum_weight.min()) <= 0:
        raise RuntimeError("The separator did not write audio.")
    estimates /= sum_weight
    on_step(total, total)
    return estimates, total


def _shared_gain(stems: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    peak = 0.0
    for audio in stems.values():
        if audio.size == 0:
            continue
        peak = max(peak, float(np.max(np.abs(audio))))
    if not np.isfinite(peak) or peak <= 1.0:
        return stems, 1.0
    gain = float(np.float32(1.0 / peak))
    scale = np.float32(gain)
    return {name: audio * scale for name, audio in stems.items()}, gain


def separate_stems(
    source: Path,
    output_dir: Path,
    progress_path: Path | None = None,
    job_id: str | None = None,
    progress_hook: callable | None = None,
    device: str = "cpu",
    model=None,
) -> dict[str, object]:
    def on_step(done: int, total: int) -> None:
        if is_cancelled(job_id):
            raise SeparationCancelled()
        if progress_path is not None:
            try:
                write_progress(progress_path, done, total)
            except OSError as cause:
                raise SeparationCancelled() from cause
        if progress_hook is not None:
            progress_hook(done, total)

    started = time.perf_counter()
    mix = load_mix(source)
    decode_s = time.perf_counter() - started
    if mix.shape[1] == 0:
        raise RuntimeError("Could not read that audio file.")
    if mix.shape[1] > MAX_SECONDS * SAMPLE_RATE:
        raise RuntimeError("Choose a file under 10 minutes.")
    stem_model = model if model is not None else load_stem_model(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        stem_model.to("cuda")
    mixture = torch.from_numpy(np.ascontiguousarray(mix)).unsqueeze(0)
    infer_started = time.perf_counter()
    estimates, chunks = _separate_segments(stem_model, mixture, device, on_step)
    infer_s = time.perf_counter() - infer_started
    if is_cancelled(job_id):
        raise SeparationCancelled()
    sources = list(stem_model.sources)
    if estimates.ndim != 4 or estimates.shape[0] != 1 or estimates.shape[1] != len(sources):
        raise RuntimeError("The separator did not write audio.")
    expected = int(mix.shape[1])
    stems: dict[str, np.ndarray] = {}
    for index, name in enumerate(sources):
        audio = estimates[0, index].detach().cpu().numpy().astype(np.float32, copy=False)
        if audio.shape[0] != 2:
            raise RuntimeError("The separator did not write audio.")
        if audio.shape[1] < expected:
            raise RuntimeError("The separator did not write audio.")
        stems[str(name)] = np.ascontiguousarray(audio[:, :expected])
    missing = [name for name in STEM_NAMES if name not in stems]
    if missing:
        raise RuntimeError("The separator did not write audio.")
    export_started = time.perf_counter()
    scaled, gain = _shared_gain({name: stems[name] for name in STEM_NAMES})
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in STEM_NAMES:
        write_stereo(output_dir / f"{name}.wav", scaled[name])
    export_s = time.perf_counter() - export_started
    timings: dict[str, object] = {
        "decodeS": decode_s,
        "inferS": infer_s,
        "exportS": export_s,
        "chunks": chunks,
        "gain": gain,
        "model": MODEL_NAME,
        "device": device,
        "backend": device,
    }
    temporary = (output_dir / "timings.json").with_suffix(".tmp")
    temporary.write_text(json.dumps(timings))
    temporary.replace(output_dir / "timings.json")
    return timings
