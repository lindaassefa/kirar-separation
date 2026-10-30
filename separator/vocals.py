"""Vocal separation with UVR-MDX-NET-Voc_FT.

The checkpoint is an MDX-Net ONNX model. It is not Demucs. Parameters come
from the UVR model table for tail hash 77d07b2667ddf05b9e3175941b4454a0:
n_fft 7680, 3072 bins, 256 frames, compensate 1.021, primary stem Vocals.

The instrumental is the mixture minus the vocal estimate. This module does
not estimate krar.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

SAMPLE_RATE = 44_100
N_FFT = 7680
HOP = 1024
DIM_F = 3072
DIM_T = 256
COMPENSATE = 1.021
N_BINS = N_FFT // 2 + 1
CHUNK = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
MODEL_PATH = Path(__file__).resolve().parent / "models" / "UVR-MDX-NET-Voc_FT.onnx"
# Measured on this M2: 8 intra-op threads were fastest (about 17 s/chunk).
# 4 threads were 17.9 s, 2 were 23.2 s, and 1 was 39.7 s. CoreML compiled
# only 151 of 178 nodes, ran slower, and changed the waveform, so it is unused.
INTRA_OP_THREADS = 8
MAX_SECONDS = 600

_sessions: dict[str, ort.InferenceSession] = {}
_window_tensor: torch.Tensor | None = None
_cancel_ids: set[str] = set()


class SeparationCancelled(Exception):
    pass


def session_for(path: Path) -> ort.InferenceSession:
    key = str(path)
    if key not in _sessions:
        if not Path(key).is_file():
            raise FileNotFoundError(f"Vocal model is missing: {key}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = INTRA_OP_THREADS
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        _sessions[key] = ort.InferenceSession(key, sess_options=options, providers=["CPUExecutionProvider"])
    return _sessions[key]


def session() -> ort.InferenceSession:
    return session_for(MODEL_PATH)


def cuda_session(path: Path | None = None) -> ort.InferenceSession:
    """CUDA-only session. A missing CUDA provider raises instead of timing the CPU."""
    target = path or MODEL_PATH
    key = f"cuda:{target}"
    if key not in _sessions:
        if not target.is_file():
            raise FileNotFoundError(f"Vocal model is missing: {target}")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider did not load.")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        runner = ort.InferenceSession(str(target), sess_options=options, providers=["CUDAExecutionProvider"])
        if not runner.get_providers() or runner.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDAExecutionProvider did not load.")
        _sessions[key] = runner
    return _sessions[key]


def decode_to_stereo_wav(source: Path, destination: Path) -> None:
    if shutil.which("afconvert"):
        completed = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@44100", "-c", "2", str(source), str(destination)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("Could not read that audio file.")
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Could not read that audio file.")
    completed = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(destination)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("Could not read that audio file.")


def read_stereo(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        if handle.getnchannels() != 2 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} is not 44.1 kHz 16-bit stereo.")
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32).reshape(-1, 2)
    return np.ascontiguousarray(samples.T / 32768)


def load_mix(source: Path) -> np.ndarray:
    try:
        return read_stereo(source)
    except (wave.Error, EOFError, ValueError, OSError):
        with tempfile.TemporaryDirectory(prefix="kirar-decode-") as temporary:
            decoded = Path(temporary) / "mix.wav"
            decode_to_stereo_wav(source, decoded)
            return read_stereo(decoded)


def shared_gain(vocals: np.ndarray, instrumental: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One gain for both stems so neither is clipped on its own.

    Independent clipping makes vocals.wav + instrumental.wav differ from the mix
    wherever either stem exceeds full scale. Scaling both by the same gain keeps
    their sum equal to that gain times the mix.
    """
    if vocals.size == 0 or instrumental.size == 0:
        return vocals, instrumental, 1.0
    peak = float(max(np.max(np.abs(vocals)), np.max(np.abs(instrumental))))
    if not np.isfinite(peak) or peak <= 1.0:
        return vocals, instrumental, 1.0
    gain = np.float32(1.0 / peak)
    return vocals * gain, instrumental * gain, float(gain)


def export_stems(output_dir: Path, vocals: np.ndarray, instrumental: np.ndarray) -> float:
    scaled_vocals, scaled_instrumental, gain = shared_gain(vocals, instrumental)
    write_stereo(output_dir / "vocals.wav", scaled_vocals)
    write_stereo(output_dir / "instrumental.wav", scaled_instrumental)
    return gain


def write_stereo(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1, 1)
    interleaved = (clipped.T * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(interleaved.tobytes())


def _window() -> torch.Tensor:
    global _window_tensor
    if _window_tensor is None:
        _window_tensor = torch.hann_window(N_FFT, periodic=True)
    return _window_tensor


def spectrogram(waveform: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    spectrum = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP, window=window, center=True, return_complex=True)
    if spectrum.shape[-1] != DIM_T:
        raise RuntimeError(f"Expected {DIM_T} frames, got {spectrum.shape[-1]}.")
    packed = torch.view_as_real(spectrum).permute(0, 3, 1, 2).reshape(1, 4, N_BINS, DIM_T)
    return packed[:, :, :DIM_F, :]


def waveform_from(packed: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    restored = torch.nn.functional.pad(packed, (0, 0, 0, N_BINS - DIM_F))
    channels = restored.reshape(2, 2, N_BINS, DIM_T).permute(0, 2, 3, 1).contiguous()
    spectrum = torch.view_as_complex(channels)
    return torch.istft(spectrum, n_fft=N_FFT, hop_length=HOP, window=window, center=True, length=CHUNK)


def infer_chunk(chunk: np.ndarray, window: torch.Tensor, runner: ort.InferenceSession) -> np.ndarray:
    features = spectrogram(torch.from_numpy(np.ascontiguousarray(chunk)), window)
    estimated = runner.run(None, {"input": features.numpy()})[0]
    wave = waveform_from(torch.from_numpy(estimated), window)
    return wave.numpy()


def chunk_count(samples: int) -> tuple[int, int, int]:
    step = CHUNK - 2 * TRIM
    pad = (step - (samples % step)) % step
    total = (samples + pad + step - 1) // step
    return step, pad, total


def request_cancel(job_id: str) -> None:
    _cancel_ids.add(job_id)


def is_cancelled(job_id: str | None) -> bool:
    return bool(job_id) and job_id in _cancel_ids


def model_waveform(
    mix: np.ndarray,
    runner: ort.InferenceSession,
    on_chunk: callable | None = None,
    job_id: str | None = None,
) -> np.ndarray:
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError("Expected stereo audio shaped [2, samples].")
    samples = mix.shape[1]
    step, pad, total = chunk_count(samples)
    padded = np.pad(mix, ((0, 0), (TRIM, TRIM + pad)))
    estimate = np.zeros((2, samples + pad), dtype=np.float32)
    window = _window()
    if on_chunk is not None:
        on_chunk(0, total)
    for index, start in enumerate(range(0, samples + pad, step)):
        if job_id is not None and job_id in _cancel_ids:
            _cancel_ids.discard(job_id)
            raise SeparationCancelled()
        piece = padded[:, start : start + CHUNK]
        if piece.shape[1] < CHUNK:
            piece = np.pad(piece, ((0, 0), (0, CHUNK - piece.shape[1])))
        estimated = infer_chunk(piece, window, runner)
        estimate[:, start : start + step] = estimated[:, TRIM : TRIM + step]
        if on_chunk is not None:
            on_chunk(index + 1, total)
    return estimate[:, :samples]


def split_stems(
    mix: np.ndarray,
    model_path: Path,
    compensate: float,
    primary: str,
    on_chunk: callable | None = None,
    job_id: str | None = None,
    runner: ort.InferenceSession | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    active = runner if runner is not None else session_for(model_path)
    estimate = model_waveform(mix, active, on_chunk=on_chunk, job_id=job_id) * compensate
    if primary == "Vocals":
        vocals = estimate
        instrumental = mix - vocals
    elif primary == "Instrumental":
        instrumental = estimate
        vocals = mix - instrumental
    else:
        raise ValueError(f"Unknown primary stem: {primary}")
    return vocals, instrumental


def separate_vocals(mix: np.ndarray, on_chunk: callable | None = None, job_id: str | None = None) -> np.ndarray:
    vocals, _instrumental = split_stems(mix, MODEL_PATH, COMPENSATE, "Vocals", on_chunk=on_chunk, job_id=job_id)
    return vocals


def write_progress(path: Path, done: int, total: int) -> None:
    payload = json.dumps({"chunksDone": done, "chunksTotal": total})
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def separate_file(
    source: Path,
    output_dir: Path,
    progress_path: Path | None = None,
    job_id: str | None = None,
    runner: ort.InferenceSession | None = None,
    progress_hook: callable | None = None,
) -> dict[str, object]:
    def on_chunk(done: int, total: int) -> None:
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
    infer_started = time.perf_counter()
    vocals, instrumental = split_stems(
        mix,
        MODEL_PATH,
        COMPENSATE,
        "Vocals",
        on_chunk=on_chunk,
        job_id=job_id,
        runner=runner,
    )
    infer_s = time.perf_counter() - infer_started
    export_started = time.perf_counter()
    gain = export_stems(output_dir, vocals, instrumental)
    export_s = time.perf_counter() - export_started
    active = runner if runner is not None else session()
    timings: dict[str, object] = {
        "decodeS": decode_s,
        "inferS": infer_s,
        "exportS": export_s,
        "chunks": chunk_count(int(mix.shape[1]))[2],
        "gain": gain,
        "providers": list(active.get_providers()),
        "backend": "cuda" if runner is not None else "cpu",
    }
    temporary = (output_dir / "timings.json").with_suffix(".tmp")
    temporary.write_text(json.dumps(timings))
    temporary.replace(output_dir / "timings.json")
    return timings


def serve() -> None:
    incoming: queue.Queue[dict[str, object]] = queue.Queue()

    def read_stdin() -> None:
        for line in sys_stdin():
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            incoming.put(message)

    threading.Thread(target=read_stdin, name="kirar-vocal-stdin", daemon=True).start()
    session()
    print(json.dumps({"event": "ready"}), flush=True)
    while True:
        message = incoming.get()
        if message.get("cmd") == "cancel":
            request_cancel(str(message.get("id", "")))
            continue
        job_id = str(message.get("id", ""))
        try:
            if job_id in _cancel_ids:
                _cancel_ids.discard(job_id)
                raise SeparationCancelled()
            separate_file(
                Path(str(message["input"])),
                Path(str(message["output"])),
                progress_path=Path(str(message["progress"])),
                job_id=job_id,
            )
        except SeparationCancelled:
            print(json.dumps({"event": "cancelled", "id": job_id}), flush=True)
            continue
        except Exception as cause:
            print(json.dumps({"event": "error", "id": job_id, "error": str(cause)}), flush=True)
            continue
        print(json.dumps({"event": "done", "id": job_id}), flush=True)


def sys_stdin():
    import sys

    return sys.stdin


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate vocals with UVR-MDX-NET-Voc_FT.")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.serve:
        serve()
        return
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    separate_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()
