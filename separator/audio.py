"""Load, resample, and write mono WAV audio for the separator."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_mono(path: str | Path, sample_rate: int) -> torch.Tensor:
    with wave.open(str(path)) as handle:
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path} is not 16-bit PCM WAV.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32).reshape(-1, channels)
    waveform = torch.from_numpy(samples.mean(axis=1).copy()) / 32768
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.contiguous()


def slice_audio(waveform: torch.Tensor, sample_rate: int, start_second: float, duration_second: float) -> torch.Tensor:
    start = int(round(start_second * sample_rate))
    length = int(round(duration_second * sample_rate))
    end = start + length
    if start < 0 or end > waveform.numel():
        raise ValueError(f"Slice {start_second}+{duration_second}s is outside an audio buffer of {waveform.numel() / sample_rate:.2f}s.")
    return waveform[start:end].contiguous()


def write_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform.detach().cpu().reshape(-1).clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
