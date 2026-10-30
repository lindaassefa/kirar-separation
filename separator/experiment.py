"""Train and evaluate a two-source masker on synthetic mixtures.

The krar target is a local recording the existing classifier consistently
labels as krar. It is not a confirmed isolated stem. The other target is a
stretch of a different local recording that the same classifier does not label
as krar. Mixtures are created here by adding the two signals at random gains.

This does not connect the Next.js endpoint and it is not a production separator.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from separator.audio import load_mono, slice_audio, write_wav
from separator.metrics import correlation, si_sdr
from separator.model import MaskSeparator

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
KRAR_PATH = Path("/Users/lindaberhe/Downloads/krar_candidate_20s.wav")
OTHER_PATH = Path("/Users/lindaberhe/Desktop/songs/forever.band/Media/Audio Files/audio [music] (1).wav")
RUN_DIR = Path(__file__).resolve().parent / "runs" / "smoke"


def pick_device() -> torch.device:
    # The M2 GPU is visible to Torch 2.4, but torch.stft on MPS requires macOS 14.
    # This machine is macOS 12.5, so the spectrogram and the training step run on CPU.
    if torch.backends.mps.is_available():
        print("mps=available stft_on_mps=unsupported_before_macos_14")
    return torch.device("cpu")


def random_chunk(source: torch.Tensor) -> torch.Tensor:
    start = random.randrange(0, source.numel() - CHUNK_SAMPLES + 1)
    return source[start : start + CHUNK_SAMPLES]


def make_mixture(krar: torch.Tensor, other: torch.Tensor, krar_gain: float, other_gain: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    krar_part = krar * krar_gain
    other_part = other * other_gain
    mixture = krar_part + other_part
    peak = torch.max(mixture.abs().max(), torch.tensor(1.0, device=mixture.device))
    return mixture / peak, krar_part / peak, other_part / peak


def batch_from(krar_source: torch.Tensor, other_source: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mixtures = []
    krar_parts = []
    other_parts = []
    for _ in range(batch_size):
        mixture, krar_part, other_part = make_mixture(
            random_chunk(krar_source),
            random_chunk(other_source),
            random.uniform(0.4, 1.0),
            random.uniform(0.4, 1.0),
        )
        mixtures.append(mixture)
        krar_parts.append(krar_part)
        other_parts.append(other_part)
    return (
        torch.stack(mixtures).to(device),
        torch.stack(krar_parts).to(device),
        torch.stack(other_parts).to(device),
    )


def fixed_examples(krar_source: torch.Tensor, other_source: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gains = ((0.7, 0.7), (1.0, 0.45), (0.45, 1.0))
    starts = range(0, krar_source.numel() - CHUNK_SAMPLES + 1, CHUNK_SAMPLES)
    mixtures = []
    krar_parts = []
    other_parts = []
    for start in starts:
        for krar_gain, other_gain in gains:
            mixture, krar_part, other_part = make_mixture(
                krar_source[start : start + CHUNK_SAMPLES],
                other_source[start : start + CHUNK_SAMPLES],
                krar_gain,
                other_gain,
            )
            mixtures.append(mixture)
            krar_parts.append(krar_part)
            other_parts.append(other_part)
    return (
        torch.stack(mixtures).to(device),
        torch.stack(krar_parts).to(device),
        torch.stack(other_parts).to(device),
    )


def evaluate(model: MaskSeparator, mixture: torch.Tensor, krar: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        krar_estimate, other_estimate = model(mixture)
    krar_score = float(si_sdr(krar_estimate, krar).mean())
    other_score = float(si_sdr(other_estimate, other).mean())
    mixture_krar = float(si_sdr(mixture, krar).mean())
    mixture_other = float(si_sdr(mixture, other).mean())
    stem_sum = krar_estimate + other_estimate
    return {
        "krar_si_sdr_db": krar_score,
        "other_si_sdr_db": other_score,
        "mixture_vs_krar_si_sdr_db": mixture_krar,
        "mixture_vs_other_si_sdr_db": mixture_other,
        "krar_improvement_db": krar_score - mixture_krar,
        "other_improvement_db": other_score - mixture_other,
        "krar_correlation_with_target": correlation(krar_estimate, krar),
        "krar_correlation_with_mixture": correlation(krar_estimate, mixture),
        "max_abs_diff_krar_vs_mixture": float((krar_estimate - mixture).abs().max()),
        "max_abs_reconstruction_error": float((stem_sum - mixture).abs().max()),
    }


def train() -> dict[str, object]:
    random.seed(7)
    torch.manual_seed(7)
    device = pick_device()
    print(f"torch={torch.__version__} device={device}")

    krar_audio = load_mono(KRAR_PATH, SAMPLE_RATE)
    other_audio = load_mono(OTHER_PATH, SAMPLE_RATE)
    krar_train = slice_audio(krar_audio, SAMPLE_RATE, 0, 15)
    krar_eval = slice_audio(krar_audio, SAMPLE_RATE, 15, 5)
    other_train = slice_audio(other_audio, SAMPLE_RATE, 16, 16)
    other_eval = slice_audio(other_audio, SAMPLE_RATE, 32, 4)
    # The held-out other slice is 4s and the krar slice is 5s. Pair equal lengths.
    pair_length = min(krar_eval.numel(), other_eval.numel())
    krar_eval = krar_eval[:pair_length]
    other_eval = other_eval[:pair_length]

    model = MaskSeparator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    steps = 300
    model.train()
    for step in range(1, steps + 1):
        mixture, krar, other = batch_from(krar_train, other_train, batch_size=4, device=device)
        krar_estimate, other_estimate = model(mixture)
        loss = F.l1_loss(krar_estimate, krar) + F.l1_loss(other_estimate, other)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                score = float(si_sdr(krar_estimate, krar).mean())
            print(f"step={step} l1={float(loss):.4f} train_batch_krar_si_sdr={score:.2f}")

    train_mixture, train_krar, train_other = fixed_examples(krar_train[: pair_length * 2], other_train[: pair_length * 2], device)
    train_metrics = evaluate(model, train_mixture, train_krar, train_other)
    heldout_mixture, heldout_krar, heldout_other = fixed_examples(krar_eval, other_eval, device)
    heldout_metrics = evaluate(model, heldout_mixture, heldout_krar, heldout_other)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": model.n_fft,
        "hop_length": model.hop_length,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, RUN_DIR / "separator.pt")
    with torch.no_grad():
        krar_estimate, other_estimate = model(heldout_mixture[:1])
    write_wav(RUN_DIR / "example_mixture.wav", heldout_mixture[0], SAMPLE_RATE)
    write_wav(RUN_DIR / "example_krar_target.wav", heldout_krar[0], SAMPLE_RATE)
    write_wav(RUN_DIR / "example_krar_estimate.wav", krar_estimate[0], SAMPLE_RATE)
    write_wav(RUN_DIR / "example_other_estimate.wav", other_estimate[0], SAMPLE_RATE)

    report = {
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "stft_device": "cpu",
        "steps": steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "krar_source": str(KRAR_PATH),
        "other_source": str(OTHER_PATH),
        "krar_train_seconds": [0, 15],
        "krar_eval_seconds": [15, 19],
        "other_train_seconds": [16, 32],
        "other_eval_seconds": [32, 36],
        "train": train_metrics,
        "heldout": heldout_metrics,
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
