"""Check that both stems share one gain and still reconstruct the mix."""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from separator.vocals import export_stems, load_mix, read_stereo, shared_gain


def pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").reshape(-1, 2).astype(np.int32)


def pcm_from_float(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1, 1)
    return (clipped.T * 32767).astype(np.int16).astype(np.int32)


def reconstruction_error(folder: Path, reference: np.ndarray) -> int:
    got = pcm(folder / "vocals.wav") + pcm(folder / "instrumental.wav")
    diff = np.abs(got - pcm_from_float(reference))
    return int(diff.max()) if diff.size else 0


def main() -> None:
    mix = np.array([[0.2, 0.9, -0.4], [0.1, -0.8, 0.5]], dtype=np.float32)
    vocals = np.array([[0.5, 1.6, -0.2], [0.0, -1.3, 0.4]], dtype=np.float32)
    instrumental = mix - vocals
    quiet_vocals = (mix * np.float32(0.5)).copy()
    quiet_instrumental = (mix * np.float32(0.25)).copy()
    returned, _, gain_one = shared_gain(quiet_vocals, quiet_instrumental)
    if gain_one != 1.0 or returned is not quiet_vocals:
        raise SystemExit(f"quiet audio was scaled: gain={gain_one}")

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        shared = folder / "shared"
        shared.mkdir()
        gain = export_stems(shared, vocals, instrumental)
        shared_error = reconstruction_error(shared, mix * np.float32(gain))
        from separator.vocals import write_stereo

        clipped = folder / "clipped"
        clipped.mkdir()
        write_stereo(clipped / "vocals.wav", vocals)
        write_stereo(clipped / "instrumental.wav", instrumental)
        clipped_error = reconstruction_error(clipped, mix)
        print(f"synthetic_gain={gain:.6f} shared_max_counts={shared_error} independent_max_counts={clipped_error}")
        if gain >= 1 or shared_error > 2 or clipped_error <= shared_error:
            raise SystemExit("shared gain did not repair an over-scale stem")

        heard = read_stereo(shared / "vocals.wav")
        if float(np.max(np.abs(heard))) > 1:
            raise SystemExit("exported vocals exceed full scale")

    clip = Path("/tmp/kirar-bench/clip10.wav")
    baseline = Path("/tmp/kirar-bench/10s/vocals.npy")
    if clip.is_file() and baseline.is_file():
        mix = load_mix(clip)
        estimate = np.load(baseline)
        frames = min(mix.shape[1], estimate.shape[1])
        mix = mix[:, :frames]
        estimate = estimate[:, :frames]
        instrumental = mix - estimate
        peak = float(max(np.max(np.abs(estimate)), np.max(np.abs(instrumental))))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            gain = export_stems(folder, estimate, instrumental)
            error = reconstruction_error(folder, mix * np.float32(gain))
        print(f"clip10_peak={peak:.6f} gain={gain:.6f} shared_max_counts={error}")
        if error > 2:
            raise SystemExit("10s stems do not reconstruct the mix")
    print("shared_gain_ok")


if __name__ == "__main__":
    sys.exit(main() or 0)
