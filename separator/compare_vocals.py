"""Compare Voc_FT and Kim Inst on three excerpts of one azmari recording.

Excerpts come from 527689__brunoauzet__ethipian-azmari-songs-around-fire.wav.
Windows were chosen with the existing classifier plus onset strength and voiced fraction.
This does not change the website and does not call the instrumental stem krar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from separator.vocals import split_stems, write_stereo

SOURCE = Path("/Users/lindaberhe/Desktop/527689__brunoauzet__ethipian-azmari-songs-around-fire.wav")
OUTPUT = Path(__file__).resolve().parent / "runs" / "vocal-compare"
MODELS = (
    {
        "name": "vocft",
        "path": Path(__file__).resolve().parent / "models" / "UVR-MDX-NET-Voc_FT.onnx",
        "compensate": 1.021,
        "primary": "Vocals",
    },
    {
        "name": "kim_inst",
        "path": Path(__file__).resolve().parent / "models" / "Kim_Inst.onnx",
        "compensate": 1.02,
        "primary": "Instrumental",
    },
)
EXCERPTS = (
    {"name": "singing", "start": 0.0, "seconds": 8.0},
    {"name": "krar", "start": 176.0, "seconds": 8.0},
    {"name": "both", "start": 596.0, "seconds": 8.0},
)


def read_pcm(path: Path) -> tuple[np.ndarray, int]:
    """Read 16-bit PCM, including WAV extensible files.

    The stdlib wave module seeks by the fmt sample rate and misses the real
    data offset in this recording, which clips the excerpt to full scale.
    """
    blob = path.read_bytes()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF WAVE file.")
    position = 12
    fmt = None
    data = None
    while position + 8 <= len(blob):
        chunk_id = blob[position : position + 4]
        size = int.from_bytes(blob[position + 4 : position + 8], "little")
        payload = blob[position + 8 : position + 8 + size]
        if chunk_id == b"fmt ":
            fmt = payload
        elif chunk_id == b"data":
            data = payload
            break
        position += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path} has no fmt or data chunk.")
    audio_format = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if audio_format == 65534:
        audio_format = int.from_bytes(fmt[24:26], "little")
    if audio_format != 1 or bits != 16:
        raise ValueError(f"{path} is not 16-bit PCM.")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32).reshape(-1, channels)
    return samples / 32768, rate


def load_excerpt(start: float, seconds: float) -> np.ndarray:
    samples, rate = read_pcm(SOURCE)
    begin = int(start * rate)
    end = begin + int(seconds * rate)
    excerpt = samples[begin:end]
    stereo = excerpt[:, :2] if excerpt.shape[1] >= 2 else np.repeat(excerpt, 2, axis=1)
    waveform = torch.from_numpy(np.ascontiguousarray(stereo.T))
    if rate != 44_100:
        waveform = torchaudio.functional.resample(waveform, rate, 44_100)
    return waveform.numpy()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(SOURCE), "excerpts": []}
    for excerpt in EXCERPTS:
        mix = load_excerpt(excerpt["start"], excerpt["seconds"])
        folder = OUTPUT / excerpt["name"]
        write_stereo(folder / "original.wav", mix)
        record = {"name": excerpt["name"], "start": excerpt["start"], "seconds": excerpt["seconds"], "models": []}
        print(f"excerpt {excerpt['name']} {excerpt['start']}s", flush=True)
        for model in MODELS:
            print(f"  {model['name']}", flush=True)
            vocals, instrumental = split_stems(mix, model["path"], model["compensate"], model["primary"])
            write_stereo(folder / f"{model['name']}_vocals.wav", vocals)
            write_stereo(folder / f"{model['name']}_instrumental.wav", instrumental)
            record["models"].append(model["name"])
        manifest["excerpts"].append(record)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
