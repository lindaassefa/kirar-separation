"""Private Modal T4 worker.

Four-stem songs use pretrained htdemucs on the GPU. The vocal/instrumental
mode still uses UVR-MDX-NET-Voc_FT. Importing this file does not start a GPU.
Deploying it is a separate approval:

    KIRAR_MODAL_BENCHMARK is not used here.
    modal deploy scripts/modal_app.py

min_containers is 0. One container at a time, a 5-minute timeout, and no
retries. The loaded model stays for three minutes after the last song, then
the container exits. There is no web endpoint.

The image is CUDA 12.4, CUDA torch 2.4.1 for htdemucs, onnxruntime-gpu 1.19.2
for the vocal fallback, and ffmpeg so the worker decodes the original upload.
That image is not deployed until a separate approval.

On 2026-10-28 a cold client wait was 128 seconds. The T4 was billed for 36
seconds and did 11.7 seconds of work. The rest was the first image pull and
float32 uploads. Warm inference was 0.225 seconds per chunk.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from separator.modal_names import APP_NAME, PROGRESS_DICT

MAX_BYTES = 120 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".aif", ".aiff", ".mp4", ".audio"}
# Three idle minutes costs about $0.037 at the measured T4 rate, then the GPU stops.
WARM_SECONDS = 180
SEPARATOR = ROOT / "separator"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("zlib1g", "ffmpeg")
    .pip_install("numpy>=1.26,<2.1")
    .run_commands(
        "pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124",
        "pip install 'demucs==4.0.1' --no-deps",
        "pip install dora-search einops 'julius>=0.2.3' lameenc openunmix pyyaml tqdm omegaconf retrying submitit treetable antlr4-python3-runtime",
        "pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124",
        "python -c \"from demucs.pretrained import get_model; get_model('htdemucs')\"",
    )
    .pip_install("onnxruntime-gpu==1.19.2")
    .add_local_dir(
        str(SEPARATOR),
        remote_path="/root/separator",
        ignore=[
            "**/.venv/**",
            "**/.modal-venv/**",
            "**/runs/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "models/*.onnx",
        ],
    )
    .add_local_file(
        str(SEPARATOR / "models" / "UVR-MDX-NET-Voc_FT.onnx"),
        "/root/separator/models/UVR-MDX-NET-Voc_FT.onnx",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    cpu=2.0,
    memory=8192,
    min_containers=0,
    max_containers=1,
    timeout=300,
    scaledown_window=WARM_SECONDS,
    retries=0,
)
class VocalSeparator:
    @modal.enter()
    def load(self) -> None:
        from separator.stems import load_stem_model

        self.stem_model = load_stem_model(device="cuda")
        self.runner = None
        self.providers: list[str] | None = None

    def _vocal_runner(self):
        if self.runner is None:
            from separator.vocals import cuda_session

            self.runner = cuda_session()
            self.providers = list(self.runner.get_providers())
        return self.runner

    @modal.method()
    def separate(self, job_id: str, audio: bytes, suffix: str) -> dict:
        from separator.vocals import SeparationCancelled, separate_file

        if not isinstance(audio, (bytes, bytearray)) or len(audio) == 0:
            raise RuntimeError("Could not read that audio file.")
        if len(audio) > MAX_BYTES:
            raise RuntimeError("Choose a file under 120 MB.")
        safe_suffix = suffix.lower() if isinstance(suffix, str) and suffix.lower() in ALLOWED_SUFFIXES else ".audio"
        progress = modal.Dict.from_name(PROGRESS_DICT, create_if_missing=True)

        def on_chunk(done: int, total: int) -> None:
            if progress.get(f"cancel:{job_id}"):
                raise SeparationCancelled()
            progress[job_id] = {"chunksDone": int(done), "chunksTotal": int(total)}

        with tempfile.TemporaryDirectory(prefix="kirar-gpu-") as temporary:
            folder = Path(temporary)
            source = folder / f"input{safe_suffix}"
            source.write_bytes(audio)
            output = folder / "out"
            output.mkdir()
            timings = separate_file(
                source,
                output,
                runner=self._vocal_runner(),
                progress_hook=on_chunk,
                job_id=job_id,
            )
            vocals = (output / "vocals.wav").read_bytes()
            instrumental = (output / "instrumental.wav").read_bytes()
        if bytes(vocals[:4]) != b"RIFF" or bytes(instrumental[:4]) != b"RIFF":
            raise RuntimeError("The separator did not write audio.")
        timings["providers"] = self.providers
        timings["backend"] = "cuda"
        return {"vocals": vocals, "instrumental": instrumental, "timings": timings}

    @modal.method()
    def separate_stems(self, job_id: str, audio: bytes, suffix: str) -> dict:
        from separator.stems import STEM_NAMES, separate_stems
        from separator.vocals import SeparationCancelled

        if not isinstance(audio, (bytes, bytearray)) or len(audio) == 0:
            raise RuntimeError("Could not read that audio file.")
        if len(audio) > MAX_BYTES:
            raise RuntimeError("Choose a file under 120 MB.")
        safe_suffix = suffix.lower() if isinstance(suffix, str) and suffix.lower() in ALLOWED_SUFFIXES else ".audio"
        progress = modal.Dict.from_name(PROGRESS_DICT, create_if_missing=True)

        def on_chunk(done: int, total: int) -> None:
            if progress.get(f"cancel:{job_id}"):
                raise SeparationCancelled()
            progress[job_id] = {"chunksDone": int(done), "chunksTotal": int(total)}

        with tempfile.TemporaryDirectory(prefix="kirar-gpu-") as temporary:
            folder = Path(temporary)
            source = folder / f"input{safe_suffix}"
            source.write_bytes(audio)
            output = folder / "out"
            output.mkdir()
            timings = separate_stems(
                source,
                output,
                progress_hook=on_chunk,
                job_id=job_id,
                device="cuda",
                model=self.stem_model,
            )
            wavs = {name: (output / f"{name}.wav").read_bytes() for name in STEM_NAMES}
        if any(bytes(wav[:4]) != b"RIFF" for wav in wavs.values()):
            raise RuntimeError("The separator did not write audio.")
        timings["device"] = "cuda"
        timings["backend"] = "cuda"
        return {**wavs, "timings": timings}
