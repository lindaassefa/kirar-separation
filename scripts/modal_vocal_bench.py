"""GPU benchmark for UVR-MDX-NET-Voc_FT on Modal.

This file does not deploy the website and does not start a GPU by itself.
`modal run` refuses to call Modal until KIRAR_MODAL_BENCHMARK=approved.
That variable is the explicit approval to spend GPU time.

The DSP is separator/vocals.py. The GPU session is CUDA only. If CUDA does
not load, the call stops before the timed clips. The laptop keeps its CPU
worker; this script does not change it.

Measured rates used only for the cost estimate. Confirmed from
`modal billing rates` on this workspace, October 28, 2026:

- T4: $0.590 per hour
- L4: $0.800 per hour
- CPU: $0.0473 per core-hour
- Memory: $0.008 per GiB-hour

Containers request 2 CPU cores and 8 GiB. scaledown_window is 2 seconds so
the idle tail is not the default 60 seconds.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "separator" / "models" / "UVR-MDX-NET-Voc_FT.onnx"
DEFAULT_CLIP10 = Path("/tmp/kirar-bench/clip10.wav")
DEFAULT_CLIP60 = Path("/tmp/kirar-bench/clip60.wav")
DEFAULT_BASE10 = Path("/tmp/kirar-bench/10s/vocals.npy")
DEFAULT_BASE60 = Path("/tmp/kirar-bench/60s/vocals.npy")
APPROVAL = "KIRAR_MODAL_BENCHMARK"
GPU_USD_PER_SECOND = {"T4": 0.590 / 3600, "L4": 0.800 / 3600}
CPU_USD_PER_CORE_SECOND = 0.0473 / 3600
RAM_USD_PER_GIB_SECOND = 0.008 / 3600
RESERVED_CORES = 2.0
RESERVED_GIB = 8.0
SONG_SECONDS = 173.0

PLAN = """
Kirar vocal GPU benchmark
-------------------------
No GPU container is started by printing this plan.

Model: separator/models/UVR-MDX-NET-Voc_FT.onnx
Graph already runs on CPU ORT 1.19.2. Input and output are float32
[batch_size, 4, 3072, 256]. CUDA is not verified until the approved run.
The first remote call uses CUDAExecutionProvider only and returns immediately
if that provider does not load.

Clips: the same 48 kHz 10s and 60s wavs. This machine resamples them with
the existing afconvert path to 44.1 kHz before upload, which is the audio
the CPU baseline was computed on.

Baseline: /tmp/kirar-bench/10s/vocals.npy and 60s/vocals.npy
Checks: max abs diff, correlation, RMS ratio. No listening claim.

Each GPU, one container:
  1. Load the ONNX session (cold start is the client wait minus server work).
  2. Separate the 10s mix.
  3. Separate the 60s mix.
STFT and overlap stay in separator/vocals.py. Only the session provider changes.

GPUs: T4, then L4. scaledown_window=2. timeout=600. retries=0. No min_containers.
This script must not be passed to `modal deploy`.

Approval, when you want the paid run:
  python3 -m venv /tmp/kirar-modal
  /tmp/kirar-modal/bin/pip install 'modal>=1.0' numpy
  /tmp/kirar-modal/bin/modal setup
  KIRAR_MODAL_BENCHMARK=approved /tmp/kirar-modal/bin/modal run scripts/modal_vocal_bench.py
"""


def _refuse_unless_approved() -> None:
    if os.environ.get(APPROVAL) != "approved":
        raise SystemExit(
            "Refusing to start Modal. This would provision a paid GPU.\n"
            "Set KIRAR_MODAL_BENCHMARK=approved only after explicit approval.\n"
            "Run: python3 scripts/modal_vocal_bench.py"
        )


def _image():
    import modal

    # CUDA 12 runtime plus cuDNN. onnxruntime-gpu 1.19.2 on PyPI expects CUDA 12 and cuDNN 9.
    # Torch is the CPU wheel so STFT matches the local separator without a second CUDA stack.
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("zlib1g")
        .pip_install("numpy>=1.26,<2.1")
        .run_commands("pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu")
        .pip_install("onnxruntime-gpu==1.19.2")
        .add_local_dir(
            str(REPO / "separator"),
            remote_path="/root/separator",
            ignore=[
                "**/.venv/**",
                "**/runs/**",
                "**/__pycache__/**",
                "**/*.pyc",
                "models/*.onnx",
            ],
        )
        .add_local_file(str(MODEL), "/root/separator/models/UVR-MDX-NET-Voc_FT.onnx")
    )


def _separate(mix_bytes: bytes | None) -> dict:
    """Load CUDA ORT, or separate one mix. mix_bytes is a float32 [2, samples] npy."""
    import io
    import resource
    import subprocess
    import threading

    import numpy as np
    import onnxruntime as ort

    started = time.perf_counter()
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    peak_gpu = {"mb": 0}

    def sample_gpu() -> None:
        while not stop.is_set():
            try:
                used = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                )
                peak_gpu["mb"] = max(peak_gpu["mb"], int(used.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.CalledProcessError):
                return
            stop.wait(0.2)

    stop = threading.Event()
    threading.Thread(target=sample_gpu, daemon=True).start()
    # Import torch, via the shared separator, before the CUDA session starts.
    from separator.vocals import COMPENSATE, chunk_count, model_waveform

    global _SESSION
    load_s = 0.0
    if _SESSION is None:
        load_started = time.perf_counter()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        # CPU is intentionally absent so a failed CUDA load cannot silently time the CPU.
        _SESSION = ort.InferenceSession(
            "/root/separator/models/UVR-MDX-NET-Voc_FT.onnx",
            sess_options=options,
            providers=["CUDAExecutionProvider"],
        )
        load_s = time.perf_counter() - load_started
    providers = _SESSION.get_providers()
    if "CUDAExecutionProvider" not in providers:
        stop.set()
        raise RuntimeError(f"CUDA execution provider did not load: {providers}")

    vocals = None
    infer_s = 0.0
    chunks = 0
    if mix_bytes is not None:
        mix = np.load(io.BytesIO(mix_bytes))
        chunks = chunk_count(int(mix.shape[1]))[2]
        infer_started = time.perf_counter()
        vocals = model_waveform(mix, _SESSION) * COMPENSATE
        infer_s = time.perf_counter() - infer_started

    stop.set()
    # Linux reports KiB. The benchmark records that unit in the key name.
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "gpu": gpu_name,
        "providers": providers,
        "model_load_s": load_s,
        "infer_s": infer_s,
        "chunks": chunks,
        "server_s": time.perf_counter() - started,
        "peak_gpu_mb": peak_gpu["mb"],
        "peak_rss_kib": peak_rss_kib,
    }
    if vocals is None:
        return report
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(vocals.astype(np.float32)))
    report["vocals_npy"] = buffer.getvalue()
    return report


_SESSION = None

try:
    import modal
except ImportError:  # The plan can be printed before the Modal client exists.
    modal = None

# Building the image is a paid Modal CPU step. Skip it unless this process is
# already inside a container or the benchmark was explicitly approved.
_in_container = bool(os.environ.get("MODAL_TASK_ID"))
_approved = os.environ.get(APPROVAL) == "approved"
if modal is not None and (_approved or _in_container):
    app = modal.App("kirar-vocal-bench")
    _bench_image = _image()

    def _function(gpu: str):
        return app.function(
            image=_bench_image,
            gpu=gpu,
            cpu=RESERVED_CORES,
            memory=int(RESERVED_GIB * 1024),
            timeout=600,
            scaledown_window=2,
            max_containers=1,
            retries=0,
            # The container imports this file. The approval env makes the same
            # functions exist there. MODAL_TASK_ID is also set by Modal itself.
            env={APPROVAL: "approved"},
        )

    @_function("T4")
    def bench_t4(mix10: bytes, mix60: bytes) -> dict:
        return _run_both(mix10, mix60)

    @_function("L4")
    def bench_l4(mix10: bytes, mix60: bytes) -> dict:
        return _run_both(mix10, mix60)

    def _run_both(mix10: bytes, mix60: bytes) -> dict:
        """One container: load CUDA, then both clips. A CUDA failure returns before either clip."""
        loaded = _separate(None)
        clips = {}
        for label, payload in (("10s", mix10), ("60s", mix60)):
            report = _separate(payload)
            clips[label] = report
        return {"load": loaded, "clips": clips}

    def _decode_mixes() -> tuple[bytes, bytes]:
        import subprocess

        python = REPO / "separator" / ".venv" / "bin" / "python"
        code = r"""
import io
from pathlib import Path
import numpy as np
from separator.vocals import load_mix
for source, dest in (("CLIP10", "OUT10"), ("CLIP60", "OUT60")):
    mix = np.ascontiguousarray(load_mix(Path(os.environ[source])).astype(np.float32))
    buf = io.BytesIO()
    np.save(buf, mix)
    Path(os.environ[dest]).write_bytes(buf.getvalue())
"""
        out10 = Path("/tmp/kirar-bench/clip10_441.npy")
        out60 = Path("/tmp/kirar-bench/clip60_441.npy")
        env = os.environ.copy()
        env.update(
            {
                "CLIP10": str(DEFAULT_CLIP10),
                "CLIP60": str(DEFAULT_CLIP60),
                "OUT10": str(out10),
                "OUT60": str(out60),
                "PYTHONPATH": str(REPO),
            }
        )
        subprocess.check_call([str(python), "-c", "import os\n" + code], cwd=REPO, env=env)
        return out10.read_bytes(), out60.read_bytes()

    def _compare(kind: str, produced: bytes, baseline: Path) -> dict:
        import io

        import numpy as np

        got = np.load(io.BytesIO(produced))
        ref = np.load(baseline)
        n = min(got.shape[1], ref.shape[1])
        a = got[:, :n].astype(np.float64).ravel()
        b = ref[:, :n].astype(np.float64).ravel()
        diff = np.abs(a - b)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        corr = float(np.dot(a, b) / denom) if denom else 0.0
        rms_a = float(np.sqrt(np.mean(a * a)))
        rms_b = float(np.sqrt(np.mean(b * b)))
        return {
            "clip": kind,
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "p99_abs": float(np.percentile(diff, 99)) if diff.size else 0.0,
            "correlation": corr,
            "rms_ratio": rms_a / rms_b if rms_b else 0.0,
            "samples": int(a.size),
        }

    def _cost(gpu: str, seconds: float) -> float:
        rate = GPU_USD_PER_SECOND[gpu]
        extra = RESERVED_CORES * CPU_USD_PER_CORE_SECOND + RESERVED_GIB * RAM_USD_PER_GIB_SECOND
        return seconds * (rate + extra)

    def _run_gpu(name: str, fn, mix10: bytes, mix60: bytes) -> float:
        print(f"\n== {name} ==", flush=True)
        wall0 = time.perf_counter()
        payload = fn.remote(mix10, mix60)
        wall = time.perf_counter() - wall0
        loaded = payload["load"]
        cold = max(0.0, wall - float(loaded["server_s"]) - sum(float(payload["clips"][k]["server_s"]) for k in ("10s", "60s")))
        print(
            f"cold_start_s={cold:.3f} model_load_s={loaded['model_load_s']:.3f} "
            f"wall_s={wall:.3f} providers={loaded['providers']} gpu={loaded['gpu']}",
            flush=True,
        )
        if "CUDAExecutionProvider" not in loaded["providers"]:
            raise RuntimeError(f"{name} did not run CUDAExecutionProvider: {loaded['providers']}")
        peak_gpu = int(loaded["peak_gpu_mb"])
        peak_rss = int(loaded["peak_rss_kib"])
        for label in ("10s", "60s"):
            report = payload["clips"][label]
            vocals = report.pop("vocals_npy")
            baseline = DEFAULT_BASE10 if label == "10s" else DEFAULT_BASE60
            quality = _compare(label, vocals, baseline)
            per_chunk = report["infer_s"] / report["chunks"] if report["chunks"] else 0.0
            peak_gpu = max(peak_gpu, int(report["peak_gpu_mb"]))
            peak_rss = max(peak_rss, int(report["peak_rss_kib"]))
            print(
                f"{label} infer_s={report['infer_s']:.3f} chunks={report['chunks']} "
                f"per_chunk_s={per_chunk:.3f} server_s={report['server_s']:.3f} "
                f"peak_gpu_mb={report['peak_gpu_mb']} peak_rss_kib={report['peak_rss_kib']}",
                flush=True,
            )
            print(
                f"{label} max_abs={quality['max_abs']:.6g} p99_abs={quality['p99_abs']:.6g} "
                f"corr={quality['correlation']:.6f} rms_ratio={quality['rms_ratio']:.4f}",
                flush=True,
            )
        infer_60 = float(payload["clips"]["60s"]["infer_s"])
        chunks_60 = int(payload["clips"]["60s"]["chunks"])
        per_chunk = infer_60 / chunks_60
        step = 261120 - 2 * 3840
        song_samples = int(round(SONG_SECONDS * 44100))
        song_pad = (step - (song_samples % step)) % step
        song_chunks = (song_samples + song_pad + step - 1) // step
        song_infer = per_chunk * song_chunks
        song_warm = _cost(name, song_infer)
        song_cold = _cost(name, cold + float(loaded["model_load_s"]) + song_infer)
        this_run = _cost(name, wall)
        print(
            f"{name} peak_gpu_mb={peak_gpu} peak_rss_mib={peak_rss / 1024:.1f} "
            f"estimated_3min_warm_usd={song_warm:.4f} with_cold_start_usd={song_cold:.4f} "
            f"this_call_usd={this_run:.4f} song_infer_s={song_infer:.1f} song_chunks={song_chunks}",
            flush=True,
        )
        return this_run

    @app.local_entrypoint()
    def main() -> None:
        _refuse_unless_approved()
        for path in (MODEL, DEFAULT_CLIP10, DEFAULT_CLIP60, DEFAULT_BASE10, DEFAULT_BASE60):
            if not path.is_file():
                raise SystemExit(f"Missing benchmark input: {path}")
        mix10, mix60 = _decode_mixes()
        only = os.environ.get("KIRAR_MODAL_ONLY", "")
        spent = 0.0
        if only != "L4":
            spent += _run_gpu("T4", bench_t4, mix10, mix60)
        # L4's hard timeout is 600s. Skip it if the measured T4 bill plus that cap would pass $2.
        l4_cap = _cost("L4", 600.0)
        if only != "T4":
            if spent + l4_cap > 2.0:
                raise SystemExit(f"Stopping before L4. Estimated spend ${spent:.2f} plus L4 cap ${l4_cap:.2f} exceeds $2.")
            _run_gpu("L4", bench_l4, mix10, mix60)


if __name__ == "__main__":
    sys.stdout.write(PLAN)
    missing = [str(path) for path in (MODEL, DEFAULT_CLIP10, DEFAULT_CLIP60, DEFAULT_BASE10, DEFAULT_BASE60) if not path.is_file()]
    if missing:
        sys.stdout.write("\nMissing local inputs:\n")
        sys.stdout.write("\n".join(missing) + "\n")
    else:
        sys.stdout.write("\nLocal inputs are present. No Modal call was made.\n")
