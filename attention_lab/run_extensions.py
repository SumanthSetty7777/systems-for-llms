"""Bounded H100 experiments for CUDA decode and attention backward; dry run by default.

Examples (use Python 3.12 and modal==1.5.1)::

    python -m attention_lab.run_extensions --stage cuda-smoke
    python -m attention_lab.run_extensions --stage cuda-smoke --execute
    python -m attention_lab.run_extensions --stage backward-study --execute

Every execution snapshots its source and stores raw evidence in a unique local
directory and the cohere-kernel-extensions Modal volume. No background GPU jobs,
automatic retries, or multiple concurrent containers are configured here.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "cohere-kernel-extensions"
CUDA_IMAGE_TAG = "nvidia/cuda:13.0.0-devel-ubuntu24.04"
# Linux/amd64 image verified against the official Docker Hub tag API, 2026-11-05.
CUDA_IMAGE = "nvidia/cuda@sha256:435220c0fef35cbf712e11999f8670a83835ef3cdd18564e5e8122f83078c88c"
PACKAGES = ["torch==2.11.0", "triton==3.6.0", "einops==0.8.2", "numpy==2.4.4",
            "modal==1.5.1", "ninja==1.11.1.4"]
STAGES = {
    "cuda-smoke": ("attention_lab.cuda_decode.study", "smoke", 420, 480),
    "cuda-study": ("attention_lab.cuda_decode.study", "study", 540, 600),
    "backward-smoke": ("attention_lab.backward_study", "smoke", 300, 480),
    "backward-study": ("attention_lab.backward_study", "study", 540, 600),
}
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
EVIDENCE_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md"}
RESOURCE_RATE = 0.001097 + 4 * 0.0000131 + 32 * 0.00000222
TIMEOUT_HEADROOM = 60


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_files():
    """Include nested CUDA sources, without accidentally uploading prior results."""
    files = []
    excluded = {"runs", "results", "source", "__pycache__", "build", "dist"}
    for directory, children, names in os.walk(ROOT / "attention_lab", followlinks=False):
        children[:] = sorted(name for name in children
                             if name not in excluded and not name.startswith("."))
        for name in children:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Source directory symlinks are not supported")
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError("Source file symlinks are not supported")
            files.append(path)
    return sorted(files)


def manifest(stage, seconds, run_id):
    module, substage, _, maximum = STAGES[stage]
    if not 60 <= seconds <= maximum:
        raise ValueError(f"{stage} time budget must be between 60 and {maximum} seconds")
    return {
        "schema_version": 1, "run_id": run_id, "stage": stage,
        "module": module, "module_stage": substage,
        "study_time_budget_seconds": seconds,
        "job_timeout_seconds": seconds + TIMEOUT_HEADROOM,
        "gpu": "H100", "python": "3.12", "packages": PACKAGES,
        "cuda_image": CUDA_IMAGE, "cuda_image_tag": CUDA_IMAGE_TAG,
        "cuda_image_verified": "2026-11-05",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files()},
        "resources": {"gpus": 1, "cpu_cores": 4, "ram_gib": 32,
                      "max_containers": 1, "retries": 0, "scaledown_window_seconds": 2},
        "volume": VOLUME_NAME,
        "cost_estimate": {
            "requested_resources_usd_at_timeout": round(RESOURCE_RATE * (seconds + TIMEOUT_HEADROOM), 4),
            "four_default_stages_usd_at_timeout": round(
                RESOURCE_RATE * sum(config[2] + TIMEOUT_HEADROOM for config in STAGES.values()), 4),
            "source": "https://modal.com/pricing", "checked": "2026-11-04",
            "scope": "Estimate, not a billing cap; excludes image builds, startup, storage, excess CPU/RAM and other apps",
        },
    }


def study_command(settings, root):
    command = [sys.executable, "-m", settings["module"], "--stage", settings["module_stage"],
               "--output-dir", str(root / "study"), "--time-budget-seconds",
               str(settings["study_time_budget_seconds"])]
    if settings["stage"].startswith("cuda-"):
        command.append("--execute")
    return command


def stop_child(child):
    """Terminate the process group, including any still-running NVCC subprocess."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=3)


def remote_study(settings, root, commit):
    """Keep exceptions inside the job so failure logs can still be returned."""
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "launch.json", settings)
    started = time.monotonic()
    outcome = {"status": "failed", "run_id": settings["run_id"]}
    try:
        commit()
        import torch
        import triton

        environment = {
            "python": sys.version, "torch": torch.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda, "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
        for label, command in (("nvidia_smi", ["nvidia-smi"]), ("nvcc", ["nvcc", "--version"]),
                               ("packages", [sys.executable, "-m", "pip", "freeze"])):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                environment[label] = {"stdout": result.stdout, "stderr": result.stderr,
                                      "returncode": result.returncode}
            except (OSError, subprocess.TimeoutExpired) as error:
                environment[label] = {"error": str(error)}
        atomic_json(root / "environment.json", environment)
        commit()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable")
        if (torch.__version__.split("+")[0] != "2.11.0" or triton.__version__ != "3.6.0"
                or not str(torch.version.cuda).startswith("13.0")):
            raise RuntimeError("Expected torch 2.11.0 / CUDA 13.0 / Triton 3.6.0")
        if environment["nvcc"].get("returncode") != 0:
            raise RuntimeError("CUDA devel image did not provide usable nvcc")
        command = study_command(settings, root)
        atomic_json(root / "command.json", {"argv": command})
        with (root / "study.log").open("w") as log:
            child = subprocess.Popen(command, cwd="/project", stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=settings["study_time_budget_seconds"])
                outcome.update(status="completed" if code == 0 else "failed", returncode=code)
            except subprocess.TimeoutExpired:
                stop_child(child)
                outcome.update(status="timed_out", returncode=child.returncode)
            finally:
                stop_child(child)
    except Exception as error:
        outcome.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        outcome["wall_seconds"] = time.monotonic() - started
        atomic_json(root / "outcome.json", outcome)
        try:
            commit()
        except Exception as error:
            outcome["volume_commit_error"] = str(error)
            atomic_json(root / "outcome.json", outcome)
    # Return raw files without truncating logs or discarding failing cases.
    evidence = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*")
                if path.is_file() and path.suffix in EVIDENCE_SUFFIXES}
    return {"outcome": outcome, "files": evidence}


def save_evidence(output, relative, content):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe returned evidence path")
    destination = output / Path(*path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def recover_volume_evidence(volume, settings, output):
    """Recover already-committed evidence if the remote call itself failed."""
    prefix = settings["run_id"] + "/"
    recovered = []
    for entry in volume.iterdir(settings["run_id"], recursive=True):
        name = entry.path.lstrip("/")
        if not name.startswith(prefix) or PurePosixPath(name).suffix not in EVIDENCE_SUFFIXES:
            continue
        relative = name[len(prefix):]
        content = b"".join(volume.read_file(entry.path))
        save_evidence(output, relative, content)
        recovered.append(relative)
    return recovered


def execute(settings, output):
    import modal
    from modal.client import Client

    if sys.version_info[:2] != (3, 12) or modal.__version__ != "1.5.1":
        raise RuntimeError("Use Python 3.12 and modal==1.5.1 for the cloud launcher")
    required = settings["module"].replace(".", "/") + ".py"
    if required not in settings["source_sha256"]:
        raise FileNotFoundError(f"Study source is not ready: {required}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing run directory: {output}")
    output.mkdir(parents=True)
    atomic_json(output / "launch.json", settings)
    volume = None

    async def preflight():
        await asyncio.wait_for(Client.from_env.aio(), timeout=25)

    try:
        # Snapshot before networking so the evidence always identifies submitted code.
        snapshots = []
        for name, expected in settings["source_sha256"].items():
            content = (ROOT / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"Source changed after manifest creation: {name}")
            snapshot = output / "source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(content)
            snapshots.append((name, snapshot))
        asyncio.run(preflight())
        image = (modal.Image.from_registry(settings["cuda_image"], add_python=settings["python"])
                 .entrypoint([])
                 .apt_install("build-essential")
                 .pip_install(*settings["packages"])
                 .env({"PYTHONPATH": "/project", "PYTHONDONTWRITEBYTECODE": "1",
                       "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0",
                       "MAX_JOBS": "4", "OMP_NUM_THREADS": "4",
                       "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions"}))
        for name, snapshot in snapshots:
            image = image.add_local_file(str(snapshot), "/project/" + name)
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
        app = modal.App(VOLUME_NAME)

        @app.function(image=image, gpu="H100", cpu=4, memory=32768,
                      volumes={"/results": volume}, min_containers=0, max_containers=1,
                      scaledown_window=2, retries=0, timeout=settings["job_timeout_seconds"],
                      serialized=True, include_source=False)
        def job(options):
            from attention_lab.run_extensions import remote_study
            return remote_study(options, Path("/results") / options["run_id"], volume.commit)

        with modal.enable_output(), app.run():
            result = job.remote(settings)
        for relative, content in result["files"].items():
            save_evidence(output, relative, content)
        return result["outcome"]
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__,
                   "error": str(error) or "Modal authentication/API request timed out",
                   "volume": VOLUME_NAME, "volume_directory": settings["run_id"],
                   "recovered_files": []}
        if volume is not None:
            try:
                failure["recovered_files"] = recover_volume_evidence(volume, settings, output)
            except Exception as recovery_error:
                failure["recovery_error"] = str(recovery_error)
        atomic_json(output / "launch-error.json", failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=list(STAGES), default="cuda-smoke")
    parser.add_argument("--time-budget-seconds", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually start the bounded H100 job")
    args = parser.parse_args(argv)
    seconds = STAGES[args.stage][2] if args.time_budget_seconds is None else args.time_budget_seconds
    run_id = args.run_id or ("cohere-" + args.stage + "-" +
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        parser.error("run ids must contain only letters, numbers, underscores and hyphens")
    try:
        settings = manifest(args.stage, seconds, run_id)
    except ValueError as error:
        parser.error(str(error))
    if not args.execute:
        print(json.dumps(settings, indent=2))
        return 0
    output = args.output_dir or ROOT / "runs" / run_id
    result = execute(settings, output)
    print(json.dumps({**result, "output": str(output)}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
