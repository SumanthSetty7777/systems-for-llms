"""Bounded single-GPU attention study on Modal; dry run unless --execute."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GPU_RATES = {"H100": 0.001097, "A100-80GB": 0.000694, "L40S": 0.000542}
STAGE_SECONDS = {"smoke": 480, "ablation": 900, "flex_hints": 300,
                 "tuning": 1200, "confirmation": 600, "block": 420}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest(stage, gpu, seconds, run_id):
    files = sorted((ROOT / "attention_lab").glob("*.py"))
    if any(path.is_symlink() for path in files):
        raise ValueError("Source symlinks are not supported")
    rate = GPU_RATES[gpu] + 4 * 0.0000131 + 32 * 0.00000222
    return {
        "schema_version": 1, "run_id": run_id, "stage": stage, "gpu": gpu,
        "job_timeout_seconds": seconds + 90,
        "study_time_budget_seconds": seconds,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in files},
        "packages": ["torch==2.11.0", "einops==0.8.2", "numpy==2.4.4", "modal==1.5.1"],
        "python": "3.12", "resources": {"gpus": 1, "cpu_cores": 4, "ram_gib": 32},
        "cost_estimate": {
            "requested_resources_usd_at_timeout": round(rate * (seconds + 90), 4),
            "source": "https://modal.com/pricing", "checked": "2026-11-04",
            "scope": "Estimate, not billing cap; excludes image builds, startup, storage, excess CPU/RAM and other apps",
        },
    }


def remote_study(settings, root, commit):
    """Run a separate child so a timeout still permits evidence retrieval."""
    import torch

    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "launch.json", settings)
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "device_count": torch.cuda.device_count(),
                   "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    try:
        process = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        environment["nvidia_smi"] = process.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        environment["nvidia_smi_error"] = str(error)
    atomic_json(root / "environment.json", environment)
    commit()
    command = [sys.executable, "-m"]
    if settings["stage"] == "block":
        command += ["attention_lab.block_benchmark"]
    else:
        command += ["attention_lab.study", "--stage", settings["stage"]]
    command += ["--output-dir", str(root / "study"), "--time-budget-seconds",
                str(settings["study_time_budget_seconds"]), "--execute"]
    if settings.get("tuning_volume_dir"):
        command += ["--tuning-dir", settings["tuning_volume_dir"]]
    started = time.monotonic()
    with (root / "study.log").open("w") as log:
        child = subprocess.Popen(command, cwd="/project", stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=settings["study_time_budget_seconds"] + 30)
            outcome = {"status": "completed" if code == 0 else "failed", "returncode": code}
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            outcome = {"status": "timed_out", "returncode": child.returncode}
    outcome.update({"wall_seconds": time.monotonic() - started, "run_id": settings["run_id"]})
    atomic_json(root / "outcome.json", outcome)
    commit()
    evidence = {str(path.relative_to(root)): path.read_text() for path in root.rglob("*.json")}
    evidence["study.log"] = (root / "study.log").read_text(errors="replace")[-100_000:]
    return {"outcome": outcome, "files": evidence}


def execute(settings, output):
    import modal
    from modal.client import Client

    if sys.version_info[:2] != (3, 12) or modal.__version__ != "1.5.1":
        raise RuntimeError("Use Python 3.12 and modal==1.5.1 for the cloud launcher")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing run directory: {output}")
    output.mkdir(parents=True)
    atomic_json(output / "launch.json", settings)

    async def preflight():
        await asyncio.wait_for(Client.from_env.aio(), timeout=25)

    try:
        asyncio.run(preflight())
        image = (modal.Image.debian_slim(python_version=settings["python"])
                 .pip_install(*settings["packages"])
                 .env({"PYTHONPATH": "/project", "PYTHONDONTWRITEBYTECODE": "1"}))
        for name, expected in settings["source_sha256"].items():
            path = ROOT / name
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"Source changed after manifest creation: {name}")
            snapshot = output / "source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(content)
            image = image.add_local_file(str(snapshot), "/project/" + name)
        volume = modal.Volume.from_name("attention-kernel-study", create_if_missing=True, version=2)
        app = modal.App("attention-kernel-study")

        @app.function(image=image, gpu=settings["gpu"], cpu=4, memory=32768,
                      volumes={"/results": volume}, min_containers=0, max_containers=1,
                      scaledown_window=2, retries=0, timeout=settings["job_timeout_seconds"],
                      serialized=True, include_source=False)
        def job(options):
            from attention_lab.run_modal import remote_study
            return remote_study(options, Path("/results") / options["run_id"], volume.commit)

        with modal.enable_output(), app.run():
            result = job.remote(settings)
        for relative, content in result["files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe returned evidence path")
            destination = output / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
        return result["outcome"]
    except Exception as error:
        atomic_json(output / "launch-error.json", {"status": "failed", "error_type": type(error).__name__,
                    "error": str(error) or "Modal authentication/API request timed out", "gpu_results_available": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=list(STAGE_SECONDS), default="smoke")
    parser.add_argument("--gpu", choices=list(GPU_RATES), default="H100")
    parser.add_argument("--time-budget-seconds", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tuning-run-id", help="Prior tuning run in attention-kernel-study volume")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    seconds = STAGE_SECONDS[args.stage] if args.time_budget_seconds is None else args.time_budget_seconds
    if not 60 <= seconds <= 1800:
        parser.error("time budget must be between 60 and 1800 seconds")
    run_id = args.run_id or "attention-" + args.stage + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for value in (run_id, args.tuning_run_id):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            parser.error("run ids must contain only letters, numbers, underscores and hyphens")
    settings = manifest(args.stage, args.gpu, seconds, run_id)
    if args.tuning_run_id:
        settings["tuning_volume_dir"] = "/results/" + args.tuning_run_id + "/study"
    if not args.execute:
        print(json.dumps(settings, indent=2))
        return 0
    output = args.output_dir or ROOT / "runs" / run_id
    result = execute(settings, output)
    print(json.dumps({**result, "output": str(output)}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
