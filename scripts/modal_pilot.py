"""Prepare or run the bounded pilot on Modal; ordinary execution is a dry run.

From the repository, using Python 3.12 with the optional Modal 1.5.1 SDK:
  python scripts/modal_pilot.py --config configs/gpu-pilot.json
  python scripts/modal_pilot.py --config configs/gpu-pilot.json \
      --execute --gpu L40S --job-timeout-seconds 1800

Only --execute uploads the listed files and starts billable resources. Results
are retained in the named Modal Volume and small reports are copied locally.
Timeouts bound execution time, not the total invoice (build/startup/storage and
other resource charges are separate). Completed GPU pilots are archived in results/.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid


REPO = Path(__file__).resolve().parents[1]
MODAL_VERSION = "1.5.1"
PACKAGES = ["torch==2.11.0", "einops==0.8.2", "numpy==2.4.4", "pytest==9.0.2",
            f"modal=={MODAL_VERSION}"]
CONFIG_OVERRIDES = {"max_steps": "max_steps", "train_warmup_steps": "warmup_steps",
                    "eval_interval": "eval_interval", "eval_batches": "eval_batches",
                    "checkpoint_interval": "checkpoint_interval"}


def apply_config_overrides(config: dict, overrides: dict) -> dict:
    """Apply explicitly requested training settings before recording the payload."""
    unknown = set(overrides) - CONFIG_OVERRIDES.keys()
    if unknown:
        raise ValueError(f"Unknown config overrides: {sorted(unknown)}")
    result = dict(config)
    for option, value in overrides.items():
        if value is None:
            continue
        minimum = 0 if option == "train_warmup_steps" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{option.replace('_', '-')} must be an integer >= {minimum}")
        result[CONFIG_OVERRIDES[option]] = value
    maximum = result.get("max_steps")
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("max-steps must be a positive integer")
    training_warmup = result.get("warmup_steps", 2)
    if type(training_warmup) is not int or not 0 <= training_warmup < maximum:
        raise ValueError("train-warmup-steps must be in [0, max_steps)")
    for key in ("eval_interval", "eval_batches", "checkpoint_interval"):
        if key in result and (type(result[key]) is not int or result[key] <= 0):
            raise ValueError(f"{key.replace('_', '-')} must be a positive integer")
    return result


def validate_options(config: dict, options: dict) -> None:
    """Validate billable-work controls without importing the GPU or cloud SDK."""
    effective = options["effective_batch_size"]
    microbatches = options["microbatches"]
    if (type(effective) is not int or effective <= 0 or not microbatches
            or any(type(b) is not int or b <= 0 or effective % b for b in microbatches)
            or len(set(microbatches)) != len(microbatches)):
        raise ValueError("microbatches must be distinct positive divisors of effective-batch-size")
    warmup = options["warmup_steps"]
    if type(warmup) is not int or not 1 <= warmup < config["max_steps"]:
        raise ValueError("warmup-steps must be positive and smaller than max_steps")
    if (type(options["case_timeout_seconds"]) is not int or options["case_timeout_seconds"] <= 0
            or type(options["job_timeout_seconds"]) is not int or options["job_timeout_seconds"] <= 60):
        raise ValueError("case timeout must be positive and job timeout must exceed 60 seconds")
    seeds = options["seeds"]
    if (not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 - 1 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct integers in [0, 2**63 - 1)")
    policies = options.get("policies")
    if policies is not None:
        if not policies or any(not isinstance(policy, str) for policy in policies):
            raise ValueError("policies must be a nonempty list of backend:checkpoint:microbatch strings")
        if len(set(policies)) != len(policies):
            raise ValueError("policies must be distinct")
        for policy in policies:
            match = re.fullmatch(r"(eager|sdpa):(none|block):([1-9][0-9]*)", policy)
            if not match or int(match[3]) not in microbatches:
                raise ValueError("policies require eager|sdpa:none|block:microbatch with a requested microbatch")


def collect_evidence(root: Path, log_limit: int = 65_536) -> dict:
    """Collect text/JSON evidence only; checkpoints never enter the client response."""
    root = Path(root)
    evidence = {"format_version": 1, "logs": {}, "truncated_logs": [],
                "manifest": None, "cases": {}, "missing_files": [], "errors": []}

    def read_json(path: Path):
        if not path.exists():
            evidence["missing_files"].append(str(path.relative_to(root)))
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            evidence["errors"].append(f"{path.relative_to(root)}: {error}")
            return None

    def read_log(path: Path):
        if not path.exists():
            evidence["missing_files"].append(str(path.relative_to(root)))
            return None
        try:
            with path.open("rb") as stream:
                size = path.stat().st_size
                stream.seek(max(0, size - log_limit))
                text = stream.read(log_limit).decode("utf-8", errors="replace")
            if size > log_limit:
                evidence["truncated_logs"].append(str(path.relative_to(root)))
            return text
        except OSError as error:
            evidence["errors"].append(f"{path.relative_to(root)}: {error}")
            return None

    for name in ("tests.log", "preflight.log", "pilot.log"):
        evidence["logs"][name] = read_log(root / name)
    evidence["config"] = read_json(root / "config.json")
    evidence["preflight"] = read_json(root / "preflight.json")
    manifest = read_json(root / "pilot" / "manifest.json")
    evidence["manifest"] = manifest
    if isinstance(manifest, dict) and isinstance(manifest.get("cases"), list):
        for case in manifest["cases"]:
            case_id = case.get("case_id") if isinstance(case, dict) else None
            if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9_-]+", case_id):
                evidence["errors"].append("manifest has an invalid case identifier")
                continue
            run = root / "pilot" / "runs" / case_id
            item = {"metadata": read_json(run / "metadata.json"),
                    "summary": read_json(run / "summary.json"),
                    "execution": read_json(root / "pilot" / "execution" / f"{case_id}.json"),
                    "metrics": [],
                    "log_tail": read_log(root / "pilot" / "logs" / f"{case_id}.log")}
            metrics = run / "metrics.jsonl"
            if metrics.exists():
                try:
                    with metrics.open() as stream:
                        for number, line in enumerate(stream, 1):
                            if line.strip():
                                try:
                                    item["metrics"].append(json.loads(line))
                                except ValueError as error:
                                    evidence["errors"].append(f"{metrics.relative_to(root)}:{number}: {error}")
                except OSError as error:
                    evidence["errors"].append(f"{metrics.relative_to(root)}: {error}")
            else:
                evidence["missing_files"].append(str(metrics.relative_to(root)))
            evidence["cases"][case_id] = item
    elif manifest is not None:
        evidence["errors"].append("manifest does not contain a cases list")
    return evidence


def prepare_payload(config_path: Path, repo: Path = REPO) -> dict:
    """Resolve explicit inputs; never mount the repository's parent or home."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    files = []

    def add_file(local: Path, remote: str) -> None:
        local = local.resolve(strict=True)
        if not local.is_file():
            raise ValueError(f"Not a regular file: {local}")
        files.append({"local": str(local), "remote": remote,
                      "bytes": local.stat().st_size})

    for folder in ("src", "tests"):
        for path in sorted((repo / folder).rglob("*.py")):
            if path.is_symlink():
                raise ValueError(f"Source symlinks are not mounted: {path}")
            add_file(path, "/project/" + path.relative_to(repo).as_posix())
    for path in sorted((repo / "configs").glob("*.json")):
        if path.is_symlink():
            raise ValueError(f"Configuration symlinks are not mounted: {path}")
        add_file(path, "/project/" + path.relative_to(repo).as_posix())
    add_file(repo / "pyproject.toml", "/project/pyproject.toml")
    add_file(repo / "scripts" / "modal_pilot.py", "/project/scripts/modal_pilot.py")
    for name in ("run_study.py", "quality_target.py"):
        study_script = repo / "scripts" / name
        if study_script.exists():
            if study_script.is_symlink():
                raise ValueError(f"Source symlinks are not mounted: {study_script}")
            add_file(study_script, f"/project/scripts/{name}")

    data = config["data"]
    if data["kind"] == "npy":
        for split in ("train", "valid"):
            key = f"{split}_path"
            remote = f"/data/{split}.npy"
            add_file(config_path.parent / data[key], remote)
            data[key] = remote
    elif data["kind"] != "synthetic":
        raise ValueError("Only npy or synthetic data is supported")
    tokenizer_paths = []
    for index, filename in enumerate(data.get("tokenizer_files", [])):
        remote = f"/data/tokenizer-{index}/{Path(filename).name}"
        add_file(config_path.parent / filename, remote)
        tokenizer_paths.append(remote)
    data["tokenizer_files"] = tokenizer_paths
    config["device"] = "cuda"
    config["precision"] = "bf16"
    return {"config": config, "files": files,
            "upload_bytes": sum(item["bytes"] for item in files),
            "python": "3.12", "packages": PACKAGES}


def execute(payload: dict, options: dict, output_dir: Path) -> dict:
    # Importing this module or preparing a payload never contacts Modal.
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("--execute requires Python 3.12 to match the serialized remote function. "
                           "Use an isolated Python 3.12 client with modal==1.5.1.")
    validate_options(payload["config"], options)
    import modal
    if modal.__version__ != MODAL_VERSION:
        raise RuntimeError(f"--execute requires modal=={MODAL_VERSION}; found {modal.__version__}")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "launch.json").write_text(
        json.dumps({"payload": payload, "options": options}, indent=2) + "\n")
    image = (modal.Image.debian_slim(python_version=payload["python"])
             .pip_install(*payload["packages"])
             .env({"PYTHONPATH": "/project/src", "PYTHONDONTWRITEBYTECODE": "1"}))
    for item in payload["files"]:
        image = image.add_local_file(item["local"], item["remote"])
    volume = modal.Volume.from_name(options["volume"], create_if_missing=True, version=2)
    app = modal.App("systems-for-llms-pilot")

    @app.function(image=image, gpu=options["gpu"], cpu=2, memory=8192,
                  volumes={"/results": volume}, min_containers=0,
                  max_containers=1, scaledown_window=2, retries=0,
                  timeout=options["job_timeout_seconds"], serialized=True,
                  include_source=False)
    def remote_job(config: dict, settings: dict) -> dict:
        import json
        import importlib.metadata
        import importlib.util
        import os
        from pathlib import Path
        import signal
        import subprocess
        import sys
        import time

        started = time.monotonic()
        root = Path("/results") / settings["run_id"]
        root.mkdir(parents=True, exist_ok=False)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        report = {"status": "started", "run_id": settings["run_id"],
                  "volume": settings["volume"], "volume_path": str(root),
                  "torch": importlib.metadata.version("torch")}

        def command(args: list[str], log_name: str, maximum: float) -> None:
            # Leave time to stop child processes, commit the volume, and return.
            remaining = settings["job_timeout_seconds"] - (time.monotonic() - started) - 30
            if remaining <= 0:
                raise TimeoutError("Pilot job time limit reached")
            with (root / log_name).open("w") as log:
                child = subprocess.Popen(args, cwd="/project", stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=min(maximum, remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError(f"Timed out: {log_name}") from None
                if code:
                    raise RuntimeError(f"Command failed with exit {code}; see {log_name}")
            volume.commit()

        reports = {}
        try:
            # A short-lived subprocess releases its CUDA context before profiling.
            preflight = """import json, sys, torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in the requested GPU container')
if not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError('This pilot requires native BF16 support')
with open(sys.argv[1], 'w') as output:
    json.dump({'cuda_available': True, 'gpu_name': torch.cuda.get_device_name(),
               'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory}, output)
"""
            command([sys.executable, "-c", preflight, str(root / "preflight.json")],
                    "preflight.log", 60)
            report.update(json.loads((root / "preflight.json").read_text()))
            command([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                     "tests"], "tests.log", 300)
            pilot_command = [sys.executable, "-m", "llmlab.pilot", "--config", str(config_path),
                             "--output-dir", str(root / "pilot"), "--microbatches",
                             *map(str, settings["microbatches"]), "--effective-batch-size",
                             str(settings["effective_batch_size"]), "--seeds", *map(str, settings["seeds"]),
                             "--warmup-steps", str(settings["warmup_steps"]), "--execute",
                             "--device", "cuda", "--precision", "bf16", "--timeout-seconds",
                             str(settings["case_timeout_seconds"])]
            if settings.get("policies") is not None:
                pilot_command.extend(["--policies", *settings["policies"]])
            command(pilot_command, "pilot.log", settings["job_timeout_seconds"])
            report["status"] = "finished"
        except Exception as error:
            report.update(status="failed", error=str(error))
        finally:
            aggregate_path = root / "pilot" / "aggregate.json"
            if aggregate_path.exists():
                try:
                    aggregate = json.loads(aggregate_path.read_text())
                    reports["aggregate.json"] = aggregate
                    report["case_status_counts"] = aggregate.get("status_counts", {})
                    report["comparability"] = aggregate.get("comparability", {})
                    cases = aggregate.get("cases", [])
                    if report["status"] == "finished" and (
                            not cases or any(case.get("status") != "completed"
                                             for case in cases)
                            or report["comparability"].get("status") != "passed"):
                        report.update(status="failed", error="Some pilot cases failed or are not comparable; see aggregate.json")
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    report.update(status="failed", error=f"Could not read aggregate: {error}")
            elif report["status"] == "finished":
                report.update(status="failed", error="Pilot did not produce aggregate.json")
            try:
                # Load the exact uploaded helper; importing it has no cloud side effects.
                spec = importlib.util.spec_from_file_location(
                    "pilot_evidence", "/project/scripts/modal_pilot.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                evidence = module.collect_evidence(root)
                reports["evidence.json"] = evidence
                (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
                report["evidence_status"] = ("partial" if evidence["errors"] or evidence["missing_files"]
                                             else "complete")
                if report["status"] == "finished" and report["evidence_status"] != "complete":
                    report.update(status="failed", error="Evidence collection is incomplete; see evidence.json")
            except Exception as error:
                report["evidence_status"] = "failed"
                report["evidence_error"] = str(error)
                if report["status"] == "finished":
                    report.update(status="failed", error="Could not collect run evidence")
            report["function_seconds"] = time.monotonic() - started
            (root / "launch-summary.json").write_text(json.dumps(report, indent=2) + "\n")
            volume.commit()
        reports["launch-summary.json"] = report
        return reports

    try:
        with modal.enable_output(), app.run():
            reports = remote_job.remote(payload["config"], options)
    except Exception as error:
        (output_dir / "launch-error.json").write_text(
            json.dumps({"status": "failed", "error": str(error),
                        "volume": options["volume"], "run_id": options["run_id"]}, indent=2) + "\n")
        raise
    for name, report in reports.items():
        (output_dir / name).write_text(json.dumps(report, indent=2) + "\n")
    return reports["launch-summary.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", default="L40S")
    parser.add_argument("--volume", default="systems-for-llms-results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--effective-batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, help="override optimizer-update horizon")
    parser.add_argument("--train-warmup-steps", type=int, help="override learning-rate warmup; zero is allowed")
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--policies", nargs="+", help="explicit backend:checkpoint:microbatch treatments")
    parser.add_argument("--case-timeout-seconds", type=int, default=120)
    parser.add_argument("--job-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        parser.error("run-id must contain 1-80 letters, numbers, underscores or hyphens")
    payload = prepare_payload(args.config)
    options = {key: value for key, value in vars(args).items()
               if key not in {"config", "execute", "output_dir", "run_id", *CONFIG_OVERRIDES}
               and not (key == "policies" and value is None)}
    try:
        payload["config"] = apply_config_overrides(
            payload["config"], {key: getattr(args, key) for key in CONFIG_OVERRIDES})
        validate_options(payload["config"], options)
    except ValueError as error:
        parser.error(str(error))
    options["run_id"] = run_id
    output_dir = args.output_dir or REPO / "runs" / ("modal-" + run_id)
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", "options": options,
                      "local_output_dir": str(output_dir), "payload": payload}, indent=2))
    if args.execute:
        report = execute(payload, options, output_dir)
        print(json.dumps(report, indent=2))
        if report["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
