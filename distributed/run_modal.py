"""Launch the isolated distributed study on Modal; default execution is a dry run."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
MODAL_VERSION = "1.5.1"
GPU_RATES = {f"{name}:{count}": count * rate
             for name, rate in (("H100", 0.001097), ("H200", 0.001261),
                                ("L40S", 0.000542), ("A100-80GB", 0.000694))
             for count in (2, 4, 8)}


def resources(options: dict) -> dict:
    count = int(options["gpu"].split(":")[-1])
    return {"gpu_count": count, "cpu_cores": max(8, count * 2), "ram_gib": max(32, count * 8)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot(files: list[dict]) -> dict:
    return {item["remote"]: hashlib.sha256(Path(item["local"]).read_bytes()).hexdigest()
            for item in files if item["remote"].startswith("/project/")}


def prepare_payload(config_path: Path, repo: Path = REPO) -> dict:
    """Use the existing explicit data mapping and add only distributed code/tests."""
    launcher = load_module(repo / "scripts/modal_pilot.py", "distributed_base_launcher")
    payload = launcher.prepare_payload(config_path, repo)
    required = [repo / "distributed/run_modal.py", repo / "distributed/run_experiments.py"]
    for name in ("run_campaign.py", "analyze.py"):
        extra = repo / "distributed" / name
        if extra.exists():
            required.append(extra)
    additional = required + sorted((repo / "distributed/llmdist").rglob("*.py"))
    additional += sorted((repo / "distributed/tests").rglob("*.py"))
    additional += sorted((repo / "distributed/configs").glob("*.json"))
    for path in additional:
        if path.is_symlink():
            raise ValueError(f"Source symlinks are not mounted: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Not a regular file: {path}")
        payload["files"].append({"local": str(resolved),
                                 "remote": "/project/" + path.relative_to(repo).as_posix(),
                                 "bytes": resolved.stat().st_size})
    remotes = [item["remote"] for item in payload["files"]]
    if len(remotes) != len(set(remotes)):
        raise ValueError("Duplicate remote payload paths")
    payload["upload_bytes"] = sum(item["bytes"] for item in payload["files"])
    payload["source_sha256"] = source_snapshot(payload["files"])
    return payload


def validate_options(options: dict) -> None:
    if options["gpu"] not in GPU_RATES:
        raise ValueError(f"gpu must be one of {sorted(GPU_RATES)}")
    seconds = options["job_timeout_seconds"]
    if type(seconds) is not int or not 600 <= seconds <= 7200:
        raise ValueError("job-timeout-seconds must be an integer between 600 and 7200")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", options["run_id"]):
        raise ValueError("run-id must contain only letters, numbers, underscores and hyphens")
    if options["stage"] not in {"smoke", "screen", "scale", "confirmation"}:
        raise ValueError("Unknown distributed stage")
    cases = options["plan"].get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Distributed plan must contain cases")
    ids = [case.get("case_id") for case in cases]
    if (any(not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case_id) for case_id in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Distributed case identifiers must be unique and safe")
    available = resources(options)["gpu_count"]
    if any(type(case.get("world_size")) is not int or not 1 <= case["world_size"] <= available for case in cases):
        raise ValueError("Planned world size must fit the GPU allocation")
    if (type(options["plan"].get("required_gpus")) is not int
            or not max(case["world_size"] for case in cases) <= options["plan"]["required_gpus"] <= available):
        raise ValueError("Plan required_gpus must cover all ranks and fit the allocation")


def cost_estimate(options: dict) -> dict:
    allocation = resources(options)
    rate = GPU_RATES[options["gpu"]] + allocation["cpu_cores"] * 0.0000131 + allocation["ram_gib"] * 0.00000222
    return {"requested_resources_usd_per_hour": round(rate * 3600, 4),
            "requested_resources_usd_at_timeout": round(rate * options["job_timeout_seconds"], 4),
            "allocated_gpu_count": allocation["gpu_count"], "cpu_cores": allocation["cpu_cores"], "ram_gib": allocation["ram_gib"],
            "source": "https://modal.com/pricing", "checked": "2026-11-01",
            "scope": "All allocated GPUs are billed even when a case uses fewer ranks. Excludes startup/builds, excess CPU/RAM, storage/egress and other apps; not a billing cap."}


def run_command(args: list[str], log: Path, timeout: float, cwd: Path) -> None:
    """Bound the whole child process group, including torchrun worker processes."""
    print(f"Running {log.name}", flush=True)
    with log.open("w") as output:
        child = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                                 cwd=cwd, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise TimeoutError(f"Timed out: {log.name}") from None
    if code:
        tail = log.read_text(errors="replace")[-4000:].strip()
        raise RuntimeError(f"Exit {code}: {log.name}" + (f"\n{tail}" if tail else ""))


def collect_evidence(root: Path, log_limit: int = 65_536) -> dict:
    """Collect full text metrics and JSON; never transfer model/optimizer tensors."""
    result = {"format_version": 1, "json": {}, "metrics": {}, "logs": {},
              "truncated_logs": [], "errors": [], "missing_files": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(root).as_posix()
        if name in {"evidence.json", "launch-summary.json"}:
            continue
        try:
            if path.suffix == ".json":
                result["json"][name] = json.loads(path.read_text())
            elif path.suffix == ".jsonl":
                result["metrics"][name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            elif path.suffix == ".log":
                with path.open("rb") as stream:
                    size = path.stat().st_size
                    stream.seek(max(0, size - log_limit))
                    result["logs"][name] = stream.read(log_limit).decode("utf-8", errors="replace")
                if size > log_limit:
                    result["truncated_logs"].append(name)
        except (OSError, ValueError) as error:
            result["errors"].append(f"{name}: {error}")
    for name in ("config.json", "plan.json", "preflight.json", "tests-report.json", "study/aggregate.json",
                 "preflight.log", "tests.log", "study.log"):
        if not (root / name).exists():
            result["missing_files"].append(name)
    plan = result["json"].get("plan.json", {})
    expected = {case["case_id"]: case for case in plan.get("cases", [])}
    aggregate = result["json"].get("study/aggregate.json", {})
    for case in aggregate.get("cases", []):
        if case.get("status") != "completed":
            continue
        case_id = case.get("case_id")
        if case_id not in expected:
            result["errors"].append(f"Unexpected completed case: {case_id}")
            continue
        directory = f"study/cases/{case_id}"
        names = [f"{directory}/summary.json"]
        for rank in range(expected[case_id]["world_size"]):
            names += [f"{directory}/rank-{rank}-{suffix}" for suffix in
                      ("metadata.json", "metrics.jsonl", "summary.json")]
        for name in names:
            if not (root / name).is_file():
                result["missing_files"].append(name)
    return result


def read_test_report(path: Path) -> dict:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    report = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0, "nonpassing": []}
    for case in cases:
        for tag, counter in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            if case.find(tag) is not None:
                report[counter] += 1
                report["nonpassing"].append({"name": case.get("name"), "class": case.get("classname"), "status": tag})
    report["passed"] = report["tests"] - report["failures"] - report["errors"] - report["skipped"]
    return report


PREFLIGHT = """import json, subprocess, sys, torch
count = int(sys.argv[2])
if torch.cuda.device_count() != count:
    raise RuntimeError(f'Expected exactly {count} CUDA GPUs; found {torch.cuda.device_count()}')
devices = []
for index in range(count):
    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f'GPU {index} lacks native BF16 support')
    properties = torch.cuda.get_device_properties(index)
    devices.append({'index': index, 'name': properties.name, 'uuid': str(getattr(properties, 'uuid', 'unavailable')), 'memory_bytes': properties.total_memory,
                    'capability': list(torch.cuda.get_device_capability(index))})
warnings = []

def diagnostic(command):
    # Provider containers can expose CUDA while restricting management queries.
    # These reports describe hardware; the mandatory CUDA/NCCL tests gate training.
    def text(value):
        return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        report = {'command': command, 'status': 'completed' if result.returncode == 0 else 'unavailable',
                  'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as error:
        report = {'command': command, 'status': 'timeout', 'returncode': None,
                  'stdout': text(error.stdout), 'stderr': text(error.stderr), 'error': str(error)}
    except OSError as error:
        report = {'command': command, 'status': 'unavailable', 'returncode': None,
                  'stdout': '', 'stderr': str(error)}
    if report['status'] != 'completed':
        warnings.append(' '.join(command) + ': ' + report['status'])
    return report

topology = diagnostic(['nvidia-smi', 'topo', '-m'])
identities = diagnostic(['nvidia-smi', '--query-gpu=uuid,name,pci.bus_id', '--format=csv,noheader'])
peer_access = []
for i in range(count):
    row = []
    for j in range(count):
        try:
            row.append(torch.cuda.can_device_access_peer(i, j) if i != j else True)
        except RuntimeError as error:
            row.append(None)
            warnings.append(f'Peer-access query {i}->{j} unavailable: {error}')
    peer_access.append(row)
with open(sys.argv[1], 'w') as output:
    json.dump({'devices': devices, 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'peer_access': peer_access,
               'nvidia_smi_topology': topology['stdout'] if topology['status'] == 'completed' else None,
               'nvidia_smi_identities': identities['stdout'] if identities['status'] == 'completed' else None,
               'nvidia_smi_diagnostics': {'topology': topology, 'identities': identities},
               'warnings': warnings}, output, indent=2)
print(json.dumps({'devices': devices, 'warnings': warnings}, indent=2))
if topology['status'] == 'completed':
    print(topology['stdout'])
"""


def run_remote(config: dict, options: dict, root: Path, commit, project: Path = Path("/project")) -> dict:
    """Remote orchestration, kept independently testable without the Modal SDK."""
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "plan.json").write_text(json.dumps(options["plan"], indent=2) + "\n")
    report = {"status": "started", "run_id": options["run_id"], "stage": options["stage"],
              "gpu": options["gpu"], "allocated_gpu_count": resources(options)["gpu_count"],
              "volume": options["volume"], "volume_path": str(root)}

    def command(args, name, cap):
        remaining = options["job_timeout_seconds"] - (time.monotonic() - started) - 45
        if remaining <= 0:
            raise TimeoutError("Distributed job time limit reached")
        run_command(args, root / name, min(cap, remaining), project)
        commit()

    reports = {}
    try:
        command([sys.executable, "-c", PREFLIGHT, str(root / "preflight.json"), str(resources(options)["gpu_count"])], "preflight.log", 90)
        command([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                 "--junitxml", str(root / "tests.xml"), "tests", "distributed/tests"], "tests.log", 600)
        test_report = read_test_report(root / "tests.xml")
        (root / "tests-report.json").write_text(json.dumps(test_report, indent=2) + "\n")
        if not test_report["tests"] or any(test_report[field] for field in ("failures", "errors", "skipped")):
            raise RuntimeError("GPU correctness suite must pass without failures or skipped tests")
        command([sys.executable, "distributed/run_experiments.py", "--plan", str(root / "plan.json"),
                 "--config", str(root / "config.json"), "--output-dir", str(root / "study"),
                 "--allocated-gpus", str(resources(options)["gpu_count"]), "--execute"],
                "study.log", options["job_timeout_seconds"])
        aggregate = json.loads((root / "study/aggregate.json").read_text())
        cases = aggregate.get("cases", [])
        expected = {case["case_id"] for case in options["plan"]["cases"]}
        if (aggregate.get("status") != "completed" or len(cases) != len(expected)
                or {case.get("case_id") for case in cases} != expected
                or any(case.get("status") != "completed" for case in cases)):
            raise RuntimeError("Distributed cases failed or are incomplete; inspect aggregate and logs")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
    finally:
        if (root / "tests.xml").exists() and not (root / "tests-report.json").exists():
            try:
                (root / "tests-report.json").write_text(json.dumps(read_test_report(root / "tests.xml"), indent=2) + "\n")
            except (OSError, ValueError, ET.ParseError):
                pass  # The original command failure and raw log remain authoritative.
        evidence = collect_evidence(root)
        reports["evidence.json"] = evidence
        report["evidence_status"] = "partial" if evidence["errors"] or evidence["missing_files"] else "complete"
        if report["status"] == "completed" and report["evidence_status"] != "complete":
            report.update(status="failed", error="Distributed evidence is incomplete")
        aggregate = evidence["json"].get("study/aggregate.json")
        if aggregate is not None:
            reports["aggregate.json"] = aggregate
            report["case_status_counts"] = {status: sum(case.get("status") == status for case in aggregate.get("cases", []))
                                             for status in sorted({case.get("status", "unknown") for case in aggregate.get("cases", [])})}
        report["function_seconds"] = time.monotonic() - started
        reports["launch-summary.json"] = report
        for name, content in reports.items():
            (root / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        commit()
    return reports


def execute(payload: dict, options: dict, output_dir: Path) -> dict:
    validate_options(options)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("--execute requires Python 3.12 and modal==1.5.1")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if source_snapshot(payload["files"]) != payload["source_sha256"]:
        raise RuntimeError("Source files changed after the launch payload was prepared")
    import modal
    if modal.__version__ != MODAL_VERSION:
        raise RuntimeError(f"--execute requires modal=={MODAL_VERSION}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "launch.json").write_text(json.dumps({"payload": payload, "options": options,
                                                      "cost_estimate": cost_estimate(options)}, indent=2) + "\n")
    image = (modal.Image.debian_slim(python_version=payload["python"]).pip_install(*payload["packages"])
             .env({"PYTHONPATH": "/project/src:/project/distributed", "PYTHONDONTWRITEBYTECODE": "1",
                   "LLMDIST_TEST_DEVICE": "cuda"}))
    for item in payload["files"]:
        image = image.add_local_file(item["local"], item["remote"])
    volume = modal.Volume.from_name(options["volume"], create_if_missing=True, version=2)
    app = modal.App("systems-for-llms-distributed")

    allocation = resources(options)
    @app.function(image=image, gpu=options["gpu"], cpu=allocation["cpu_cores"], memory=allocation["ram_gib"] * 1024,
                  volumes={"/results": volume}, min_containers=0, max_containers=1,
                  scaledown_window=2, retries=0, timeout=options["job_timeout_seconds"],
                  serialized=True, include_source=False)
    def remote_job(config, settings):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("distributed_remote", "/project/distributed/run_modal.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_remote(config, settings, Path("/results") / settings["run_id"], volume.commit)

    try:
        with modal.enable_output(), app.run():
            reports = remote_job.remote(payload["config"], options)
    except Exception as error:
        (output_dir / "launch-error.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        raise
    for name, content in reports.items():
        (output_dir / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return reports["launch-summary.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "screen", "scale", "confirmation"], default="smoke")
    parser.add_argument("--config", type=Path, default=REPO / "configs/scale-100m-s1024.json")
    parser.add_argument("--gpu", choices=sorted(GPU_RATES), default="H100:2")
    parser.add_argument("--job-timeout-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    experiment = load_module(REPO / "distributed/run_experiments.py", "distributed_experiment_plan")
    plan = experiment.build_plan(args.stage)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("distributed-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    options = {"stage": args.stage, "plan": plan, "gpu": args.gpu, "job_timeout_seconds": args.job_timeout_seconds,
               "run_id": run_id, "volume": "systems-for-llms-results"}
    try:
        validate_options(options)
        payload = prepare_payload(args.config)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    output = (args.output_dir or REPO / "runs" / run_id).resolve()
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", "output_dir": str(output),
                      "options": options, "payload": payload, "cost_estimate": cost_estimate(options)}, indent=2), flush=True)
    if args.execute:
        summary = execute(payload, options, output)
        print(json.dumps(summary, indent=2), flush=True)
        if summary["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
