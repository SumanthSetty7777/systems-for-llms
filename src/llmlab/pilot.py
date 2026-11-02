"""Plan a fixed-batch attention/memory pilot; execute only when requested."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from .config import TrainConfig, load_config
from .train import source_identity


def _sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive_int(value, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def make_plan(config: TrainConfig, *, microbatches: list[int], effective_batch_size: int,
              seeds: list[int], warmup_steps: int, device: str = "cuda",
              precision: str | None = None, order_seed: int = 0,
              timeout_seconds: float = 600, policies: list[str] | None = None) -> dict:
    """Resolve every case without loading data or starting training."""
    config.validate()
    _positive_int(effective_batch_size, "effective_batch_size")
    _positive_int(warmup_steps, "warmup_steps")
    if warmup_steps >= config.max_steps:
        raise ValueError("profiling warmup_steps must be less than max_steps")
    if not microbatches or not seeds:
        raise ValueError("microbatches and seeds must not be empty")
    if len(set(microbatches)) != len(microbatches) or len(set(seeds)) != len(seeds):
        raise ValueError("duplicate microbatches or seeds would repeat the same case")
    for microbatch in microbatches:
        _positive_int(microbatch, "microbatch")
        if effective_batch_size % microbatch:
            raise ValueError("every microbatch must divide effective_batch_size")
    selected = None
    if policies is not None:
        if not policies:
            raise ValueError("policies must not be empty")
        selected = set()
        for policy in policies:
            match = (re.fullmatch(r"(eager|sdpa):(none|block):([1-9][0-9]*)", policy)
                     if isinstance(policy, str) else None)
            if match is None:
                raise ValueError("policies must use backend:checkpoint:microbatch, "
                                 "for example sdpa:block:8")
            backend, checkpointing, batch = match.groups()
            microbatch = int(batch)
            if microbatch not in microbatches:
                raise ValueError("every policy microbatch must belong to microbatches")
            treatment = (backend, checkpointing, microbatch)
            if treatment in selected:
                raise ValueError("duplicate policies would repeat the same case")
            selected.add(treatment)
    for seed in seeds:
        if type(seed) is not int or not 0 <= seed < 2**63 - 1:
            raise ValueError("seeds must be integers in [0, 2**63 - 1)")
    if type(order_seed) is not int:
        raise ValueError("order_seed must be an integer")
    if device not in {"cuda", "cpu"}:
        raise ValueError("pilot device must be cuda, or cpu for an explicit harness check")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) \
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    # A plan's nested configs must keep the Python API's data paths meaningful.
    config = replace(config, data=replace(
        config.data,
        train_path=str(Path(config.data.train_path).resolve()) if config.data.train_path else None,
        valid_path=str(Path(config.data.valid_path).resolve()) if config.data.valid_path else None,
        tokenizer_files=tuple(str(Path(path).resolve()) for path in config.data.tokenizer_files),
    ))
    source = source_identity()
    cases = []
    for seed in sorted(seeds):
        for backend in ("eager", "sdpa"):
            for checkpointing in ("none", "block"):
                for microbatch in sorted(microbatches):
                    if selected is not None and (backend, checkpointing, microbatch) not in selected:
                        continue
                    accumulation = effective_batch_size // microbatch
                    resolved = replace(
                        config, seed=seed, device=device, batch_size=microbatch,
                        grad_accum_steps=accumulation,
                        precision=config.precision if precision is None else precision,
                        model=replace(config.model, attention_backend=backend,
                                      activation_checkpointing=checkpointing),
                    )
                    resolved.validate()
                    case_id = f"{backend}-{checkpointing}-mb{microbatch}-acc{accumulation}-seed{seed}"
                    config_dict = resolved.to_dict()
                    cases.append({
                        "case_id": case_id,
                        "labels": {"attention_backend": backend,
                                   "activation_checkpointing": checkpointing,
                                   "batch_size": microbatch, "grad_accum_steps": accumulation,
                                   "seed": seed, "device": device, "precision": resolved.precision},
                        "config": config_dict, "config_sha256": _sha256(config_dict),
                        "source_sha256": source["python_source_sha256"],
                        "max_steps": resolved.max_steps,
                        "tokens_per_step": effective_batch_size * resolved.model.context_length,
                        "config_path": f"configs/{case_id}.json",
                        "run_path": f"runs/{case_id}",
                    })
    random.Random(order_seed).shuffle(cases)
    plan = {
        "format_version": 1, "source": source,
        "effective_batch_size": effective_batch_size, "warmup_steps": warmup_steps,
        "order_seed": order_seed, "timeout_seconds": timeout_seconds,
        "attention_dispatch": "sdpa uses automatic PyTorch dispatch; no particular kernel is guaranteed",
        "warmup_scope": "exclude optimizer steps <= warmup_steps from timing statistics; keep the training schedule unchanged",
        "cases": cases,
    }
    plan["plan_sha256"] = _sha256(plan)
    return plan


def write_plan(plan: dict, output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    for case in plan["cases"]:
        _write_json(root / case["config_path"], case["config"])
    _write_json(root / "manifest.json", plan)
    return root


def _finite(value, *, positive: bool = False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def _p90(values: list[float]) -> float:
    values = sorted(values)
    position = 0.9 * (len(values) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def summarize_case(root: str | Path, case: dict, warmup_steps: int) -> dict:
    """Summarize one case, retaining failures and partial/missing output explicitly."""
    root = Path(root)
    run_dir = root / "runs" / case["case_id"]
    row = {"case_id": case["case_id"], **case["labels"],
           "config_sha256": case["config_sha256"], "status": "completed",
           "effective_batch_size": case["labels"]["batch_size"] * case["labels"]["grad_accum_steps"],
           "tokens_per_step": case["tokens_per_step"], "tokens_seen": None,
           "data_identity": None, "data_identity_sha256": None, "batch_stream_sha256": None,
           "measured_steps": 0, "expected_measured_steps": case["max_steps"] - warmup_steps,
           "step_seconds_median": None, "step_seconds_p90": None,
           "tokens_per_second": None, "last_train_loss": None, "valid_loss": None,
           "runner_times": {}, "subprocess_wall_seconds": None,
           "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
    errors = []
    execution = {}
    summary = {}
    failed = False

    def problem(status: str, message: str) -> None:
        priority = {"completed": 0, "insufficient_samples": 1, "missing_output": 2, "invalid_output": 3}
        if not failed and priority[status] > priority[row["status"]]:
            row["status"] = status
        errors.append(message)

    def number(value, name: str, *, required: bool = False, nonnegative: bool = False):
        if value is None and not required:
            return None
        if not _finite(value) or (nonnegative and value < 0):
            problem("invalid_output", f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
            return None
        return value

    try:
        execution_path = root / "execution" / f"{case['case_id']}.json"
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            row["returncode"] = execution.get("returncode")
            if execution.get("status") in {"failed", "timeout"} or execution.get("returncode") not in {None, 0}:
                row["status"] = "timeout" if execution.get("status") == "timeout" else "failed"
                failed = True
                errors.append(execution.get("error") or f"process exited with code {execution.get('returncode')}")
            elif execution.get("status") != "completed":
                problem("insufficient_samples", "subprocess completion was not recorded")
            row["subprocess_wall_seconds"] = number(execution.get("wall_seconds"), "subprocess wall time", nonnegative=True)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("status") == "failed":
                if not failed:
                    row["status"] = "failed"
                failed = True
                errors.append(summary.get("error", "runner failed"))
            elif summary.get("status") != "completed":
                problem("insufficient_samples", "runner did not complete its planned updates")
            else:
                if type(summary.get("step")) is not int or summary["step"] != case["max_steps"]:
                    problem("invalid_output", "completed summary step does not match the plan")
                if (type(summary.get("tokens_seen")) is not int
                        or summary["tokens_seen"] != case["max_steps"] * case["tokens_per_step"]):
                    problem("invalid_output", "completed summary tokens_seen does not match the plan")
            row["tokens_seen"] = number(summary.get("tokens_seen"), "summary tokens_seen", nonnegative=True)
            row["valid_loss"] = number(summary.get("valid_loss"), "validation loss", required=summary.get("status") == "completed")
            row["runner_times"] = {key: number(summary.get(key), key, nonnegative=True) for key in (
                "setup_seconds", "train_seconds", "eval_seconds", "checkpoint_seconds", "wall_seconds")}
            for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
                row[key] = number(summary.get(key), key, nonnegative=True)
        else:
            problem("missing_output", "summary.json is missing")

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("config_sha256") != case["config_sha256"]:
                problem("invalid_output", "metadata config identity does not match the plan")
            if metadata.get("source", {}).get("python_source_sha256") != case["source_sha256"]:
                problem("invalid_output", "metadata source identity does not match the plan")
            identity = metadata.get("data_identity")
            if isinstance(identity, dict) and identity:
                row["data_identity_sha256"] = _sha256(identity)
                row["data_identity"] = identity
            row["gpu_name"] = metadata.get("gpu_name")
            row["gpu_count"] = number(metadata.get("gpu_count"), "gpu_count", nonnegative=True)
            row["workload"] = metadata.get("workload")
        else:
            problem("missing_output", "metadata.json is missing")

        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            problem("missing_output", "metrics.jsonl is missing")
            records = []
        else:
            records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        train = [record for record in records if record.get("event") == "train"]
        for index, record in enumerate(train, 1):
            if type(record.get("step")) is not int or record["step"] != index or index > case["max_steps"]:
                raise ValueError("training steps must be a unique consecutive sequence starting at 1")
            if not _finite(record.get("step_seconds"), positive=True) or not _finite(record.get("loss")):
                raise ValueError("training duration must be positive and finite; loss must be finite")
            if type(record.get("tokens_seen")) is not int or record["tokens_seen"] != index * case["tokens_per_step"]:
                raise ValueError("tokens_seen does not match the planned effective batch")
        measured = [record for record in train if record["step"] > warmup_steps]
        row["measured_steps"] = len(measured)
        if measured:
            durations = [record["step_seconds"] for record in measured]
            row["step_seconds_median"] = statistics.median(durations)
            row["step_seconds_p90"] = _p90(durations)
            row["tokens_per_second"] = case["tokens_per_step"] * len(measured) / sum(durations)
        if train:
            row["last_train_loss"] = train[-1]["loss"]
        hashes = [record.get("batch_start_sha256") for record in train]
        row["batch_stream_sha256"] = (
            hashlib.sha256("".join(hashes).encode()).hexdigest()
            if len(hashes) == case["max_steps"]
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes) else None
        )
        if len(measured) < row["expected_measured_steps"] and row["status"] == "completed":
            problem("insufficient_samples", "fewer post-warmup training rows than planned")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        problem("invalid_output", str(error))
    if errors:
        row["error"] = "; ".join(str(error) for error in errors)
    if row["status"] == "timeout":
        row["failure_kind"] = "timeout"
    elif row["status"] == "failed":
        log_path = root / "logs" / f"{case['case_id']}.log"
        try:
            log_tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
        except OSError:
            log_tail = ""
        text = row.get("error", "") + log_tail
        row["failure_kind"] = "oom" if "out of memory" in text.lower() or "outofmemoryerror" in text.lower() else "process_error"
    return row


def aggregate_results(root: str | Path, plan: dict) -> dict:
    rows = [summarize_case(root, case, plan["warmup_steps"]) for case in plan["cases"]]
    groups = []
    for seed in sorted({row["seed"] for row in rows}):
        group_rows = [row for row in rows if row["seed"] == seed]
        complete = [row for row in group_rows if row["status"] == "completed"]
        reasons = []
        for key in ("data_identity_sha256", "batch_stream_sha256"):
            if any(row[key] is None for row in complete):
                reasons.append(f"completed cases lack valid {key} evidence")
            if len({row[key] for row in complete if row[key] is not None}) > 1:
                reasons.append(f"completed cases disagree on {key}")
        status = "failed" if reasons else "passed"
        if len(complete) != len(group_rows):
            reasons.append("not all planned cases completed with valid output")
            if status != "failed":
                status = "insufficient"
        groups.append({"seed": seed, "status": status, "reasons": reasons,
                       "case_ids": [row["case_id"] for row in group_rows]})
    comparability_status = ("failed" if any(group["status"] == "failed" for group in groups)
                            else "insufficient" if any(group["status"] != "passed" for group in groups)
                            else "passed")
    return {"format_version": 1, "plan_sha256": plan["plan_sha256"],
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "comparability": {"status": comparability_status, "seed_groups": groups,
                              "reasons": [f"seed {group['seed']}: {reason}" for group in groups for reason in group["reasons"]]},
            "timing_scope": "per-case complete optimizer-step durations after profiling warmup; includes batch preparation and transfer, excludes evaluation/checkpoint time",
            "throughput_scope": "measured valid tokens divided by summed measured step durations",
            "percentile_method": "linear interpolation at 0.9 * (sample_count - 1)",
            "interpretation": "descriptive pilot measurements; CPU/synthetic cases are harness checks; no speedup or quality-equivalence claim",
            "cases": rows}


def execute_plan(output_dir: str | Path, *, allow_cpu: bool = False) -> dict:
    root = Path(output_dir).resolve()
    plan = json.loads((root / "manifest.json").read_text())
    identity = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("format_version") != 1 or _sha256(identity) != plan.get("plan_sha256"):
        raise ValueError("manifest version or identity is invalid")
    if source_identity()["python_source_sha256"] != plan["source"]["python_source_sha256"]:
        raise ValueError("source changed after planning; inspect and create a new plan")
    for case in plan["cases"]:
        if not re.fullmatch(r"[a-z0-9_-]+", case["case_id"]):
            raise ValueError("invalid case identifier")
        config_path = root / "configs" / f"{case['case_id']}.json"
        resolved = load_config(config_path)
        if _sha256(resolved.to_dict()) != case["config_sha256"]:
            raise ValueError(f"resolved config changed for {case['case_id']}")
        if resolved.device == "cpu":
            if not allow_cpu:
                raise ValueError("CPU execution requires --allow-cpu and is a harness check only")
        elif resolved.device != "cuda":
            raise ValueError("pilot execution supports CUDA or explicitly allowed CPU only")
        elif not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no pilot cases were started")
    for directory in ("runs", "logs", "execution"):
        (root / directory).mkdir(exist_ok=True)
    package = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(package.parent) + os.pathsep + environment.get("PYTHONPATH", "")
    for case in plan["cases"]:
        case_id = case["case_id"]
        execution_path = root / "execution" / f"{case_id}.json"
        # Re-entering an interrupted plan only runs untouched cases. Never overwrite evidence.
        if execution_path.exists() or (root / "runs" / case_id).exists():
            continue
        command = [sys.executable, "-m", "llmlab.train", "--config",
                   str(root / "configs" / f"{case_id}.json"), "--output-dir",
                   str(root / "runs" / case_id)]
        result = {"status": "running", "command": command, "returncode": None}
        _write_json(execution_path, result)
        started = time.perf_counter()
        with (root / "logs" / f"{case_id}.log").open("w") as log:
            try:
                completed = subprocess.run(command, cwd=package.parent.parent, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           timeout=plan["timeout_seconds"], check=False)
                result.update(status="completed" if completed.returncode == 0 else "failed",
                              returncode=completed.returncode)
                if completed.returncode:
                    result["error"] = f"process exited with code {completed.returncode}; see logs/{case_id}.log"
            except subprocess.TimeoutExpired:
                result.update(status="timeout", error=f"case exceeded {plan['timeout_seconds']} seconds")
            except OSError as error:
                result.update(status="failed", error=str(error))
        result["wall_seconds"] = time.perf_counter() - started
        _write_json(execution_path, result)
        _write_json(root / "aggregate.json", aggregate_results(root, plan))
    aggregate = aggregate_results(root, plan)
    _write_json(root / "aggregate.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--run-plan", type=Path, help="execute a previously inspected plan")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--microbatches", nargs="+", type=int)
    parser.add_argument("--policies", nargs="+",
                        help="select backend:checkpoint:microbatch combinations; default is the full matrix")
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--warmup-steps", type=int, help="timing exclusion only (default: 5); does not change LR warmup")
    parser.add_argument("--order-seed", type=int, help="case-order seed (default: 0)")
    parser.add_argument("--timeout-seconds", type=float, help="per-case subprocess timeout (default: 600)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="pilot device (default: cuda)")
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="permit CPU execution for harness checks")
    args = parser.parse_args()
    if args.run_plan is not None:
        if any(value is not None for value in (args.output_dir, args.microbatches,
                                               args.effective_batch_size, args.seeds, args.precision,
                                               args.warmup_steps, args.order_seed, args.timeout_seconds,
                                               args.device, args.policies)) or args.execute:
            parser.error("--run-plan uses the saved configuration; only --allow-cpu may accompany it")
        aggregate = execute_plan(args.run_plan, allow_cpu=args.allow_cpu)
        print(json.dumps(aggregate["status_counts"], sort_keys=True))
        return
    if args.output_dir is None or args.microbatches is None or args.effective_batch_size is None:
        parser.error("planning requires --output-dir, --microbatches, and --effective-batch-size")
    config = load_config(args.config)
    plan = make_plan(config, microbatches=args.microbatches,
                     effective_batch_size=args.effective_batch_size,
                     seeds=[config.seed] if args.seeds is None else args.seeds,
                     warmup_steps=5 if args.warmup_steps is None else args.warmup_steps,
                     device="cuda" if args.device is None else args.device,
                     precision=args.precision, order_seed=0 if args.order_seed is None else args.order_seed,
                     timeout_seconds=600 if args.timeout_seconds is None else args.timeout_seconds,
                     policies=args.policies)
    root = write_plan(plan, args.output_dir)
    print(f"Planned {len(plan['cases'])} cases: {root / 'manifest.json'}")
    if args.execute:
        aggregate = execute_plan(root, allow_cpu=args.allow_cpu)
        print(json.dumps(aggregate["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
