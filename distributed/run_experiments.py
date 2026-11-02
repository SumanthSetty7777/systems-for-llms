"""Plan and execute matched-batch distributed comparisons inside one GPU host."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "tiny": {"vocab_size": 128, "d_model": 64, "num_layers": 2, "num_heads": 4, "d_ff": 128},
    "100m": {"vocab_size": 10000, "d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 2048},
    "324m": {"vocab_size": 10000, "d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 2752},
}


def plan_hash(plan):
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "plan_sha256"},
                                    sort_keys=True).encode()).hexdigest()


def build_plan(stage):
    cases = []

    def add(mode, world, *, model="324m", context=1024, global_batch=32, microbatch=4,
            checkpointing="none", sync="last", bucket=25, seed=1337, steps=60, precision="bf16"):
        case_id = f"{model}-s{context}-{mode}-w{world}-mb{microbatch}-{checkpointing}-{sync}-b{bucket}-{precision}-seed{seed}"
        cases.append({"case_id": case_id, "mode": mode, "world_size": world, "model": model,
                      "context_length": context, "global_batch_size": global_batch,
                      "microbatch_size": microbatch, "checkpointing": checkpointing,
                      "sync_policy": sync, "bucket_cap_mb": bucket, "seed": seed,
                      "steps": steps, "precision": precision, "warmup_steps": 1 if model == "tiny" else 5,
                      "timeout_seconds": 120 if model == "tiny" else 240})

    if stage == "smoke":
        for mode in ("single", "naive", "ddp", "zero1", "fsdp", "tp"):
            add(mode, 1 if mode == "single" else 2, model="tiny", context=32,
                global_batch=8, microbatch=2, steps=5, precision="fp32")
        for mode in ("single", "ddp", "zero1", "tp"):
            add(mode, 1 if mode == "single" else 2, model="tiny", context=32,
                global_batch=8, microbatch=2, steps=5, checkpointing="block", precision="bf16")
    elif stage == "screen":
        for model in ("100m", "324m"):
            for checkpointing in ("none", "block"):
                for mode in ("single", "ddp", "zero1", "fsdp", "tp"):
                    add(mode, 1 if mode == "single" else 2, model=model,
                        global_batch=8, checkpointing=checkpointing, steps=30)
    elif stage == "scale":
        for checkpointing in ("none", "block"):
            add("single", 1, checkpointing=checkpointing)
            for mode in ("ddp", "zero1", "fsdp", "tp"):
                for world in (2, 4, 8):
                    add(mode, world, checkpointing=checkpointing)
        add("naive", 8)
        add("ddp", 8, microbatch=1)
        add("ddp", 8, microbatch=1, sync="every")
        for bucket in (1, 100):
            add("ddp", 8, microbatch=1, bucket=bucket)
        for checkpointing in ("none", "block"):
            for mode in ("single", "ddp"):
                add(mode, 1 if mode == "single" else 8, context=4096, global_batch=8,
                    microbatch=1, checkpointing=checkpointing)
        for mode in ("zero1", "fsdp", "tp"):
            add(mode, 8, context=4096, global_batch=8, microbatch=1, checkpointing="block")
    elif stage == "confirmation":
        for seed in (2027, 2028, 2029):
            for mode in ("single", "ddp", "zero1", "fsdp", "tp"):
                add(mode, 1 if mode == "single" else 8, seed=seed)
            # Repeat the checkpointed memory comparison on the same seeds.
            for mode in ("ddp", "fsdp"):
                add(mode, 8, checkpointing="block", seed=seed)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # A fixed order is recorded before launch; repetitions use separate seeds.
    random.Random(336).shuffle(cases)
    plan = {"format_version": 1, "stage": stage, "cases": cases, "case_count": len(cases),
            "required_gpus": max(case["world_size"] for case in cases), "order_seed": 336,
            "comparability": "Same model/context/seed has identical global batch, optimizer schedule, validation windows and ordered examples; rank-local microbatch and parallelism differ",
            "cost_scope": "Active GPU-seconds describe algorithmic resource use; all allocated GPUs remain billable during lower-world-size baselines",
            "interpretation": "Fixed-global-batch strong scaling; short screens do not establish time to convergence or a quality-equivalence claim",
            "failure_policy": "Preserve the first failure and stop; subsequent cases remain not_run. No automatic retry."}
    validate_plan(plan)
    plan["plan_sha256"] = plan_hash(plan)
    return plan


def validate_plan(plan):
    if plan.get("plan_sha256") and plan["plan_sha256"] != plan_hash(plan):
        raise ValueError("Distributed plan hash mismatch")
    cases = plan["cases"]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Plan must have distinct nonempty cases")
    layouts = {}
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["case_id"]):
            raise ValueError("Case identifier must be safe and nonempty")
        world, batch, micro = (case[key] for key in ("world_size", "global_batch_size", "microbatch_size"))
        if any(type(n) is not int for n in (world, batch, micro)) or world not in (1, 2, 4, 8) or batch <= 0 or micro <= 0:
            raise ValueError("Invalid rank or batch layout")
        if case["mode"] not in {"single", "naive", "ddp", "zero1", "fsdp", "tp"}:
            raise ValueError("Unknown parallelism mode")
        if (case["mode"] == "single") != (world == 1):
            raise ValueError("Single mode and one-rank layout must coincide")
        if batch % (micro * (1 if case["mode"] == "tp" else world)):
            raise ValueError("Global batch must divide across ranks and microbatches")
        model = MODELS[case["model"]]
        if case["mode"] == "tp" and (model["num_heads"] % world or model["d_ff"] % world):
            raise ValueError("TP requires divisible attention heads and feed-forward width")
        if case["sync_policy"] not in {"last", "every"} or (case["sync_policy"] == "every" and case["mode"] not in {"ddp", "zero1"}):
            raise ValueError("Invalid synchronization policy")
        if case["checkpointing"] not in {"none", "block"} or case["precision"] not in {"fp32", "bf16"}:
            raise ValueError("Invalid checkpointing or precision policy")
        if not math.isfinite(case["bucket_cap_mb"]) or case["bucket_cap_mb"] <= 0:
            raise ValueError("Bucket capacity must be positive and finite")
        if type(case["context_length"]) is not int or case["context_length"] <= 0 or case["timeout_seconds"] <= 0:
            raise ValueError("Invalid context or timeout")
        if not 0 <= case["warmup_steps"] < case["steps"]:
            raise ValueError("Invalid timing warmup")
        group = (case["model"], case["context_length"], case["seed"], case["precision"])
        layout = (batch, case["steps"], case["warmup_steps"])
        if group in layouts and layouts[group] != layout:
            raise ValueError("Comparable policies must share global batch, steps and timing warmup")
        layouts[group] = layout
    if plan["case_count"] != len(cases) or plan["required_gpus"] != max(case["world_size"] for case in cases):
        raise ValueError("Plan case or GPU count mismatch")


def case_config(base, case, device="cuda"):
    config = deepcopy(base)
    config["model"].update(MODELS[case["model"]])
    config["model"].update(context_length=case["context_length"], attention_backend="sdpa",
                           activation_checkpointing=case["checkpointing"])
    config.update(seed=case["seed"], device=device, precision=case["precision"], num_threads=2,
                  batch_size=case["global_batch_size"], grad_accum_steps=1,
                  max_steps=case["steps"], warmup_steps=min(5, case["steps"] - 1),
                  eval_interval=case["steps"], eval_batches=8, eval_batch_size=1,
                  checkpoint_interval=case["steps"])
    if case["model"] == "tiny":
        config["data"] = {"kind": "synthetic", "synthetic_tokens": 4096}
    return config


def command_for(case, config_path, destination, device="cuda"):
    command = [sys.executable]
    if case["world_size"] > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc-per-node={case['world_size']}", "--max-restarts=0", "--module", "llmdist.train"]
    else:
        command += ["-m", "llmdist.train"]
    command += ["--config", str(config_path), "--output-dir", str(destination),
                "--mode", case["mode"], "--global-batch-size", str(case["global_batch_size"]),
                "--microbatch-size", str(case["microbatch_size"]), "--warmup-steps", str(case["warmup_steps"]),
                "--sync-policy", case["sync_policy"], "--bucket-cap-mb", str(case["bucket_cap_mb"]),
                "--device", device]
    return command


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect_case(path, case):
    summary = json.loads((path / "summary.json").read_text())
    if summary["status"] != "completed" or summary["step"] != case["steps"]:
        raise ValueError("Case did not complete all planned updates")
    metadata, streams = [], []
    for rank in range(case["world_size"]):
        meta = json.loads((path / f"rank-{rank}-metadata.json").read_text())
        rows = [json.loads(line) for line in (path / f"rank-{rank}-metrics.jsonl").read_text().splitlines()]
        rank_summary = json.loads((path / f"rank-{rank}-summary.json").read_text())
        train = [row for row in rows if row["event"] == "train"]
        if [row["step"] for row in train] != list(range(1, case["steps"] + 1)) or rank_summary["status"] != "completed":
            raise ValueError("Missing rank training updates or summary")
        for key in ("mode", "world_size", "global_batch_size", "microbatch_size", "sync_policy", "bucket_cap_mb"):
            if meta[key] != case[key]:
                raise ValueError(f"Executed layout differs from the plan: {key}")
        if meta["rank"] != rank or meta["timing_warmup_steps"] != case["warmup_steps"] or meta["profile_steps"] != 0:
            raise ValueError("Executed rank, warmup or profiling differs from the plan")
        config = meta["config"]
        expected_model = dict(MODELS[case["model"]], context_length=case["context_length"],
                              activation_checkpointing=case["checkpointing"], attention_backend="sdpa")
        if any(config["model"][key] != value for key, value in expected_model.items()):
            raise ValueError("Executed model or checkpointing differs from the plan")
        if any(config[key] != case[key] for key in ("seed", "precision")) or config["max_steps"] != case["steps"]:
            raise ValueError("Executed seed, precision or step horizon differs from the plan")
        tokens = case["global_batch_size"] * case["context_length"]
        if any(row["tokens_seen"] != row["step"] * tokens for row in train):
            raise ValueError("Training token accounting differs from the plan")
        for row in train:
            if not all(math.isfinite(row[key]) for key in ("loss", "gradient_norm", "step_seconds", "learning_rate")) or row["step_seconds"] <= 0:
                raise ValueError("Nonfinite training metric or invalid step time")
        actual_hash = hashlib.sha256("".join(row["batch_start_sha256"] for row in train).encode()).hexdigest()
        common = ("status", "mode", "world_size", "global_batch_size", "microbatch_size", "step",
                  "tokens_seen", "tokens_per_step", "measured_steps", "batch_stream_sha256")
        if any(rank_summary[key] != summary[key] for key in common) or actual_hash != summary["batch_stream_sha256"]:
            raise ValueError("Rank summary or batch-stream digest differs from actual metrics")
        streams.append([(row["batch_start_sha256"], row["learning_rate"], row["tokens_seen"]) for row in train])
        metadata.append(meta)
    for key in ("mode", "world_size", "global_batch_size", "microbatch_size", "sync_policy"):
        if summary[key] != case[key]:
            raise ValueError(f"Summary differs from the plan: {key}")
    measured = [row["step_seconds"] for row in train if row["step"] > case["warmup_steps"]]
    if (summary["tokens_seen"] != case["steps"] * tokens or summary["tokens_per_step"] != tokens
            or summary["measured_steps"] != len(measured)
            or not math.isclose(summary["measured_train_seconds"], sum(measured), rel_tol=1e-10)
            or not math.isclose(summary["tokens_per_second"], len(measured) * tokens / sum(measured), rel_tol=1e-10)
            or not math.isfinite(summary["valid_loss"])):
        raise ValueError("Summary timing, token accounting or validation loss is invalid")
    if any(stream != streams[0] for stream in streams):
        raise ValueError("Global batch or learning-rate stream differs across ranks")
    for key in ("data_identity", "source", "initial_parameter_sha256"):
        if any(meta[key] != metadata[0][key] for meta in metadata):
            raise ValueError(f"Rank provenance mismatch: {key}")
    signature = {key: metadata[0][key] for key in ("data_identity", "source", "initial_parameter_sha256")}
    signature["batch_stream_sha256"] = summary["batch_stream_sha256"]
    signature["learning_rate_stream_sha256"] = hashlib.sha256(json.dumps(streams[0]).encode()).hexdigest()
    return summary, signature


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=3)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def add_scaling_metrics(rows, allocated_gpus):
    baselines = {(r["model"], r["context_length"], r["seed"], r["precision"], r["checkpointing"]): r
                 for r in rows if r["status"] == "completed" and r["mode"] == "single"}
    for row in rows:
        if "subprocess_seconds" in row:
            row["allocated_gpu_seconds_subprocess"] = allocated_gpus * row["subprocess_seconds"]
        if row["status"] != "completed":
            continue
        row["active_gpu_seconds_measured"] = row["world_size"] * row["measured_train_seconds"]
        baseline = baselines.get((row["model"], row["context_length"], row["seed"], row["precision"], row["checkpointing"]))
        if baseline:
            speedup = row["tokens_per_second"] / baseline["tokens_per_second"]
            row["throughput_speedup"] = speedup
            row["strong_scaling_efficiency"] = speedup / row["world_size"]
            row["baseline_case_id"] = baseline["case_id"]


def run_plan(plan, base, output, *, device="cuda", allocated_gpus=None):
    validate_plan(plan)
    allocated_gpus = plan["required_gpus"] if allocated_gpus is None else allocated_gpus
    if type(allocated_gpus) is not int or allocated_gpus < plan["required_gpus"]:
        raise ValueError("Allocated GPU count must cover all planned ranks")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("configs", "cases", "logs"):
        (output / folder).mkdir()
    save(output / "plan.json", plan)
    rows, signatures = [], {}
    aggregate = {"status": "running", "planned_count": len(plan["cases"]), "cases": rows,
                 "plan_sha256": plan["plan_sha256"], "allocated_gpus": allocated_gpus}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(REPO / "src"), str(REPO / "distributed"))),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    child = None
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminated(signum, frame):
        stop_child(child)
        raise TimeoutError("Outer job terminated; active worker group stopped")

    signal.signal(signal.SIGTERM, terminated)
    for index, case in enumerate(plan["cases"]):
        row = dict(case, status="running")
        rows.append(row)
        save(output / "aggregate.json", aggregate)
        config_path, destination = output / "configs" / (case["case_id"] + ".json"), output / "cases" / case["case_id"]
        save(config_path, case_config(base, case, device))
        log_path = output / "logs" / (case["case_id"] + ".log")
        print(f"[{index + 1}/{len(plan['cases'])}] {case['case_id']}", flush=True)
        started = time.monotonic()
        try:
            with log_path.open("w") as log:
                child = subprocess.Popen(command_for(case, config_path, destination, device),
                                         stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env, start_new_session=True)
                try:
                    code = child.wait(timeout=case["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    stop_child(child)
                    row["status"] = "timeout"
                    raise TimeoutError("Distributed case exceeded its limit") from None
            if code:
                tail = log_path.read_text(errors="replace")[-16000:]
                row["status"] = "oom" if "out of memory" in tail.lower() else "failed"
                raise RuntimeError(f"Worker exited {code}; see {log_path.name}")
            summary, signature = inspect_case(destination, case)
            group = (case["model"], case["context_length"], case["seed"], case["precision"])
            if group in signatures and signature != signatures[group]:
                raise ValueError("Cross-policy source, initialization, dataset or batch stream mismatch")
            signatures[group] = signature
            row.update(summary, comparability="passed")
        except Exception as error:
            if row["status"] == "running":
                row["status"] = "invalid"
            row["error"] = str(error)
            rows.extend(dict(later, status="not_run") for later in plan["cases"][index + 1:])
        finally:
            row["subprocess_seconds"] = time.monotonic() - started
            add_scaling_metrics(rows, allocated_gpus)
            aggregate["status"] = "completed" if len(rows) == len(plan["cases"]) and all(r["status"] == "completed" for r in rows) else "running"
            if row["status"] != "completed":
                aggregate["status"] = "failed"
            save(output / "aggregate.json", aggregate)
        if row["status"] != "completed":
            break
    signal.signal(signal.SIGTERM, previous_handler)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "screen", "scale", "confirmation"], default="smoke")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--allocated-gpus", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text()) if args.plan else build_plan(args.stage)
    validate_plan(plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    result = run_plan(plan, json.loads(args.config.read_text()), args.output_dir,
                      device=args.device, allocated_gpus=args.allocated_gpus)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
