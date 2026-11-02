"""Run a bounded study stage sequentially; default execution only prints its plan."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_FILE = REPO / "results/2026-11-01-100m-quality-dev/frozen-target.json"
QUALITY_STAGES = {"quality-confirmation", "confirmation"}
spec = importlib.util.spec_from_file_location("study_launcher", REPO / "scripts/modal_pilot.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

QUALITY_PROTOCOL = {
    "workload": "100M/1K TinyStories",
    "policies": ["sdpa:none:8", "sdpa:none:1", "sdpa:block:8"],
    "development_seed": 1337,
    "confirmation_seeds": [2027, 2028, 2029],
    "max_steps": 1200,
    "tokens_per_step": 8192,
    "train_warmup_steps": 50,
    "eval_interval": 50,
    "eval_batches": 64,
    "eval_batch_size": 1,
    "checkpoint_interval": 1200,
    "allocated_memory_budget_bytes": 3 * 2**30,
    "memory_scope": "allocated memory only; does not establish fit on a physical 3 GiB GPU",
    "target_rule": "ceil(max(development validation losses at step 600) / 0.05) * 0.05",
    "target_step": 600,
    "target_rounding": "0.05",
    "confirmation_gate": "freeze numeric target and development evidence hashes before confirmation",
    "time_to_target_scope": "first observed eval wall_seconds; no interpolation; includes setup and preceding eval/checkpoints, excludes interpreter startup",
}


def build_plan(stage: str) -> dict:
    def job(name, config, microbatches, effective, *, seeds=None, case_timeout=120,
            job_timeout=1800, overrides=None, policies=None, allow_limits=False):
        return {"name": name, "config": f"configs/{config}.json",
                "microbatches": microbatches, "effective_batch_size": effective,
                "seeds": [1337] if seeds is None else seeds, "warmup_steps": 5,
                "case_timeout_seconds": case_timeout, "job_timeout_seconds": job_timeout,
                "overrides": overrides or {}, "policies": policies,
                "allow_resource_limits": allow_limits,
                "case_count": len(policies or microbatches) * (1 if policies else 4) * len(seeds or [1337])}

    if stage == "development":
        jobs = [
            job("22m-4k-screen", "scale-22m-s4096", [1, 2], 2, allow_limits=True),
            job("324m-1k-screen", "scale-324m-s1024", [1, 8], 8,
                case_timeout=180, job_timeout=2400, allow_limits=True),
            job("324m-4k-screen", "scale-324m-s4096", [1, 2], 2,
                case_timeout=300, job_timeout=3600, allow_limits=True),
            job("100m-1k-quality-dev", "scale-100m-s1024", [1, 8], 8,
                case_timeout=600, job_timeout=2400,
                policies=QUALITY_PROTOCOL["policies"],
                overrides={key: QUALITY_PROTOCOL[key] for key in
                           ("max_steps", "train_warmup_steps", "eval_interval",
                            "eval_batches", "checkpoint_interval")}),
        ]
    elif stage in {"timing-confirmation", *QUALITY_STAGES}:
        quality_jobs = [
            job(f"100m-1k-quality-seed{seed}", "scale-100m-s1024", [1, 8], 8,
                seeds=[seed], case_timeout=600, job_timeout=2400,
                policies=QUALITY_PROTOCOL["policies"],
                overrides={key: QUALITY_PROTOCOL[key] for key in
                           ("max_steps", "train_warmup_steps", "eval_interval",
                            "eval_batches", "checkpoint_interval")})
            for seed in QUALITY_PROTOCOL["confirmation_seeds"]
        ]
        timing_jobs = [
            job("100m-1k-timing", "scale-100m-s1024", [1, 8], 8,
                seeds=[2027, 2028, 2029], case_timeout=180,
                job_timeout=2400, overrides={"max_steps": 60, "checkpoint_interval": 60}),
            job("100m-4k-timing", "scale-100m-s4096", [1, 2], 2,
                seeds=[2027, 2028, 2029], case_timeout=180,
                job_timeout=2400, overrides={"max_steps": 60, "checkpoint_interval": 60}),
        ]
        jobs = (quality_jobs if stage == "quality-confirmation" else
                timing_jobs if stage == "timing-confirmation" else quality_jobs + timing_jobs)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    seconds = sum(job["job_timeout_seconds"] for job in jobs)
    rate = 0.000542 + 2 * 0.0000131 + 8 * 0.00000222
    return {"format_version": 1, "stage": stage, "gpu": "L40S",
            "concurrent_jobs": 1, "jobs": jobs, "quality_protocol": QUALITY_PROTOCOL,
            "case_count": sum(job["case_count"] for job in jobs),
            "max_function_seconds_sum": seconds,
            "nominal_requested_resources_usd_at_time_limits": round(seconds * rate, 4),
            "pricing_source": "https://modal.com/pricing",
            "pricing_checked": "2026-11-01",
            "cost_scope": "estimate at requested GPU/CPU/RAM only; excludes startup/build, excess usage, storage/egress and other apps; not a billing cap",
            "working_budget_usd": 50}


def prepare_job(job: dict, run_id: str):
    payload = launcher.prepare_payload(REPO / job["config"])
    payload["config"] = launcher.apply_config_overrides(payload["config"], job["overrides"])
    options = {key: job[key] for key in ("microbatches", "effective_batch_size", "seeds",
               "warmup_steps", "case_timeout_seconds", "job_timeout_seconds")}
    options.update(gpu="L40S", volume="systems-for-llms-results", run_id=run_id)
    if job["policies"] is not None:
        options["policies"] = job["policies"]
    launcher.validate_options(payload["config"], options)
    return payload, options


def source_snapshot(payloads: list[dict]) -> dict:
    # The data arrays are identified in every remote case. Lock local code/configs
    # across this stage so edits cannot silently change later uploaded jobs.
    files = {item["local"] for payload in payloads for item in payload["files"]
             if item["remote"].startswith("/project/")}
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(files)}


def validate_frozen_target(path: Path, repo: Path = REPO) -> dict:
    # Keep all planning and target validation usable without importing Torch or Modal.
    target_spec = importlib.util.spec_from_file_location("study_quality_target", repo / "scripts/quality_target.py")
    target_module = importlib.util.module_from_spec(target_spec)
    target_spec.loader.exec_module(target_module)
    return target_module.validate_frozen_target(path, repo=repo)


def target_snapshot(path: Path) -> dict:
    path = path.resolve()
    original = path.read_bytes()
    target = validate_frozen_target(path, repo=REPO)
    if path.read_bytes() != original:
        raise RuntimeError("Frozen quality target changed while it was being validated")
    return {"path": str(path), "sha256": hashlib.sha256(original).hexdigest(), "target": target}


def validate_quality_jobs(jobs: list[dict], prepared: list, target: dict) -> None:
    if target["quality_protocol"] != QUALITY_PROTOCOL:
        raise ValueError("Quality protocol differs from the frozen development protocol")

    def fixed_config(config):
        result = json.loads(json.dumps(config))
        for key in ("seed", "batch_size", "grad_accum_steps"):
            result.pop(key, None)
        result["model"].pop("activation_checkpointing", None)
        return result

    reference = fixed_config(target["reference_config"])
    for job, (payload, options) in zip(jobs, prepared):
        if job["policies"] is None:
            continue  # Timing runs deliberately use a separate, shorter protocol.
        if payload["packages"] != target["reference_payload_packages"]:
            raise ValueError(f"{job['name']} packages differ from the frozen development environment")
        if fixed_config(payload["config"]) != reference:
            raise ValueError(f"{job['name']} config differs from the frozen development reference")
        if (options.get("policies") != QUALITY_PROTOCOL["policies"]
                or options["microbatches"] != [1, 8]
                or options["effective_batch_size"] != 8
                or options["seeds"] != job["seeds"]
                or len(options["seeds"]) != 1
                or options["seeds"][0] not in QUALITY_PROTOCOL["confirmation_seeds"]
                or options["effective_batch_size"] * payload["config"]["model"]["context_length"]
                   != QUALITY_PROTOCOL["tokens_per_step"]):
            raise ValueError(f"{job['name']} options differ from the frozen quality protocol")


def classify_result(summary: dict, aggregate: dict, job: dict, evidence: dict | None = None) -> str:
    if summary.get("evidence_status") not in {"complete", "partial"}:
        return "blocked"
    cases = aggregate.get("cases", [])
    if len(cases) != job["case_count"]:
        return "blocked"
    if evidence is not None:
        if evidence.get("errors"):
            return "blocked"
        allowed_missing = {f"pilot/runs/{c['case_id']}/{filename}" for c in cases
                           if c.get("failure_kind") in {"oom", "timeout"}
                           for filename in ("metadata.json", "summary.json", "metrics.jsonl")}
        if set(evidence.get("missing_files", [])) - allowed_missing:
            return "blocked"
    comparability = aggregate.get("comparability", {}).get("status")
    if (summary.get("status") == "finished" and summary.get("evidence_status") == "complete"
            and comparability == "passed" and all(c.get("status") == "completed" for c in cases)):
        return "completed"
    if not job["allow_resource_limits"] or comparability not in {"passed", "insufficient"}:
        return "blocked"
    limited = [c for c in cases if c.get("status") != "completed"]
    if limited and all(c.get("status") in {"failed", "timeout"}
                       and c.get("failure_kind") in {"oom", "timeout"} for c in limited):
        return "completed_with_resource_limits"
    return "blocked"


def save(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "timing-confirmation", "quality-confirmation", "confirmation"], default="development")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-file", type=Path, default=DEFAULT_TARGET_FILE,
                        help="Validated frozen development target required by quality confirmation stages")
    args = parser.parse_args()
    plan = build_plan(args.stage)
    if args.stage in QUALITY_STAGES:
        plan["frozen_quality_target"] = target_snapshot(args.target_file)
    campaign_id = datetime.now(timezone.utc).strftime("study-%Y%m%dT%H%M%SZ")
    output = (args.output_dir or REPO / "runs" / (campaign_id + "-" + args.stage)).resolve()
    prepared = [prepare_job(job, f"{campaign_id}-{job['name']}") for job in plan["jobs"]]
    if args.stage in QUALITY_STAGES:
        validate_quality_jobs(plan["jobs"], prepared, plan["frozen_quality_target"]["target"])
    plan["source_sha256"] = source_snapshot([payload for payload, _ in prepared])
    plan["resolved_jobs"] = [{"name": job["name"], "config": payload["config"], "options": options}
                             for job, (payload, options) in zip(plan["jobs"], prepared)]
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", "output_dir": str(output),
                      "plan": plan}, indent=2), flush=True)
    if not args.execute:
        return
    if sys.version_info[:2] != (3, 12):
        parser.error("--execute requires Python 3.12 and modal==1.5.1")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "campaign.json", plan)
    state = {"status": "running", "stage": args.stage, "jobs": []}
    save(output / "campaign-summary.json", state)
    try:
        for index, (job, (payload, options)) in enumerate(zip(plan["jobs"], prepared), 1):
            if args.stage in QUALITY_STAGES:
                if target_snapshot(args.target_file) != plan["frozen_quality_target"]:
                    raise RuntimeError("Frozen quality target changed during this stage; remaining jobs were not launched")
            current = [prepare_job(item, resolved[1]["run_id"])
                       for item, resolved in zip(plan["jobs"], prepared)]
            if source_snapshot([item[0] for item in current]) != plan["source_sha256"]:
                raise RuntimeError("Project code/configs changed during this stage; remaining jobs were not launched")
            if args.stage in QUALITY_STAGES:
                validate_quality_jobs(plan["jobs"], current, plan["frozen_quality_target"]["target"])
            print(f"\n[{index}/{len(prepared)}] Starting {job['name']} ({job['case_count']} cases)", flush=True)
            destination = output / job["name"]
            summary = launcher.execute(payload, options, destination)
            aggregate_path = destination / "aggregate.json"
            aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {}
            evidence = json.loads((destination / "evidence.json").read_text())
            status = classify_result(summary, aggregate, job, evidence)
            state["jobs"].append({"name": job["name"], "status": status,
                                  "run_id": options["run_id"], "output_dir": str(destination),
                                  "function_seconds": summary.get("function_seconds"),
                                  "case_status_counts": summary.get("case_status_counts")})
            save(output / "campaign-summary.json", state)
            print(json.dumps(state["jobs"][-1], indent=2), flush=True)
            if status == "blocked":
                raise RuntimeError(f"{job['name']} failed a check; inspect its returned evidence before continuing")
        state["status"] = ("finished_with_resource_limits" if any(
            j["status"] == "completed_with_resource_limits" for j in state["jobs"]) else "finished")
    except Exception as error:
        state.update(status="blocked", error=str(error))
        save(output / "campaign-summary.json", state)
        raise
    save(output / "campaign-summary.json", state)
    print(f"\nStage finished. Results: {output}", flush=True)
    if args.stage == "development":
        print("Inspect development curves and freeze the target before launching quality confirmation.", flush=True)


if __name__ == "__main__":
    main()
