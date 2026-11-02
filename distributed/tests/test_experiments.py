"""Plan/evidence audits plus a real, network-free single-process execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_experiments.py"
spec = importlib.util.spec_from_file_location("distributed_experiments_test", SCRIPT)
experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiments)


@pytest.mark.parametrize("stage", ["smoke", "screen", "scale", "confirmation"])
def test_plans_are_repeatable_and_use_valid_matched_global_batches(stage):
    plan = experiments.build_plan(stage)
    assert plan == experiments.build_plan(stage)
    experiments.validate_plan(plan)
    assert plan["plan_sha256"] == experiments.plan_hash(plan)
    groups = {}
    for case in plan["cases"]:
        group = (case["model"], case["context_length"], case["seed"], case["precision"])
        layout = (case["global_batch_size"], case["steps"])
        assert groups.setdefault(group, layout) == layout
        divisor = case["microbatch_size"] * (1 if case["mode"] == "tp" else case["world_size"])
        assert case["global_batch_size"] % divisor == 0
        if stage == "scale":
            assert case["global_batch_size"] * case["context_length"] == 32768
    if stage in {"scale", "confirmation"}:
        assert plan["required_gpus"] == 8


def test_confirmation_pairs_scaling_and_checkpointed_memory_policies_on_fresh_seeds():
    plan = experiments.build_plan("confirmation")
    assert plan["case_count"] == 21
    assert {case["seed"] for case in plan["cases"]} == {2027, 2028, 2029}
    policies = {("single", 1, "none"), ("ddp", 8, "none"), ("zero1", 8, "none"),
                ("fsdp", 8, "none"), ("tp", 8, "none"),
                ("ddp", 8, "block"), ("fsdp", 8, "block")}
    for seed in (2027, 2028, 2029):
        paired = [case for case in plan["cases"] if case["seed"] == seed]
        assert {(case["mode"], case["world_size"], case["checkpointing"]) for case in paired} == policies
        # Baseline and memory pairs differ only in the declared systems policies.
        for case in paired:
            assert (case["model"], case["context_length"], case["global_batch_size"],
                    case["microbatch_size"], case["steps"], case["warmup_steps"],
                    case["precision"], case["sync_policy"], case["bucket_cap_mb"]) == (
                        "324m", 1024, 32, 4, 60, 5, "bf16", "last", 25)


@pytest.mark.parametrize("field,value", [
    ("case_id", "../escape"), ("sync_policy", "unknown"),
    ("checkpointing", "unknown"), ("bucket_cap_mb", 0),
    ("precision", "fp8"), ("global_batch_size", 16),
])
def test_invalid_plan_controls_or_unmatched_group_are_rejected(field, value):
    plan = experiments.build_plan("smoke")
    plan.pop("plan_sha256")
    case = next(item for item in plan["cases"] if item["mode"] == "ddp" and item["precision"] == "fp32")
    case[field] = value
    with pytest.raises(ValueError):
        experiments.validate_plan(plan)


def test_plan_hash_detects_a_changed_policy():
    plan = experiments.build_plan("scale")
    plan["cases"][0]["seed"] += 1
    with pytest.raises(ValueError, match="hash"):
        experiments.validate_plan(plan)


def test_commands_use_runner_cli_and_requested_world_size():
    case = next(item for item in experiments.build_plan("scale")["cases"]
                if item["mode"] == "ddp" and item["world_size"] == 8)
    command = experiments.command_for(case, Path("config.json"), Path("results"))
    assert "torch.distributed.run" in command and "--nproc-per-node=8" in command
    assert command[command.index("--module") + 1] == "llmdist.train"
    for flag, expected in (("--mode", "ddp"), ("--global-batch-size", "32"),
                           ("--microbatch-size", str(case["microbatch_size"])),
                           ("--sync-policy", case["sync_policy"]), ("--device", "cuda")):
        assert command[command.index(flag) + 1] == expected
    single = dict(case, mode="single", world_size=1)
    command = experiments.command_for(single, "config.json", "results", device="cpu")
    assert "torch.distributed.run" not in command
    assert command[1:3] == ["-m", "llmdist.train"]
    assert command[command.index("--device") + 1] == "cpu"


@pytest.fixture(scope="module")
def real_single_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("experiment-integration") / "study"
    plan = experiments.build_plan("smoke")
    case = next(item for item in plan["cases"] if item["mode"] == "single" and item["precision"] == "fp32")
    plan.update(cases=[case], case_count=1, required_gpus=1)
    plan["plan_sha256"] = experiments.plan_hash(plan)
    base = json.loads((SCRIPT.parents[1] / "configs/smoke.json").read_text())
    aggregate = experiments.run_plan(plan, base, output, device="cpu", allocated_gpus=1)
    assert aggregate["status"] == "completed", aggregate
    return output / "cases" / case["case_id"], case, aggregate


def test_actual_single_cpu_artifacts_pass_full_inspection(real_single_run):
    path, case, aggregate = real_single_run
    summary, signature = experiments.inspect_case(path, case)
    assert summary["status"] == "completed"
    assert summary["tokens_seen"] == case["steps"] * case["global_batch_size"] * case["context_length"]
    assert summary["tokens_per_step"] == case["global_batch_size"] * case["context_length"]
    assert summary["measured_steps"] == case["steps"] - case["warmup_steps"]
    assert aggregate["cases"][0]["comparability"] == "passed"
    assert signature["batch_stream_sha256"] == summary["batch_stream_sha256"]
    metadata = json.loads((path / "rank-0-metadata.json").read_text())
    assert metadata["device"] == "cpu"
    assert metadata["config"]["model"]["activation_checkpointing"] == case["checkpointing"]
    assert summary["tokens_per_second"] > 0


@pytest.mark.parametrize("filename", ["rank-0-metadata.json", "rank-0-metrics.jsonl", "rank-0-summary.json"])
def test_missing_rank_evidence_is_rejected(tmp_path, real_single_run, filename):
    original, case, _ = real_single_run
    path = tmp_path / "case"
    shutil.copytree(original, path)
    (path / filename).unlink()
    with pytest.raises((OSError, ValueError)):
        experiments.inspect_case(path, case)


@pytest.mark.parametrize("field,value", [
    ("mode", "ddp"), ("microbatch_size", 1), ("sync_policy", "every"),
    ("bucket_cap_mb", 100), ("precision", "bf16"), ("checkpointing", "block"),
    ("seed", 2027), ("context_length", 16),
])
def test_executed_policy_mismatch_is_rejected(tmp_path, real_single_run, field, value):
    original, case, _ = real_single_run
    path = tmp_path / "case"
    shutil.copytree(original, path)
    metadata_path = path / "rank-0-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if field in {"precision", "seed"}:
        metadata["config"][field] = value
    elif field in {"checkpointing", "context_length"}:
        key = "activation_checkpointing" if field == "checkpointing" else field
        metadata["config"]["model"][key] = value
    else:
        metadata[field] = value
    experiments.save(metadata_path, metadata)
    with pytest.raises(ValueError):
        experiments.inspect_case(path, case)


@pytest.mark.parametrize("field,value", [("tokens_seen", 1), ("batch_stream_sha256", "wrong")])
def test_summary_cannot_disagree_with_rank_metrics(tmp_path, real_single_run, field, value):
    original, case, _ = real_single_run
    path = tmp_path / "case"
    shutil.copytree(original, path)
    summary = json.loads((path / "summary.json").read_text())
    summary[field] = value
    experiments.save(path / "summary.json", summary)
    with pytest.raises(ValueError):
        experiments.inspect_case(path, case)


def test_cross_policy_signature_includes_learning_rate_stream(tmp_path, real_single_run):
    original, case, _ = real_single_run
    _, original_signature = experiments.inspect_case(original, case)
    path = tmp_path / "case"
    shutil.copytree(original, path)
    metrics_path = path / "rank-0-metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    for row in rows:
        if row["event"] == "train":
            row["learning_rate"] *= 2
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    try:
        _, changed_signature = experiments.inspect_case(path, case)
    except ValueError:
        # Rejecting a schedule that disagrees with the frozen config is stronger.
        return
    assert changed_signature != original_signature


def test_active_compute_and_reserved_capacity_have_distinct_cost_scopes():
    common = dict(status="completed", model="324m", context_length=1024, seed=1337,
                  precision="bf16", checkpointing="none", global_batch_size=32)
    single = dict(common, case_id="single", mode="single", world_size=1, tokens_per_second=100,
                  measured_train_seconds=2, subprocess_seconds=3)
    distributed = dict(common, case_id="ddp", mode="ddp", world_size=8, tokens_per_second=400,
                       measured_train_seconds=0.5, subprocess_seconds=1)
    failed = dict(common, case_id="failed", status="failed", mode="fsdp", world_size=8)
    experiments.add_scaling_metrics([single, distributed, failed], allocated_gpus=8)
    assert single["active_gpu_seconds_measured"] == 2
    assert single["allocated_gpu_seconds_subprocess"] == 24
    assert distributed["active_gpu_seconds_measured"] == 4
    assert distributed["allocated_gpu_seconds_subprocess"] == 8
    assert distributed["throughput_speedup"] == 4
    assert distributed["strong_scaling_efficiency"] == 0.5
    assert distributed["baseline_case_id"] == "single"
    assert "throughput_speedup" not in failed


def test_timeout_stops_later_cases_and_retains_allocated_time(tmp_path):
    plan = experiments.build_plan("smoke")
    single = next(case for case in plan["cases"] if case["mode"] == "single" and case["precision"] == "fp32")
    plan["cases"].remove(single)
    plan["cases"].insert(0, single)
    single["timeout_seconds"] = 0.001
    plan["plan_sha256"] = experiments.plan_hash(plan)
    base = json.loads((SCRIPT.parents[1] / "configs/smoke.json").read_text())
    result = experiments.run_plan(plan, base, tmp_path / "timeout", device="cpu", allocated_gpus=2)
    assert result["status"] == "failed"
    assert result["cases"][0]["status"] == "timeout"
    assert all(row["status"] == "not_run" for row in result["cases"][1:])
    row = result["cases"][0]
    assert row["allocated_gpu_seconds_subprocess"] == 2 * row["subprocess_seconds"]
    assert "tokens_per_second" not in row


@pytest.mark.parametrize("count", [-1, 0, 1, 1.5])
def test_invalid_allocation_never_creates_output(tmp_path, count):
    output = tmp_path / "invalid-allocation"
    with pytest.raises(ValueError, match="Allocated GPU count"):
        experiments.run_plan(experiments.build_plan("smoke"), {}, output, allocated_gpus=count)
    assert not output.exists()
