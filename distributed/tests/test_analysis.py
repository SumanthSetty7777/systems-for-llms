"""Reports must preserve failures and derive comparisons from actual evidence."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load("distributed_analysis_under_test", "analyze.py")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _case(case_id, mode="single", world=1, checkpoint="none", seed=1337, **changes):
    return {"case_id": case_id, "mode": mode, "world_size": world, "model": "324m",
            "context_length": 1024, "global_batch_size": 32, "microbatch_size": 4,
            "checkpointing": checkpoint, "sync_policy": "last", "bucket_cap_mb": 25,
            "seed": seed, "steps": 3, "precision": "bf16", "warmup_steps": 1,
            "timeout_seconds": 240, **changes}


def _fixture(root, cases=None, completed=None):
    cases = cases or [_case("single-none"), _case("ddp-none", "ddp", 2),
                      _case("single-block", checkpoint="block"), _case("ddp-block", "ddp", 2, "block"),
                      _case("zero1-none", "zero1", 2), _case("fsdp-failed", "fsdp", 8)]
    completed = set(completed if completed is not None else [case["case_id"] for case in cases[:-1]])
    plan = {"format_version": 1, "stage": "scale", "cases": cases, "case_count": len(cases)}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    packed = {"plan.json": plan, "preflight.json": {"torch": "test-torch", "cuda": "test-cuda",
              "devices": [{"index": index, "name": "Test GPU", "uuid": "GPU-test-uuid" if index == 0 else f"GPU-test-{index}",
                           "memory_bytes": 80 * 1024**3} for index in range(8)],
              "nvidia_smi_topology": "GPU0 GPU1\nGPU0 X NV18\nGPU1 NV18 X"},
              "tests-report.json": {"tests": 11, "passed": 11, "failures": 0, "errors": 0, "skipped": 0}}
    evidence = {"format_version": 1, "json": packed, "metrics": {}, "logs": {}, "errors": [], "missing_files": [], "truncated_logs": []}
    rows = []
    for case in cases:
        if case["case_id"] not in completed:
            rows.append(dict(case, status="not_run"))
            continue
        seconds = (2 if case["checkpointing"] == "block" else 1) / (1 if case["mode"] == "single" else 1.5)
        tokens = case["global_batch_size"] * case["context_length"]
        metrics = [{"event": "train", "step": step, "step_seconds": 99 if step <= case["warmup_steps"] else seconds,
                    "tokens_seen": step * tokens, "batch_start_sha256": f"batch-{step}",
                    "learning_rate": 0.001 / step, "loss": 5 - step * 0.1, "gradient_norm": 1.2}
                   for step in range(1, case["steps"] + 1)]
        batch_hash = hashlib.sha256("".join(item["batch_start_sha256"] for item in metrics).encode()).hexdigest()
        ranks = [{"rank": rank, "peak_cuda_allocated_bytes": (10 + rank) * 1024**3,
                  "peak_cuda_reserved_bytes": (12 + rank) * 1024**3,
                  "optimizer_state_bytes": (1 if case["mode"] == "zero1" else 2) * 1024**3}
                 for rank in range(case["world_size"])]
        count = case["steps"] - case["warmup_steps"]
        summary = {"status": "completed", "mode": case["mode"], "world_size": case["world_size"], "step": case["steps"],
                   "tokens_seen": case["steps"] * tokens, "tokens_per_step": tokens, "measured_steps": count,
                   "batch_stream_sha256": batch_hash, "measured_train_seconds": seconds * count,
                   "tokens_per_second": tokens / seconds, "valid_loss": 4.1, "ranks": ranks,
                   "peak_cuda_allocated_bytes": max(item["peak_cuda_allocated_bytes"] for item in ranks),
                   "peak_cuda_reserved_bytes": max(item["peak_cuda_reserved_bytes"] for item in ranks),
                   "optimizer_state_bytes_max_rank": max(item["optimizer_state_bytes"] for item in ranks),
                   "optimizer_state_bytes_sum": sum(item["optimizer_state_bytes"] for item in ranks)}
        row = dict(case, **summary, comparability="passed", subprocess_seconds=15)
        rows.append(row)
        prefix = f"study/cases/{case['case_id']}/"
        packed[prefix + "summary.json"] = deepcopy(summary)
        for rank in range(case["world_size"]):
            meta = {key: case[key] for key in ("mode", "world_size", "global_batch_size", "microbatch_size", "sync_policy", "bucket_cap_mb")}
            meta.update(rank=rank, timing_warmup_steps=case["warmup_steps"], profile_steps=0,
                        config={"seed": case["seed"], "precision": case["precision"],
                                "model": {"context_length": case["context_length"], "activation_checkpointing": case["checkpointing"]}},
                        source={"sha256": "source-test-sha", "files": {"distributed/llmdist/train.py": "train-test-sha"}},
                        data_identity={"sha256": "data-test-sha"}, initial_parameter_sha256="initial-test-sha",
                        timing_scope="synchronized rank maximum", peak_memory_scope="training and evaluation")
            packed[prefix + f"rank-{rank}-metadata.json"] = meta
            packed[prefix + f"rank-{rank}-summary.json"] = deepcopy(summary)
            evidence["metrics"][prefix + f"rank-{rank}-metrics.jsonl"] = deepcopy(metrics)
    aggregate = {"status": "completed" if len(completed) == len(cases) else "failed", "planned_count": len(cases),
                 "cases": rows, "plan_sha256": plan["plan_sha256"], "allocated_gpus": 8}
    packed["study/aggregate.json"] = deepcopy(aggregate)
    _save(root / "scale/aggregate.json", aggregate)
    _save(root / "scale/evidence.json", evidence)
    _save(root / "scale/launch-summary.json", {"status": aggregate["status"], "function_seconds": 120, "allocated_gpu_count": 8})
    _save(root / "scale/launch.json", {"payload": {"source_sha256": {"/project/train.py": "artifact-test-sha"}},
          "cost_estimate": {"requested_resources_usd_per_hour": 36, "allocated_gpu_count": 8,
                            "source": "https://modal.com/pricing", "checked": "2026-11-01"}})
    return aggregate, evidence


def test_measured_speedups_match_checkpoint_policy_and_max_rank_memory(tmp_path):
    _fixture(tmp_path)
    result = analysis.build_summary(tmp_path)
    rows = {row["case_id"]: row for row in result["cases"]}
    assert rows["ddp-none"]["throughput_speedup"] == pytest.approx(1.5)
    assert rows["ddp-block"]["throughput_speedup"] == pytest.approx(1.5)
    assert rows["ddp-block"]["baseline_case_id"] == "single-block"
    assert rows["ddp-none"]["strong_scaling_efficiency"] == pytest.approx(0.75)
    assert rows["ddp-none"]["tokens_per_second"] == pytest.approx(32768 * 1.5)
    assert rows["ddp-none"]["active_gpu_seconds_per_update"] == pytest.approx(2 / 1.5)
    assert rows["single-none"]["allocated_gpu_seconds_subprocess"] == 8 * 15
    assert rows["ddp-none"]["peak_cuda_allocated_bytes"] == 11 * 1024**3
    assert rows["zero1-none"]["optimizer_state_bytes_sum"] == 2 * 1024**3
    assert result["recorded_function_time_estimate_usd"] == pytest.approx(1.2)
    assert "One-seed" in result["interpretation"]
    assert result["hardware"]["devices"][0]["uuid"] == "GPU-test-uuid"


def test_unavailable_topology_is_reported_without_losing_measurements(tmp_path):
    _, evidence = _fixture(tmp_path)
    hardware = evidence["json"]["preflight.json"]
    hardware["nvidia_smi_topology"] = None
    hardware["warnings"] = ["nvidia-smi topo -m: unavailable"]
    _save(tmp_path / "scale/evidence.json", evidence)
    result = analysis.build_summary(tmp_path)
    assert any("nvidia-smi topo -m" in warning for warning in result["warnings"])
    assert any("does not identify link type" in warning for warning in result["warnings"])
    measured = next(row for row in result["cases"] if row["case_id"] == "ddp-none")
    assert measured["throughput_speedup"] == pytest.approx(1.5)


def test_missing_or_mismatched_baselines_never_gain_ratios(tmp_path):
    _fixture(tmp_path, [_case("single"), _case("different-seed", "ddp", 2, seed=2027),
                       _case("different-checkpoint", "ddp", 2, "block"), _case("different-source", "ddp", 2)],
             completed={"single", "different-seed", "different-checkpoint", "different-source"})
    evidence = analysis._read(tmp_path / "scale/evidence.json")
    for rank in range(2):
        evidence["json"][f"study/cases/different-source/rank-{rank}-metadata.json"]["source"]["sha256"] = "changed-source"
    _save(tmp_path / "scale/evidence.json", evidence)
    result = analysis.build_summary(tmp_path)
    for row in result["cases"][1:]:
        assert row["throughput_speedup"] is None
        assert row["strong_scaling_efficiency"] is None
        assert row["baseline_case_id"] is None
    assert result["cases"][-1]["baseline_status"] == "provenance mismatch"


@pytest.mark.parametrize("kind", ["aggregate", "time", "rank-stream", "memory", "plan"])
def test_rejects_inconsistent_evidence_before_creating_output(tmp_path, kind):
    run = tmp_path / "run"
    aggregate, evidence = _fixture(run)
    if kind == "aggregate":
        aggregate["cases"][0]["tokens_per_second"] = 999
        _save(run / "scale/aggregate.json", aggregate)
    elif kind == "time":
        evidence["metrics"]["study/cases/single-none/rank-0-metrics.jsonl"][-1]["step_seconds"] = 7
    elif kind == "rank-stream":
        evidence["metrics"]["study/cases/ddp-none/rank-1-metrics.jsonl"][-1]["batch_start_sha256"] = "wrong-batch"
    elif kind == "memory":
        evidence["json"]["study/cases/ddp-none/summary.json"]["optimizer_state_bytes_max_rank"] = 0
    else:
        evidence["json"]["plan.json"]["cases"][0]["seed"] = 99
    _save(run / "scale/evidence.json", evidence)
    with pytest.raises(ValueError, match="mismatch"):
        analysis.write_report(run, tmp_path / "report", plots=False)
    assert not (tmp_path / "report").exists()


def test_missing_input_and_zero_completed_cases_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        analysis.build_summary(tmp_path)
    _fixture(tmp_path, completed=set())
    with pytest.raises(ValueError, match="No completed"):
        analysis.build_summary(tmp_path)


def test_portable_report_preserves_all_38_planned_cases_and_failure(tmp_path):
    experiments = _load("distributed_plan_for_analysis_test", "run_experiments.py")
    planned = experiments.build_plan("scale")["cases"]
    completed = {case["case_id"] for case in planned if case["mode"] == "single"}
    run, output = tmp_path / "run", tmp_path / "report"
    aggregate, evidence = _fixture(run, planned, completed=completed)
    failed = next(row for row in aggregate["cases"] if row["status"] == "not_run")
    failed.update(status="oom", error="CUDA out of memory", tokens_per_second=999)
    evidence["json"]["study/aggregate.json"] = aggregate
    _save(run / "scale/aggregate.json", aggregate)
    _save(run / "scale/evidence.json", evidence)
    result = analysis.write_report(run, output, plots=False)
    assert len(result["cases"]) == result["planned_count"] == 38
    assert result["case_status_counts"] == {"completed": 4, "oom": 1, "not_run": 33}
    assert "tokens_per_second" not in next(row for row in result["cases"] if row["status"] == "oom")
    readme = (output / "README.md").read_text()
    assert all(case["case_id"] in readme for case in planned)
    assert "All allocated GPUs are billed" in readme
    assert "CUDA out of memory" in readme
    assert "GPU-test-uuid" in readme
    assert "passed=11" in readme
    assert result["figures"] == []
    assert result["artifact_source_sha256"] == {"/project/train.py": "artifact-test-sha"}
    for name, checksum in result["input_files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    with pytest.raises(FileExistsError):
        analysis.write_report(run, output, plots=False)


def test_canonical_plot_selection_does_not_repeat_reference_or_mix_ablations(tmp_path):
    cases = [_case("reference"), _case("ddp", "ddp", 2), _case("microbatch", "ddp", 8, microbatch_size=1),
             _case("sync-every", "ddp", 8, sync_policy="every"), _case("bucket", "ddp", 8, bucket_cap_mb=1),
             _case("4k-reference", context_length=4096, global_batch_size=8, microbatch_size=1),
             _case("4k-ddp", "ddp", 8, context_length=4096, global_batch_size=8, microbatch_size=1)]
    _fixture(tmp_path, cases, completed={case["case_id"] for case in cases})
    groups = analysis.plot_series(analysis.build_summary(tmp_path))
    assert len(groups) == 2
    assert {row["case_id"] for _, rows in groups for row in rows} == {"reference", "ddp", "4k-reference", "4k-ddp"}
    assert all(sum(row["mode"] == "single" for row in rows) == 1 for _, rows in groups)


def test_cli_runs_without_plot_dependency(tmp_path):
    _fixture(tmp_path / "run")
    result = subprocess.run([sys.executable, str(Path(analysis.__file__)), "--run-dir", str(tmp_path / "run"),
                             "--output-dir", str(tmp_path / "report"), "--no-plots"],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["case_status_counts"]["completed"] == 5
    assert (tmp_path / "report/summary.json").exists()


def _standalone_fixture(root):
    cases = []
    for seed in (2027, 2028, 2029):
        for mode, world, checkpoint in (("single", 1, "none"), ("ddp", 8, "none"), ("zero1", 8, "none"),
                                        ("fsdp", 8, "none"), ("tp", 8, "none"), ("ddp", 8, "block"), ("fsdp", 8, "block")):
            cases.append(_case(f"{mode}-{checkpoint}-{seed}", mode, world, checkpoint, seed))
    aggregate, evidence = _fixture(root, cases, completed={case["case_id"] for case in cases})
    plan = evidence["json"]["plan.json"]
    plan["stage"] = "confirmation"
    plan["plan_sha256"] = analysis._hash({key: value for key, value in plan.items() if key != "plan_sha256"})
    aggregate["plan_sha256"] = plan["plan_sha256"]
    evidence["json"]["study/aggregate.json"] = deepcopy(aggregate)
    _save(root / "scale/aggregate.json", aggregate)
    _save(root / "scale/evidence.json", evidence)
    for path in (root / "scale").iterdir():
        path.rename(root / path.name)
    (root / "scale").rmdir()
    launch = analysis._read(root / "launch.json")
    launch["options"] = {"plan": plan, "stage": "confirmation", "run_id": "confirmation-test"}
    _save(root / "launch.json", launch)
    report = analysis._read(root / "launch-summary.json")
    report.update(stage="confirmation", run_id="confirmation-test", evidence_status="complete", case_status_counts={"completed": 21})
    _save(root / "launch-summary.json", report)
    return aggregate, evidence


def test_standalone_confirmation_pairs_seeds_and_preserves_cost_and_inputs(tmp_path):
    run, output = tmp_path / "run", tmp_path / "report"
    _standalone_fixture(run)
    summary = analysis.write_report(run, output, plots=False)
    assert summary["stage"] == "confirmation"
    assert summary["input_layout"] == "standalone"
    assert summary["case_status_counts"] == {"completed": 21}
    assert summary["seeds"] == [2027, 2028, 2029]
    assert len(summary["policy_statistics"]) == len(summary["paired_comparisons"]) == 7
    assert summary["recorded_function_time_estimate_usd"] == pytest.approx(1.2)
    assert [row["stage"] for row in summary["cost_stages"]] == ["confirmation"]
    comparisons = {row["comparison"] + row["numerator"]["checkpointing"]: row for row in summary["paired_comparisons"]}
    ddp = comparisons["ddp_vs_singlenone"]
    assert ddp["paired_count"] == 3
    assert ddp["metrics"]["throughput_ratio"] == pytest.approx({"count": 3, "mean": 1.5, "median": 1.5, "sample_sd": 0, "min": 1.5, "max": 1.5})
    assert comparisons["ddp_block_vs_noneblock"]["metrics"]["throughput_ratio"]["mean"] == pytest.approx(0.5)
    assert comparisons["fsdp_block_vs_noneblock"]["paired_count"] == 3
    readme = (output / "README.md").read_text()
    assert "Paired seed comparisons" in readme
    assert "inputs/confirmation/evidence.json" in readme
    assert "sample standard deviation across seeds" in readme
    assert set(summary["input_files_sha256"]) == {f"inputs/confirmation/{name}" for name in ("aggregate.json", "evidence.json", "launch.json", "launch-summary.json")}
    for name, checksum in summary["input_files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum


def test_paired_ratios_are_aggregated_after_matching_seeds(tmp_path):
    _standalone_fixture(tmp_path)
    rows = analysis.build_summary(tmp_path)["cases"]
    for seed, reference, ddp in ((2027, 10, 20), (2028, 100, 300), (2029, 1000, 4000)):
        next(row for row in rows if row["seed"] == seed and row["mode"] == "single")["tokens_per_second"] = reference
        next(row for row in rows if row["seed"] == seed and row["mode"] == "ddp" and row["checkpointing"] == "none")["tokens_per_second"] = ddp
    paired = next(group for group in analysis.paired_comparisons(rows) if group["comparison"] == "ddp_vs_single")
    assert [pair["throughput_ratio"] for pair in paired["pairs"]] == [2, 3, 4]
    assert paired["metrics"]["throughput_ratio"] == {"count": 3, "mean": 3, "median": 3, "sample_sd": 1, "min": 2, "max": 4}
    assert paired["metrics"]["throughput_ratio"]["mean"] != pytest.approx((20 + 300 + 4000) / (10 + 100 + 1000))
    # The same seed cannot silently be paired across source/config changes.
    altered = next(row for row in rows if row["seed"] == 2028 and row["mode"] == "ddp" and row["checkpointing"] == "none")
    altered["provenance_sha256"] = "different-source-or-config"
    paired = next(group for group in analysis.paired_comparisons(rows) if group["comparison"] == "ddp_vs_single")
    assert paired["seeds"] == [2027, 2029]


@pytest.mark.parametrize("kind", ["hardware", "launch-plan", "skipped-test", "missing-evidence", "launch-counts"])
def test_completed_confirmation_rejects_contradictory_gate_and_launch(tmp_path, kind):
    _, evidence = _standalone_fixture(tmp_path)
    if kind == "hardware":
        evidence["json"]["preflight.json"]["devices"].pop()
    elif kind == "launch-plan":
        launch = analysis._read(tmp_path / "launch.json")
        launch["options"]["plan"]["cases"][0]["seed"] = 999
        _save(tmp_path / "launch.json", launch)
    elif kind == "skipped-test":
        evidence["json"]["tests-report.json"].update(passed=10, skipped=1)
    elif kind == "missing-evidence":
        evidence["missing_files"] = ["tests.log"]
    else:
        report = analysis._read(tmp_path / "launch-summary.json")
        report["case_status_counts"] = {"completed": 20}
        _save(tmp_path / "launch-summary.json", report)
    _save(tmp_path / "evidence.json", evidence)
    with pytest.raises(ValueError):
        analysis.build_summary(tmp_path)
