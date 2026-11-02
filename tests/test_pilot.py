import copy
import hashlib
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from llmlab import pilot
from llmlab.config import DataConfig, ModelConfig, TrainConfig
from llmlab.data import load_data


@pytest.fixture
def config():
    return TrainConfig(
        model=ModelConfig(context_length=8),
        data=DataConfig(kind="synthetic", synthetic_tokens=256),
        seed=7,
        device="cpu",
        batch_size=2,
        max_steps=5,
        warmup_steps=2,
        eval_interval=2,
        checkpoint_interval=3,
    )


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    source = {
        "python_source_sha256": "a" * 64,
        "git_head": "b" * 40,
        "git_status": "",
    }
    monkeypatch.setattr(pilot, "source_identity", lambda: copy.deepcopy(source))


def plan(config, **changes):
    options = {
        "microbatches": [1, 2],
        "effective_batch_size": 4,
        "seeds": [7, 11],
        "warmup_steps": 1,
        "order_seed": 17,
    }
    options.update(changes)
    return pilot.make_plan(config, **options)


def test_plan_changes_only_experimental_controls(config):
    result = plan(config, precision="bf16")
    assert len(result["cases"]) == 16
    assert len({case["case_id"] for case in result["cases"]}) == 16
    seen = set()
    for case in result["cases"]:
        resolved = case["config"]
        labels = case["labels"]
        assert resolved["batch_size"] * resolved["grad_accum_steps"] == 4
        assert case["tokens_per_step"] == 4 * config.model.context_length
        assert resolved["device"] == labels["device"] == "cuda"
        assert resolved["precision"] == labels["precision"] == "bf16"
        assert resolved["max_steps"] == case["max_steps"] == config.max_steps
        # Profiling warmup must not rewrite the learning-rate schedule.
        assert resolved["warmup_steps"] == config.warmup_steps
        assert result["warmup_steps"] == 1
        assert resolved["data"] == config.to_dict()["data"]
        for key in ("learning_rate", "min_learning_rate", "weight_decay", "grad_clip",
                    "eval_interval", "eval_batch_size", "eval_batches", "checkpoint_interval"):
            assert resolved[key] == config.to_dict()[key]
        for key in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            assert resolved["model"][key] == config.to_dict()["model"][key]
        assert labels["batch_size"] == resolved["batch_size"]
        assert labels["grad_accum_steps"] == resolved["grad_accum_steps"]
        assert labels["seed"] == resolved["seed"]
        assert labels["attention_backend"] == resolved["model"]["attention_backend"]
        assert labels["activation_checkpointing"] == resolved["model"]["activation_checkpointing"]
        expected_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
        assert case["config_sha256"] == expected_hash
        assert case["source_sha256"] == "a" * 64
        seen.add((labels["attention_backend"], labels["activation_checkpointing"],
                  labels["batch_size"], labels["seed"]))
    assert seen == {
        (backend, checkpointing, microbatch, seed)
        for backend in ("eager", "sdpa")
        for checkpointing in ("none", "block")
        for microbatch in (1, 2)
        for seed in (7, 11)
    }


def test_case_order_is_randomized_reproducibly(config):
    first, repeated = plan(config), plan(config)
    changed = plan(config, order_seed=18)
    assert first == repeated
    first_order = [case["case_id"] for case in first["cases"]]
    changed_order = [case["case_id"] for case in changed["cases"]]
    assert set(first_order) == set(changed_order)
    assert first_order != changed_order
    assert first["plan_sha256"] != changed["plan_sha256"]


def test_selected_policies_keep_only_requested_treatments_and_preserve_controls(config):
    policies = ["sdpa:none:1", "sdpa:none:2", "sdpa:block:2"]
    selected = plan(config, policies=policies)
    full = plan(config)
    expected = {(backend, checkpointing, batch, seed)
                for backend, checkpointing, batch in [("sdpa", "none", 1),
                                                       ("sdpa", "none", 2),
                                                       ("sdpa", "block", 2)]
                for seed in (7, 11)}
    assert len(selected["cases"]) == 6
    assert {(c["labels"]["attention_backend"], c["labels"]["activation_checkpointing"],
             c["labels"]["batch_size"], c["labels"]["seed"]) for c in selected["cases"]} == expected
    original_cases = {c["case_id"]: c for c in full["cases"]}
    assert all(c == original_cases[c["case_id"]] for c in selected["cases"])
    assert selected == plan(config, policies=list(reversed(policies)))
    assert selected["plan_sha256"] != full["plan_sha256"]
    identity = {key: value for key, value in selected.items() if key != "plan_sha256"}
    assert selected["plan_sha256"] == pilot._sha256(identity)


def test_explicit_full_policy_selection_preserves_default_plan_and_shuffle(config):
    policies = [f"{backend}:{checkpointing}:{batch}"
                for backend in ("eager", "sdpa") for checkpointing in ("none", "block")
                for batch in (1, 2)]
    assert plan(config, policies=policies) == plan(config)


@pytest.mark.parametrize("policies", [
    [], ["sdpa:none:1", "sdpa:none:1"], ["sdpa:none:4"], ["sdpa:none:3"],
    ["flash:none:1"], ["sdpa:layer:1"], ["sdpa:none:0"], ["sdpa:none:-1"],
    ["sdpa:none:1.0"], ["sdpa:none:01"], ["sdpa:none"], ["sdpa:none:1:extra"],
    [None], [1],
])
def test_invalid_policy_selections_are_rejected(config, policies):
    with pytest.raises(ValueError):
        plan(config, policies=policies)


def test_policy_selection_cli_writes_only_requested_cases(config, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "selected-plan"
    monkeypatch.setattr(pilot.sys, "argv", [
        "llmlab.pilot", "--config", str(config_path), "--output-dir", str(output),
        "--microbatches", "1", "2", "--effective-batch-size", "4", "--warmup-steps", "1",
        "--policies", "sdpa:none:1", "sdpa:block:2",
    ])
    pilot.main()
    result = json.loads((output / "manifest.json").read_text())
    assert {case["case_id"] for case in result["cases"]} == {
        "sdpa-none-mb1-acc4-seed7", "sdpa-block-mb2-acc2-seed7",
    }
    assert len(list((output / "configs").glob("*.json"))) == 2
    assert not (output / "execution").exists()


@pytest.mark.parametrize("changes", [
    {"microbatches": []},
    {"microbatches": [1, 1]},
    {"microbatches": [0]},
    {"microbatches": [3]},
    {"microbatches": [True]},
    {"effective_batch_size": 0},
    {"seeds": []},
    {"seeds": [7, 7]},
    {"seeds": [-1]},
    {"warmup_steps": 0},
    {"warmup_steps": 5},
    {"timeout_seconds": 0},
    {"timeout_seconds": math.inf},
])
def test_invalid_pilot_settings_are_rejected(config, changes):
    with pytest.raises(ValueError):
        plan(config, **changes)


def test_writing_plan_does_not_start_training_or_overwrite(config, tmp_path, monkeypatch):
    result = plan(config)
    def unexpected_run(*args, **kwargs):
        raise AssertionError("planning must not launch a subprocess")
    monkeypatch.setattr(pilot.subprocess, "run", unexpected_run)
    destination = tmp_path / "pilot"
    pilot.write_plan(result, destination)
    assert json.loads((destination / "manifest.json").read_text()) == result
    for case in result["cases"]:
        path = destination / "configs" / f"{case['case_id']}.json"
        assert json.loads(path.read_text()) == case["config"]
    assert not (destination / "runs").exists() or not list((destination / "runs").iterdir())
    with pytest.raises(FileExistsError):
        pilot.write_plan(result, destination)


def case_output(config, root, case=None):
    case = plan(config)["cases"][0] if case is None else case
    run_dir = root / "runs" / case["case_id"]
    run_dir.mkdir(parents=True)
    metadata = {
        "config": case["config"],
        "config_sha256": case["config_sha256"],
        "source": {"python_source_sha256": case["source_sha256"]},
        "device": "cuda",
        "precision": case["labels"]["precision"],
        "workload": "synthetic pipeline check",
        "data_identity": load_data(replace(config, seed=case["labels"]["seed"]))[2],
    }
    summary = {
        "status": "completed", "step": 5,
        "tokens_seen": 5 * case["tokens_per_step"],
        "valid_loss": 2.3,
        "setup_seconds": 3.0, "train_seconds": 110.0,
        "eval_seconds": 5.0, "checkpoint_seconds": 7.0,
        "wall_seconds": 125.0,
        "peak_cuda_allocated_bytes": 1000, "peak_cuda_reserved_bytes": 2000,
    }
    metrics = [
        {"event": "train", "step": step, "step_seconds": duration,
         "tokens_seen": step * case["tokens_per_step"], "loss": 4.0 - 0.1 * step,
         "gradient_norm": 0.25,
         "batch_start_sha256": hashlib.sha256(f"{case['labels']['seed']}:{step}".encode()).hexdigest()}
        for step, duration in enumerate([100.0, 1.0, 2.0, 3.0, 4.0], start=1)
    ]
    metrics.insert(2, {"event": "eval", "step": 2, "valid_loss": 2.5,
                       "step_seconds": 999.0})
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)
    write_metrics(run_dir, metrics)
    return case, run_dir, metadata, summary, metrics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_metrics(run_dir, metrics):
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))


def test_aggregation_excludes_warmup_but_retains_full_runner_cost(config, tmp_path):
    case, _, _, _, _ = case_output(config, tmp_path)
    row = pilot.summarize_case(tmp_path, case, warmup_steps=1)
    assert row["status"] == "completed"
    assert row["measured_steps"] == 4
    assert row["step_seconds_median"] == 2.5
    assert row["step_seconds_p90"] == pytest.approx(3.7)
    assert row["tokens_per_second"] == pytest.approx(4 * case["tokens_per_step"] / 10)
    assert row["last_train_loss"] == 3.5
    assert row["valid_loss"] == 2.3
    assert row["runner_times"]["train_seconds"] == 110.0
    assert row["runner_times"]["wall_seconds"] == 125.0
    assert row["peak_cuda_allocated_bytes"] == 1000
    assert row["peak_cuda_reserved_bytes"] == 2000


def test_missing_case_is_reported(config, tmp_path):
    case = plan(config)["cases"][0]
    row = pilot.summarize_case(tmp_path, case, warmup_steps=1)
    assert row["status"] == "missing_output"
    assert row["measured_steps"] == 0
    assert row.get("step_seconds_median") is None


def test_truncated_metrics_are_not_reported_as_complete(config, tmp_path):
    case, directory, _, _, metrics = case_output(config, tmp_path)
    write_metrics(directory, [row for row in metrics if row["step"] == 1])
    row = pilot.summarize_case(tmp_path, case, warmup_steps=1)
    assert row["status"] == "insufficient_samples"
    assert row["measured_steps"] == 0
    assert row.get("tokens_per_second") is None


@pytest.mark.parametrize("status,error", [
    ("failed", "CUDA out of memory"),
    ("timeout", "case exceeded timeout"),
])
def test_execution_failure_takes_priority_over_partial_metrics(config, tmp_path, status, error):
    case, directory, _, summary, metrics = case_output(config, tmp_path)
    summary.update(status="failed", step=2, error=error)
    write_json(directory / "summary.json", summary)
    write_metrics(directory, [row for row in metrics if row["step"] <= 2])
    write_json(tmp_path / "execution" / f"{case['case_id']}.json", {
        "status": status, "returncode": 1 if status == "failed" else None,
        "error": error, "wall_seconds": 120.0,
    })
    row = pilot.summarize_case(tmp_path, case, warmup_steps=1)
    assert row["status"] == status
    assert error in row["error"]
    assert row["measured_steps"] == 1
    assert row["subprocess_wall_seconds"] == 120.0


def test_timeout_before_runner_writes_output_is_preserved(config, tmp_path):
    case = plan(config)["cases"][0]
    write_json(tmp_path / "execution" / f"{case['case_id']}.json", {
        "status": "timeout", "returncode": None,
        "error": "case exceeded timeout", "wall_seconds": 600.0,
    })
    row = pilot.summarize_case(tmp_path, case, warmup_steps=1)
    assert row["status"] == "timeout"
    assert row["measured_steps"] == 0
    assert "timeout" in row["error"]
    assert row["subprocess_wall_seconds"] == 600.0


@pytest.mark.parametrize("problem", ["duration_zero", "duration_nan", "loss_inf", "duplicate", "tokens"])
def test_invalid_metrics_are_not_summarized_as_success(config, tmp_path, problem):
    case, directory, _, _, metrics = case_output(config, tmp_path)
    if problem == "duration_zero":
        metrics[1]["step_seconds"] = 0.0
    elif problem == "duration_nan":
        metrics[1]["step_seconds"] = math.nan
    elif problem == "loss_inf":
        metrics[1]["loss"] = math.inf
    elif problem == "duplicate":
        metrics.append(copy.deepcopy(metrics[1]))
    else:
        metrics[1]["tokens_seen"] += 1
    write_metrics(directory, metrics)
    assert pilot.summarize_case(tmp_path, case, warmup_steps=1)["status"] == "invalid_output"


def test_truncated_json_is_reported_without_crashing_aggregation(config, tmp_path):
    case, directory, _, _, _ = case_output(config, tmp_path)
    (directory / "metrics.jsonl").write_text('{"event":"train",')
    assert pilot.summarize_case(tmp_path, case, warmup_steps=1)["status"] == "invalid_output"


@pytest.mark.parametrize("identity", ["config", "source"])
def test_mismatched_result_identity_is_rejected(config, tmp_path, identity):
    case, directory, metadata, _, _ = case_output(config, tmp_path)
    if identity == "config":
        metadata["config_sha256"] = "c" * 64
    else:
        metadata["source"]["python_source_sha256"] = "d" * 64
    write_json(directory / "metadata.json", metadata)
    assert pilot.summarize_case(tmp_path, case, warmup_steps=1)["status"] == "invalid_output"


@pytest.mark.parametrize("changes", [
    {"valid_loss": None}, {"valid_loss": math.nan}, {"valid_loss": math.inf},
    {"step": 4}, {"tokens_seen": 1},
])
def test_completed_summary_requires_final_counts_and_finite_validation(config, tmp_path, changes):
    case, directory, _, summary, _ = case_output(config, tmp_path)
    summary.update(changes)
    write_json(directory / "summary.json", summary)
    assert pilot.summarize_case(tmp_path, case, warmup_steps=1)["status"] == "invalid_output"


def test_comparability_pairs_only_cases_with_the_same_seed(config, tmp_path):
    result = plan(config)
    for case in result["cases"]:
        case_output(config, tmp_path, case)
    aggregate = pilot.aggregate_results(tmp_path, result)
    check = aggregate["comparability"]
    assert check["status"] == "passed"
    assert not check["reasons"]
    assert {group["seed"] for group in check["seed_groups"]} == {7, 11}
    for group in check["seed_groups"]:
        assert group["status"] == "passed"
        assert not group["reasons"]
        assert set(group["case_ids"]) == {
            case["case_id"] for case in result["cases"] if case["labels"]["seed"] == group["seed"]
        }
    for row in aggregate["cases"]:
        expected = load_data(replace(config, seed=row["seed"]))[2]
        assert row["data_identity"] == expected
        assert row["data_identity_sha256"] == hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("problem", ["missing_hash", "invalid_hash", "missing_data", "data", "stream"])
def test_comparability_requires_complete_matching_evidence(config, tmp_path, problem):
    result = plan(config, microbatches=[1], seeds=[7])
    for case in result["cases"]:
        _, directory, metadata, _, metrics = case_output(config, tmp_path, case)
    # Corrupt the final case, including a warmup row: every update needs evidence.
    if problem in {"missing_hash", "invalid_hash", "stream"}:
        if problem == "missing_hash":
            metrics[0].pop("batch_start_sha256")
        else:
            metrics[0]["batch_start_sha256"] = ("x" if problem == "invalid_hash" else "f") * 64
        write_metrics(directory, metrics)
    else:
        if problem == "missing_data":
            metadata.pop("data_identity")
        else:
            metadata["data_identity"]["train"]["sha256"] = "f" * 64
        write_json(directory / "metadata.json", metadata)
    aggregate = pilot.aggregate_results(tmp_path, result)
    assert aggregate["status_counts"] == {"completed": 4}
    assert aggregate["comparability"]["status"] == "failed"
    assert aggregate["comparability"]["reasons"]
    assert aggregate["comparability"]["seed_groups"][0]["status"] == "failed"
    if problem in {"missing_hash", "invalid_hash"}:
        assert aggregate["cases"][-1]["batch_stream_sha256"] is None


@pytest.mark.parametrize("missing_kind", ["missing", "failed"])
def test_missing_or_failed_cases_make_comparability_insufficient(config, tmp_path, missing_kind):
    result = plan(config, microbatches=[1], seeds=[7])
    for case in result["cases"][1:]:
        case_output(config, tmp_path, case)
    if missing_kind == "failed":
        case, directory, metadata, summary, metrics = case_output(config, tmp_path, result["cases"][0])
        summary.update(status="failed", error="CUDA out of memory")
        metadata["data_identity"]["train"]["sha256"] = "f" * 64
        metrics[0]["batch_start_sha256"] = "f" * 64
        write_json(directory / "summary.json", summary)
        write_json(directory / "metadata.json", metadata)
        write_metrics(directory, metrics)
    check = pilot.aggregate_results(tmp_path, result)["comparability"]
    assert check["status"] == "insufficient"
    assert check["reasons"]
    assert check["seed_groups"][0]["status"] == "insufficient"


def test_comparability_with_no_completed_cases_is_insufficient(config, tmp_path):
    check = pilot.aggregate_results(tmp_path, plan(config))["comparability"]
    assert check["status"] == "insufficient"
    assert all(group["status"] == "insufficient" for group in check["seed_groups"])


def test_completed_evidence_mismatch_takes_priority_over_missing_cases(config, tmp_path):
    result = plan(config, microbatches=[1], seeds=[7])
    for case in result["cases"][1:]:
        _, directory, metadata, _, _ = case_output(config, tmp_path, case)
    metadata["data_identity"]["train"]["sha256"] = "f" * 64
    write_json(directory / "metadata.json", metadata)
    assert pilot.aggregate_results(tmp_path, result)["comparability"]["status"] == "failed"


def test_executor_refuses_cpu_without_explicit_harness_flag(config, tmp_path, monkeypatch):
    root = pilot.write_plan(plan(config, device="cpu", microbatches=[1], seeds=[7]), tmp_path / "pilot")
    def unexpected_run(*args, **kwargs):
        raise AssertionError("unapproved CPU execution must not start")
    monkeypatch.setattr(pilot.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="allow-cpu"):
        pilot.execute_plan(root)
    assert not (root / "execution").exists()


def test_executor_continues_after_failures_and_preserves_evidence_on_reentry(config, tmp_path, monkeypatch):
    result = plan(config, device="cpu", microbatches=[1], seeds=[7], timeout_seconds=10)
    root = pilot.write_plan(result, tmp_path / "pilot")
    cases = {case["case_id"]: case for case in result["cases"]}
    calls = []

    def fake_run(command, **kwargs):
        case_id = Path(command[command.index("--config") + 1]).stem
        calls.append(case_id)
        assert command[1:3] == ["-m", "llmlab.train"]
        assert kwargs["timeout"] == 10
        assert kwargs["stderr"] == subprocess.STDOUT
        kwargs["stdout"].write(f"attempt {len(calls)}\n")
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if len(calls) == 2:
            kwargs["stdout"].write("CUDA out of memory\n")
            return subprocess.CompletedProcess(command, 1)
        case_output(config, root, cases[case_id])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    aggregate = pilot.execute_plan(root, allow_cpu=True)
    assert calls == [case["case_id"] for case in result["cases"]]
    assert [row["status"] for row in aggregate["cases"]] == ["timeout", "failed", "completed", "completed"]
    assert aggregate["cases"][1]["failure_kind"] == "oom"
    for row in aggregate["cases"]:
        execution = json.loads((root / "execution" / f"{row['case_id']}.json").read_text())
        assert row["subprocess_wall_seconds"] == execution["wall_seconds"] >= 0
    evidence = {path: path.read_bytes() for folder in ("execution", "logs", "runs")
                for path in (root / folder).rglob("*") if path.is_file()}
    assert pilot.execute_plan(root, allow_cpu=True) == aggregate
    assert len(calls) == len(result["cases"])
    assert all(path.read_bytes() == contents for path, contents in evidence.items())


@pytest.mark.parametrize("option,value", [
    ("--timeout-seconds", "1"), ("--warmup-steps", "1"),
    ("--order-seed", "0"), ("--device", "cuda"),
    ("--policies", "sdpa:none:1"),
])
def test_run_plan_cli_rejects_explicit_planning_options_before_execution(monkeypatch, option, value):
    def unexpected_execute(*args, **kwargs):
        raise AssertionError("invalid CLI arguments must not execute a plan")
    monkeypatch.setattr(pilot, "execute_plan", unexpected_execute)
    monkeypatch.setattr(pilot.sys, "argv", ["llmlab.pilot", "--run-plan", "saved-plan", option, value])
    with pytest.raises(SystemExit) as error:
        pilot.main()
    assert error.value.code == 2


def test_nonfinite_runner_duration_is_invalid_but_aggregate_remains_json_safe(config, tmp_path):
    result = plan(config)
    _, directory, _, summary, _ = case_output(config, tmp_path, result["cases"][0])
    summary["train_seconds"] = math.nan
    write_json(directory / "summary.json", summary)
    aggregate = pilot.aggregate_results(tmp_path, result)
    assert aggregate["cases"][0]["status"] == "invalid_output"
    assert aggregate["cases"][0]["runner_times"]["train_seconds"] is None
    json.dumps(aggregate, allow_nan=False)
