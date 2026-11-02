"""Study selection and failure boundaries; no cloud requests."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_study.py"
spec = importlib.util.spec_from_file_location("study_test_module", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_development_excludes_completed_screens_and_only_selects_three_quality_policies():
    plan = study.build_plan("development")
    assert [j["name"] for j in plan["jobs"]] == [
        "22m-4k-screen", "324m-1k-screen", "324m-4k-screen", "100m-1k-quality-dev"]
    assert plan["case_count"] == 27
    quality = plan["jobs"][-1]
    assert quality["seeds"] == [1337]
    assert set(quality["policies"]) == {"sdpa:none:8", "sdpa:none:1", "sdpa:block:8"}
    assert quality["overrides"]["max_steps"] == 1200
    assert quality["overrides"]["eval_interval"] == 50
    assert quality["overrides"]["eval_batches"] == 64


def test_confirmation_seeds_are_separate_from_development():
    plan = study.build_plan("timing-confirmation")
    assert plan["case_count"] == 48
    assert all(j["seeds"] == [2027, 2028, 2029] for j in plan["jobs"])


def test_quality_confirmation_retains_full_training_protocol_per_seed():
    development = study.build_plan("development")["jobs"][-1]
    plan = study.build_plan("quality-confirmation")
    assert plan["case_count"] == 9
    assert [job["seeds"] for job in plan["jobs"]] == [[2027], [2028], [2029]]
    for job in plan["jobs"]:
        assert job["case_count"] == 3
        assert job["overrides"] == development["overrides"]
        assert job["overrides"]["max_steps"] == 1200
        assert job["policies"] == development["policies"]
        assert job["effective_batch_size"] == 8
        assert (job["case_timeout_seconds"], job["job_timeout_seconds"]) == (600, 2400)
        assert not job["allow_resource_limits"]


def test_combined_confirmation_runs_quality_before_timing():
    plan = study.build_plan("confirmation")
    assert plan["case_count"] == 57
    assert plan["jobs"][:3] == study.build_plan("quality-confirmation")["jobs"]
    assert plan["jobs"][3:] == study.build_plan("timing-confirmation")["jobs"]
    assert plan["max_function_seconds_sum"] == 5 * 2400
    assert plan["nominal_requested_resources_usd_at_time_limits"] < 50


@pytest.fixture
def frozen_quality(monkeypatch, tmp_path):
    # Fixtures isolate planning from the large local training arrays and cloud SDK.
    config = json.loads((study.REPO / "configs/scale-100m-s1024.json").read_text())
    config["data"].update(train_path="/data/train.npy", valid_path="/data/valid.npy",
                          tokenizer_files=["/data/tokenizer-0/vocab.pkl", "/data/tokenizer-1/merges.pkl"])
    overrides = study.build_plan("development")["jobs"][-1]["overrides"]
    config = study.launcher.apply_config_overrides(config, overrides)
    target = {"target_validation_loss": 2.4, "reference_config": config,
              "reference_payload_packages": list(study.launcher.PACKAGES),
              "quality_protocol": json.loads(json.dumps(study.QUALITY_PROTOCOL))}
    path = tmp_path / "frozen-target.json"
    path.write_text(json.dumps(target))
    monkeypatch.setattr(study, "validate_frozen_target", lambda path, repo: json.loads(path.read_text()))

    def prepare(job, run_id):
        resolved = study.launcher.apply_config_overrides(json.loads(json.dumps(config)), job["overrides"])
        options = {key: job[key] for key in ("microbatches", "effective_batch_size", "seeds", "policies")}
        options["run_id"] = run_id
        return {"files": [], "config": resolved, "packages": list(study.launcher.PACKAGES)}, options

    monkeypatch.setattr(study, "prepare_job", prepare)
    return path, target


@pytest.mark.parametrize("stage", ["quality-confirmation", "confirmation"])
def test_quality_target_is_required_even_for_dry_run(monkeypatch, tmp_path, stage):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--stage", stage,
                                     "--target-file", str(tmp_path / "missing.json")])
    monkeypatch.setattr(study, "prepare_job", lambda *args: pytest.fail("prepared before validating target"))
    monkeypatch.setattr(study.launcher, "execute", lambda *args: pytest.fail("launched without frozen target"))
    with pytest.raises(FileNotFoundError):
        study.main()


def test_confirmation_dry_run_freezes_target_and_hash(monkeypatch, tmp_path, capsys, frozen_quality):
    path, target = frozen_quality
    output = tmp_path / "confirmation"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--stage", "confirmation", "--target-file", str(path),
                                     "--output-dir", str(output)])
    monkeypatch.setattr(study.launcher, "execute", lambda *args: pytest.fail("dry run launched GPU work"))
    study.main()
    result = json.loads(capsys.readouterr().out)
    frozen = result["plan"]["frozen_quality_target"]
    assert frozen == {"path": str(path.resolve()), "target": target,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert result["plan"]["case_count"] == 57
    assert not output.exists()


def test_stale_target_evidence_blocks_before_execution(monkeypatch, tmp_path, frozen_quality):
    path, _ = frozen_quality
    output = tmp_path / "confirmation"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--stage", "quality-confirmation", "--execute",
                                     "--target-file", str(path), "--output-dir", str(output)])
    def stale(path, repo):
        raise ValueError("development evidence hash mismatch")
    monkeypatch.setattr(study, "validate_frozen_target", stale)
    monkeypatch.setattr(study.launcher, "execute", lambda *args: pytest.fail("launched with stale evidence"))
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        study.main()
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("learning_rate", 0.001), ("max_steps", 600),
                                        ("eval_batches", 4), ("model.d_model", 512),
                                        ("data.train_path", "/data/other.npy")])
def test_quality_configs_must_match_frozen_development(frozen_quality, field, value):
    _, target = frozen_quality
    job = study.build_plan("quality-confirmation")["jobs"][0]
    payload, options = study.prepare_job(job, "check-config")
    config = payload["config"]
    if "." in field:
        parent, key = field.split(".")
        config[parent][key] = value
    else:
        config[field] = value
    with pytest.raises(ValueError, match="frozen development reference"):
        study.validate_quality_jobs([job], [(payload, options)], target)


def test_quality_config_allows_only_planned_policy_and_seed_changes(frozen_quality):
    _, target = frozen_quality
    job = study.build_plan("quality-confirmation")["jobs"][0]
    payload, options = study.prepare_job(job, "check-policy")
    payload["config"].update(seed=2027, batch_size=1, grad_accum_steps=8)
    payload["config"]["model"]["activation_checkpointing"] = "block"
    study.validate_quality_jobs([job], [(payload, options)], target)
    options["seeds"] = [1337]
    with pytest.raises(ValueError, match="frozen quality protocol"):
        study.validate_quality_jobs([job], [(payload, options)], target)


def test_quality_protocol_must_match_the_frozen_target(frozen_quality):
    _, target = frozen_quality
    job = study.build_plan("quality-confirmation")["jobs"][0]
    prepared = [study.prepare_job(job, "check-protocol")]
    target["quality_protocol"]["target_step"] = 1000
    with pytest.raises(ValueError, match="frozen development protocol"):
        study.validate_quality_jobs([job], prepared, target)


def test_quality_packages_must_match_the_frozen_environment(frozen_quality):
    _, target = frozen_quality
    job = study.build_plan("quality-confirmation")["jobs"][0]
    payload, options = study.prepare_job(job, "check-packages")
    payload["packages"][0] = "torch==0.0.0"
    with pytest.raises(ValueError, match="frozen development environment"):
        study.validate_quality_jobs([job], [(payload, options)], target)


@pytest.mark.parametrize("change", ["target_bytes", "development_evidence"])
def test_confirmation_revalidates_target_before_each_job(monkeypatch, tmp_path, frozen_quality, change):
    path, _ = frozen_quality
    output = tmp_path / "confirmation"
    calls = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--stage", "quality-confirmation", "--execute",
                                     "--target-file", str(path), "--output-dir", str(output)])
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))

    def validate(path, repo):
        if calls and change == "development_evidence":
            raise ValueError("development evidence hash mismatch")
        return json.loads(path.read_text())

    def complete(payload, options, destination):
        calls.append(options["run_id"])
        destination.mkdir()
        (destination / "aggregate.json").write_text(json.dumps({"comparability": {"status": "passed"},
            "cases": [{"status": "completed"} for _ in range(3)]}))
        (destination / "evidence.json").write_text(json.dumps({"errors": [], "missing_files": []}))
        if change == "target_bytes":
            path.write_text(path.read_text() + "\n")  # Same parsed target; file identity still changed.
        return {"status": "finished", "evidence_status": "complete"}

    monkeypatch.setattr(study, "validate_frozen_target", validate)
    monkeypatch.setattr(study.launcher, "execute", complete)
    with pytest.raises((ValueError, RuntimeError), match="(target changed|evidence hash mismatch)"):
        study.main()
    assert len(calls) == 1
    state = json.loads((output / "campaign-summary.json").read_text())
    assert state["status"] == "blocked"
    assert len(state["jobs"]) == 1


def test_only_explicit_screen_resource_limits_allow_continuation():
    job = {"case_count": 2, "allow_resource_limits": True}
    summary = {"status": "failed", "evidence_status": "partial"}
    aggregate = {"comparability": {"status": "insufficient"}, "cases": [
        {"status": "completed"}, {"status": "failed", "failure_kind": "oom"}]}
    assert study.classify_result(summary, aggregate, job) == "completed_with_resource_limits"
    assert study.classify_result(dict(summary, evidence_status="failed"), aggregate, job) == "blocked"
    assert study.classify_result(summary, aggregate, dict(job, allow_resource_limits=False)) == "blocked"
    aggregate["cases"][1]["failure_kind"] = "process_error"
    assert study.classify_result(summary, aggregate, job) == "blocked"
    aggregate["cases"][1] = {"status": "timeout", "failure_kind": "timeout"}
    assert study.classify_result(summary, aggregate, job) == "completed_with_resource_limits"
    aggregate["comparability"]["status"] = "failed"
    assert study.classify_result(summary, aggregate, job) == "blocked"


def test_incomplete_jobs_or_missing_success_evidence_block_continuation():
    job = {"case_count": 2, "allow_resource_limits": True}
    summary = {"status": "finished", "evidence_status": "complete"}
    aggregate = {"comparability": {"status": "passed"}, "cases": [{"status": "completed"}]}
    assert study.classify_result(summary, aggregate, job) == "blocked"
    aggregate["cases"].append({"status": "completed"})
    assert study.classify_result(summary, aggregate, job) == "completed"
    summary["evidence_status"] = "partial"
    assert study.classify_result(summary, aggregate, job) == "blocked"


def test_source_snapshot_detects_changed_sources(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("old")
    payload = {"files": [{"local": str(path), "remote": "/project/src/source.py"}]}
    original = study.source_snapshot([payload])
    path.write_text("new")
    assert study.source_snapshot([payload]) != original


def test_dry_run_does_not_create_run_directory(monkeypatch, tmp_path, capsys):
    output = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--stage", "development", "--output-dir", str(output)])
    def prepare(job, run_id):
        return {"files": [], "config": job["overrides"]}, {"run_id": run_id}
    monkeypatch.setattr(study, "prepare_job", prepare)
    monkeypatch.setattr(study.launcher, "execute", lambda *args: (_ for _ in ()).throw(
        AssertionError("dry run contacted the launcher")))
    study.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry_run"
    assert not output.exists()


def test_quality_overrides_reach_payload(monkeypatch):
    base = {"max_steps": 30, "warmup_steps": 5, "eval_batch_size": 1,
            "eval_batches": 4, "checkpoint_interval": 30, "eval_interval": 30}
    monkeypatch.setattr(study.launcher, "prepare_payload", lambda path: {"config": base.copy(), "files": []})
    job = study.build_plan("development")["jobs"][-1]
    payload, options = study.prepare_job(job, "test-quality")
    config = payload["config"]
    assert (config["max_steps"], config["warmup_steps"], config["eval_batches"]) == (1200, 50, 64)
    assert config["eval_batch_size"] == 1
    assert options["policies"] == job["policies"]
    assert base["max_steps"] == 30


def test_only_limited_case_artifacts_may_be_missing():
    job = {"case_count": 2, "allow_resource_limits": True}
    summary = {"status": "failed", "evidence_status": "partial"}
    aggregate = {"comparability": {"status": "insufficient"}, "cases": [
        {"case_id": "good", "status": "completed"},
        {"case_id": "oom", "status": "failed", "failure_kind": "oom"}]}
    evidence = {"errors": [], "missing_files": ["pilot/runs/oom/summary.json"]}
    assert study.classify_result(summary, aggregate, job, evidence) == "completed_with_resource_limits"
    evidence["missing_files"] = ["pilot/runs/good/metrics.jsonl"]
    assert study.classify_result(summary, aggregate, job, evidence) == "blocked"
    evidence["missing_files"] = ["tests.log"]
    assert study.classify_result(summary, aggregate, job, evidence) == "blocked"
    evidence.update(missing_files=[], errors=["metadata could not be parsed"])
    assert study.classify_result(summary, aggregate, job, evidence) == "blocked"


def test_existing_output_directory_prevents_repeat_launch(monkeypatch, tmp_path):
    output = tmp_path / "already-launched"
    output.mkdir()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--execute", "--output-dir", str(output)])
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(study, "prepare_job", lambda job, run_id: ({"files": [], "config": {}}, {"run_id": run_id}))
    monkeypatch.setattr(study.launcher, "execute", lambda *args: (_ for _ in ()).throw(
        AssertionError("repeat attempt launched billable work")))
    with pytest.raises(FileExistsError):
        study.main()


def test_unexpected_failure_stops_later_jobs(monkeypatch, tmp_path):
    output = tmp_path / "campaign"
    calls = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--execute", "--output-dir", str(output)])
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(study, "prepare_job", lambda job, run_id: ({"files": [], "config": {}}, {"run_id": run_id}))
    def fail(payload, options, destination):
        calls.append(options["run_id"])
        destination.mkdir()
        (destination / "aggregate.json").write_text(json.dumps({"cases": [], "comparability": {"status": "failed"}}))
        (destination / "evidence.json").write_text(json.dumps({"errors": [], "missing_files": []}))
        return {"status": "failed", "evidence_status": "complete"}
    monkeypatch.setattr(study.launcher, "execute", fail)
    with pytest.raises(RuntimeError, match="failed a check"):
        study.main()
    assert len(calls) == 1
    state = json.loads((output / "campaign-summary.json").read_text())
    assert state["status"] == "blocked"
    assert len(state["jobs"]) == 1
