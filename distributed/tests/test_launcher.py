"""Cloud-launch boundaries and evidence retention, without any cloud requests."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_modal.py"
spec = importlib.util.spec_from_file_location("distributed_launcher_test", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def settings(gpu="H100:2", stage="smoke", world_size=2):
    return {"gpu": gpu, "stage": stage, "job_timeout_seconds": 900, "volume": "results-test",
            "run_id": "distributed-test", "plan": {"stage": stage, "required_gpus": world_size,
              "cases": [{"case_id": "ddp-test", "world_size": world_size}]}}


@pytest.fixture
def payload_repo(tmp_path):
    repo = tmp_path / "repo"
    for directory in ("src/llmlab", "tests", "scripts", "configs", "distributed/llmdist", "distributed/tests"):
        (repo / directory).mkdir(parents=True)
    shutil.copyfile(SCRIPT.parents[1] / "scripts/modal_pilot.py", repo / "scripts/modal_pilot.py")
    (repo / "pyproject.toml").write_text("# project\n")
    (repo / "src/llmlab/train.py").write_text("# training\n")
    (repo / "tests/test_model.py").write_text("# model checks\n")
    (repo / "distributed/llmdist/train.py").write_text("# distributed training\n")
    (repo / "distributed/tests/test_train.py").write_text("# distributed checks\n")
    for name in ("run_modal.py", "run_experiments.py"):
        (repo / "distributed" / name).write_text("# explicit script\n")
    (repo / "distributed/secret.env").write_text("not uploaded\n")
    (repo / "distributed/private.py").write_text("# not an enumerated script\n")
    (repo / "distributed/private.json").write_text("{}")
    (repo / "distributed/private.pt").write_bytes(b"not uploaded")
    config = {"data": {"kind": "synthetic", "tokenizer_files": []}, "max_steps": 10}
    path = repo / "configs/smoke.json"
    path.write_text(json.dumps(config))
    return repo, path


def test_payload_only_mounts_enumerated_code_and_data(payload_repo):
    repo, path = payload_repo
    (repo / "distributed/run_campaign.py").write_text("# sequential bounded stages\n")
    (repo / "distributed/analyze.py").write_text("# evidence analysis\n")
    payload = launcher.prepare_payload(path, repo)
    paths = {item["remote"] for item in payload["files"]}
    assert {"/project/distributed/run_modal.py", "/project/distributed/run_experiments.py",
            "/project/distributed/run_campaign.py", "/project/distributed/analyze.py", "/project/distributed/llmdist/train.py",
            "/project/distributed/tests/test_train.py"} <= paths
    assert not any("private" in path or "secret" in path for path in paths)
    assert payload["source_sha256"] == launcher.source_snapshot(payload["files"])
    assert payload["upload_bytes"] == sum(item["bytes"] for item in payload["files"])
    assert payload["config"]["device"] == "cuda" and payload["config"]["precision"] == "bf16"


def test_payload_rejects_distributed_symlink(payload_repo):
    repo, path = payload_repo
    (repo / "distributed/llmdist/leak.py").symlink_to(repo / "distributed/secret.env")
    with pytest.raises(ValueError, match="symlinks"):
        launcher.prepare_payload(path, repo)


@pytest.mark.parametrize("change", [
    lambda value: value.update(gpu="H100:1"),
    lambda value: value.update(job_timeout_seconds=0),
    lambda value: value.update(job_timeout_seconds=9000),
    lambda value: value.update(job_timeout_seconds=True),
    lambda value: value.update(run_id="../bad"),
    lambda value: value["plan"].update(cases=[]),
    lambda value: value["plan"].update(required_gpus=8),
    lambda value: value["plan"]["cases"][0].update(world_size=8),
    lambda value: value["plan"]["cases"].append(dict(value["plan"]["cases"][0])),
])
def test_invalid_controls_rejected_before_sdk(change):
    value = settings()
    change(value)
    with pytest.raises(ValueError):
        launcher.validate_options(value)


def test_scale_counts_all_allocated_gpus_in_estimate():
    value = settings("H100:8", "scale", 8)
    launcher.validate_options(value)
    assert launcher.resources(value) == {"gpu_count": 8, "cpu_cores": 16, "ram_gib": 64}
    cost = launcher.cost_estimate(value)
    assert cost["requested_resources_usd_at_timeout"] == pytest.approx(8.2149)
    assert cost["allocated_gpu_count"] == 8


def test_h200_confirmation_accounts_for_all_reserved_resources():
    value = settings("H200:8", "confirmation", 8)
    value["job_timeout_seconds"] = 1800
    launcher.validate_options(value)
    assert launcher.resources(value) == {"gpu_count": 8, "cpu_cores": 16, "ram_gib": 64}
    cost = launcher.cost_estimate(value)
    expected = 1800 * (8 * 0.001261 + 16 * 0.0000131 + 64 * 0.00000222)
    assert cost["requested_resources_usd_at_timeout"] == pytest.approx(round(expected, 4))
    assert cost["requested_resources_usd_at_timeout"] == pytest.approx(18.7914)
    assert cost["allocated_gpu_count"] == 8


def test_wrong_python_does_not_create_output_or_import_sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "version_info", (3, 14, 0))
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Python 3.12"):
        launcher.execute({}, settings(), output)
    assert not output.exists()


def test_existing_output_rejected_before_sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "version_info", (3, 12, 0))
    with pytest.raises(FileExistsError):
        launcher.execute({}, settings(), tmp_path)


def test_changed_source_rejected_before_sdk(monkeypatch, tmp_path, payload_repo):
    monkeypatch.setattr(launcher.sys, "version_info", (3, 12, 0))
    repo, config = payload_repo
    payload = launcher.prepare_payload(config, repo)
    (repo / "distributed/llmdist/train.py").write_text("# changed\n")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Source files changed"):
        launcher.execute(payload, settings(), output)
    assert not output.exists()


def test_preflight_failure_prevents_tests_and_experiments(monkeypatch, tmp_path):
    calls, commits = [], []
    def fail(args, log, timeout, cwd):
        calls.append(log.name)
        log.write_text("missing CUDA GPU\n")
        raise RuntimeError("preflight failure")
    monkeypatch.setattr(launcher, "run_command", fail)
    reports = launcher.run_remote({}, settings(), tmp_path / "run", lambda: commits.append(True), project=tmp_path)
    assert calls == ["preflight.log"]
    assert reports["launch-summary.json"]["status"] == "failed"
    assert "preflight failure" in reports["launch-summary.json"]["error"]
    assert reports["evidence.json"]["logs"]["preflight.log"] == "missing CUDA GPU\n"
    assert commits


def test_command_failure_includes_actual_log_tail(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="specific diagnostic failure"):
        launcher.run_command([sys.executable, "-c", "print('specific diagnostic failure'); raise SystemExit(7)"],
                             log, 10, tmp_path)
    assert "specific diagnostic failure" in log.read_text()


@pytest.mark.parametrize("failure", ["tests.log", "study.log"])
def test_remote_failure_retains_logs_and_stops(monkeypatch, tmp_path, failure):
    calls = []
    def command(args, log, timeout, cwd):
        calls.append(log.name)
        log.write_text("output\n")
        if log.name == "preflight.log":
            (log.parent / "preflight.json").write_text('{"devices": [0, 1]}')
        if log.name == "tests.log":
            assert timeout == 600
            (log.parent / "tests.xml").write_text('<testsuites><testsuite><testcase name="ok"/></testsuite></testsuites>')
        if log.name == failure:
            raise RuntimeError("expected command failure")
    monkeypatch.setattr(launcher, "run_command", command)
    reports = launcher.run_remote({}, settings(), tmp_path / "run", lambda: None, project=tmp_path)
    assert reports["launch-summary.json"]["status"] == "failed"
    assert calls[-1] == failure
    assert failure in reports["evidence.json"]["logs"]


@pytest.mark.parametrize("status", ["completed", "oom"])
def test_remote_aggregate_controls_success_and_collects_metrics(monkeypatch, tmp_path, status):
    def command(args, log, timeout, cwd):
        log.write_text("output\n")
        if log.name == "preflight.log":
            assert args[-1] == "2"
            (log.parent / "preflight.json").write_text('{"devices": [0, 1]}')
        if log.name == "tests.log":
            (log.parent / "tests.xml").write_text('<testsuites><testsuite><testcase name="ok"/></testsuite></testsuites>')
        if log.name == "study.log":
            assert args[args.index("--allocated-gpus") + 1] == "2"
            study = log.parent / "study"
            study.mkdir()
            (study / "aggregate.json").write_text(json.dumps({"status": "completed" if status == "completed" else "failed",
                      "cases": [{"case_id": "ddp-test", "status": status}]}))
            case = study / "cases/ddp-test"
            case.mkdir(parents=True)
            (case / "summary.json").write_text('{}')
            for rank in (0, 1):
                (case / f"rank-{rank}-metrics.jsonl").write_text('{"event":"train","step":1}\n')
                (case / f"rank-{rank}-metadata.json").write_text('{}')
                (case / f"rank-{rank}-summary.json").write_text('{}')
            (case / "checkpoint.pt").write_bytes(b"do not return tensors")
    monkeypatch.setattr(launcher, "run_command", command)
    reports = launcher.run_remote({}, settings(), tmp_path / "run", lambda: None, project=tmp_path)
    assert reports["launch-summary.json"]["status"] == ("completed" if status == "completed" else "failed")
    assert reports["evidence.json"]["metrics"]["study/cases/ddp-test/rank-0-metrics.jsonl"] == [{"event": "train", "step": 1}]
    assert reports["evidence.json"]["json"]["tests-report.json"]["passed"] == 1
    assert "do not return tensors" not in json.dumps(reports)


def test_malformed_metric_is_retained_as_evidence_error(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("not JSON\n")
    result = launcher.collect_evidence(tmp_path)
    assert result["errors"] and "metrics.jsonl" in result["errors"][0]


def test_evidence_log_tail_is_bounded(tmp_path):
    (tmp_path / "long.log").write_text("x" * 1000)
    result = launcher.collect_evidence(tmp_path, log_limit=20)
    assert result["logs"]["long.log"] == "x" * 20
    assert result["truncated_logs"] == ["long.log"]


def test_completed_case_requires_every_rank_evidence(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(settings()["plan"]))
    (tmp_path / "study").mkdir()
    (tmp_path / "study/aggregate.json").write_text(json.dumps({"cases": [{"case_id": "ddp-test", "status": "completed"}]}))
    evidence = launcher.collect_evidence(tmp_path)
    assert "study/cases/ddp-test/rank-1-metrics.jsonl" in evidence["missing_files"]
    assert "study/cases/ddp-test/summary.json" in evidence["missing_files"]


def test_gpu_test_skips_block_experiments(monkeypatch, tmp_path):
    calls = []
    def command(args, log, timeout, cwd):
        calls.append(log.name)
        log.write_text("output\n")
        if log.name == "preflight.log":
            (log.parent / "preflight.json").write_text('{"devices": [0, 1]}')
        if log.name == "tests.log":
            (log.parent / "tests.xml").write_text('<testsuites><testsuite><testcase name="gpu"><skipped/></testcase></testsuite></testsuites>')
    monkeypatch.setattr(launcher, "run_command", command)
    result = launcher.run_remote({}, settings(), tmp_path / "run", lambda: None, project=tmp_path)
    assert calls == ["preflight.log", "tests.log"]
    assert result["launch-summary.json"]["status"] == "failed"
    assert result["evidence.json"]["json"]["tests-report.json"]["skipped"] == 1
