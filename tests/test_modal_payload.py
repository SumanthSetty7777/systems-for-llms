"""Local-only launcher boundaries; no Modal SDK or network is used."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_pilot.py"
spec = importlib.util.spec_from_file_location("modal_pilot_local", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def payload_fixture(tmp_path, kind="npy"):
    repo = tmp_path / "repo"
    (repo / "src" / "llmlab").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()
    (repo / "src" / "llmlab" / "train.py").write_text("# source\n")
    (repo / "tests" / "test_model.py").write_text("# tests\n")
    (repo / "pyproject.toml").write_text("# metadata\n")
    (repo / "scripts" / "modal_pilot.py").write_text("# launcher\n")
    (repo / "secret.env").write_text("not part of upload\n")
    (repo / "src" / "unrelated.json").write_text("not part of upload\n")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.npy", "valid.npy", "vocab.pkl"):
        (data / name).write_bytes(name.encode())
    config = {"data": {"kind": kind, "train_path": "../data/train.npy",
                       "valid_path": "../data/valid.npy",
                       "tokenizer_files": ["../data/vocab.pkl"]},
              "device": "cpu", "precision": "fp32", "max_steps": 10}
    path = repo / "config.json"
    path.write_text(json.dumps(config))
    return repo, path, config


def test_payload_uses_only_explicit_project_and_data_files(tmp_path):
    repo, path, original = payload_fixture(tmp_path)
    payload = launcher.prepare_payload(path, repo)
    assert {item["remote"] for item in payload["files"]} == {
        "/project/src/llmlab/train.py", "/project/tests/test_model.py",
        "/project/pyproject.toml", "/data/train.npy", "/data/valid.npy",
        "/project/scripts/modal_pilot.py",
        "/data/tokenizer-0/vocab.pkl",
    }
    assert payload["config"]["data"]["train_path"] == "/data/train.npy"
    assert payload["config"]["data"]["valid_path"] == "/data/valid.npy"
    assert payload["config"]["device"] == "cuda"
    assert payload["config"]["precision"] == "bf16"
    assert json.loads(path.read_text()) == original
    assert payload["upload_bytes"] == sum(Path(item["local"]).stat().st_size
                                           for item in payload["files"])


def test_payload_refuses_missing_data_before_launch(tmp_path):
    repo, path, _ = payload_fixture(tmp_path)
    (tmp_path / "data" / "train.npy").unlink()
    with pytest.raises(FileNotFoundError):
        launcher.prepare_payload(path, repo)


def test_payload_refuses_source_symlinks(tmp_path):
    repo, path, _ = payload_fixture(tmp_path)
    (repo / "src" / "linked.py").symlink_to(repo / "secret.env")
    with pytest.raises(ValueError, match="symlinks"):
        launcher.prepare_payload(path, repo)


def test_dry_run_cli_makes_no_output_directory(tmp_path):
    output = tmp_path / "should-not-exist"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config",
         str(SCRIPT.parents[1] / "configs" / "smoke.json"),
         "--output-dir", str(output), "--run-id", "dry-run-check"],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report["mode"] == "dry_run"
    assert report["options"]["run_id"] == "dry-run-check"
    assert not output.exists()


def test_execute_rejects_wrong_python_before_modal_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "version_info", (3, 14, 0))
    output = tmp_path / "should-not-exist"
    with pytest.raises(RuntimeError, match="Python 3.12"):
        launcher.execute({}, {}, output)
    assert not output.exists()


def test_payload_includes_only_named_optional_campaign_script(tmp_path):
    repo, path, _ = payload_fixture(tmp_path)
    (repo / "scripts" / "run_study.py").write_text("# bounded campaign\n")
    (repo / "scripts" / "quality_target.py").write_text("# frozen target gate\n")
    (repo / "scripts" / "private_script.py").write_text("# not mounted\n")
    payload = launcher.prepare_payload(path, repo)
    remotes = {item["remote"] for item in payload["files"]}
    assert "/project/scripts/run_study.py" in remotes
    assert "/project/scripts/quality_target.py" in remotes
    assert "/project/scripts/private_script.py" not in remotes


def test_training_overrides_resolve_in_payload_without_changing_source_config(tmp_path):
    output = tmp_path / "should-not-exist"
    config = SCRIPT.parents[1] / "configs" / "smoke.json"
    before = config.read_bytes()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config),
         "--output-dir", str(output), "--max-steps", "1200",
         "--train-warmup-steps", "50", "--eval-interval", "50",
         "--eval-batches", "64", "--checkpoint-interval", "1200",
         "--microbatches", "1", "8", "--effective-batch-size", "8",
         "--policies", "sdpa:none:1", "sdpa:none:8", "sdpa:block:8"],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    actual = report["payload"]["config"]
    assert {key: actual[key] for key in (
        "max_steps", "warmup_steps", "eval_interval", "eval_batches", "checkpoint_interval"
    )} == {"max_steps": 1200, "warmup_steps": 50, "eval_interval": 50,
           "eval_batches": 64, "checkpoint_interval": 1200}
    assert report["options"]["warmup_steps"] == 5  # timing warmup remains separate
    assert report["options"]["policies"] == ["sdpa:none:1", "sdpa:none:8", "sdpa:block:8"]
    assert "max_steps" not in report["options"]
    assert config.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("flags", [
    ["--max-steps", "0"],
    ["--max-steps", "4"],  # fewer updates than the default timing warmup
    ["--train-warmup-steps", "-1"],
    ["--max-steps", "20", "--train-warmup-steps", "20"],
    ["--eval-interval", "0"],
    ["--eval-batches", "-1"],
    ["--checkpoint-interval", "0"],
    ["--policies", "sdpa:none:3"],  # not requested and does not divide E=8
    ["--policies", "sdpa:none:01"],
    ["--policies", "sdpa:none:1", "sdpa:none:1"],
    ["--policies", "flash:none:1"],
    ["--policies", "sdpa:layer:1"],
])
def test_invalid_overrides_or_policies_rejected_before_execute(tmp_path, flags):
    output = tmp_path / "should-not-exist"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config",
         str(SCRIPT.parents[1] / "configs" / "smoke.json"), "--execute",
         "--output-dir", str(output), *flags], capture_output=True, text=True,
    )
    assert result.returncode == 2  # argparse validation, before SDK import/execute
    assert not output.exists()


def test_default_overrides_preserve_config_and_zero_training_warmup_is_valid():
    config = {"max_steps": 20, "warmup_steps": 2, "eval_batches": 4}
    assert launcher.apply_config_overrides(config, {}) == config
    assert launcher.apply_config_overrides(config, {"train_warmup_steps": 0})["warmup_steps"] == 0
    with pytest.raises(ValueError, match="integer"):
        launcher.apply_config_overrides(config, {"max_steps": True})
    with pytest.raises(ValueError, match="Unknown"):
        launcher.apply_config_overrides(config, {"learning_rate": 0.1})


def evidence_fixture(root):
    case_id = "sdpa-none-mb1-acc8-seed1337"
    run = root / "pilot" / "runs" / case_id
    run.mkdir(parents=True)
    (root / "pilot" / "execution").mkdir()
    (root / "pilot" / "logs").mkdir()
    (root / "pilot" / "manifest.json").write_text(json.dumps({"cases": [{"case_id": case_id}]}))
    (root / "config.json").write_text('{"max_steps": 2}')
    (root / "preflight.json").write_text('{"cuda_available": true}')
    for name in ("tests.log", "preflight.log", "pilot.log"):
        (root / name).write_text("121 passed\n")
    (run / "metadata.json").write_text('{"source": {"python_source_sha256": "source"}}')
    (run / "summary.json").write_text('{"status": "completed", "step": 2}')
    (root / "pilot" / "execution" / f"{case_id}.json").write_text('{"returncode": 0}')
    (root / "pilot" / "logs" / f"{case_id}.log").write_text("case log\n")
    metrics = [{"event": "train", "step": 1, "wall_seconds": 0.5},
               {"event": "eval", "step": 2, "wall_seconds": 1.2, "valid_loss": 4.2}]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, metrics)) + "\n")
    (run / "checkpoint.pt").write_bytes(b"checkpoint must never enter reports")
    return case_id, run, metrics


def test_evidence_retains_curves_logs_and_provenance_but_never_checkpoints(tmp_path):
    case_id, _, metrics = evidence_fixture(tmp_path)
    evidence = launcher.collect_evidence(tmp_path)
    assert evidence["errors"] == evidence["missing_files"] == evidence["truncated_logs"] == []
    assert evidence["cases"][case_id]["metrics"] == metrics
    assert evidence["cases"][case_id]["metadata"]["source"]["python_source_sha256"] == "source"
    assert evidence["logs"]["tests.log"] == "121 passed\n"
    assert "checkpoint must never" not in json.dumps(evidence)


def test_evidence_preserves_partial_metrics_and_records_corruption_and_missing_files(tmp_path):
    case_id, run, metrics = evidence_fixture(tmp_path)
    with (run / "metrics.jsonl").open("a") as stream:
        stream.write('{"unfinished":\n')
    (run / "summary.json").unlink()
    (root_log := tmp_path / "pilot" / "logs" / f"{case_id}.log").write_text("prefix---out of memory")
    evidence = launcher.collect_evidence(tmp_path, log_limit=13)
    assert evidence["cases"][case_id]["metrics"] == metrics
    assert evidence["cases"][case_id]["summary"] is None
    assert evidence["cases"][case_id]["log_tail"] == "out of memory"
    assert str(root_log.relative_to(tmp_path)) in evidence["truncated_logs"]
    assert any("summary.json" in path for path in evidence["missing_files"])
    assert any("metrics.jsonl:3" in error for error in evidence["errors"])


def test_evidence_recovers_test_failures_before_plan_exists(tmp_path):
    (tmp_path / "tests.log").write_text("1 failed, 120 passed\n")
    evidence = launcher.collect_evidence(tmp_path)
    assert evidence["logs"]["tests.log"] == "1 failed, 120 passed\n"
    assert evidence["manifest"] is None
    assert evidence["cases"] == {}
    assert "pilot/manifest.json" in evidence["missing_files"]


def test_evidence_rejects_path_traversal_in_case_identifiers(tmp_path):
    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot" / "manifest.json").write_text(
        json.dumps({"cases": [{"case_id": "../../outside"}]}))
    evidence = launcher.collect_evidence(tmp_path)
    assert evidence["cases"] == {}
    assert "invalid case identifier" in evidence["errors"][0]
