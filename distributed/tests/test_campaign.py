"""The large allocation is conditional on the small correctness gate passing."""

import builtins
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_campaign.py"
spec = importlib.util.spec_from_file_location("distributed_campaign_test", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    source = tmp_path / "train.py"
    source.write_text("# frozen training implementation\n")
    files = [{"local": str(source), "remote": "/project/distributed/llmdist/train.py", "bytes": source.stat().st_size}]
    payload = {"files": files, "source_sha256": campaign.launcher.source_snapshot(files), "config": {}}
    monkeypatch.setattr(campaign.launcher, "prepare_payload", lambda path: payload)
    result = campaign.prepare_campaign(tmp_path / "config.json", tmp_path / "campaign")
    return result, source


def test_prepare_freezes_both_stages_and_cost_without_output(prepared):
    result, _ = prepared
    definition = result["definition"]
    assert not Path(definition["output_dir"]).exists()
    stages = definition["stages"]
    assert [(s["name"], s["options"]["gpu"], s["options"]["job_timeout_seconds"]) for s in stages] == [
        ("smoke", "H100:2", 900), ("scale", "H100:8", 2400)]
    assert [s["options"]["plan"]["required_gpus"] for s in stages] == [2, 8]
    assert all(s["options"]["plan"]["plan_sha256"] for s in stages)
    assert definition["requested_resources_usd_at_timeouts"] == pytest.approx(24.0393)
    assert definition["requested_resources_usd_at_timeouts"] < definition["working_budget_usd"]


def test_dry_run_never_imports_sdk_or_creates_output(monkeypatch, prepared, capsys):
    result, _ = prepared
    output = Path(result["definition"]["output_dir"])
    monkeypatch.setattr(campaign, "prepare_campaign", lambda config, destination: result)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output-dir", str(output)])
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name != "modal", "A dry run must not import the cloud SDK"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    campaign.main()
    assert json.loads(capsys.readouterr().out)["mode"] == "dry_run"
    assert not output.exists()


def test_over_budget_rejected_before_output(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.launcher, "prepare_payload", lambda path: {})
    monkeypatch.setattr(campaign.launcher, "cost_estimate", lambda options: {"requested_resources_usd_at_timeout": 30})
    with pytest.raises(ValueError, match="working budget"):
        campaign.prepare_campaign(tmp_path / "config.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_output_is_never_overwritten(monkeypatch, prepared):
    result, _ = prepared
    output = Path(result["definition"]["output_dir"])
    output.mkdir()
    (output / "keep.txt").write_text("existing evidence")
    calls = []
    monkeypatch.setattr(campaign.launcher, "execute", lambda *args: calls.append(args))
    with pytest.raises(FileExistsError):
        campaign.execute_campaign(result)
    assert not calls
    assert (output / "keep.txt").read_text() == "existing evidence"


@pytest.mark.parametrize("mutation", ["source", "plan", "allocation"])
def test_frozen_changes_rejected_before_any_launch(monkeypatch, prepared, mutation):
    result, source = prepared
    if mutation == "source":
        source.write_text("# changed implementation\n")
    elif mutation == "plan":
        result["definition"]["stages"][0]["options"]["plan"]["cases"][0]["steps"] += 1
    else:
        result["definition"]["stages"][1]["options"]["gpu"] = "H100:4"
    calls = []
    monkeypatch.setattr(campaign.launcher, "execute", lambda *args: calls.append(args))
    with pytest.raises((ValueError, RuntimeError), match="changed"):
        campaign.execute_campaign(result)
    assert not calls
    assert not Path(result["definition"]["output_dir"]).exists()


def test_completed_campaign_is_sequential_and_persists_results(monkeypatch, prepared):
    result, _ = prepared
    calls = []

    def execute(payload, options, output):
        persisted = json.loads((output.parent / "campaign.json").read_text())
        assert persisted["status"] == "running"
        assert persisted["stages"][len(calls)]["status"] == "running"
        if calls:
            assert persisted["stages"][0]["status"] == "completed"
        calls.append(options["stage"])
        output.mkdir()
        return {"status": "completed", "function_seconds": 100, "case_status_counts": {"completed": len(options["plan"]["cases"])}}

    monkeypatch.setattr(campaign.launcher, "execute", execute)
    completed = campaign.execute_campaign(result)
    assert calls == ["smoke", "scale"]
    assert completed["status"] == "completed"
    disk = json.loads((Path(completed["definition"]["output_dir"]) / "campaign.json").read_text())
    assert disk == completed
    assert all(s["requested_resource_usd_for_reported_function_time"] > 0 for s in disk["stages"])
    with pytest.raises(ValueError, match="once"):
        campaign.execute_campaign(completed)


@pytest.mark.parametrize("failure", ["failed_summary", "exception"])
def test_failed_smoke_blocks_eight_gpu_allocation(monkeypatch, prepared, failure):
    result, _ = prepared
    calls = []

    def execute(payload, options, output):
        calls.append(options["stage"])
        if failure == "exception":
            raise RuntimeError("correctness gate failed")
        return {"status": "failed", "function_seconds": 10, "error": "GPU test failed"}

    monkeypatch.setattr(campaign.launcher, "execute", execute)
    failed = campaign.execute_campaign(result)
    assert calls == ["smoke"]
    assert failed["status"] == "failed"
    assert failed["stages"][1] == {"name": "scale", "status": "not_run", "blocked_by": "smoke"}
    saved = json.loads((Path(failed["definition"]["output_dir"]) / "campaign.json").read_text())
    assert saved["stages"][0]["error"]


def test_source_change_after_smoke_blocks_scale(monkeypatch, prepared):
    result, source = prepared
    calls = []

    def execute(payload, options, output):
        calls.append(options["stage"])
        source.write_text("# change after the correctness gate\n")
        return {"status": "completed", "function_seconds": 10}

    monkeypatch.setattr(campaign.launcher, "execute", execute)
    failed = campaign.execute_campaign(result)
    assert calls == ["smoke"]
    assert failed["stages"][0]["status"] == "completed"
    assert failed["stages"][1]["status"] == "failed"
    assert "Source files changed" in failed["stages"][1]["error"]


def test_scale_failure_is_retained_without_retry(monkeypatch, prepared):
    result, _ = prepared
    calls = []

    def execute(payload, options, output):
        calls.append(options["stage"])
        return {"status": "completed" if options["stage"] == "smoke" else "failed", "function_seconds": 10}

    monkeypatch.setattr(campaign.launcher, "execute", execute)
    failed = campaign.execute_campaign(result)
    assert calls == ["smoke", "scale"]
    assert failed["status"] == "failed"
    assert failed["stages"][0]["status"] == "completed"
    assert failed["stages"][1]["status"] == "failed"
