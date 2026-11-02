"""Frozen-target integrity checks without cloud SDKs or production data."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/quality_target.py"
spec = importlib.util.spec_from_file_location("quality_target_test", SCRIPT)
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def modify(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


@pytest.fixture
def development(tmp_path):
    repo, archive = tmp_path / "repo", tmp_path / "archive"
    package = repo / "src/llmlab"
    package.mkdir(parents=True)
    (package / "train.py").write_text("# Synthetic training source for gate tests\n")
    archive.mkdir()
    data_files, data_hashes = [], {}
    for name in ("train.npy", "valid.npy", "vocab.pkl", "merges.pkl"):
        path = tmp_path / name
        path.write_bytes(("fixture " + name).encode())
        remote = "/data/" + name
        data_files.append({"local": str(path), "remote": remote})
        data_hashes[name] = target.file_sha256(path)
    protocol = {"development_seed": 1337, "max_steps": 1200, "tokens_per_step": 8192,
                "train_warmup_steps": 50, "eval_interval": 50, "eval_batches": 64,
                "eval_batch_size": 1, "checkpoint_interval": 1200,
                "target_step": 600, "target_rounding": "0.05",
                "target_rule": "ceil(max(development validation losses at step 600) / 0.05) * 0.05",
                "policies": list(target.POLICIES)}
    reference = {"seed": 1337, "max_steps": 1200, "warmup_steps": 50,
                 "eval_interval": 50, "eval_batches": 64, "eval_batch_size": 1,
                 "checkpoint_interval": 1200, "batch_size": 8, "grad_accum_steps": 1,
                 "learning_rate": 0.0003, "model": {"context_length": 1024,
                 "attention_backend": "sdpa", "activation_checkpointing": "none"},
                 "data": {"kind": "npy", "train_path": "/data/train.npy",
                 "valid_path": "/data/valid.npy", "tokenizer_files": ["/data/vocab.pkl", "/data/merges.pkl"]}}
    options = {"run_id": "fixture-quality-dev", "seeds": [1337], "policies": list(target.POLICIES)}
    identity = {"kind": "npy", "train": {"sha256": data_hashes["train.npy"]},
                "valid": {"sha256": data_hashes["valid.npy"]},
                "tokenizer_files": [{"name": name, "sha256": data_hashes[name]}
                                    for name in ("vocab.pkl", "merges.pkl")]}
    source = target.training_source_sha256(repo)
    evidence = {"errors": [], "missing_files": [], "cases": {}, "manifest": {"cases": []}}
    aggregate = {"comparability": {"status": "passed"}, "cases": []}
    losses = [2.3875473272055387, 2.3878817595541477, 2.3883473575115204]
    for policy, loss in zip(target.POLICIES, losses):
        backend, checkpoint, microbatch = policy.split(":")
        config = copy.deepcopy(reference)
        config["batch_size"], config["grad_accum_steps"] = int(microbatch), 8 // int(microbatch)
        config["model"]["activation_checkpointing"] = checkpoint
        config["data"]["synthetic_tokens"] = 4096
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        case_id = f"{backend}-{checkpoint}-mb{microbatch}"
        metrics = []
        for step in range(1, 1201):
            metrics.append({"event": "train", "step": step, "tokens_seen": step * 8192,
                            "batch_start_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
                            "learning_rate": 0.0003, "loss": 3.0, "wall_seconds": step / 10})
            if step % 50 == 0:
                metrics.append({"event": "eval", "step": step, "valid_tokens": 65536,
                                "valid_loss": loss if step == 600 else 3.0,
                                "wall_seconds": step / 10 + 0.01})
        evidence["cases"][case_id] = {"metadata": {"config": config, "config_sha256": config_hash,
            "data_identity": copy.deepcopy(identity), "packages": {"torch": "2.11.0"},
            "parameter_count": 100313856, "source": {"python_source_sha256": source}},
            "execution": {"returncode": 0}, "summary": {"status": "completed", "step": 1200},
            "metrics": metrics}
        evidence["manifest"]["cases"].append({"case_id": case_id, "config": copy.deepcopy(config),
                                              "config_sha256": config_hash, "source_sha256": source})
        aggregate["cases"].append({"case_id": case_id, "config_sha256": config_hash, "status": "completed"})
    write_json(archive / "campaign.json", {"stage": "development", "quality_protocol": protocol,
        "resolved_jobs": [{"config": reference, "options": options}]})
    write_json(archive / "launch.json", {"payload": {"config": reference, "files": data_files,
        "packages": ["torch==2.11.0"]}, "options": options})
    write_json(archive / "launch-summary.json", {"run_id": options["run_id"],
        "status": "finished", "evidence_status": "complete"})
    write_json(archive / "aggregate.json", aggregate)
    write_json(archive / "evidence.json", evidence)
    return repo, archive


def test_freeze_computes_exact_prespecified_target_and_revalidates(development):
    repo, archive = development
    frozen = target.freeze_target(archive, repo)
    assert frozen["target_validation_loss"] == 2.4
    assert frozen["target_loss_decimal"] == "2.40"
    assert frozen["reference_config"]["data"].get("synthetic_tokens") is None
    assert set(frozen["input_sha256"]) == set(target.INPUTS)
    assert target.validate_frozen_target(archive / "frozen-target.json", repo) == frozen


@pytest.mark.parametrize(("loss", "expected"), [(2.4, "2.40"), (2.4000000000000004, "2.45")])
def test_decimal_ceiling_does_not_round_down_at_boundary(development, loss, expected):
    _, archive = development
    def change(evidence):
        for case in evidence["cases"].values():
            for event in case["metrics"]:
                if event["event"] == "eval" and event["step"] == 600:
                    event["valid_loss"] = loss
    modify(archive / "evidence.json", change)
    assert target.derive_target(archive)["target_loss_decimal"] == expected


@pytest.mark.parametrize("event_kind", ["train", "eval"])
def test_incomplete_metrics_cannot_freeze(development, event_kind):
    repo, archive = development
    def change(evidence):
        case = next(iter(evidence["cases"].values()))
        case["metrics"] = [event for event in case["metrics"]
                           if not (event["event"] == event_kind and event["step"] == 600)]
    modify(archive / "evidence.json", change)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        target.freeze_target(archive, repo)
    assert not (archive / "frozen-target.json").exists()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("batch_start_sha256", "changed", "ordered batches"),
    ("learning_rate", 0.01, "ordered batches"),
    ("tokens_seen", 1, "training-token"),
    ("loss", float("nan"), "Nonfinite"),
])
def test_bad_or_unpaired_training_metrics_are_rejected(development, field, value, message):
    _, archive = development
    def change(evidence):
        case = list(evidence["cases"].values())[1]
        case["metrics"][0][field] = value
    modify(archive / "evidence.json", change)
    with pytest.raises(ValueError, match=message):
        target.derive_target(archive)


@pytest.mark.parametrize(("field", "message"), [
    ("source", "training source"), ("config", "fixed configs"), ("data_identity", "data_identity"),
])
def test_mismatched_case_provenance_is_rejected(development, field, message):
    _, archive = development
    def change(evidence):
        metadata = list(evidence["cases"].values())[1]["metadata"]
        if field == "source":
            metadata[field]["python_source_sha256"] = "changed"
        elif field == "config":
            metadata[field]["learning_rate"] = 0.1
        else:
            metadata[field]["train"]["sha256"] = "changed"
    modify(archive / "evidence.json", change)
    with pytest.raises(ValueError, match=message):
        target.derive_target(archive)


@pytest.mark.parametrize("filename", ["frozen-target.json", "evidence.json"])
def test_changed_frozen_value_or_raw_evidence_is_rejected(development, filename):
    repo, archive = development
    target.freeze_target(archive, repo)
    if filename == "frozen-target.json":
        modify(archive / filename, lambda value: value.update(target_validation_loss=2.5))
    else:
        # Even a semantically irrelevant edit changes the archived input hash.
        with (archive / filename).open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match="target or its development evidence changed"):
        target.validate_frozen_target(archive / "frozen-target.json", repo)


@pytest.mark.parametrize("changed_file", ["train.npy", "vocab.pkl", "source"])
def test_current_training_inputs_must_still_match_development(development, changed_file):
    repo, archive = development
    frozen = target.freeze_target(archive, repo)
    path = (repo / "src/llmlab/train.py" if changed_file == "source" else
            next(Path(item["path"]) for item in frozen["data_files"] if Path(item["path"]).name == changed_file))
    path.write_bytes(b"changed after development")
    with pytest.raises(ValueError, match="changed since development"):
        target.validate_frozen_target(archive / "frozen-target.json", repo)


def test_freeze_never_overwrites_an_existing_target(development):
    repo, archive = development
    target.freeze_target(archive, repo)
    original = (archive / "frozen-target.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        target.freeze_target(archive, repo)
    assert (archive / "frozen-target.json").read_bytes() == original


def test_changed_prelaunch_protocol_cannot_select_a_new_target(development):
    _, archive = development
    modify(archive / "campaign.json", lambda value: value["quality_protocol"].update(target_step=550))
    with pytest.raises(ValueError, match="Unexpected development protocol"):
        target.derive_target(archive)


def test_launch_must_match_prelaunch_campaign(development):
    _, archive = development
    modify(archive / "campaign.json", lambda value: value["resolved_jobs"][0]["config"].update(learning_rate=0.2))
    with pytest.raises(ValueError, match="prelaunch campaign"):
        target.derive_target(archive)
