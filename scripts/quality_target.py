"""Freeze and verify the development-derived target before quality confirmation.

This module uses only the standard library and never contacts a cloud service.
The five input JSON files are retained unchanged beside frozen-target.json.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INPUTS = ("campaign.json", "launch.json", "launch-summary.json", "aggregate.json", "evidence.json")
POLICIES = ("sdpa:none:8", "sdpa:none:1", "sdpa:block:8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_source_sha256(repo: Path) -> str:
    # Keep the algorithm identical to llmlab.train.source_identity, without Torch.
    package = repo / "src/llmlab"
    paths = sorted(package.rglob("*.py"))
    require(bool(paths), "Training source is missing")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fixed_config(config: dict) -> dict:
    config = json.loads(json.dumps(config))
    for key in ("batch_size", "grad_accum_steps"):
        config.pop(key, None)
    config["model"].pop("activation_checkpointing", None)
    # The launch JSON omits this unused default added by the training dataclass.
    if config["data"].get("kind") == "npy" and config["data"].get("synthetic_tokens") == 4096:
        config["data"].pop("synthetic_tokens")
    return config


def derive_target(archive: Path) -> dict:
    """Recompute the target and provenance from complete paired development data."""
    raw = {name: (archive / name).read_bytes() for name in INPUTS}
    campaign, launch, summary, aggregate, evidence = (json.loads(raw[name]) for name in INPUTS)
    protocol = campaign["quality_protocol"]
    expected = {"development_seed": 1337, "max_steps": 1200, "tokens_per_step": 8192,
                "train_warmup_steps": 50, "eval_interval": 50, "eval_batches": 64,
                "eval_batch_size": 1, "checkpoint_interval": 1200,
                "target_step": 600, "target_rounding": "0.05",
                "target_rule": "ceil(max(development validation losses at step 600) / 0.05) * 0.05"}
    require(all(protocol.get(key) == value for key, value in expected.items())
            and protocol["policies"] == list(POLICIES), "Unexpected development protocol or target rule")
    require(campaign["stage"] == "development", "Target must use the development campaign")
    require(summary["status"] == "finished" and summary.get("evidence_status") == "complete",
            "Development must finish with complete evidence")
    require(not evidence["errors"] and not evidence["missing_files"], "Development evidence is incomplete")
    require(aggregate["comparability"]["status"] == "passed", "Development comparability did not pass")
    rows = aggregate["cases"]
    require(len(rows) == 3 and all(row["status"] == "completed" for row in rows),
            "All three development policies must complete")
    ids = [row["case_id"] for row in rows]
    require(len(set(ids)) == 3 and set(ids) == set(evidence["cases"]), "Development case identities differ")
    reference = launch["payload"]["config"]
    options = launch["options"]
    require(summary["run_id"] == options["run_id"], "Development run IDs differ")
    require(reference["seed"] == 1337 and options["seeds"] == [1337]
            and options["policies"] == list(POLICIES), "Development seed or launch policies differ")
    for field, config_field in (("max_steps", "max_steps"), ("train_warmup_steps", "warmup_steps"),
                                ("eval_interval", "eval_interval"), ("eval_batches", "eval_batches"),
                                ("eval_batch_size", "eval_batch_size"),
                                ("checkpoint_interval", "checkpoint_interval")):
        require(reference[config_field] == protocol[field], f"Development config differs: {field}")
    recorded_jobs = [job for job in campaign["resolved_jobs"]
                     if job["options"]["run_id"] == options["run_id"]]
    require(len(recorded_jobs) == 1 and recorded_jobs[0]["config"] == reference
            and recorded_jobs[0]["options"] == options, "Launch differs from the prelaunch campaign")
    manifest = {case["case_id"]: case for case in evidence["manifest"]["cases"]}
    require(set(manifest) == set(ids), "Development manifest case identities differ")
    expected_train, expected_eval = list(range(1, 1201)), list(range(50, 1201, 50))
    expected_valid_tokens = reference["model"]["context_length"] * 64
    require(reference["model"]["context_length"] == 1024, "Quality development context must be 1K")
    observed_policies, at_target_step, paired_stream = [], {}, None
    reference_metadata = None
    for row in rows:
        case_id = row["case_id"]
        case = evidence["cases"][case_id]
        metadata, config = case["metadata"], case["metadata"]["config"]
        require(_fixed_config(config) == _fixed_config(reference), "Development fixed configs differ")
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        require(config_hash == metadata["config_sha256"] == row["config_sha256"]
                == manifest[case_id]["config_sha256"] and config == manifest[case_id]["config"],
                "Development config hash or manifest differs")
        require(config["batch_size"] * config["grad_accum_steps"] == 8,
                "Development effective batch differs")
        observed_policies.append(f"{config['model']['attention_backend']}:{config['model']['activation_checkpointing']}:{config['batch_size']}")
        require(case["execution"]["returncode"] == 0 and case["summary"]["status"] == "completed"
                and case["summary"]["step"] == 1200, "Development subprocess did not complete")
        train = [event for event in case["metrics"] if event["event"] == "train"]
        evaluations = [event for event in case["metrics"] if event["event"] == "eval"]
        require([event["step"] for event in train] == expected_train, "Missing or duplicate training updates")
        require([event["step"] for event in evaluations] == expected_eval, "Missing or duplicate evaluations")
        require(all(event["tokens_seen"] == event["step"] * 8192 for event in train),
                "Development training-token schedule differs")
        require(all(event["valid_tokens"] == expected_valid_tokens for event in evaluations),
                "Development validation-token schedule differs")
        for event in case["metrics"]:
            require(all(math.isfinite(value) for value in event.values() if isinstance(value, (float, int))),
                    "Nonfinite development metric")
        stream = [(event["batch_start_sha256"], event["learning_rate"], event["tokens_seen"]) for event in train]
        if paired_stream is None:
            paired_stream, reference_metadata = stream, metadata
        require(stream == paired_stream, "Development ordered batches or learning rates differ")
        for field in ("data_identity", "packages", "parameter_count"):
            require(metadata[field] == reference_metadata[field], f"Development {field} differs")
        source = metadata["source"]["python_source_sha256"]
        require(source == reference_metadata["source"]["python_source_sha256"]
                == manifest[case_id]["source_sha256"], "Development training source differs")
        at_target_step[case_id] = next(event["valid_loss"] for event in evaluations if event["step"] == 600)
    require(sorted(observed_policies) == sorted(POLICIES), "Development policy set differs")
    increment = Decimal("0.05")
    loss = (max(Decimal(str(value)) for value in at_target_step.values()) / increment).to_integral_value(
        rounding=ROUND_CEILING) * increment
    identity = reference_metadata["data_identity"]
    require(identity["kind"] == reference["data"]["kind"] == "npy", "Quality study requires NPY data")
    files = {item["remote"]: item["local"] for item in launch["payload"]["files"]}
    data_files = [{"path": files[reference["data"][f"{split}_path"]], "sha256": identity[split]["sha256"]}
                  for split in ("train", "valid")]
    for remote, tokenizer in zip(reference["data"]["tokenizer_files"], identity["tokenizer_files"], strict=True):
        data_files.append({"path": files[remote], "sha256": tokenizer["sha256"]})
    return {"format_version": 1, "target_validation_loss": float(loss), "target_loss_decimal": str(loss),
            "quality_protocol": protocol, "development_run_id": options["run_id"],
            "development_losses_at_target_step": at_target_step, "reference_config": reference,
            "reference_data_identity": identity,
            "reference_source_sha256": reference_metadata["source"]["python_source_sha256"],
            "reference_packages": reference_metadata["packages"],
            "reference_payload_packages": launch["payload"]["packages"], "data_files": data_files,
            "input_sha256": {name: hashlib.sha256(raw[name]).hexdigest() for name in INPUTS}}


def _check_current_inputs(target: dict, repo: Path) -> None:
    require(training_source_sha256(repo) == target["reference_source_sha256"],
            "Training source changed since development; confirmation was not launched")
    for item in target["data_files"]:
        require(file_sha256(Path(item["path"])) == item["sha256"],
                f"Dataset or tokenizer changed since development: {item['path']}")


def freeze_target(archive: Path, repo: Path = REPO) -> dict:
    destination = archive / "frozen-target.json"
    require(not destination.exists(), "Frozen target already exists; it will not be overwritten")
    target = derive_target(archive)
    _check_current_inputs(target, repo)
    target["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    with destination.open("x") as stream:
        stream.write(json.dumps(target, indent=2, allow_nan=False) + "\n")
    return target


def validate_frozen_target(path: Path, repo: Path = REPO) -> dict:
    target = json.loads(path.read_text())
    require(isinstance(target.get("frozen_at_utc"), str), "Frozen target timestamp is missing")
    require(datetime.fromisoformat(target["frozen_at_utc"]).tzinfo is not None,
            "Frozen target timestamp must include its timezone")
    require({key: value for key, value in target.items() if key != "frozen_at_utc"} == derive_target(path.parent),
            "Frozen target or its development evidence changed")
    _check_current_inputs(target, repo)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--freeze", action="store_true", help="Create a new frozen target; never overwrite")
    args = parser.parse_args()
    target = freeze_target(args.archive) if args.freeze else validate_frozen_target(args.archive / "frozen-target.json")
    print(json.dumps({"target_validation_loss": target["target_validation_loss"],
                      "frozen_at_utc": target["frozen_at_utc"], "status": "validated"}, indent=2))


if __name__ == "__main__":
    main()
