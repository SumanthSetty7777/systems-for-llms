"""Run the two-GPU correctness gate, then the eight-GPU scaling study."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid


REPO = Path(__file__).resolve().parents[1]
BUDGET_USD = 50.0
STAGES = (("smoke", "H100:2", 900), ("scale", "H100:8", 2400))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load(REPO / "distributed/run_modal.py", "distributed_campaign_launcher")
experiment = _load(REPO / "distributed/run_experiments.py", "distributed_campaign_experiment")


def definition_hash(definition):
    return hashlib.sha256(json.dumps(definition, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_campaign(config_path, output_dir):
    """Freeze both plans and their explicit payload before any cloud request."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    config = Path(config_path).resolve()
    payload = launcher.prepare_payload(config)
    campaign_id = datetime.now(timezone.utc).strftime("distributed-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    stages = []
    for stage, gpu, seconds in STAGES:
        options = {"stage": stage, "gpu": gpu, "job_timeout_seconds": seconds,
                   "volume": "systems-for-llms-results", "run_id": campaign_id + "-" + stage,
                   "plan": experiment.build_plan(stage)}
        launcher.validate_options(options)
        experiment.validate_plan(options["plan"])
        stages.append({"name": stage, "options": options, "output_dir": str(output / stage),
                       "resources": launcher.resources(options), "cost_estimate": launcher.cost_estimate(options)})
    cost = round(sum(stage["cost_estimate"]["requested_resources_usd_at_timeout"] for stage in stages), 4)
    if cost > BUDGET_USD:
        raise ValueError(f"Requested-resource estimate ${cost:.2f} exceeds the ${BUDGET_USD:.2f} working budget")
    definition = {"campaign_id": campaign_id, "output_dir": str(output), "config_path": str(config),
                  "working_budget_usd": BUDGET_USD, "requested_resources_usd_at_timeouts": cost,
                  "payload": payload, "stages": stages,
                  "cost_scope": "All allocated GPUs count during every case. This estimate excludes image builds/startup, excess CPU/RAM, storage/egress and other apps; it is not an account billing cap.",
                  "failure_policy": "Stop at the first failed gate or stage; no retries and no automatic confirmation stage."}
    return {"format_version": 1, "status": "planned", "definition": definition,
            "definition_sha256": definition_hash(definition),
            "stages": [{"name": stage["name"], "status": "not_run"} for stage in stages]}


def _save(output, campaign):
    pending = output / ".campaign.json.tmp"
    pending.write_text(json.dumps(campaign, indent=2, allow_nan=False) + "\n")
    pending.replace(output / "campaign.json")


def _verify_definition(campaign):
    definition = campaign["definition"]
    if campaign["definition_sha256"] != definition_hash(definition):
        raise ValueError("Frozen campaign definition changed; prepare a new campaign")
    payload = definition["payload"]
    if launcher.source_snapshot(payload["files"]) != payload["source_sha256"]:
        raise RuntimeError("Source files changed after the campaign was frozen; prepare a new campaign")
    for stage in definition["stages"]:
        launcher.validate_options(stage["options"])
        experiment.validate_plan(stage["options"]["plan"])


def execute_campaign(campaign):
    """Execute only the frozen two-stage definition, preserving every outcome."""
    _verify_definition(campaign)
    if campaign["status"] != "planned":
        raise ValueError("A campaign can only execute once")
    definition = campaign["definition"]
    output = Path(definition["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    campaign["status"] = "running"
    _save(output, campaign)
    for index, stage in enumerate(definition["stages"]):
        record = campaign["stages"][index]
        try:
            _verify_definition(campaign)
            record["status"] = "running"
            _save(output, campaign)
            options = stage["options"]
            print(f"Starting {stage['name']}: {options['gpu']}, {len(options['plan']['cases'])} cases, {options['job_timeout_seconds']}s job limit", flush=True)
            summary = launcher.execute(definition["payload"], options, Path(stage["output_dir"]))
            record["summary"] = summary
            record["status"] = summary["status"]
            if isinstance(summary.get("function_seconds"), (int, float)):
                record["requested_resource_usd_for_reported_function_time"] = round(
                    summary["function_seconds"] * stage["cost_estimate"]["requested_resources_usd_per_hour"] / 3600, 4)
            if summary["status"] != "completed":
                raise RuntimeError(f"{stage['name']} did not pass; later stages are blocked")
        except Exception as error:
            record.update(status="failed", error=str(error))
            campaign["status"] = "failed"
            for pending in campaign["stages"][index + 1:]:
                pending["blocked_by"] = stage["name"]
            _save(output, campaign)
            return campaign
        _save(output, campaign)
    campaign["status"] = "completed"
    _save(output, campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/scale-324m-s1024.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        campaign = prepare_campaign(args.config, args.output_dir)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", **campaign}, indent=2), flush=True)
    if args.execute:
        result = execute_campaign(campaign)
        print(json.dumps({"status": result["status"], "output_dir": result["definition"]["output_dir"],
                          "stages": result["stages"]}, indent=2), flush=True)
        if result["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
