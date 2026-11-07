"""Run the complete staged attention study, stopping when a gate fails."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .run_modal import ROOT, STAGE_SECONDS, atomic_json, execute, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=["H100", "A100-80GB", "L40S"], default="H100")
    parser.add_argument("--estimated-resource-budget-usd", type=float, default=6.0)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    prefix = args.run_id or "attention-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        parser.error("invalid run id")
    stages = [manifest(stage, args.gpu, seconds, prefix + "-" + stage)
              for stage, seconds in STAGE_SECONDS.items()]
    estimate = sum(stage["cost_estimate"]["requested_resources_usd_at_timeout"] for stage in stages)
    if args.estimated_resource_budget_usd <= 0 or estimate > args.estimated_resource_budget_usd:
        parser.error(f"planned resource estimate {estimate:.2f} exceeds budget guard")
    plan = {"run_id": prefix, "gpu": args.gpu, "stages": stages,
            "estimated_requested_resources_usd": round(estimate, 4),
            "scope": "Resource estimate, not a billing cap. Builds/startup/excess CPU or RAM/storage and other apps excluded. One GPU at a time, no automatic retries."}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    root = ROOT / "runs" / prefix
    if root.exists():
        raise FileExistsError(f"Use a new run id: {root}")
    root.mkdir(parents=True)
    atomic_json(root / "campaign-plan.json", plan)
    outcomes = []
    for settings in stages:
        stage = settings["stage"]
        if stage in ("confirmation", "block"):
            settings["tuning_volume_dir"] = "/results/" + prefix + "-tuning/study"
        output = root / stage
        try:
            result = execute(settings, output)
        except Exception as error:
            outcomes.append({"stage": stage, "status": "launch_failed", "error_type": type(error).__name__,
                             "error": str(error) or "Modal authentication/API request timed out"})
            atomic_json(root / "campaign-summary.json", {"status": "stopped", "outcomes": outcomes})
            print(json.dumps(outcomes[-1]))
            return 2
        summary_path = output / "study" / "summary.json"
        study = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        outcomes.append({"stage": stage, "remote": result, "study": study})
        # Tuning may expose resource-infeasible candidates. Other stages must
        # have complete, correct executions and every requested baseline.
        acceptable = study.get("status") == "passed" or (
            stage == "tuning" and study.get("status") == "completed_with_failures"
            and study.get("passed_cases", 0) > 0)
        acceptable &= study.get("not_run_cases", 1) == 0
        acceptable &= study.get("flex_failed_cases", 0) == 0
        atomic_json(root / "campaign-summary.json", {"status": "running" if acceptable else "stopped",
                                                     "outcomes": outcomes})
        if not acceptable:
            print(json.dumps({"status": "stopped", "stage": stage, "output": str(output)}))
            return 2
    atomic_json(root / "campaign-summary.json", {"status": "completed", "outcomes": outcomes})
    print(json.dumps({"status": "completed", "output": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
