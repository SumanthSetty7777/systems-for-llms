"""Finite attention experiments with per-case evidence and explicit dry runs.

Time budgets are checked between cases. An external process/container timeout
must bound a hung CUDA call. Search comparisons replay measured tables offline;
they are not repeated GPU searches or evidence of end-to-end tuning speed.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

from .masks import MaskSpec
from .search import CandidateResult, KernelConfig, default_configs, search


DEFAULT = KernelConfig(64, 64, 4, 2)
VARIANTS = {
    "csr_masked": {"implementation": "csr", "use_full_blocks": False},
    "csr_full": {"implementation": "csr", "use_full_blocks": True},
    "interval_masked": {"implementation": "interval", "use_full_blocks": False},
    "interval_split": {"implementation": "interval", "use_full_blocks": True},
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _documents(length):
    """Packed documents with irregular boundaries and multiple lengths."""
    sizes = (97, 211, 53, 389)
    result = []
    label = 0
    while len(result) < length:
        result.extend([label] * min(sizes[label % len(sizes)], length - len(result)))
        label += 1
    return result


def _workload(name, length, head_dim, pattern, *, dtype="bfloat16", heads=8, window=127):
    spec = MaskSpec(causal=True,
                    window_left=window if pattern == "window" else None,
                    same_document=pattern == "document")
    return {"workload_id": name, "length": length, "head_dim": head_dim,
            "batch": 1, "heads": heads, "dtype": dtype,
            "spec": asdict(spec), "pattern": pattern,
            "document_ids": _documents(length) if spec.same_document else None}


def _case(workload, seed, variant, config=DEFAULT, *, flex=False, selection=None):
    config = asdict(config)
    identity = "-".join(str(config[key]) for key in ("block_q", "block_k", "num_warps", "num_stages"))
    return {**workload, "case_id": f"{workload['workload_id']}-s{seed}-{variant}-{identity}",
            "seed": seed, "variant": variant, "config": config,
            "include_flex": flex, "selection": selection, **VARIANTS[variant]}


def build_plan(stage, *, winners=None):
    """Return deterministic cases; no imports, allocations or GPU calls occur."""
    cases = []
    if stage == "smoke":
        shapes = [(128, 64, "float16"), (257, 128, "float16"),
                  (128, 128, "bfloat16"), (257, 64, "bfloat16")]
        for index, (length, dimension, dtype) in enumerate(shapes):
            for pattern in ("causal", "window", "document"):
                workload = _workload(f"smoke-{index}-{pattern}", length, dimension,
                                     pattern, dtype=dtype, heads=2)
                variants = ("csr_full",) if pattern == "document" else ("csr_full", "interval_split")
                for variant in variants:
                    flex = index == 0 and pattern in ("causal", "document") and variant == "csr_full"
                    cases.append(_case(workload, 0, variant, flex=flex))
    elif stage == "ablation":
        for length in (1024, 4096):
            for dimension in (64, 128):
                for pattern in ("causal", "window", "document"):
                    workload = _workload(f"n{length}-d{dimension}-{pattern}", length, dimension, pattern)
                    variants = list(VARIANTS)[:2] if pattern == "document" else list(VARIANTS)
                    for seed in (11, 12, 13):
                        # Deterministic permutation reduces always-first variant bias.
                        order = variants.copy()
                        random.Random(seed).shuffle(order)
                        for variant in order:
                            cases.append(_case(workload, seed, variant, flex=variant == "csr_full"))
    elif stage == "flex_hints":
        # A separate study gives FlexAttention the same proved contiguous-block
        # fact used by interval lowering; the original ablation is unchanged.
        for length in (1024, 4096):
            for dimension in (64, 128):
                for pattern in ("causal", "window"):
                    workload = _workload(f"n{length}-d{dimension}-{pattern}", length, dimension, pattern)
                    for seed in (11, 12, 13):
                        case = _case(workload, seed, "interval_split", flex=True)
                        case["flex_contiguous"] = True
                        cases.append(case)
    elif stage == "tuning":
        workloads = [
            _workload("tune-causal", 4096, 64, "causal"),
            _workload("tune-window128", 4096, 64, "window"),
            _workload("tune-window512", 4096, 128, "window", window=511),
            _workload("tune-document", 4096, 64, "document"),
        ]
        for workload in workloads:
            variant = "csr_full" if workload["pattern"] == "document" else "interval_split"
            configs = list(default_configs())
            random.Random(2026).shuffle(configs)
            for index, config in enumerate(configs):
                cases.append(_case(workload, 23, variant, config, flex=index == 0))
    elif stage == "confirmation":
        workloads = [
            (_workload("holdout-tail257", 257, 64, "window"), "tune-window128"),
            (_workload("holdout-tail4097", 4097, 128, "window", window=511), "tune-window512"),
            (_workload("holdout-causal8192", 8192, 64, "causal"), "tune-causal"),
            (_workload("holdout-document8192", 8192, 64, "document"), "tune-document"),
        ]
        for workload, source in workloads:
            variant = "csr_full" if workload["pattern"] == "document" else "interval_split"
            for seed in (101, 102, 103):
                base = _case(workload, seed, variant, flex=True, selection="fixed_default")
                base["case_id"] += "-default"
                cases.append(base)
                if winners and source in winners:
                    tuned = _case(workload, seed, variant, KernelConfig(**winners[source]["config"]),
                                  selection={"policy": "fastest_correct_training_configuration",
                                             "source_workload": source,
                                             "source_case_id": winners[source]["case_id"]})
                    tuned["case_id"] += "-transferred"
                    cases.append(tuned)
    else:
        raise ValueError("unknown study stage")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case identifiers")
    return {"schema_version": 1, "stage": stage, "cases": cases,
            "selection_policy": "Confirmation transfers winners from matching training families; held-out measurements never select configurations.",
            "timing_mode": "cuda_graph", "warmup": 5, "repeats": 20, "inner_repeats": 20,
            "baseline_reuse": "Flex results may be paired only within an identical workload and input seed; baseline_source_case_id records the source. SDPA is remeasured per case."}


def _eligible(measurement):
    return CandidateResult(measurement.get("status") == "passed" and measurement.get("correct") is True,
                           measurement.get("latency_ms"), error=measurement.get("error")).eligible


def load_winners(directory):
    records = [json.loads(path.read_text()) for path in sorted((Path(directory) / "cases").glob("*.json"))]
    winners = {}
    for record in records:
        case, measured = record["case"], record["measurement"]
        if _eligible(measured):
            old = winners.get(case["workload_id"])
            if old is None or measured["latency_ms"] < old["latency_ms"]:
                winners[case["workload_id"]] = {"case_id": case["case_id"], "config": case["config"],
                                               "latency_ms": measured["latency_ms"]}
    return winners


def replay_search(records):
    """Replay finite measured tables; omit incomplete tables from comparisons."""
    groups = defaultdict(list)
    for record in records:
        groups[record["case"]["workload_id"]].append(record)
    reports = []
    for workload, group in sorted(groups.items()):
        table = {KernelConfig(**r["case"]["config"]): r["measurement"] for r in group}
        if set(table) != set(default_configs()):
            reports.append({"workload_id": workload, "status": "incomplete_table", "measured_configs": len(table)})
            continue
        valid = [value["latency_ms"] for value in table.values() if _eligible(value)]
        if not valid:
            reports.append({"workload_id": workload, "status": "no_correct_configuration"})
            continue
        oracle = min(valid)
        trials = []
        for method in ("random", "evolution"):
            for budget in (4, 8, 12):
                for seed in range(20):
                    def evaluate(config):
                        measured = table[config]
                        return CandidateResult(_eligible(measured), measured.get("latency_ms"),
                                               error=measured.get("error"))
                    selected = search(evaluate, budget=budget, method=method, seed=seed)
                    best = selected.best
                    trials.append({"method": method, "budget": budget, "search_seed": seed,
                                   "best_latency_ms": best.result.latency_ms if best else None,
                                   "relative_regret": best.result.latency_ms / oracle - 1 if best else None,
                                   "best_config": asdict(best.config) if best else None})
        reports.append({"workload_id": workload, "status": "completed", "oracle_latency_ms": oracle,
                        "eligible_configurations": len(valid), "trials": trials})
    return {"methodology": "OFFLINE replay of fixed GPU-measured latency tables. No new GPU calls; excludes timing noise, compile cost, thermal/order effects and end-to-end search overhead.",
            "workloads": reports}


def _fatal(measurement):
    error = (str(measurement.get("error", "")) + " "
             + str(measurement.get("flex", {}).get("error", ""))).lower()
    return any(text in error for text in ("illegal memory access", "device-side assert", "context is destroyed"))


def run_study(plan, output_dir, *, time_budget_seconds=900, evaluate=None):
    """Execute and checkpoint evidence; injection supports CPU orchestration tests."""
    if evaluate is None:
        from .benchmark import run_case
        evaluate = run_case
    output_dir = Path(output_dir)
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)
    if any((output_dir / "cases").iterdir()):
        raise FileExistsError("Case output directory must be empty; use a new run directory")
    if not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
        raise ValueError("time budget must be finite and positive")
    _write_json(output_dir / "plan.json", plan)
    started = time.monotonic()
    records = []
    summary = {"schema_version": 1, "stage": plan["stage"], "status": "running",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "planned_cases": len(plan["cases"]), "completed_cases": 0, "passed_cases": 0,
               "failed_cases": 0, "time_budget_seconds": time_budget_seconds,
               "flex_requested_cases": 0, "flex_passed_cases": 0, "flex_failed_cases": 0,
               "python": platform.python_version(),
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob("*.py"))}}
    for case in plan["cases"]:
        if time.monotonic() - started >= time_budget_seconds:
            summary["status"] = "time_budget_exhausted"
            break
        case_start = time.monotonic()
        kwargs = {key: case[key] for key in ("length", "head_dim", "batch", "heads", "dtype", "document_ids", "seed", "implementation", "use_full_blocks", "include_flex")}
        kwargs["flex_contiguous"] = case.get("flex_contiguous", False)
        kwargs.update(case["config"])
        kwargs.update({key: plan[key] for key in ("warmup", "repeats", "inner_repeats", "timing_mode")})
        kwargs["spec"] = MaskSpec(**case["spec"])
        try:
            measured = evaluate(**kwargs)
        except Exception as exc:
            measured = {"status": "failed", "correct": False, "latency_ms": None,
                        "error_type": type(exc).__name__, "error": str(exc)}
        record = {"case": case, "measurement": measured,
                  "wall_seconds": time.monotonic() - case_start}
        _write_json(output_dir / "cases" / (case["case_id"] + ".json"), record)
        records.append(record)
        summary["completed_cases"] += 1
        summary["passed_cases" if _eligible(measured) else "failed_cases"] += 1
        flex_failed = case["include_flex"] and measured.get("flex", {}).get("status") != "passed"
        if case["include_flex"]:
            summary["flex_requested_cases"] += 1
            summary["flex_failed_cases" if flex_failed else "flex_passed_cases"] += 1
        summary["elapsed_seconds"] = time.monotonic() - started
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps({"case": case["case_id"], "status": measured.get("status"),
                          "latency_ms": measured.get("latency_ms"),
                          "progress": f"{len(records)}/{len(plan['cases'])}"}), flush=True)
        if _fatal(measured) or (plan["stage"] == "smoke" and not _eligible(measured)):
            summary["status"] = "stopped_after_failure"
            break
        if plan["stage"] == "smoke" and flex_failed:
            summary["status"] = "stopped_after_baseline_failure"
            break
    else:
        summary["status"] = ("completed_with_failures" if summary["failed_cases"] else
                             "completed_with_baseline_failures" if summary["flex_failed_cases"] else "passed")
    # Pair optional Flex baselines by identical semantics and random input seed.
    sources = {(r["case"]["workload_id"], r["case"]["seed"]): r
               for r in records if r["measurement"].get("flex", {}).get("status") == "passed"}
    pairs = []
    for record in records:
        case, measured = record["case"], record["measurement"]
        source = sources.get((case["workload_id"], case["seed"]))
        flex = source["measurement"]["flex"] if source else {}
        flex_latency = flex.get("latency_ms")
        pairs.append({"case_id": case["case_id"],
                      "baseline_source_case_id": source["case"]["case_id"] if source else None,
                      "flex_latency_ms": flex_latency,
                      "flex_speedup": flex_latency / measured["latency_ms"]
                      if _eligible(measured) and isinstance(flex_latency, (int, float)) and flex_latency > 0 else None})
    _write_json(output_dir / "baseline_pairs.json", pairs)
    if plan["stage"] == "tuning":
        _write_json(output_dir / "search_replay.json", replay_search(records))
        _write_json(output_dir / "winners.json", load_winners(output_dir))
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["not_run_cases"] = summary["planned_cases"] - summary["completed_cases"]
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "ablation", "flex_hints", "tuning", "confirmation"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-budget-seconds", type=float, default=900)
    parser.add_argument("--tuning-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    winners = load_winners(args.tuning_dir) if args.tuning_dir else None
    plan = build_plan(args.stage, winners=winners)
    if not args.execute:
        _write_json(args.output_dir / "plan.json", plan)
        print(json.dumps({"status": "dry_run", "stage": args.stage, "cases": len(plan["cases"]),
                          "plan": str(args.output_dir / "plan.json")}))
        return 0
    import torch
    if not torch.cuda.is_available():
        report = {"status": "not_run", "error": "A CUDA GPU is required; no GPU experiment was run"}
        _write_json(args.output_dir / "summary.json", report)
        print(json.dumps(report))
        return 2
    summary = run_study(plan, args.output_dir, time_budget_seconds=args.time_budget_seconds)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
