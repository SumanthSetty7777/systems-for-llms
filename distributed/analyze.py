"""Build a portable report from completed distributed campaign or stage evidence.

The summary uses only the standard library. Matplotlib is optional and imported
only when writing figures. This command never launches compute or changes runs.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


MATCH_KEYS = ("model", "context_length", "seed", "precision", "checkpointing",
              "global_batch_size", "steps", "warmup_steps")
METHODS = {"ddp": "DDP", "zero1": "DDP + sharded AdamW", "fsdp": "FSDP", "tp": "Tensor parallel"}
POLICY_KEYS = tuple(key for key in MATCH_KEYS if key != "seed") + (
    "mode", "world_size", "microbatch_size", "sync_policy", "bucket_cap_mb")
TIMING_NOTE = ("Each optimizer step follows an inter-rank barrier and device synchronization. "
               "Reported time is the maximum rank time, including data, backward synchronization, "
               "clipping, update and loss reduction; it excludes the timing-max reduction, "
               "evaluation, setup and checkpoint writes. This measures synchronized steps, "
               "not uninterrupted end-to-end training throughput.")
COST_NOTE = ("Active GPU-seconds per update = participating GPUs × measured training seconds / "
             "measured updates. All allocated GPUs are billed even during a one-GPU baseline. "
             "Allocated GPU-seconds for subprocesses include process startup and evaluation, "
             "but not all launch overhead. Function-time dollar estimates use the launch's "
             "recorded requested-resource rate; they are not invoices and exclude image builds, "
             "startup, excess CPU/RAM, storage/egress and other apps.")


def _read(path):
    with Path(path).open() as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Evidence mismatch: {label}")


def _close(actual, expected, label):
    if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Evidence mismatch: {label}")


def _case_evidence(row, evidence):
    """Recheck recorded rank streams and recompute performance from timed updates."""
    case_id = row["case_id"]
    prefix = f"study/cases/{case_id}/"
    packed = evidence["json"]
    recorded = packed[prefix + "summary.json"]
    _same(recorded["status"], "completed", case_id + " completion")
    _same(row.get("comparability"), "passed", case_id + " runner comparability")
    tokens = row["context_length"] * row["global_batch_size"]
    metadata, streams, timings = [], [], []
    for rank in range(row["world_size"]):
        meta = packed[prefix + f"rank-{rank}-metadata.json"]
        rank_summary = packed[prefix + f"rank-{rank}-summary.json"]
        metrics = evidence["metrics"][prefix + f"rank-{rank}-metrics.jsonl"]
        train = [event for event in metrics if event.get("event") == "train"]
        _same([event["step"] for event in train], list(range(1, row["steps"] + 1)), case_id + " updates")
        _same(meta["rank"], rank, case_id + " rank")
        for key in ("mode", "world_size", "global_batch_size", "microbatch_size", "sync_policy", "bucket_cap_mb"):
            _same(meta[key], row[key], case_id + " " + key)
        _same(meta["timing_warmup_steps"], row["warmup_steps"], case_id + " warmup")
        _same(meta["profile_steps"], 0, case_id + " profiling")
        for key in ("seed", "precision"):
            _same(meta["config"][key], row[key], case_id + " " + key)
        _same(meta["config"]["model"]["context_length"], row["context_length"], case_id + " context")
        _same(meta["config"]["model"]["activation_checkpointing"], row["checkpointing"], case_id + " checkpointing")
        for event in train:
            _same(event["tokens_seen"], event["step"] * tokens, case_id + " token accounting")
            _positive(event["step_seconds"], case_id + " step time")
            for key in ("loss", "gradient_norm", "learning_rate"):
                if not math.isfinite(event[key]):
                    raise ValueError(f"Nonfinite {key}: {case_id}")
        for key in ("status", "mode", "world_size", "step", "tokens_seen", "tokens_per_step", "measured_steps", "batch_stream_sha256"):
            _same(rank_summary[key], recorded[key], case_id + " rank summary " + key)
        stream = [(event["batch_start_sha256"], event["learning_rate"], event["tokens_seen"]) for event in train]
        streams.append(stream)
        timings.append([event["step_seconds"] for event in train if event["step"] > row["warmup_steps"]])
        metadata.append(meta)
    if not timings[0]:
        raise ValueError(f"No measured updates: {case_id}")
    for meta, stream, measured in zip(metadata, streams, timings):
        _same(stream, streams[0], case_id + " rank batch/LR stream")
        _same(measured, timings[0], case_id + " rank synchronized times")
        for key in ("source", "data_identity", "initial_parameter_sha256"):
            _same(meta[key], metadata[0][key], case_id + " rank provenance " + key)
    batch_hash = hashlib.sha256("".join(item[0] for item in streams[0]).encode()).hexdigest()
    seconds, updates = sum(timings[0]), len(timings[0])
    throughput = tokens * updates / seconds
    for target in (recorded, row):
        for key, expected in (("step", row["steps"]), ("tokens_per_step", tokens),
                              ("tokens_seen", row["steps"] * tokens), ("measured_steps", updates),
                              ("batch_stream_sha256", batch_hash)):
            _same(target[key], expected, case_id + " " + key)
        _close(target["measured_train_seconds"], seconds, case_id + " measured time")
        _close(target["tokens_per_second"], throughput, case_id + " throughput")
    ranks = recorded["ranks"]
    _same(sorted(item["rank"] for item in ranks), list(range(row["world_size"])), case_id + " memory ranks")
    derived = {"measured_train_seconds": seconds, "measured_steps": updates, "tokens_per_second": throughput,
               "step_seconds_mean": seconds / updates,
               "active_gpu_seconds_per_update": row["world_size"] * seconds / updates,
               "active_gpu_seconds_measured": row["world_size"] * seconds}
    for source, target, operation in (("peak_cuda_allocated_bytes", "peak_cuda_allocated_bytes", max),
                                      ("peak_cuda_reserved_bytes", "peak_cuda_reserved_bytes", max),
                                      ("optimizer_state_bytes", "optimizer_state_bytes_max_rank", max),
                                      ("optimizer_state_bytes", "optimizer_state_bytes_sum", sum)):
        values = [item[source] for item in ranks]
        if any(value is not None and (type(value) is not int or value < 0) for value in values):
            raise ValueError(f"Invalid memory accounting: {case_id}")
        value = operation(values) if all(value is not None for value in values) else None
        _same(recorded[target], value, case_id + " " + target)
        _same(row[target], value, case_id + " aggregate " + target)
        derived[target] = value
    if type(recorded.get("valid_loss")) not in (int, float) or not math.isfinite(recorded["valid_loss"]):
        raise ValueError(f"Nonfinite validation loss: {case_id}")
    _same(row.get("valid_loss"), recorded["valid_loss"], case_id + " validation loss")
    # Preserve optimizer/evaluation settings as well as the full model layout.
    signature = {key: metadata[0][key] for key in ("source", "data_identity", "initial_parameter_sha256")}
    signature.update(training_config=metadata[0]["config"], stream_sha256=_hash(streams[0]))
    exact_provenance = _hash(signature)
    # Checkpointing is the treatment in block/none pairs; every other config,
    # source, initialization and data/LR stream field must still match.
    signature["training_config"] = dict(signature["training_config"])
    signature["training_config"]["model"] = dict(signature["training_config"]["model"])
    signature["training_config"]["model"].pop("activation_checkpointing", None)
    derived.update(provenance_sha256=exact_provenance,
                   checkpoint_comparison_provenance_sha256=_hash(signature), source=metadata[0]["source"],
                   timing_scope=metadata[0].get("timing_scope"),
                   peak_memory_scope=metadata[0].get("peak_memory_scope"),
                   evidence_verified=True, ranks=ranks)
    return derived


def _cost_rows(run_dir, campaign, stage_root=None, stage=None):
    definitions = {stage["name"]: stage for stage in campaign.get("definition", {}).get("stages", [])}
    names = [stage] if stage_root is not None else sorted(set(definitions) | {"scale"})
    rows = []
    for name in names:
        root = stage_root if stage_root is not None else run_dir / name
        launch = _read(root / "launch.json") if (root / "launch.json").is_file() else {}
        report = _read(root / "launch-summary.json") if (root / "launch-summary.json").is_file() else {}
        estimate = launch.get("cost_estimate", definitions.get(name, {}).get("cost_estimate", {}))
        seconds, hourly = report.get("function_seconds"), estimate.get("requested_resources_usd_per_hour")
        cost = None
        if seconds is not None and hourly is not None:
            cost = _positive(seconds, name + " function time") * _positive(hourly, name + " hourly rate") / 3600
        rows.append({"stage": name, "status": report.get("status", "not_reported"),
                     "function_seconds": seconds, "allocated_gpu_count": report.get("allocated_gpu_count", estimate.get("allocated_gpu_count")),
                     "requested_resource_usd_for_function_time": cost, "recorded_estimate": estimate})
    return rows


def _statistics(values):
    values = list(values)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def policy_statistics(rows):
    """Summarize each policy across seeds; step observations are not replicates."""
    groups = {}
    for row in rows:
        if row["status"] == "completed":
            groups.setdefault(tuple(row[key] for key in POLICY_KEYS), []).append(row)
    result = []
    for key, cases in sorted(groups.items()):
        seeds = [row["seed"] for row in cases]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Repeated policy has more than one case for a seed")
        fields = ("tokens_per_second", "peak_cuda_allocated_bytes", "optimizer_state_bytes_max_rank",
                  "active_gpu_seconds_per_update", "valid_loss", "throughput_speedup", "strong_scaling_efficiency")
        metrics = {field: _statistics(row[field] for row in cases)
                   for field in fields if all(row.get(field) is not None for row in cases)}
        result.append({"policy": dict(zip(POLICY_KEYS, key)), "seeds": sorted(seeds),
                       "case_ids": [row["case_id"] for row in sorted(cases, key=lambda row: row["seed"])],
                       "metrics": metrics})
    return result


def paired_comparisons(rows):
    """Compare canonical policies using matched seeds, config and provenance."""
    base_keys = tuple(key for key in MATCH_KEYS if key not in {"seed", "checkpointing"})
    base_keys += ("microbatch_size", "sync_policy", "bucket_cap_mb")
    canonical = [row for row in rows if row["status"] == "completed"
                 and row["microbatch_size"] == (1 if row["context_length"] == 4096 else 4)
                 and row["sync_policy"] == "last" and row["bucket_cap_mb"] == 25]
    lookup = {}
    for row in canonical:
        key = tuple(row[name] for name in base_keys) + (row["seed"], row["mode"], row["world_size"], row["checkpointing"])
        if key in lookup:
            raise ValueError("Ambiguous canonical policy for paired comparisons")
        lookup[key] = row
    groups = {}
    for row in canonical:
        references = []
        if row["mode"] in {"ddp", "tp"}:
            references.append((row["mode"] + "_vs_single", "single", 1, row["checkpointing"]))
        if row["mode"] in {"zero1", "fsdp"}:
            references.append((row["mode"] + "_vs_ddp", "ddp", row["world_size"], row["checkpointing"]))
        if row["mode"] in {"ddp", "fsdp"} and row["checkpointing"] == "block":
            references.append((row["mode"] + "_block_vs_none", row["mode"], row["world_size"], "none"))
        base = tuple(row[name] for name in base_keys)
        for name, mode, world, checkpoint in references:
            reference = lookup.get(base + (row["seed"], mode, world, checkpoint))
            if reference is None:
                continue
            signature = ("provenance_sha256" if checkpoint == row["checkpointing"]
                         else "checkpoint_comparison_provenance_sha256")
            if row[signature] != reference[signature]:
                continue
            key = base + (name, row["world_size"], row["checkpointing"], world, checkpoint)
            group = groups.setdefault(key, {"comparison": name, "settings": dict(zip(base_keys, base)),
                                            "numerator": {"mode": row["mode"], "world_size": row["world_size"], "checkpointing": row["checkpointing"]},
                                            "denominator": {"mode": mode, "world_size": world, "checkpointing": checkpoint}, "pairs": []})
            values = {}
            for field, metric in (("tokens_per_second", "throughput_ratio"),
                                  ("peak_cuda_allocated_bytes", "allocated_memory_ratio"),
                                  ("optimizer_state_bytes_max_rank", "optimizer_state_ratio"),
                                  ("active_gpu_seconds_per_update", "active_gpu_seconds_ratio")):
                if row.get(field) is not None and reference.get(field) is not None and reference[field] > 0:
                    values[metric] = row[field] / reference[field]
            values["validation_loss_difference"] = row["valid_loss"] - reference["valid_loss"]
            if mode == "single":
                values["strong_scaling_efficiency"] = values["throughput_ratio"] / row["world_size"]
            group["pairs"].append({"seed": row["seed"], "numerator_case_id": row["case_id"],
                                    "denominator_case_id": reference["case_id"], **values})
    result = []
    for key, group in sorted(groups.items()):
        group["pairs"].sort(key=lambda pair: pair["seed"])
        group["seeds"] = [pair["seed"] for pair in group["pairs"]]
        group["paired_count"] = len(group["pairs"])
        fields = set.intersection(*(set(pair) for pair in group["pairs"])) - {"seed", "numerator_case_id", "denominator_case_id"}
        group["metrics"] = {field: _statistics(pair[field] for pair in group["pairs"]) for field in sorted(fields)}
        result.append(group)
    return result


def build_summary(run_dir):
    """Read a campaign or a standalone stage; recheck all completed rank evidence."""
    run_dir = Path(run_dir)
    standalone = (run_dir / "aggregate.json").is_file()
    stage_root = run_dir if standalone else run_dir / "scale"
    aggregate = _read(stage_root / "aggregate.json")
    evidence = _read(stage_root / "evidence.json")
    _same(evidence["json"]["study/aggregate.json"], aggregate, "local and packed aggregate")
    plan = evidence["json"]["plan.json"]
    _same(plan["plan_sha256"], _hash({key: value for key, value in plan.items() if key != "plan_sha256"}), "plan SHA256")
    _same(aggregate["plan_sha256"], plan["plan_sha256"], "aggregate plan SHA256")
    _same(aggregate["planned_count"], len(plan["cases"]), "planned count")
    cases = {row["case_id"]: row for row in aggregate["cases"]}
    if len(cases) != len(aggregate["cases"]) or len({case["case_id"] for case in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("Duplicate case identifier")
    if set(cases) - {case["case_id"] for case in plan["cases"]}:
        raise ValueError("Aggregate contains cases absent from the plan")
    if not any(row.get("status") == "completed" for row in cases.values()):
        raise ValueError("No completed cases to analyze")
    allocated = _positive(aggregate["allocated_gpus"], "allocated GPUs")
    launch = _read(stage_root / "launch.json") if (stage_root / "launch.json").is_file() else {}
    launch_report = _read(stage_root / "launch-summary.json") if (stage_root / "launch-summary.json").is_file() else {}
    if launch_report:
        _same(launch_report["status"], aggregate["status"], "launch and aggregate status")
        _same(launch_report["allocated_gpu_count"], allocated, "launch allocation")
        if "stage" in launch_report:
            _same(launch_report["stage"], plan["stage"], "launch stage")
        if "case_status_counts" in launch_report:
            _same(launch_report["case_status_counts"], dict(Counter(row["status"] for row in cases.values())), "launch case counts")
    settings = launch.get("options", {})
    if settings:
        _same(settings["plan"], plan, "launched plan")
        _same(settings["stage"], plan["stage"], "launched stage")
        if launch_report.get("run_id"):
            _same(settings["run_id"], launch_report["run_id"], "launch run identity")
    if aggregate["status"] == "completed":
        _same(len(cases), len(plan["cases"]), "completed planned count")
        _same({row["status"] for row in cases.values()}, {"completed"}, "completed case statuses")
        if evidence.get("errors") or evidence.get("missing_files"):
            raise ValueError("Completed run has incomplete packed evidence")
        if launch_report.get("evidence_status") not in (None, "complete"):
            raise ValueError("Completed launch reports incomplete evidence")
    rows, warnings = [], []
    for planned in plan["cases"]:
        original = cases.get(planned["case_id"], dict(planned, status="pending"))
        for key, value in planned.items():
            _same(original.get(key), value, planned["case_id"] + " planned " + key)
        if original["world_size"] > allocated:
            raise ValueError("Case uses more GPUs than the allocation")
        row = dict(original)
        for key in ("throughput_speedup", "strong_scaling_efficiency", "baseline_case_id",
                    "active_gpu_seconds_per_update", "active_gpu_seconds_measured", "allocated_gpu_seconds_subprocess"):
            row.pop(key, None)
        row.update(throughput_speedup=None, strong_scaling_efficiency=None, baseline_case_id=None)
        if row.get("subprocess_seconds") is not None:
            row["allocated_gpu_seconds_subprocess"] = allocated * _positive(row["subprocess_seconds"], "subprocess time")
        if row["status"] == "completed":
            row.update(_case_evidence(row, evidence))
        else:
            # A partial or failed run must not appear as a zero or stale measurement.
            for key in ("tokens_per_second", "measured_train_seconds", "measured_steps", "valid_loss",
                        "step_seconds_mean", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                        "optimizer_state_bytes_max_rank", "optimizer_state_bytes_sum"):
                row.pop(key, None)
        rows.append(row)
    baselines = {}
    for row in rows:
        if row["status"] == "completed" and row["mode"] == "single":
            key = tuple(row[name] for name in MATCH_KEYS)
            if key in baselines:
                raise ValueError("Ambiguous single-GPU baseline")
            baselines[key] = row
    for row in rows:
        if row["status"] != "completed":
            continue
        baseline = baselines.get(tuple(row[name] for name in MATCH_KEYS))
        if baseline is None:
            row["baseline_status"] = "missing matched completed baseline"
            warnings.append(f"{row['case_id']}: no matched completed single-GPU baseline; ratios omitted")
        elif baseline["provenance_sha256"] != row["provenance_sha256"]:
            row["baseline_status"] = "provenance mismatch"
            warnings.append(f"{row['case_id']}: baseline model/source/data/initialization/batch/LR provenance differs; ratios omitted")
        else:
            ratio = row["tokens_per_second"] / baseline["tokens_per_second"]
            row.update(baseline_status="matched", baseline_case_id=baseline["case_id"],
                       throughput_speedup=ratio, strong_scaling_efficiency=ratio / row["world_size"])
    for kind in ("errors", "missing_files", "truncated_logs"):
        warnings.extend(f"Evidence {kind}: {value}" for value in evidence.get(kind, []))
    hardware = evidence["json"].get("preflight.json", {})
    devices = hardware.get("devices", [])
    if devices:
        _same(sorted(device["index"] for device in devices), list(range(allocated)), "hardware allocation")
        for device in devices:
            _positive(device["memory_bytes"], "GPU memory")
        uuids = [device.get("uuid") for device in devices]
        if all(uuid and uuid != "unavailable" for uuid in uuids) and len(set(uuids)) != len(uuids):
            raise ValueError("Duplicate GPU UUID in preflight evidence")
    warnings.extend(f"Hardware diagnostic: {value}" for value in hardware.get("warnings", []))
    if not hardware.get("nvidia_smi_topology"):
        warnings.append("Physical GPU interconnect topology was not reported; peer-access capability alone does not identify link type or bandwidth.")
    campaign = _read(run_dir / "campaign.json") if (run_dir / "campaign.json").is_file() else {}
    payload = launch.get("payload", campaign.get("definition", {}).get("payload", {}))
    test_report = evidence["json"].get("tests-report.json")
    if not test_report or any(test_report.get(key, 0) for key in ("failures", "errors", "skipped")):
        if aggregate["status"] == "completed":
            raise ValueError("Completed run lacks a passing, zero-skip correctness gate")
        warnings.append("The packed correctness gate is absent or has nonpassing/skipped tests; inspect it before making correctness claims.")
    if test_report:
        _same(test_report["tests"], sum(test_report.get(key, 0) for key in ("passed", "failures", "errors", "skipped")), "correctness test totals")
        _positive(test_report["tests"], "correctness test count")
    seeds = sorted({row["seed"] for row in rows if row["status"] == "completed"})
    costs = _cost_rows(run_dir, campaign, stage_root if standalone else None, plan["stage"])
    recorded_costs = [row["requested_resource_usd_for_function_time"] for row in costs if row["requested_resource_usd_for_function_time"] is not None]
    return {"format_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "stage": plan["stage"], "run_status": aggregate["status"], "planned_count": len(plan["cases"]),
            "run_id": launch_report.get("run_id"), "input_layout": "standalone" if standalone else "campaign",
            "evidence_input_prefix": f"inputs/{plan['stage'] if standalone else 'scale'}",
            "case_status_counts": dict(Counter(row["status"] for row in rows)), "allocated_gpus": allocated,
            "seeds": seeds, "interpretation": ("One-seed performance screen; independent-seed confirmation is not included." if len(seeds) == 1
                                               else "Repeated-seed confirmation: means, sample standard deviations and ranges describe the recorded seeds. Each comparison uses within-seed ratios; no confidence intervals or population claims are inferred from these repeats."),
            "limitations": ["Short fixed-global-batch timing runs do not establish convergence, quality equivalence, or research novelty.",
                            "Single-host measurements do not establish multi-node network scaling.",
                            "TP uses all ranks on one microbatch; DDP/FSDP/sharded AdamW partition the global batch. All use the recorded global batch.",
                            "Sharded AdamW shards optimizer state only; its full parameters and gradients remain replicated.",
                            "TP and FSDP initialize a full FP32 model on every GPU before sharding. Training memory savings do not demonstrate initialization of models larger than one GPU can hold; reserved memory can retain the setup allocator cache."],
            "timing_note": TIMING_NOTE, "cost_note": COST_NOTE,
            "baseline_match_fields": list(MATCH_KEYS), "plan_sha256": plan["plan_sha256"],
            "hardware": hardware, "correctness_tests": test_report,
            "artifact_source_sha256": payload.get("source_sha256", {}),
            "training_source_sha256": sorted({row["source"]["sha256"] for row in rows if row["status"] == "completed"}),
            "cost_stages": costs, "recorded_function_time_estimate_usd": sum(recorded_costs) if recorded_costs else None,
            "cost_estimate_complete": all(row["requested_resource_usd_for_function_time"] is not None for row in costs),
            "warnings": warnings, "cases": rows, "policy_statistics": policy_statistics(rows),
            "paired_comparisons": paired_comparisons(rows)}


def plot_series(summary):
    """Select canonical sweeps; keep microbatch/sync/bucket ablations in the table."""
    groups = {}
    for row in summary["cases"]:
        if row["status"] != "completed" or row["mode"] not in {"single", *METHODS}:
            continue
        micro = 1 if row["context_length"] == 4096 else 4
        if row["microbatch_size"] != micro or row["sync_policy"] != "last" or row["bucket_cap_mb"] != 25:
            continue
        key = tuple(row[name] for name in MATCH_KEYS)
        groups.setdefault(key, []).append(row)
    return sorted(groups.items())


def _plot(summary, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [], "Matplotlib is unavailable; numerical tables were written. Install matplotlib or use --no-plots."
    if summary["stage"] == "confirmation" and len(summary["seeds"]) > 1:
        return _plot_repeats(summary, output, plt), None
    groups = plot_series(summary)
    if not groups:
        return [], "No completed canonical sweep cases are available for figures."
    files = []
    for name, field, ylabel, scale in (("scaling", "tokens_per_second", "Thousands of tokens / second", 1000),
                                      ("memory", "peak_cuda_allocated_bytes", "Max rank allocated (GiB)", 1024 ** 3)):
        columns = min(2, len(groups))
        nrows = math.ceil(len(groups) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(6.2 * columns, 4.2 * nrows), squeeze=False)
        for ax, (key, cases) in zip(axes.flat, groups):
            for mode in ("single", *METHODS):
                selected = sorted((row for row in cases if row["mode"] == mode and row.get(field) is not None), key=lambda row: row["world_size"])
                if not selected:
                    continue
                ax.plot([row["world_size"] for row in selected], [row[field] / scale for row in selected],
                        marker="o", linestyle="None" if mode == "single" else "-",
                        color="black" if mode == "single" else None,
                        label="Single GPU reference" if mode == "single" else METHODS[mode])
            model, context, seed, precision, checkpoint, batch, steps, warmup = key
            ax.set_title(f"{model} · context {context:,} · checkpoint {checkpoint}\n{precision} · seed {seed} · global batch {batch}")
            ax.set_xlabel("Participating GPUs")
            ax.set_ylabel(ylabel)
            ax.set_xticks([1, 2, 4, 8])
            values = [row[field] / scale for row in cases if row.get(field) is not None]
            ax.set_ylim(0, max(values, default=1) * 1.12 or 1)
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        devices = summary.get("hardware", {}).get("devices", [])
        names = sorted({device.get("name", "unreported GPU") for device in devices})
        host = f"{summary['allocated_gpus']} GPU host: {', '.join(names)}" if names else "GPU identity unreported"
        fig.suptitle(f"Distributed performance screen · {host}", fontsize=14)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.text(0.5, 0.015, "Missing cases are omitted. Single-GPU reference is plotted once; method lines contain measured ranks only.", ha="center", fontsize=9)
        for extension in ("png", "svg"):
            filename = f"{name}.{extension}"
            fig.savefig(output / filename, dpi=170)
            files.append(filename)
        plt.close(fig)
    return files, None


def _plot_repeats(summary, output, plt):
    """Show every seed and the policy mean without treating steps as repeats."""
    policies = [group for group in summary["policy_statistics"]
                if group["policy"]["sync_policy"] == "last" and group["policy"]["bucket_cap_mb"] == 25]
    order = {("single", "none"): 0, ("ddp", "none"): 1, ("zero1", "none"): 2,
             ("fsdp", "none"): 3, ("tp", "none"): 4, ("ddp", "block"): 5, ("fsdp", "block"): 6}
    policies.sort(key=lambda group: (group["policy"]["context_length"],
                                     order.get((group["policy"]["mode"], group["policy"]["checkpointing"]), 99)))
    if not policies:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5, len(policies) * 0.55)), sharey=True)
    case_lookup = {row["case_id"]: row for row in summary["cases"]}
    labels = []
    for group in policies:
        policy = group["policy"]
        mode = "Single GPU" if policy["mode"] == "single" else METHODS.get(policy["mode"], policy["mode"])
        labels.append(f"{mode} · {policy['world_size']} GPU · {policy['checkpointing']}")
    seed_markers = ("o", "s", "^")
    for ax, field, divisor, label in zip(axes, ("tokens_per_second", "peak_cuda_allocated_bytes"),
                                        (1000, 1024**3), ("Thousands of tokens / second", "Max rank allocated memory (GiB)")):
        means = [group["metrics"][field]["mean"] / divisor for group in policies]
        ax.barh(range(len(policies)), means, height=0.62, color="#cad9e7", label="Mean across seeds")
        for index, seed in enumerate(summary["seeds"]):
            y, x = [], []
            for row_index, group in enumerate(policies):
                selected = [case_lookup[case_id] for case_id in group["case_ids"] if case_lookup[case_id]["seed"] == seed]
                if selected:
                    y.append(row_index + (index - (len(summary["seeds"]) - 1) / 2) * 0.13)
                    x.append(selected[0][field] / divisor)
            ax.scatter(x, y, s=34, marker=seed_markers[index % len(seed_markers)], label=f"Seed {seed}", zorder=3)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(means) * 1.15)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower right")
    devices = summary.get("hardware", {}).get("devices", [])
    names = ", ".join(sorted({device.get("name", "unreported GPU") for device in devices}))
    contexts = ", ".join(f"{context:,}" for context in sorted({group["policy"]["context_length"] for group in policies}))
    models = ", ".join(sorted({group["policy"]["model"] for group in policies}))
    horizons = ", ".join(str(steps) for steps in sorted({group["policy"]["steps"] for group in policies}))
    fig.suptitle(f"{models.upper()} distributed confirmation · {summary['allocated_gpus']} × {names}\nContext {contexts} · {len(summary['seeds'])} seeds · {horizons} updates per case", fontsize=13)
    fig.text(0.5, 0.02, "Bars: policy means. Markers: individual seeds. Synchronized step throughput; memory includes training and evaluation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.9))
    files = []
    for extension in ("png", "svg"):
        filename = f"confirmation.{extension}"
        fig.savefig(output / filename, dpi=170)
        files.append(filename)
    plt.close(fig)
    return files


def _fmt(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _readme(summary, files):
    counts = ", ".join(f"{count} {status}" for status, count in sorted(summary["case_status_counts"].items()))
    names = sorted({device.get("name", "unreported") for device in summary["hardware"].get("devices", [])})
    lines = ["# Distributed training performance study", "", f"{summary['planned_count']} planned cases: {counts}.", "",
             f"Measured hardware: **{summary['allocated_gpus']}-GPU host, {', '.join(names) or 'identity unreported'}**. Every case shares this allocation, including the single-GPU reference.", "",
             summary["interpretation"], "", *["- " + item for item in summary["limitations"]], "",
             "## Measurement and comparisons", "", summary["timing_note"], "",
             "Speedup is measured throughput divided by a completed single-GPU baseline with matching model, context, seed, precision, checkpointing, global batch, update horizon and warmup. Model/source/data/initialization and batch/LR streams must also match. Efficiency is speedup divided by participating GPUs. Missing or mismatched baselines have no ratio.", "",
             "Peak CUDA allocated/reserved memory is the maximum across ranks and includes training and evaluation after model/wrapper setup, excluding final state export. Optimizer bytes count actual optimizer tensors, including any CPU step counters; they are not all GPU-resident bytes.", "",
             "## Measured policies", "", "Canonical 1K cases use microbatch 4; 4K cases use microbatch 1. Both use final-microbatch synchronization and a 25 MiB DDP bucket setting. Other microbatch, synchronization and bucket experiments remain in the full table.", ""]
    for image in ("confirmation.png", "scaling.png", "memory.png"):
        if image in files:
            lines += [f"![{image.removesuffix('.png')} comparison]({image})", ""]
    if len(summary["seeds"]) > 1:
        lines += ["## Paired seed comparisons", "",
                  "Each ratio is computed within one matched seed before aggregation. SD is the sample standard deviation across seeds; ranges retain all repeats. No confidence interval is claimed. Memory reductions are 1 minus the numerator/denominator ratio, so a negative value means memory increased. Validation-loss differences describe this short horizon and do not establish equal convergence.", "",
                  "| Numerator / denominator | Checkpoint policies | GPU counts | Pairs | Throughput ratio mean ± SD | Throughput median [min, max] | Allocated memory reduction | Optimizer state reduction |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for group in summary["paired_comparisons"]:
            numerator, denominator = group["numerator"], group["denominator"]
            metric = group["metrics"]["throughput_ratio"]
            mean_sd = f"{metric['mean']:.3f} ± {_fmt(metric['sample_sd'], 3)}"
            memory = group["metrics"].get("allocated_memory_ratio")
            optimizer = group["metrics"].get("optimizer_state_ratio")
            values = [f"{numerator['mode']} / {denominator['mode']}",
                      f"{numerator['checkpointing']} / {denominator['checkpointing']}",
                      f"{numerator['world_size']} / {denominator['world_size']}", group["paired_count"], mean_sd,
                      f"{metric['median']:.3f} [{metric['min']:.3f}, {metric['max']:.3f}]",
                      "—" if memory is None else f"{1 - memory['mean']:.2%}",
                      "—" if optimizer is None else f"{1 - optimizer['mean']:.2%}"]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        lines += ["", "The machine-readable summary retains every paired seed ratio, validation-loss difference, policy mean, median, sample SD and range. Checkpoint-treatment pairs allow only the activation-checkpointing config field to differ; all other model/config, source, data, initialization and batch/LR provenance must match.", ""]
    lines += ["## Every planned case", "", "GPU-s/update is active measured GPU time, not billed allocation time. Dashes indicate unmeasured or unavailable values.", "",
              "| Case | Status | tokens/s | Speedup | Efficiency | Max allocated GiB | Max reserved GiB | Max optimizer GiB | Sum optimizer GiB | GPU-s/update | Valid loss |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["cases"]:
        gib = lambda key: row[key] / 1024 ** 3 if row.get(key) is not None else None
        efficiency = row.get("strong_scaling_efficiency")
        values = [row["case_id"], row["status"], _fmt(row.get("tokens_per_second"), 0), _fmt(row.get("throughput_speedup")),
                  "—" if efficiency is None else f"{efficiency:.1%}", _fmt(gib("peak_cuda_allocated_bytes")),
                  _fmt(gib("peak_cuda_reserved_bytes")), _fmt(gib("optimizer_state_bytes_max_rank")),
                  _fmt(gib("optimizer_state_bytes_sum")), _fmt(row.get("active_gpu_seconds_per_update"), 3), _fmt(row.get("valid_loss"), 4)]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "## Allocation and cost", "", summary["cost_note"], "",
              f"The measured host allocated **{summary['allocated_gpus']} GPUs** for every case.", "",
              "| Stage | Status | Function seconds | Allocated GPUs | Requested-resource estimate USD |", "|---|---|---:|---:|---:|"]
    for row in summary["cost_stages"]:
        lines.append(f"| {_cell(row['stage'])} | {_cell(row['status'])} | {_fmt(row['function_seconds'])} | {row['allocated_gpu_count'] or '—'} | {_fmt(row['requested_resource_usd_for_function_time'], 4)} |")
    lines += ["", f"Sum of available function-time estimates: ${_fmt(summary['recorded_function_time_estimate_usd'], 4)}. " +
              ("Every listed stage has a recorded rate and function time." if summary["cost_estimate_complete"] else "Some stage estimates are unavailable; this is not a complete campaign cost."), "",
              "## Hardware and correctness evidence", ""]
    hardware = summary["hardware"]
    lines += [f"PyTorch: `{hardware.get('torch', 'unreported')}`; CUDA: `{hardware.get('cuda', 'unreported')}`.", "",
              "| GPU index | Name | UUID | Memory GiB |", "|---:|---|---|---:|"]
    for device in hardware.get("devices", []):
        memory = device.get("memory_bytes")
        lines.append(f"| {device.get('index', '—')} | {_cell(device.get('name', 'unreported'))} | `{_cell(device.get('uuid', 'unreported'))}` | {_fmt(memory / 1024**3 if memory is not None else None)} |")
    if hardware.get("nvidia_smi_topology"):
        lines += ["", "Recorded `nvidia-smi topo -m`:", "", "```text", hardware["nvidia_smi_topology"].replace("```", "'''"), "```"]
    tests = summary["correctness_tests"]
    lines += ["", "Packed stage correctness gate: " + (", ".join(f"{key}={tests.get(key, 'unreported')}" for key in ("tests", "passed", "failures", "errors", "skipped")) if tests else "unreported") + ".", "",
              "## Reproducibility", "", f"Plan SHA256: `{summary['plan_sha256']}`.", "",
              "Training source SHA256: " + ", ".join(f"`{digest}`" for digest in summary["training_source_sha256"]) + ".", "",
              f"[Machine-readable summary](summary.json) contains per-file artifact source hashes, rank memory measurements, baseline choices, and cost-rate provenance. [Raw aggregate]({summary['evidence_input_prefix']}/aggregate.json) and [packed evidence]({summary['evidence_input_prefix']}/evidence.json) preserve the original JSON, full metric streams, and bounded logs. All copied inputs have SHA256 checksums in `summary.json`. These files can be reviewed without the original machine or cloud volume.", ""]
    for row in summary["cases"]:
        if row.get("error"):
            lines.append(f"- `{row['case_id']}`: {_cell(row['error'])}")
    if summary["warnings"]:
        lines += ["", "## Evidence limitations", "", *["- " + _cell(item) for item in summary["warnings"]], ""]
    return "\n".join(lines) + "\n"


def write_report(run_dir, output_dir, *, plots=True):
    """Create an exclusive output directory after validating the input evidence."""
    run_dir, output = Path(run_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    summary = build_summary(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    copied = {}
    relative_inputs = ("campaign.json", "scale/aggregate.json", "scale/evidence.json", "scale/launch.json", "scale/launch-summary.json",
                       "smoke/aggregate.json", "smoke/evidence.json", "smoke/launch.json", "smoke/launch-summary.json")
    if summary["input_layout"] == "standalone":
        relative_inputs = ("aggregate.json", "evidence.json", "launch.json", "launch-summary.json")
    for relative in relative_inputs:
        source = run_dir / relative
        if source.is_file():
            input_name = (f"inputs/{summary['stage']}/{relative}" if summary["input_layout"] == "standalone" else "inputs/" + relative)
            destination = output / input_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied[input_name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    summary["input_files_sha256"] = copied
    files, warning = _plot(summary, output) if plots else ([], None)
    if warning:
        summary["warnings"].append(warning)
    summary["figures"] = files
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(_readme(summary, files))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        summary = write_report(args.run_dir, args.output_dir, plots=not args.no_plots)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir), "case_status_counts": summary["case_status_counts"],
                      "figures": summary["figures"], "warnings": summary["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
