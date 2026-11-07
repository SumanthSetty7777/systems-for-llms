"""Summarize recorded GPU experiments without selecting favorable subsets.

Each study directory remains a separate run. Comparisons require matching
semantics, input seed, timing protocol and hardware/software. This tool never
runs GPU work and never interprets missing measurements as successes.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def _read(path):
    return json.loads(Path(path).read_text())


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _eligible(measurement):
    return measurement.get("status") == "passed" and measurement.get("correct") is True and _positive(measurement.get("latency_ms"))


def _key(record, *, configuration=True, environment=True):
    case, measured = record["case"], record["measurement"]
    key = {name: case.get(name) for name in (
        "workload_id", "length", "head_dim", "batch", "heads", "dtype", "spec", "document_ids", "seed")}
    if environment:
        key.update({name: measured.get(name) for name in ("hardware", "software", "timing_scope", "warmup", "repeats", "inner_repeats")})
    if configuration:
        key["config"] = case.get("config")
    return json.dumps(key, sort_keys=True)


def _stats(pairs):
    values = [pair["ratio"] for pair in pairs]
    return {"paired_cases": len(values),
            "geomean_speedup": math.exp(statistics.mean(map(math.log, values))) if values else None,
            "median_speedup": statistics.median(values) if values else None,
            "min_speedup": min(values) if values else None,
            "max_speedup": max(values) if values else None,
            "regressions": sum(value < 1 for value in values),
            "regressions_over_5_percent": sum(value < 1 / 1.05 for value in values),
            "improvements": sum(value > 1 for value in values),
            "ties": sum(value == 1 for value in values)}


def _pair(left, right, name):
    return {"comparison": name, "workload_id": right["case"]["workload_id"],
            "seed": right["case"]["seed"], "left_case_id": left["case"]["case_id"],
            "right_case_id": right["case"]["case_id"],
            "left_latency_ms": left["measurement"]["latency_ms"],
            "right_latency_ms": right["measurement"]["latency_ms"],
            "ratio": left["measurement"]["latency_ms"] / right["measurement"]["latency_ms"]}


def _comparisons(records, *, confirmation=False):
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        case = record["case"]
        label = case.get("variant")
        if confirmation:
            selection = case.get("selection")
            label = "fixed_default" if selection == "fixed_default" else (
                "transferred" if isinstance(selection, dict) and selection.get("policy") == "fastest_correct_training_configuration" else None)
        if label is not None:
            groups[_key(record, configuration=not confirmation, environment=False)][label].append(record)
    comparisons = [("fixed_default", "transferred")] if confirmation else [
        ("csr_masked", "csr_full"), ("csr_full", "interval_split"),
        ("interval_masked", "interval_split")]
    result = []
    for left_label, right_label in comparisons:
        pairs, missing, failed, ambiguous, incompatible, not_applicable = [], 0, 0, 0, 0, 0
        for group in groups.values():
            if left_label not in group and right_label not in group:
                continue
            example = next(iter(group.values()))[0]
            if right_label.startswith("interval") and example["case"].get("spec", {}).get("same_document"):
                not_applicable += 1
                continue
            left, right = group.get(left_label, []), group.get(right_label, [])
            if not left or not right:
                missing += 1
            elif len(left) != 1 or len(right) != 1:
                ambiguous += 1
            elif not all(_eligible(r["measurement"]) for r in (left[0], right[0])):
                failed += 1
            elif _key(left[0], configuration=not confirmation) != _key(right[0], configuration=not confirmation):
                incompatible += 1
            else:
                pairs.append(_pair(left[0], right[0], f"{left_label}/{right_label}"))
        by_workload = defaultdict(list)
        for pair in pairs:
            by_workload[pair["workload_id"]].append(pair)
        result.append({"comparison": f"{left_label}/{right_label}", **_stats(pairs),
                       "missing_counterpart_groups": missing, "failed_groups": failed,
                       "ambiguous_groups": ambiguous, "incompatible_environment_groups": incompatible,
                       "not_applicable_groups": not_applicable, "pairs": pairs,
                       "by_workload": {key: _stats(value) for key, value in sorted(by_workload.items())}})
    return result


def _baselines(records, directory):
    by_id = {record["case"]["case_id"]: record for record in records}
    pairs_path = directory / "baseline_pairs.json"
    sources = {item["case_id"]: item.get("baseline_source_case_id")
               for item in _read(pairs_path)} if pairs_path.exists() else {}
    groups = defaultdict(lambda: {"pairs": [], "unavailable": 0, "candidate_failed": 0})
    for record in records:
        case, measured = record["case"], record["measurement"]
        # Tuning configurations stay separate; do not report their best only.
        config = "-".join(str(case["config"][key]) for key in ("block_q", "block_k", "num_warps", "num_stages"))
        selection = case.get("selection")
        selection_label = "transferred" if isinstance(selection, dict) else selection
        label = f"{case['variant']}:{config}" + (f":{selection_label}" if selection_label else "")
        for baseline in ("sdpa", "flex"):
            if baseline == "sdpa":
                latency = measured.get("baseline_latency_ms")
                source = record
                available = measured.get("baseline_correct") is True and _positive(latency)
                mode = measured.get("baseline_mask_mode", "unspecified")
                profile, options, inferred = mode, None, False
            else:
                source = record if _eligible(measured.get("flex", {})) else by_id.get(sources.get(case["case_id"]))
                flex = source["measurement"].get("flex", {}) if source else measured.get("flex", {})
                available = source is not None and _key(source, configuration=False) == _key(record, configuration=False) and _eligible(flex)
                latency = flex.get("latency_ms")
                mode = flex.get("backend_requested", "unspecified")
                options = flex.get("kernel_options")
                inferred = "baseline_variant" not in flex
                profile = flex.get("baseline_variant", "triton_default_options_unrecorded" if mode == "TRITON" and options is None else "unspecified")
            options_key = json.dumps(options, sort_keys=True)
            group = groups[(label, baseline, profile, options_key)]
            if not _eligible(measured):
                group["candidate_failed"] += 1
                continue
            if not available:
                group["unavailable"] += 1
                continue
            group["pairs"].append({"workload_id": case["workload_id"], "seed": case["seed"],
                                    "case_id": case["case_id"], "baseline_source_case_id": source["case"]["case_id"],
                                    "baseline_mode": mode, "baseline_latency_ms": latency,
                                    "baseline_variant": profile, "kernel_options": options,
                                    "baseline_variant_inferred": inferred,
                                    "candidate_latency_ms": measured["latency_ms"],
                                    "ratio": latency / measured["latency_ms"]})
    return [{"candidate": key[0], "baseline": key[1], "baseline_variant": key[2],
             "kernel_options": json.loads(key[3]), **_stats(value["pairs"]),
             "baseline_unavailable": value["unavailable"], "candidate_failed": value["candidate_failed"],
             "pairs": value["pairs"]} for key, value in sorted(groups.items())]


def _block_variant_eligible(value, field="block_latency_ms"):
    return value.get("status") == "passed" and value.get("correct") is True and _positive(value.get(field))


def _block_eligible(measured):
    return (measured.get("status") == "passed" and measured.get("correct") is True
            and measured.get("reference_finite") is True and measured.get("inputs_weights_unchanged") is True
            and all(_block_variant_eligible(measured.get("variants", {}).get(name, {}))
                    for name in ("custom_interval_split", "sdpa", "flex_triton")))


def _block_comparisons(records):
    groups = defaultdict(lambda: {"pairs": [], "failed_or_unavailable": 0})
    for record in records:
        case, measured = record["case"], record["measurement"]
        variants = measured.get("variants", {})
        candidate = variants.get("custom_interval_split", {})
        for baseline_name in ("sdpa", "flex_triton"):
            baseline = variants.get(baseline_name, {})
            # The block implementation explicitly requests only BACKEND=TRITON.
            profile = measured.get("flex_baseline_variant", "triton_default") if baseline_name == "flex_triton" else measured.get("baseline_mask_mode", "unspecified")
            options = measured.get("flex_kernel_options", {"BACKEND": "TRITON"}) if baseline_name == "flex_triton" else None
            for scope, field in (("single_block_forward", "block_latency_ms"),
                                 ("isolated_attention_forward", "attention_latency_ms")):
                group = groups[(baseline_name, profile, json.dumps(options, sort_keys=True), scope)]
                valid = (measured.get("inputs_weights_unchanged") is True and measured.get("reference_finite") is True
                         and _block_variant_eligible(candidate, field) and _block_variant_eligible(baseline, field))
                if not valid:
                    group["failed_or_unavailable"] += 1
                    continue
                group["pairs"].append({"case_id": case["case_id"], "seed": case["seed"],
                    "length": case["length"], "head_dim": case["head_dim"], "pattern": case["pattern"],
                    "config": case["config"], "selection": case.get("selection"),
                    "baseline_variant": profile, "kernel_options": options,
                    "candidate_latency_ms": candidate[field], "baseline_latency_ms": baseline[field],
                    "ratio": baseline[field] / candidate[field],
                    "isolated_attention_fraction": measured.get(f"isolated_attention_fraction_{baseline_name}"),
                    "amdahl_estimate": measured.get(f"amdahl_estimate_vs_{baseline_name}")})
    return [{"baseline": key[0], "baseline_variant": key[1], "kernel_options": json.loads(key[2]),
             "scope": key[3], **_stats(value["pairs"]), "failed_or_unavailable": value["failed_or_unavailable"],
             "pairs": value["pairs"],
             "interpretation": "Identical weights/operators and input seed within each case. One BF16 inference Transformer block, not a whole model. Isolated attention and block times are separate measurements; Amdahl estimates are illustrative only."}
            for key, value in sorted(groups.items())]


def _rank(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def _association(rows, field):
    pairs = [(row[field], row["latency_ms"]) for row in rows
             if isinstance(row.get(field), (int, float)) and not isinstance(row[field], bool) and math.isfinite(row[field])]
    if len(pairs) < 3:
        return {"samples": len(pairs), "spearman_rho": None}
    x, y = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    if len(set(x)) == 1 or len(set(y)) == 1:
        rho = None
    else:
        rho = statistics.correlation(x, y)
    return {"samples": len(pairs), "spearman_rho": rho}


def _tuning(records):
    groups = defaultdict(list)
    failures = defaultdict(list)
    for record in records:
        case, measured = record["case"], record["measurement"]
        key = case["workload_id"]
        if not _eligible(measured):
            failures[key].append({"case_id": case["case_id"], "config": case["config"],
                                  "status": measured.get("status"), "error": measured.get("error")})
            continue
        metadata = measured.get("kernel_metadata", {})
        groups[key].append({"case_id": case["case_id"], "config": case["config"],
                            "latency_ms": measured["latency_ms"],
                            **{field: metadata.get(field) for field in ("n_regs", "n_spills", "shared_memory_bytes")}})
    return [{"workload_id": workload, "valid_configurations": len(groups[workload]),
             "failed_configurations": failures[workload],
             "rows_by_latency": sorted(groups[workload], key=lambda row: row["latency_ms"]),
             "resource_associations": {field: _association(groups[workload], field)
                                       for field in ("n_regs", "n_spills", "shared_memory_bytes")},
             "scope": "Within-workload associations across configurations; compiler metadata, not Nsight occupancy measurements or causal attribution."}
            for workload in sorted(set(groups) | set(failures))]


def _replay(directory):
    path = directory / "search_replay.json"
    if not path.exists():
        return None
    replay = _read(path)
    groups = defaultdict(list)
    workload_stats = []
    for workload in replay.get("workloads", []):
        local = defaultdict(list)
        for trial in workload.get("trials", []):
            key = (trial["method"], trial["budget"])
            value = trial.get("relative_regret")
            value = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
            local[key].append(value)
            groups[key].append(value)
        workload_stats.append({"workload_id": workload["workload_id"], "status": workload["status"],
                               "groups": [_regrets(key, values) for key, values in sorted(local.items())]})
    return {"methodology": replay.get("methodology"), "workloads": workload_stats,
            "groups": [_regrets(key, values) for key, values in sorted(groups.items())]}


def _regrets(key, values):
    valid = [value for value in values if value is not None]
    return {"method": key[0], "budget": key[1], "trials": len(values), "failed_trials": len(values) - len(valid),
            "mean_relative_regret": statistics.mean(valid) if valid else None,
            "median_relative_regret": statistics.median(valid) if valid else None,
            "max_relative_regret": max(valid) if valid else None}


def _setup(records):
    groups = defaultdict(list)
    for record in records:
        case, measured = record["case"], record["measurement"]
        if not _eligible(measured):
            continue
        flex = measured.get("flex", {})
        groups[case["variant"]].append({"case_id": case["case_id"],
            "schedule_compile_ms": measured.get("schedule_compile_ms"),
            "first_call_ms": measured.get("first_call_ms"),
            "flex_initial_mask_setup_ms": flex.get("initial_mask_setup_ms"),
            "flex_mask_setup_ms_this_call": flex.get("mask_setup_ms_this_call"),
            "flex_setup_reused": flex.get("mask_setup_reused"),
            "flex_first_call_ms": flex.get("first_call_ms")})
    result = []
    for variant, rows in sorted(groups.items()):
        statistics_by_field = {}
        for field in ("schedule_compile_ms", "first_call_ms", "flex_initial_mask_setup_ms", "flex_mask_setup_ms_this_call", "flex_first_call_ms"):
            values = [row[field] for row in rows if isinstance(row[field], (int, float)) and not isinstance(row[field], bool)
                      and math.isfinite(row[field]) and row[field] >= 0]
            statistics_by_field[field] = {"samples": len(values),
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None, "max": max(values) if values else None}
        result.append({"variant": variant, "statistics_ms": statistics_by_field, "cases": rows,
                       "scope": "Successful candidates only. First calls include setup, JIT or cache lookup, launch and synchronization; caches are not reset. Neither first-call value is pure compile time. Repeated Flex initial-mask setup values are historical cached construction times, not new costs."})
    return result


def analyze_directory(directory):
    directory = Path(directory)
    paths = sorted((directory / "cases").glob("*.json"))
    records = [_read(path) for path in paths]
    ids = [record["case"]["case_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case IDs in {directory}")
    plan = _read(directory / "plan.json") if (directory / "plan.json").exists() else {}
    status = _read(directory / "summary.json") if (directory / "summary.json").exists() else {}
    stage = plan.get("stage", status.get("stage", directory.parent.name))
    is_block = stage == "block" or any("variants" in record["measurement"] and "variant" not in record["case"] for record in records)
    eligible = _block_eligible if is_block else _eligible
    errors = [{"case_id": record["case"]["case_id"], "status": record["measurement"].get("status"),
               "error": record["measurement"].get("error")} for record in records if not eligible(record["measurement"])]
    requested = records if is_block else [record for record in records if record["case"].get("include_flex")]
    flex_failures = sum(not _block_variant_eligible(record["measurement"].get("variants", {}).get("flex_triton", {})) for record in requested) if is_block else sum(not _eligible(record["measurement"].get("flex", {})) for record in requested)
    planned = len(plan["cases"]) if "cases" in plan else status.get("planned_cases")
    environments = {json.dumps({name: r["measurement"].get(name) for name in ("hardware", "software", "timing_scope")}, sort_keys=True) for r in records}
    return {"directory": str(directory), "stage": stage, "status": status.get("status", "not_recorded"),
            "planned_cases": planned, "recorded_cases": len(records),
            "not_recorded_cases": max(0, planned - len(records)) if isinstance(planned, int) else None,
            "passed_cases": len(records) - len(errors), "failed_cases": len(errors), "failures": errors,
            "flex_requested": len(requested),
            "flex_failed_or_unavailable": flex_failures,
            "environments": [json.loads(value) for value in sorted(environments)],
            "source_sha256": status.get("source_sha256"),
            "evidence_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "ablations": _comparisons(records) if stage == "ablation" else [],
            "confirmation": _comparisons(records, confirmation=True) if stage == "confirmation" else [],
            "baselines": [] if is_block else _baselines(records, directory),
            "block_comparisons": _block_comparisons(records) if is_block else [],
            "setup_observations": [] if is_block else _setup(records),
            "tuning": _tuning(records) if stage == "tuning" else [],
            "search_replay": _replay(directory)}


def analyze(root):
    root = Path(root).resolve()
    directories = {path.parent for path in root.rglob("cases") if path.is_dir()}
    if root.name == "cases":
        directories.add(root.parent)
    if (root / "cases").is_dir():
        directories.add(root)
    if not directories:
        raise ValueError(f"No cases directories found in {root}")
    return {"schema_version": 1, "input": str(root),
            "scope": "Forward-only CUDA-graph replay timings unless a recorded environment states otherwise. Attention microbenchmarks and single-Transformer-block studies are reported separately. Excludes Python dispatch, mask setup, compilation, backward and full-model execution. Ratios greater than one favor the denominator; geometric means weight paired workload/input-seed median timings equally. No best-only filtering, cross-run pooling, or pooling of default and hinted FlexAttention baselines.",
            "studies": [analyze_directory(path) for path in sorted(directories)]}


def _fmt(value):
    return "—" if value is None else f"{value:.4g}"


def markdown(summary):
    lines = ["# Attention GPU experiment report", "", summary["scope"], ""]
    for study in summary["studies"]:
        lines.extend([f"## {study['stage']}: {study['directory']}", "",
                      f"Status: {study['status']}. Recorded {study['recorded_cases']} / {study['planned_cases']} planned cases; "
                      f"passed {study['passed_cases']}, failed {study['failed_cases']}. "
                      f"Requested FlexAttention failures/unavailable: {study['flex_failed_or_unavailable']} / {study['flex_requested']}.", ""])
        for env in study["environments"]:
            hardware, software = env.get("hardware") or {}, env.get("software") or {}
            lines.append(f"Environment: {hardware.get('name', 'unrecorded GPU')}; PyTorch {software.get('torch', '?')}; "
                         f"Triton {software.get('triton', '?')}; CUDA {software.get('cuda_runtime', '?')}. {env.get('timing_scope') or ''}")
        lines.append("")
        for label in ("ablations", "confirmation"):
            if study[label]:
                lines.extend([f"### {label.capitalize()}", "", "| Ratio | Pairs | Geomean | Range | Regressions | Missing / failed / ambiguous / incompatible |",
                              "|---|---:|---:|---:|---:|---:|"])
                for item in study[label]:
                    lines.append(f"| {item['comparison']} | {item['paired_cases']} | {_fmt(item['geomean_speedup'])}× | "
                                 f"{_fmt(item['min_speedup'])}–{_fmt(item['max_speedup'])}× | {item['regressions']} | "
                                 f"{item['missing_counterpart_groups']} / {item['failed_groups']} / {item['ambiguous_groups']} / {item['incompatible_environment_groups']} |")
                lines.extend(["", "Document masks are excluded as inapplicable to interval specialization; their counts remain in SUMMARY.json.", ""])
        if study["block_comparisons"]:
            lines.extend(["### One-block and isolated-attention comparisons", "", "These measurements use the same BF16 block weights and input seed within each case. They measure one inference block, not whole-model or application latency.", "",
                          "| Baseline | Scope | Pairs | Geomean | Range | Regressions | Failed / unavailable |",
                          "|---|---|---:|---:|---:|---:|---:|"])
            for item in study["block_comparisons"]:
                lines.append(f"| {item['baseline']} / {item['baseline_variant']} | {item['scope']} | {item['paired_cases']} | {_fmt(item['geomean_speedup'])}× | "
                             f"{_fmt(item['min_speedup'])}–{_fmt(item['max_speedup'])}× | {item['regressions']} | {item['failed_or_unavailable']} |")
            lines.extend(["", "Configured tiles were fixed before block evaluation. Full-block ratios and isolated-attention ratios remain separate; illustrative Amdahl estimates and exact per-case source values are in SUMMARY.json.", ""])
        if study["baselines"]:
            lines.extend(["### External baselines", "", "| Candidate | Baseline / options | Pairs | Geomean | Range | Regressions | Unavailable / candidate failures |",
                          "|---|---|---:|---:|---:|---:|---:|"])
        for item in study["baselines"]:
            options = json.dumps(item["kernel_options"], sort_keys=True) if item["kernel_options"] is not None else "options unrecorded" if item["baseline"] == "flex" else ""
            lines.append(f"| {item['candidate']} | {item['baseline']} / {item['baseline_variant']} {options} | {item['paired_cases']} | {_fmt(item['geomean_speedup'])}× | "
                         f"{_fmt(item['min_speedup'])}–{_fmt(item['max_speedup'])}× | {item['regressions']} | "
                         f"{item['baseline_unavailable']} / {item['candidate_failed']} |")
        lines.extend(["", "SDPA uses native causal/dense dispatch where applicable and an explicit boolean mask otherwise; its selected backend is not inspected. FlexAttention requests the Triton backend. Exact baseline source IDs and mask modes are in SUMMARY.json.", ""])
        if study["setup_observations"]:
            lines.extend(["### Setup observations", "", "These successful-call measurements mix fresh compilation and cache reuse. First-call time is not pure compilation time; setup costs are excluded from the warmed speedup tables.", "",
                          "| Variant | CPU schedule compile median (ms) | First call median (ms) | First call range (ms) |", "|---|---:|---:|---:|"])
            for item in study["setup_observations"]:
                fields = item["statistics_ms"]
                lines.append(f"| {item['variant']} | {_fmt(fields['schedule_compile_ms']['median'])} | {_fmt(fields['first_call_ms']['median'])} | "
                             f"{_fmt(fields['first_call_ms']['min'])}–{_fmt(fields['first_call_ms']['max'])} |")
            lines.append("")
        if study["tuning"]:
            lines.extend(["### Tuning resource observations", "", "These are within-workload associations, not causal explanations or measured occupancy.", ""])
            for workload in study["tuning"]:
                rows = workload["rows_by_latency"]
                lines.append(f"- {workload['workload_id']}: {len(rows)} valid configurations, {len(workload['failed_configurations'])} failed. "
                             + (f"Latency range {_fmt(rows[0]['latency_ms'])}–{_fmt(rows[-1]['latency_ms'])} ms. " if rows else "")
                             + "; ".join(f"{name} Spearman ρ={_fmt(value['spearman_rho'])} (n={value['samples']})"
                                         for name, value in workload["resource_associations"].items()))
            lines.append("")
        replay = study.get("search_replay")
        if replay:
            lines.extend(["### Offline search replay", "", replay.get("methodology") or "Offline replay only.", "",
                          "| Method | Budget | Trials | Failed | Mean regret | Median regret |", "|---|---:|---:|---:|---:|---:|"])
            for item in replay["groups"]:
                lines.append(f"| {item['method']} | {item['budget']} | {item['trials']} | {item['failed_trials']} | "
                             f"{_fmt(item['mean_relative_regret'])} | {_fmt(item['median_relative_regret'])} |")
            lines.append("")
        if study["failures"]:
            lines.extend(["### Failures", ""])
            for item in study["failures"]:
                error = str(item.get("error") or item["status"]).replace("\n", " ")
                lines.append(f"- {item['case_id']}: {error}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = analyze(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "REPORT.md").write_text(markdown(summary))
    print(json.dumps({"studies": len(summary["studies"]), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
