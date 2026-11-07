"""Recompute extension results from explicit runs; no GPU or ML dependencies.

python -m attention_lab.analyze_extensions --decode-dir runs/DECODE \
    --backward-dir runs/BACKWARD --output-dir reports/extensions

Only the two explicit final runs contribute performance aggregates. Optional
--diagnostic-dir paths retain the history of failures without pooling them.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def ratio_summary(values):
    values = [float(value) for value in values if value is not None and math.isfinite(value) and value > 0]
    return {"pairs": len(values), "wins": sum(value > 1 for value in values),
            "geomean_speedup": math.exp(statistics.fmean(map(math.log, values))) if values else None,
            "median_speedup": statistics.median(values) if values else None,
            "min_speedup": min(values) if values else None, "max_speedup": max(values) if values else None}


def median(timing):
    """Recompute medians from saved samples rather than trusting derived fields."""
    samples = timing.get("samples_ms", [])
    if not samples:
        return None
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Invalid raw timing sample")
    return statistics.median(samples)


def run_metadata(root):
    return {"path": str(root.resolve()), "outcome": read_json(root / "outcome.json", {}),
            "launch": read_json(root / "launch.json", {}),
            "environment": read_json(root / "environment.json", {})}


def shape(record):
    return {key: record[key] for key in ("batch", "query_heads", "kv_heads", "groups", "length", "dim", "dtype")
            if key in record}


def error_records(value, prefix=""):
    """Keep numerical errors and explicit failure strings; do not lose zero-gradient edge cases."""
    found = []
    if isinstance(value, dict):
        if "max_abs_error" in value or "relative_rms_error" in value:
            found.append({"path": prefix, **value})
        for key, child in value.items():
            if key in ("error", "error_type", "traceback") and isinstance(child, str):
                found.append({"path": prefix + "/" + key, "text": child})
            elif isinstance(child, (dict, list)):
                found.extend(error_records(child, prefix + "/" + key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(error_records(child, prefix + "/" + str(index)))
    return found


def decode_analysis(root):
    result = run_metadata(root)
    summary = read_json(root / "study/summary.json", {})
    records = read_json(root / "study/benchmark.json", [])
    correctness = read_json(root / "study/preflight/correctness.json", [])
    if not correctness:
        correctness = read_json(root / "study/correctness.json", [])
    eligible = result["outcome"].get("status") == "completed" and summary.get("status") == "completed"
    result.update(status=summary.get("status", "pending"), aggregate_eligible=eligible,
                  cases=len(records), planned_cases=summary.get("planned_cases"),
                  correctness={"record_count": len(correctness), "reported": summary.get("preflight", {}),
                               "finite_records": sum(row.get("finite") is True for row in correctness),
                               "max_abs_error": max((row.get("max_abs_error", 0) for row in correctness), default=None),
                               "raw_errors": correctness})
    per_split = defaultdict(list)
    baseline_pairs = defaultdict(list)
    per_case = []
    best_tile_pairs = []
    for index, record in enumerate(records):
        timings = {key: median(value) for key, value in record.get("timings", {}).items()}
        timings = {key: value for key, value in timings.items() if value is not None}
        token = {key: value for key, value in timings.items() if key.startswith("cuda_token_split_")}
        tile = {key: value for key, value in timings.items() if key.startswith("cuda_tile_split_")}
        custom = {**token, **tile}
        fixed = {}
        for name, tile_ms in tile.items():
            split = name.rsplit("_", 1)[1]
            other = timings.get("cuda_token_split_" + split)
            if other is not None:
                fixed[split] = other / tile_ms
                if eligible:
                    per_split[split].append(other / tile_ms)
        row = {"case_index": index, "shape": shape(record), "seed": record.get("seed"),
               "medians_ms": timings, "fixed_split_token_over_tile": fixed,
               "raw_errors": record.get("checks", {}), "setup": record.get("setup", {})}
        if custom:
            best = min(custom, key=custom.get)
            row["best_observed"] = {"variant": best, "median_ms": custom[best]}
            row["baseline_over_best_observed"] = {}
            baseline_names = ["cuda_token_split_1", "cuda_tile_split_1"] + sorted(
                key for key in timings if not key.startswith("cuda_token_split_") and not key.startswith("cuda_tile_split_"))
            for name in baseline_names:
                if name in timings:
                    ratio = timings[name] / custom[best]
                    row["baseline_over_best_observed"][name] = ratio
                    if eligible:
                        baseline_pairs[name].append(ratio)
        if token and tile:
            row["best_token_over_best_tile"] = min(token.values()) / min(tile.values())
            if eligible:
                best_tile_pairs.append(row["best_token_over_best_tile"])
        per_case.append(row)
    distinct_shapes = len({json.dumps(shape(record), sort_keys=True) for record in records})
    result.update(distinct_shapes=distinct_shapes, seeds=sorted({record["seed"] for record in records}),
                  fixed_split={key: ratio_summary(values) for key, values in sorted(per_split.items(), key=lambda item:int(item[0]))},
                  fixed_split_pooled=ratio_summary([value for values in per_split.values() for value in values]),
                  best_token_over_best_tile=ratio_summary(best_tile_pairs),
                  best_observed_baselines={key: {**ratio_summary(values), "unpaired_cases": len(records)-len(values)}
                                           for key, values in baseline_pairs.items()}, per_case=per_case)
    return result


def backward_analysis(root):
    result = run_metadata(root)
    summary = read_json(root / "study/summary.json", {})
    cases = summary.get("cases", [])
    eligible = result["outcome"].get("status") == "completed" and summary.get("status") == "passed"
    result.update(status=summary.get("status", "pending"), aggregate_eligible=eligible,
                  cases=len(cases), planned_cases=summary.get("planned_cases"),
                  passed_cases=sum(case.get("status") == "passed" for case in cases),
                  kinds=dict(Counter(case.get("kind", "unknown") for case in cases)),
                  gpu_unittests=[case for case in cases if case.get("kind") == "gpu_unittests"],
                  numerical_cases=[case for case in cases if case.get("kind") == "correctness"],
                  raw_errors=error_records(cases), limitations=summary.get("limitations", []))
    pairs = defaultdict(list)
    rows = []
    for index, case in enumerate(cases):
        if case.get("kind") not in ("attention_training", "transformer_optimizer_step"):
            continue
        variants = case.get("variants", {})
        custom = variants.get("structured_triton", {})
        operations = ("step",) if case["kind"] == "transformer_optimizer_step" else ("backward_only", "forward_backward")
        for operation in operations:
            custom_record = custom.get(operation, {})
            custom_ms = median(custom_record.get("timing", {}))
            for name, baseline in variants.items():
                if name == "structured_triton":
                    continue
                baseline_record = baseline.get(operation, {})
                baseline_ms = median(baseline_record.get("timing", {}))
                custom_mode = custom_record.get("timing", {}).get("mode")
                baseline_mode = baseline_record.get("timing", {}).get("mode")
                valid = (case.get("status") == "passed" and custom.get("status") == "passed"
                         and baseline.get("status") == "passed" and custom_ms is not None
                         and baseline_ms is not None and custom_mode == baseline_mode)
                ratio = baseline_ms / custom_ms if valid else None
                row = {"case_index": index, "kind": case["kind"], "n": case.get("n"), "d": case.get("d"),
                       "batch": case.get("batch"), "heads": case.get("heads"), "spec": case.get("spec"),
                       "operation": operation, "baseline": name, "custom_ms": custom_ms,
                       "baseline_ms": baseline_ms, "baseline_over_custom": ratio,
                       "timing_mode": custom_mode, "baseline_timing_mode": baseline_mode,
                       "aggregate_eligible": eligible and valid, "scope": case.get("scope"),
                       "custom_memory": {key: value for key, value in custom_record.items() if key != "timing"},
                       "baseline_memory": {key: value for key, value in baseline_record.items() if key != "timing"}}
                rows.append(row)
                if eligible and valid:
                    pairs[(operation, name, custom_mode)].append(ratio)
    result["per_case"] = rows
    result["aggregates"] = [{"operation": operation, "baseline": name, "timing_mode": mode,
                             **ratio_summary(values)} for (operation, name, mode), values in pairs.items()]
    result["optimizer_validation"] = [{"case_index": index, "status": case.get("status"),
                                      "reference": case.get("optimizer_correctness_reference"),
                                      "variants": {name: {key: record[key] for key in ("status", "errors", "optimizer_observability") if key in record}
                                                   for name, record in case.get("variants", {}).items()}}
                                     for index, case in enumerate(cases) if case.get("kind") == "transformer_optimizer_step"]
    return result


def diagnostic(root):
    result = run_metadata(root)
    summary = read_json(root / "study/summary.json", {})
    result.update(aggregate_eligible=False, status=summary.get("status", "unknown"),
                  raw_errors=error_records(summary), reason="Explicit diagnostic run; excluded from every performance aggregate")
    return result


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    clean = lambda value: str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def mask_label(spec):
    spec = spec or {}
    labels = ["causal"] if spec.get("causal") else []
    if spec.get("window_left") is not None:
        labels.append(f"window({spec.get('window_left')},{spec.get('window_right')})")
    if spec.get("same_document"):
        labels.append("same-document")
    return "+".join(labels) or "dense"


def baseline_label(name):
    return {"flex_decode": "compiled FlexAttention (Triton backend)",
            "sdpa_flash": "PyTorch Flash SDPA",
            "flex_attention_triton": "compiled FlexAttention (Triton backend)"}.get(name, name)


def markdown(report):
    decode, backward = report["decode"], report["backward"]
    lines = ["# CUDA decode and structured-attention backward experiments", "",
             "Ratios are baseline latency / custom latency; values above 1 favor the custom implementation. "
             "Medians are independently recomputed from raw timing samples. Geometric means weight each measured pair equally.", "",
             "## Runs and environment", "", table(["Run", "Status", "Aggregate eligible", "GPU", "PyTorch", "CUDA", "Triton"],
             [[label, result["status"], result["aggregate_eligible"], ", ".join(result["environment"].get("devices", [])),
               result["environment"].get("torch"), result["environment"].get("cuda"), result["environment"].get("triton")]
              for label, result in (("Decode", decode), ("Backward", backward))]), "",
             f"Decode evidence: `{decode['path']}`. Backward evidence: `{backward['path']}`.", "",
             "## CUDA decode: fixed-split tiling ablation", "",
             f"{decode['cases']} shape/seed cases, {decode['distinct_shapes']} distinct shapes, seeds {decode['seeds']}. "
             "Each row compares token-at-a-time and tiled kernels at the same split count; both include the merge kernel.", "",
             table(["Split", "Pairs", "Wins", "Geomean speedup", "Min", "Max"],
                   [[split, stat["pairs"], stat["wins"], fmt(stat["geomean_speedup"]), fmt(stat["min_speedup"]), fmt(stat["max_speedup"])]
                    for split, stat in decode["fixed_split"].items()]), "",
             f"Pooled fixed-split comparison: **{fmt(decode['fixed_split_pooled']['geomean_speedup'])}×** "
             f"across {decode['fixed_split_pooled']['pairs']} pairs; {decode['fixed_split_pooled']['wins']} wins. "
             "These include correlated splits/seeds and are not independent statistical trials.", "",
             "## CUDA decode: best observed configurations", "",
             "The split is chosen using these same measurements. This is an oracle selection result, not a held-out dispatcher or production speedup.", "",
             table(["Baseline", "Paired cases", "Missing", "Wins", "Geomean speedup", "Min", "Max"],
                   [[baseline_label(name), stat["pairs"], stat["unpaired_cases"], stat["wins"], fmt(stat["geomean_speedup"]), fmt(stat["min_speedup"]), fmt(stat["max_speedup"])]
                    for name, stat in decode["best_observed_baselines"].items()]), "",
             f"Best tile versus best token: {fmt(decode['best_token_over_best_tile']['geomean_speedup'])}× geometric mean. "
             "A ratio below 1 against SDPA means the custom kernel is slower than that optimized baseline. "
             "The raw key `flex_decode` names compiled FlexAttention with the Triton backend; it does not establish use of a dedicated FlexDecoding kernel.", "",
             "## Training attention: per-case GPU graph replay", "",
             table(["Case", "N", "D", "Mask", "Operation", "Baseline", "Custom ms", "Baseline ms", "Speedup"],
                   [[row["case_index"], row["n"], row["d"], mask_label(row["spec"]), row["operation"], baseline_label(row["baseline"]),
                     fmt(row["custom_ms"]), fmt(row["baseline_ms"]), fmt(row["baseline_over_custom"])]
                    for row in backward["per_case"] if row["kind"] == "attention_training"]), "",
             "## Complete optimizer steps: matched eager CUDA events", "",
             "This measurement includes parameter restoration, forward/loss, backward, SGD and host submission gaps. "
             "It is separate from attention-only graph replay and is not an end-to-end model-training throughput claim.", "",
             table(["Case", "N", "D", "Mask", "Baseline", "Custom ms", "Baseline ms", "Speedup", "Mode"],
                   [[row["case_index"], row["n"], row["d"], mask_label(row["spec"]), baseline_label(row["baseline"]),
                     fmt(row["custom_ms"]), fmt(row["baseline_ms"]), fmt(row["baseline_over_custom"]), row["timing_mode"]]
                    for row in backward["per_case"] if row["kind"] == "transformer_optimizer_step"]), "",
             "## Correctness", "",
             f"Decode preflight: {decode['correctness']['record_count']} recorded comparisons, "
             f"{decode['correctness']['finite_records']} finite; maximum absolute error {decode['correctness']['max_abs_error']}. "
             "These are tolerance checks, not bitwise equivalence.", "",
             f"Backward study: {backward['passed_cases']}/{backward['cases']} study cases passed "
             f"({backward['planned_cases']} planned). Case types: {backward['kinds']}.", ""]
    for unit in backward["gpu_unittests"]:
        lines += [f"GPU unittest suite: {unit.get('tests_run')} tests, status {unit.get('status')}, "
                  f"skipped {len(unit.get('skipped', []))}. {unit.get('coverage', '')}", ""]
    lines += ["Raw output/dQ/dK/dV errors, optimizer gradient/update-delta comparisons, and nonzero-update checks "
              "are retained in `summary.json`. Near-zero reference gradients can make relative RMS errors large; "
              "absolute errors and the recorded tolerances must be inspected together.", "", "## Scope and limitations", "",
              "- One H100 and the recorded software versions; no cross-device claim or statistical confidence interval.",
              "- Decode benchmarks use fixed synthetic BF16 shapes and CUDA graph replay; preallocated scratch and warm kernel execution include split/merge work but exclude compilation, allocation and Python dispatch.",
              "- Setup and first-call measurements remain in JSON and are separate from warmed latency.",
              "- Best observed configurations reuse the evaluation measurements for selection.",
              "- Performance ratios from diagnostic, failed or incomplete runs are excluded from all aggregates."]
    lines += ["- " + item for item in backward["limitations"]]
    if report["diagnostics"]:
        lines += ["", "## Diagnostic runs excluded from aggregates", ""]
        for item in report["diagnostics"]:
            errors = [row["text"] for row in item["raw_errors"] if row["path"].endswith("/error")]
            lines += [f"- `{item['path']}`: {item['status']}; {len(errors)} error records retained in JSON."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-dir", type=Path, required=True)
    parser.add_argument("--backward-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.decode_dir, args.backward_dir, *args.diagnostic_dir):
        if not path.is_dir():
            parser.error(f"Run directory does not exist: {path}")
    report = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "decode": decode_analysis(args.decode_dir), "backward": backward_analysis(args.backward_dir),
              "diagnostics": [diagnostic(path) for path in args.diagnostic_dir],
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(markdown(report))
    print(json.dumps({"output": str(args.output_dir.resolve()), "decode_eligible": report["decode"]["aggregate_eligible"],
                      "backward_eligible": report["backward"]["aggregate_eligible"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
