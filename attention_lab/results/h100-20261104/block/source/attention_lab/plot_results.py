"""Plot recorded attention experiments, retaining distinct stage runs.

No GPU calls occur. Every timing comes from saved case JSON. Failed or missing
measurements are counted, never replaced or silently treated as successes.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

from .analyze_study import analyze_directory


COLORS = ("#2369A8", "#D77A26", "#24977F", "#9467A8", "#C45763")


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _groups(pairs):
    grouped = defaultdict(list)
    for pair in pairs:
        if _positive(pair.get("ratio")):
            grouped[pair["workload_id"]].append(pair["ratio"])
    return {name: {"mean": math.exp(statistics.mean(math.log(value) for value in values)),
                   "low": min(values), "high": max(values), "count": len(values)}
            for name, values in grouped.items()}


def _workload_order(name):
    match = re.search(r"n(\d+)-d(\d+)-(\w+)", name)
    return (int(match[1]), int(match[2]), match[3]) if match else (0, 0, name)


def _label(name):
    match = re.search(r"n(\d+)-d(\d+)-(\w+)", name)
    if not match:
        return name.replace("tune-", "").replace("window", "window ")
    length = int(match[1])
    length_text = f"{length // 1024}K" if length % 1024 == 0 else str(length)
    pattern = "window 128" if match[3] == "window" else match[3]
    return f"{length_text} · D{match[2]} · {pattern}"


def load_studies(root):
    root = Path(root).resolve()
    directories = {path.parent for path in root.rglob("cases") if path.is_dir()}
    if (root / "cases").is_dir():
        directories.add(root)
    studies = []
    for directory in sorted(directories):
        paths = sorted((directory / "cases").glob("*.json"))
        if not paths:
            continue
        records = [json.loads(path.read_text()) for path in paths]
        report = analyze_directory(directory)
        try:
            relative = directory.relative_to(root)
        except ValueError:
            relative = directory
        run = str(relative.parent if relative.name == "study" else relative)
        studies.append({"run": run, "directory": directory, "records": records, "report": report})
    return studies


def _empty(axis, title, detail="No valid measurements available"):
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    axis.text(0.5, 0.5, detail, ha="center", va="center", transform=axis.transAxes,
              color="#788493", fontsize=11, wrap=True)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _style_ratio_axis(axis, labels, xlabel):
    axis.set_yticks(range(len(labels)), [_label(name) for name in labels])
    axis.invert_yaxis()
    axis.axvline(1, color="#7F8A96", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#E5EAF0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.tick_params(axis="y", length=0)


def _range_series(axis, datasets, *, colors=COLORS):
    names = sorted({name for _, grouped in datasets for name in grouped}, key=_workload_order)
    width = min(0.22, 0.7 / max(len(datasets), 1))
    for index, (label, grouped) in enumerate(datasets):
        y, means, left, right = [], [], [], []
        for position, name in enumerate(names):
            if name not in grouped:
                continue
            stats = grouped[name]
            y.append(position + (index - (len(datasets) - 1) / 2) * width)
            means.append(stats["mean"])
            left.append(stats["mean"] - stats["low"])
            right.append(stats["high"] - stats["mean"])
        axis.errorbar(means, y, xerr=[left, right], fmt="o", markersize=5,
                      capsize=3, elinewidth=1.5, color=colors[index % len(colors)],
                      label=label, zorder=3)
    return names


def plot_ablation(axis, studies):
    datasets = []
    for study in studies:
        if study["report"]["stage"] != "ablation":
            continue
        for comparison in study["report"]["ablations"]:
            if comparison["comparison"] == "csr_full/interval_split" and comparison["pairs"]:
                datasets.append((study["run"], _groups(comparison["pairs"])))
    title = "A  Specializing mask traversal improves the kernel"
    if not datasets:
        _empty(axis, title)
        return
    names = _range_series(axis, datasets)
    _style_ratio_axis(axis, names, "CSR full-block latency / interval-split latency  (×)")
    axis.set_title(title, loc="left", fontweight="bold", pad=22)
    axis.text(0, 1.015, "Geometric mean of paired seeds; whiskers show minimum–maximum",
              transform=axis.transAxes, fontsize=9, color="#657180")
    if len(datasets) > 1:
        axis.legend(fontsize=7, loc="lower right", frameon=False)


def plot_tuning(axis, studies):
    groups = defaultdict(list)
    failed, missing = 0, 0
    for study in studies:
        if study["report"]["stage"] != "tuning":
            continue
        for record in study["records"]:
            case, measured = record["case"], record["measurement"]
            if measured.get("status") != "passed" or measured.get("correct") is not True or not _positive(measured.get("latency_ms")):
                failed += 1
                continue
            spills = measured.get("kernel_metadata", {}).get("n_spills")
            if not isinstance(spills, (int, float)) or isinstance(spills, bool) or not math.isfinite(spills) or spills < 0:
                missing += 1
                continue
            groups[(study["run"], case["workload_id"])].append((spills + 1, measured["latency_ms"] * 1000))
    title = "B  Compiler resource use helps explain configuration trade-offs"
    if not groups:
        _empty(axis, title, f"No valid latency / spill pairs\nFailed cases: {failed}; missing metadata: {missing}")
        return
    workload_names = sorted({name for _, name in groups})
    runs = sorted({run for run, _ in groups})
    markers = ("o", "s", "^", "D", "P")
    for (run, name), points in sorted(groups.items()):
        label = _label(name) + (f" [{run}]" if len(runs) > 1 else "")
        x, y = zip(*points)
        axis.scatter(x, y, s=30, alpha=0.8, linewidths=0.4, edgecolors="white",
                     color=COLORS[workload_names.index(name) % len(COLORS)],
                     marker=markers[runs.index(run) % len(markers)], label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Compiler-reported spills + 1  (log scale)")
    axis.set_ylabel("Warmed forward latency  (µs, log scale)")
    axis.set_title(title, loc="left", fontweight="bold", pad=22)
    count = sum(len(points) for points in groups.values())
    axis.text(0, 1.015, f"All {count} valid configurations · {failed} failed · {missing} missing metadata; association, not causation",
              transform=axis.transAxes, fontsize=9, color="#657180")
    axis.grid(which="major", color="#E5EAF0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8, frameon=False, loc="best")


def plot_blocks(axis, studies):
    datasets = []
    for study in studies:
        if study["report"]["stage"] != "block":
            continue
        grouped = {"attention": defaultdict(list), "block": defaultdict(list)}
        for record in study["records"]:
            case, measured = record["case"], record["measurement"]
            if measured.get("status") != "passed" or measured.get("correct") is not True:
                continue
            variants = measured.get("variants", {})
            custom, baseline = variants.get("custom_interval_split", {}), variants.get("flex_triton", {})
            if any(value.get("status") != "passed" or value.get("correct") is not True for value in (custom, baseline)):
                continue
            pattern = case.get("pattern", "window" if case.get("mask", {}).get("window_left") is not None else "causal")
            name = f"n{case['length']}-d{case['head_dim']}-{pattern}"
            for scope, field in (("attention", "attention_latency_ms"), ("block", "block_latency_ms")):
                if _positive(custom.get(field)) and _positive(baseline.get(field)):
                    grouped[scope][name].append({"workload_id": name, "ratio": baseline[field] / custom[field]})
        for scope in ("attention", "block"):
            pairs = [pair for values in grouped[scope].values() for pair in values]
            if pairs:
                label = "Isolated attention" if scope == "attention" else "Full Transformer block"
                datasets.append((label + (f" [{study['run']}]" if sum(s["report"]["stage"] == "block" for s in studies) > 1 else ""), _groups(pairs)))
    title = "C  Kernel gains translate differently inside a Transformer block"
    if not datasets:
        _empty(axis, title, "Transformer-block measurements are not available yet")
        return
    names = _range_series(axis, datasets)
    _style_ratio_axis(axis, names, "FlexAttention Triton latency / custom latency  (×)")
    axis.set_title(title, loc="left", fontweight="bold", pad=22)
    axis.text(0, 1.015, "Same weights and operators; separately timed attention and one inference block",
              transform=axis.transAxes, fontsize=9, color="#657180")
    axis.legend(fontsize=8, frameon=False, loc="lower right")


def plot_flex_hints(axis, studies):
    datasets = []
    unavailable = 0
    for study in studies:
        stage = study["report"]["stage"]
        if stage not in ("ablation", "flex_hints"):
            continue
        for baseline in study["report"]["baselines"]:
            if not baseline["candidate"].startswith("interval_split:64-64-4-2") or baseline["baseline"] != "flex":
                continue
            unavailable += baseline["baseline_unavailable"]
            if not baseline["pairs"]:
                continue
            label = "Flex default" if stage == "ablation" else "Flex + contiguity hint"
            datasets.append((f"{label} [{study['run']}]", _groups(baseline["pairs"])))
    title = "D  Stronger baseline: give FlexAttention the contiguity fact"
    if not datasets:
        _empty(axis, title)
        return
    names = _range_series(axis, datasets)
    _style_ratio_axis(axis, names, "FlexAttention Triton latency / interval-split latency  (×)")
    axis.set_title(title, loc="left", fontweight="bold", pad=22)
    axis.text(0, 1.015, f"Separate runs; normalize within each run · {unavailable} unavailable pairs",
              transform=axis.transAxes, fontsize=9, color="#657180")
    # Keep the full source-run identity visible without pooling distributions.
    axis.legend(fontsize=7, frameon=False, loc="lower right")


def make_figure(root, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "attention-lab-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    studies = load_studies(root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#D5DCE5",
                         "text.color": "#243246", "axes.labelcolor": "#48576B",
                         "xtick.color": "#526174", "ytick.color": "#526174",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    plot_ablation(axes[0, 0], studies)
    plot_tuning(axes[0, 1], studies)
    plot_blocks(axes[1, 0], studies)
    plot_flex_hints(axes[1, 1], studies)
    devices = sorted({record["measurement"].get("hardware", {}).get("name", "unknown GPU")
                      for study in studies for record in study["records"]})
    total = sum(study["report"]["recorded_cases"] for study in studies)
    failed = sum(study["report"]["failed_cases"] for study in studies)
    figure.suptitle("Attention specialization: measured GPU trade-offs", x=0.055, y=0.985,
                     ha="left", fontsize=20, fontweight="bold")
    figure.text(0.055, 0.949, " / ".join(devices) + "  ·  warm CUDA graph replay  ·  forward only",
                fontsize=11, color="#657180")
    figure.text(0.055, 0.022,
                f"Ratios above 1 favor the custom kernel. All available workloads shown; runs remain separate. "
                f"{total} recorded cases, {failed} failed cases.\n"
                "Whiskers are observed seed ranges, not confidence intervals. Compiler spill counts are not Nsight measurements. "
                "Kernel and block results exclude compilation and application overhead.",
                fontsize=8.5, color="#657180", linespacing=1.6)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.88, wspace=0.43, hspace=0.45)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return {"output": str(output.resolve()), "study_runs": len(studies),
            "recorded_cases": total, "failed_cases": failed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("--output must name a .png file")
    print(json.dumps(make_figure(args.input, args.output)))


if __name__ == "__main__":
    main()
