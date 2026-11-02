"""Plot sampled quality curves and observed target attainment from returned evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#4477AA", "#228833", "#CC6677", "#AA3377", "#66CCEE"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    cases = list(evidence["cases"].values())
    cases.sort(key=lambda case: (case["metadata"]["config"]["batch_size"],
                                 case["metadata"]["config"]["model"]["activation_checkpointing"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), gridspec_kw={"height_ratios": [1.5, 1]})
    labels, colors, times, memory = [], [], [], []
    handles = []
    for index, case in enumerate(cases):
        config = case["metadata"]["config"]
        evaluations = sorted((row for row in case["metrics"] if row["event"] == "eval"), key=lambda row: row["step"])
        if not evaluations or case["summary"]["status"] != "completed":
            raise ValueError("Quality plots require completed cases with evaluation evidence")
        tokens_per_step = case["metadata"]["tokens_per_optimizer_step"]
        checkpointing = config["model"]["activation_checkpointing"]
        label = f"{checkpointing.capitalize()} · mb{config['batch_size']}/acc{config['grad_accum_steps']}"
        color = COLORS[index % len(COLORS)]
        labels.append(label)
        colors.append(color)
        losses = [row["valid_loss"] for row in evaluations]
        style = ("-", "--", ":")[index % 3]
        handle, = axes[0, 0].plot([row["step"] * tokens_per_step / 1e6 for row in evaluations], losses,
                                 color=color, linestyle=style, marker="o", markersize=3,
                                 linewidth=1.7, label=label)
        handles.append(handle)
        axes[0, 1].plot([row["wall_seconds"] for row in evaluations], losses,
                        color=color, linestyle=style, marker="o", markersize=3, linewidth=1.7)
        crossing = next((row for row in evaluations if row["valid_loss"] <= args.target), None)
        times.append(crossing["wall_seconds"] if crossing else None)
        memory.append(case["summary"]["peak_cuda_allocated_bytes"] / 2**30)
        if crossing:
            axes[0, 0].scatter(crossing["step"] * tokens_per_step / 1e6, crossing["valid_loss"],
                                color=color, s=55, edgecolor="white", zorder=4)
            axes[0, 1].scatter(crossing["wall_seconds"], crossing["valid_loss"],
                                color=color, s=55, edgecolor="white", zorder=4)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
        axis.set_axisbelow(True)
    for axis in axes[0]:
        axis.axhline(args.target, color="#666666", linestyle="--", linewidth=1)
        axis.set_ylabel("Validation cross-entropy")
        axis.set_ylim(bottom=2.05)
    axes[0, 0].set(title="Loss versus training tokens", xlabel="Training tokens (millions)")
    axes[0, 0].text(.96, .92, "All three curves nearly overlap", transform=axes[0, 0].transAxes,
                      ha="right", fontsize=10, color="#555555")
    axes[0, 1].set(title="Loss versus elapsed time", xlabel="Evaluation wall time (seconds)")
    axes[0, 1].text(.97, args.target + .05, f"Target {args.target:.2f}", transform=axes[0, 1].get_yaxis_transform(),
                      ha="right", fontsize=10, color="#555555")
    for axis, values, title, xlabel, unit in (
        (axes[1, 0], times, "First observed target attainment", "Elapsed seconds", " s"),
        (axes[1, 1], memory, "Peak allocated memory", "GiB", " GiB"),
    ):
        width = [value if value is not None else 0 for value in values]
        axis.barh(labels, width, color=colors, height=.55)
        axis.invert_yaxis()
        axis.set(title=title, xlabel=xlabel)
        axis.set_xlim(0, max(width) * 1.25)
        for y, value in enumerate(values):
            if unit == " GiB":
                axis.text(value - .12, y, f"{value:.2f}{unit}", ha="right", va="center",
                          fontsize=10, color="white", fontweight="bold")
            else:
                axis.text((value or 0) + max(width) * .025, y,
                          f"{value:.1f}{unit}" if value is not None else "Unreached", va="center", fontsize=10)
    axes[1, 1].axvline(3, color="#666666", linestyle="--", linewidth=1)
    axes[1, 1].text(.98, .96, "Dashed line: 3 GiB allocated budget", transform=axes[1, 1].transAxes,
                   ha="right", va="top", fontsize=9, color="#555555")
    fig.suptitle("100M / 1K · quality development on one L40S", fontsize=17, y=.98)
    fig.text(.5, .935, "Seed 1337 · 1,200 updates · 9.83M tokens per policy · 64 fixed validation windows every 50 updates",
             ha="center", fontsize=10, color="#555555")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
    fig.text(.5, .025, "Points are observed evaluations; connecting lines do not estimate target crossings. All first crossings: update 600.\n"
             "Elapsed time includes setup and prior evaluations. Allocated memory excludes external CUDA overhead. Development evidence only.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(top=.86, bottom=.17, left=.12, right=.98, hspace=.55, wspace=.46)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output_dir / f"quality-curves.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
