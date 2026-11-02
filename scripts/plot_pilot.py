"""Plot completed pilot cases. Requires matplotlib; no GPU dependencies."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--title", default="Training throughput and GPU memory")
    args = parser.parse_args()
    aggregate = json.loads((args.run_dir / "aggregate.json").read_text())
    cases = sorted(
        (c for c in aggregate["cases"] if c["status"] == "completed"),
        key=lambda c: c["tokens_per_second"], reverse=True,
    )
    if not cases:
        parser.error("No completed cases to plot")
    labels = [
        f"{c['attention_backend'].upper()}  |  microbatch {c['batch_size']}  |  "
        + ("block checkpoint" if c["activation_checkpointing"] == "block" else "no checkpoint")
        for c in cases
    ]
    colors = ["#147d70" if c["attention_backend"] == "sdpa" else "#697b94" for c in cases]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), sharey=True)
    metrics = (
        ([c["tokens_per_second"] / 1000 for c in cases], "Throughput (thousand tokens/s)", "{:.1f}"),
        ([c["peak_cuda_allocated_bytes"] / 2**30 for c in cases], "Peak allocated CUDA memory (GiB)", "{:.2f}"),
    )
    for ax, (values, label, fmt) in zip(axes, metrics):
        bars = ax.barh(range(len(cases)), values, height=0.61, color=colors)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=5, fontsize=10)
        ax.set_xlabel(label, labelpad=10)
        ax.set_xlim(0, max(values) * 1.19)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e3e7eb", linewidth=0.7)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(cases)), labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(args.title, x=0.035, ha="left", y=0.97, fontsize=18, fontweight="bold")
    count = len(cases)
    fig.text(0.035, 0.905, f"{count}/{len(aggregate['cases'])} cases completed · "
             "Each row is one run; bars do not represent repeated-run uncertainty.", color="#52606d")
    fig.text(0.035, 0.045, "Throughput excludes setup, evaluation and checkpoint saves. "
             "Memory includes training and evaluation.\n"
             "Microbatch changes preserve the effective token batch within this pilot.",
             fontsize=9, color="#52606d", linespacing=1.6)
    fig.subplots_adjust(left=0.34, right=0.975, top=0.84, bottom=0.19, wspace=0.2)
    for suffix in ("png", "svg"):
        output = args.run_dir / f"throughput-memory.{suffix}"
        fig.savefig(output, dpi=180, facecolor="white")
        print(output.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
