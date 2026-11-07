"""Run correctness-gated kernel search on a local CUDA GPU.

Every candidate uses the same inputs and reference checks. A winning candidate
is checked again with a fresh input seed outside the search evaluation budget.
No cloud jobs are launched, and this module does not contain assumed timings.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .benchmark import run_case
from .masks import MaskSpec
from .search import CandidateResult, default_configs, search


def run_experiment(*, budget=12, method="evolution", search_seed=0, **case):
    """Search one shape/mask, then independently remeasure the selected winner.

    ``case`` supplies the non-configuration arguments of benchmark.run_case.
    The candidate budget excludes the single confirmation evaluation. Neither
    correctness nor performance on unseen shapes is inferred from this run.
    """
    candidates = default_configs()
    if type(budget) is not int or not 1 <= budget <= len(candidates):
        raise ValueError(f"budget must be between 1 and {len(candidates)}")
    forbidden = {"block_q", "block_k", "num_warps", "num_stages"} & case.keys()
    if forbidden:
        raise ValueError("kernel configurations are selected by search")
    measurements = []

    def evaluate(config):
        start = time.perf_counter()
        try:
            measurement = run_case(**case, **asdict(config))
        except Exception as exc:
            measurement = {"status": "failed", "correct": False,
                           "latency_ms": None, "error_type": type(exc).__name__,
                           "error": str(exc)}
        measurements.append({"config": asdict(config), "measurement": measurement,
                             "wall_time_ms": (time.perf_counter() - start) * 1000})
        return CandidateResult(
            correct=measurement.get("correct") is True and measurement.get("status") == "passed",
            latency_ms=measurement.get("latency_ms"),
            # First-call cost includes more than JIT and must not be relabeled.
            compile_ms=measurement.get("compile_time_ms"),
            error=measurement.get("error"),
        )

    started = time.perf_counter()
    result = search(evaluate, candidates, budget=budget, method=method, seed=search_seed)
    search_wall_ms = (time.perf_counter() - started) * 1000
    confirmation = None
    if result.best is not None:
        confirm_case = {**case, "seed": case.get("seed", 0) + 1}
        confirm_start = time.perf_counter()
        try:
            confirm_measurement = run_case(**confirm_case, **asdict(result.best.config))
        except Exception as exc:
            confirm_measurement = {"status": "failed", "correct": False,
                                   "error_type": type(exc).__name__, "error": str(exc)}
        confirmation = {"config": asdict(result.best.config),
                        "measurement": confirm_measurement,
                        "wall_time_ms": (time.perf_counter() - confirm_start) * 1000,
                        "counts_toward_search_budget": False}
    confirmed = confirmation is not None and CandidateResult(
        correct=confirmation["measurement"].get("correct") is True
        and confirmation["measurement"].get("status") == "passed",
        latency_ms=confirmation["measurement"].get("latency_ms"),
        error=confirmation["measurement"].get("error"),
    ).eligible
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if confirmed else "failed",
        "search": result.to_dict(),
        "measurements": measurements,
        "confirmation": confirmation,
        "search_wall_time_ms": search_wall_ms,
        "total_wall_time_ms": (time.perf_counter() - started) * 1000,
        "scope": "Single shape and mask; selected winner confirmed with fresh input seed. GPU forward only. No held-out-shape or search-method superiority claim.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["evolution", "random"], default="evolution")
    parser.add_argument("--budget", type=int, choices=range(1, 25), default=12)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mask", choices=["dense", "causal", "sliding", "document", "combined"], default="sliding")
    parser.add_argument("--window-left", type=int, default=127)
    parser.add_argument("--window-right", type=int, default=0)
    parser.add_argument("--document-length", type=int, default=97)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.length, args.batch, args.heads, args.document_length, args.repeats) < 1:
        parser.error("shape, document length and repeats must be positive")
    if min(args.window_left, args.window_right, args.warmup) < 0:
        parser.error("windows and warmup must be nonnegative")
    spec = MaskSpec(
        causal=args.mask in ("causal", "combined"),
        window_left=args.window_left if args.mask in ("sliding", "combined") else None,
        window_right=args.window_right if args.mask in ("sliding", "combined") else None,
        same_document=args.mask in ("document", "combined"),
    )
    documents = [i // args.document_length for i in range(args.length)] if spec.same_document else None
    report = {"schema_version": 1, "status": "not_run"}
    try:
        import torch
        if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
            raise RuntimeError("A CUDA GPU is required; no search or cloud job was run")
        report = run_experiment(
            method=args.method, budget=args.budget, search_seed=args.search_seed,
            length=args.length, head_dim=args.head_dim, batch=args.batch, heads=args.heads,
            spec=spec, document_ids=documents, device=args.device, dtype=args.dtype,
            seed=args.input_seed, warmup=args.warmup, repeats=args.repeats,
        )
    except Exception as exc:
        report.update({"error_type": type(exc).__name__, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
