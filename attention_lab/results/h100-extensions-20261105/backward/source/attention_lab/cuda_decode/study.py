"""Correctness and bounded, paired GPU measurements for CUDA split-KV decode."""

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

from .kernel import prepare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reference(q, k, v, lengths, *, scale=None):
    """FP32 token-level attention, without physically repeating KV heads."""
    import torch

    batch, heads, dim = q.shape
    kv_heads, capacity = k.shape[1:3]
    groups = heads // kv_heads
    queries = q.float().reshape(batch, kv_heads, groups, dim)
    scale = dim ** -0.5 if scale is None else scale
    scores = torch.einsum("bhgd,bhnd->bhgn", queries, k.float()) * scale
    allowed = torch.arange(capacity, device=q.device)[None, :] < lengths[:, None]
    scores = scores.masked_fill(~allowed[:, None, None, :], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.einsum("bhgn,bhnd->bhgd", probabilities, v.float()).reshape_as(q)


def make_inputs(batch, query_heads, groups, length, dim, dtype, seed, *, ragged=False):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, query_heads, dim), device="cuda", dtype=dtype, generator=generator)
    shape = (batch, query_heads // groups, length, dim)
    k = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    sizes = [length] * batch
    if ragged:
        sizes = [min(length, value) for value in (0, 1, max(1, length // 2), length)]
        if batch != 4:
            raise ValueError("ragged correctness fixture uses batch four")
    lengths = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    return q, k, v, lengths


def error_stats(actual, expected):
    import torch

    delta = actual.float() - expected
    return {"max_abs_error": delta.abs().max().item(),
            "rms_error": delta.square().mean().sqrt().item(),
            "finite": bool(torch.isfinite(actual).all().item())}


def check(actual, expected, dtype):
    import torch

    tolerance = 0.02 if dtype == torch.bfloat16 else 0.004
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
    return {**error_stats(actual, expected), "atol": tolerance, "rtol": tolerance}


def smoke_variant(output, deadline, variant):
    import torch

    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (64, 128):
            for groups in (1, 2, 4, 8):
                for length in (1, 31, 32, 33, 127, 513):
                    if time.monotonic() >= deadline:
                        return records, "time_budget_exhausted"
                    inputs = make_inputs(4, 8, groups, length, dim, dtype, 2027, ragged=True)
                    expected = reference(*inputs)
                    for splits in (1, 3, 8):
                        operation = prepare(*inputs, splits=splits, variant=variant)
                        measured = check(operation(), expected, dtype)
                        records.append({"dtype": str(dtype), "dim": dim, "groups": groups,
                                        "length": length, "splits": splits, **measured})
                    write_json(output / "correctness.json", records)
    # Exercise every merge warp and the public custom-scale path; most of the
    # 128 partitions are empty, with a completely empty request as well.
    inputs = make_inputs(4, 8, 2, 33, 128, torch.float16, 93, ragged=True)
    scale = 0.37
    operation = prepare(*inputs, splits=128, scale=scale, variant=variant)
    measured = check(operation(), reference(*inputs, scale=scale), inputs[0].dtype)
    records.append({"case": "maximum_splits_custom_scale", "splits": 128,
                    "scale": scale, **measured})
    # The API must submit onto a caller's non-default stream. Event dependency
    # makes this check independent of legacy-default-stream synchronization.
    inputs = make_inputs(4, 8, 4, 513, 64, torch.bfloat16, 91, ragged=True)
    operation = prepare(*inputs, splits=8, variant=variant)
    expected = reference(*inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = operation().clone()
    torch.cuda.current_stream().wait_stream(stream)
    measured = check(result, expected, inputs[0].dtype)
    records.append({"case": "non_default_stream", **measured})
    # Replay after changing the valid lengths in place exercises captured
    # dynamic metadata, including all-empty and partially-filled split ranges.
    with torch.cuda.stream(stream):
        operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    inputs[3].copy_(torch.tensor([513, 256, 0, 3], device="cuda", dtype=torch.int32))
    graph.replay()
    measured = check(operation.output, reference(*inputs), inputs[0].dtype)
    records.append({"case": "graph_replay_changed_lengths", **measured})
    write_json(output / "correctness.json", records)
    return records, "completed"


def smoke(output, deadline):
    records = []
    for variant in ("token", "tile"):
        cases, status = smoke_variant(output / variant, deadline, variant)
        records.extend({"variant": variant, **case} for case in cases)
        write_json(output / "correctness.json", records)
        if status != "completed":
            return records, status
    return records, "completed"


def _geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values)) if values else None


def benchmark(output, deadline):
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import flex_attention
    from attention_lab.benchmark import _time_cuda

    # Seed replicates measure sensitivity to inputs, not independent runtime
    # environments. Full valid lengths permit optimized unmasked baselines.
    cases = [
        (1, 32, 8, 1024, 128), (1, 32, 8, 4096, 128),
        (1, 32, 8, 16384, 128), (1, 32, 8, 65536, 128),
        (8, 32, 8, 4096, 128), (8, 32, 8, 16384, 128),
        (32, 32, 8, 4096, 128), (1, 16, 1, 16384, 64),
        (1, 32, 4, 4096, 64), (8, 32, 4, 4096, 64),
    ]
    write_json(output / "plan.json", {"cases": cases, "seeds": [211, 212, 213],
                                     "splits": [1, 2, 4, 8, 16, 32],
                                     "variants": ["token", "tile"],
                                     "baselines": {"sdpa_flash": "forced Flash SDPA",
                                                   "flex_decode": "FlexAttention BACKEND=TRITON_DECODE",
                                                   "flex_triton_attention": "FlexAttention BACKEND=TRITON"},
                                     "selection": "best observed split per case; no held-out dispatch claim"})
    records = []
    for case_index, (batch, heads, groups, length, dim) in enumerate(cases):
        # This is an isolated study process. Reset Dynamo's in-memory state
        # between shapes to avoid silently exhausting its recompilation cap.
        # Disk caches remain available; the recorded setup is not a cold-cache
        # compiler benchmark. A full graph prevents an eager fallback.
        factory_started = time.perf_counter()
        torch.compiler.reset()
        flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        factory_ms = (time.perf_counter() - factory_started) * 1000
        for seed in (211, 212, 213):
            if time.monotonic() >= deadline:
                return records, "time_budget_exhausted"
            inputs = make_inputs(batch, heads, groups, length, dim, torch.bfloat16, seed)
            q, k, v, lengths = inputs
            expected = reference(*inputs)
            record = {"batch": batch, "query_heads": heads, "kv_heads": heads // groups,
                      "groups": groups, "length": length, "dim": dim,
                      "seed": seed, "dtype": "torch.bfloat16", "timings": {}, "checks": {},
                      "setup": {}, "flex_factory_ms": factory_ms}
            operators = {}
            for variant, splits in ((variant, splits) for variant in ("token", "tile")
                                    for splits in (1, 2, 4, 8, 16, 32)):
                torch.cuda.synchronize()
                setup_started = time.perf_counter()
                operation = prepare(*inputs, splits=splits, variant=variant)
                torch.cuda.synchronize()
                prepare_ms = (time.perf_counter() - setup_started) * 1000
                first_call_started = time.perf_counter()
                result = operation()
                torch.cuda.synchronize()
                first_call_ms = (time.perf_counter() - first_call_started) * 1000
                name = f"cuda_{variant}_split_{splits}"
                operators[name] = operation
                record["checks"][name] = check(result, expected, q.dtype)
                record["setup"][name] = {"prepare_ms": prepare_ms, "first_call_ms": first_call_ms}
            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return functional.scaled_dot_product_attention(
                        q.unsqueeze(2), k, v, enable_gqa=True).squeeze(2)
            def flex_decode():
                return flex(q.unsqueeze(2), k, v, enable_gqa=True,
                            kernel_options={"BACKEND": "TRITON_DECODE"}).squeeze(2)
            def flex_triton_attention():
                return flex(q.unsqueeze(2), k, v, enable_gqa=True,
                            kernel_options={"BACKEND": "TRITON"}).squeeze(2)
            for name, function in (("sdpa_flash", flash), ("flex_decode", flex_decode),
                                   ("flex_triton_attention", flex_triton_attention)):
                try:
                    torch.cuda.synchronize()
                    first_call_started = time.perf_counter()
                    result = function()
                    torch.cuda.synchronize()
                    record["setup"][name] = {
                        "first_call_ms": (time.perf_counter() - first_call_started) * 1000}
                    record["checks"][name] = check(result, expected, q.dtype)
                    operators[name] = function
                except (RuntimeError, NotImplementedError) as error:
                    record.setdefault("unsupported_baselines", {})[name] = str(error)
            # Rotate operator order with the input seed so every paired repeat
            # does not run custom kernels before the optimized baselines.
            names = list(operators)
            offset = seed % len(names)
            names = names[offset:] + names[:offset]
            for name in names:
                record["timings"][name] = _time_cuda(
                    operators[name], device=q.device, warmup=3, repeats=12,
                    mode="cuda_graph", inner_repeats=10)
            cuda_names = [name for name in names if name.startswith("cuda_")]
            best = min(cuda_names, key=lambda name: record["timings"][name]["median_ms"])
            best_ms = record["timings"][best]["median_ms"]
            best_operation = operators[best]
            best_by_variant = {
                variant: min([name for name in cuda_names if name.startswith(f"cuda_{variant}_")],
                             key=lambda name: record["timings"][name]["median_ms"])
                for variant in ("token", "tile")}
            record.update({"best_observed": best, "scratch_bytes": best_operation.scratch_bytes,
                           "best_observed_by_variant": best_by_variant,
                           "best_tile_over_best_token":
                               record["timings"][best_by_variant["token"]]["median_ms"] /
                               record["timings"][best_by_variant["tile"]]["median_ms"],
                           "tile_over_token_same_split": {
                               str(splits): record["timings"][f"cuda_token_split_{splits}"]["median_ms"] /
                                            record["timings"][f"cuda_tile_split_{splits}"]["median_ms"]
                               for splits in (1, 2, 4, 8, 16, 32)},
                           "cache_bytes": (k.numel() + v.numel()) * k.element_size(),
                           "speedups": {name: timing["median_ms"] / best_ms
                                        for name, timing in record["timings"].items()
                                        if name in ("cuda_token_split_1", "cuda_tile_split_1", "sdpa_flash",
                                                    "flex_decode", "flex_triton_attention")}})
            record["effective_kv_bandwidth_gbs"] = record["cache_bytes"] / (best_ms * 1e6)
            records.append(record)
            write_json(output / "cases" / f"case-{case_index:02d}-seed-{seed}.json", record)
            write_json(output / "benchmark.json", records)
            del operators, operation, best_operation, inputs, q, k, v, lengths, expected
            gc.collect()
    return records, "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-budget-seconds", type=int, default=600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.time_budget_seconds <= 0:
        parser.error("time budget must be positive")
    if not args.execute:
        print(json.dumps({"stage": args.stage, "output_dir": str(args.output_dir),
                          "time_budget_seconds": args.time_budget_seconds, "execute": False}))
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    deadline = started + args.time_budget_seconds
    write_json(args.output_dir / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "timing_scope": "warmed CUDA graphs; includes all scratch traffic and split merge; excludes compile, Python dispatch and workspace allocation",
        "setup_scope": "synchronized wall time; prepare includes CUDA extension load/cache and workspace allocation; first call includes baseline compile/cache lookup where applicable; disk caches are preserved",
        "flex_compilation": "fullgraph, static shapes; explicit TRITON_DECODE and TRITON baselines; in-process compiler state reset between shapes, not between seed replicates",
        "prior_run_correction": "cohere-cuda-study-20261105T182625Z-06dfba used the raw key flex_decode for BACKEND=TRITON, which is standard FlexAttention, not FlexDecoding; that immutable run is diagnostic only for this comparison",
        "attention_scope": "single-token forward, BF16/FP16 contiguous KV, variable per-request valid lengths, GQA without physical KV repeat",
    })
    try:
        preflight = None
        if args.stage == "smoke":
            records, status = smoke(args.output_dir, deadline)
        else:
            preflight_records, status = smoke(args.output_dir / "preflight", deadline)
            preflight = {"status": status, "cases": len(preflight_records), "planned_cases": 582,
                         "max_abs_error": max((case["max_abs_error"] for case in preflight_records),
                                              default=None)}
            write_json(args.output_dir / "preflight" / "summary.json", preflight)
            if status == "completed":
                records, status = benchmark(args.output_dir, deadline)
            else:
                records = []
        summary = {"status": status, "stage": args.stage, "cases": len(records),
                   "wall_seconds": time.monotonic() - started}
        if args.stage == "study":
            summary["preflight"] = preflight
            summary["planned_cases"] = 30
            summary["geomean_best_tile_over_best_token"] = _geomean(
                [case["best_tile_over_best_token"] for case in records])
            summary["paired_baselines_best_observed"] = {}
            for baseline in ("cuda_token_split_1", "cuda_tile_split_1", "sdpa_flash", "flex_decode",
                             "flex_triton_attention"):
                paired = [case for case in records if baseline in case["speedups"]]
                summary["paired_baselines_best_observed"][baseline] = {
                    "paired_cases": len(paired),
                    "unpaired_completed_cases": len(records) - len(paired),
                    "distinct_shapes": len({(case["batch"], case["query_heads"],
                                             case["groups"], case["length"], case["dim"])
                                            for case in paired}),
                    "geomean_speedup": _geomean([case["speedups"][baseline]
                                                 for case in paired]),
                }
        else:
            summary["planned_cases"] = 582
            summary["max_abs_error"] = max((case["max_abs_error"] for case in records),
                                            default=None)
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary))
        if status != "completed":
            # Preserve useful partial records but never let the orchestration
            # mistake an incomplete sweep for a successful validation run.
            sys.exit(2)
    except Exception as error:
        write_json(args.output_dir / "summary.json", {
            "status": "failed", "stage": args.stage,
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
