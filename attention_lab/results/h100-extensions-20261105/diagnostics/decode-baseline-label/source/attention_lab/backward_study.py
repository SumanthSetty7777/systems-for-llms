"""Bounded, reproducible training-attention study; requires an existing CUDA GPU.

No resources are provisioned by this module. Results include negative results.
GPU graph timing excludes Python dispatch, setup, compilation and dense oracles.
"""

import argparse
from dataclasses import asdict
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

from .backward import training_attention
from .benchmark import _configure_dynamo_for_study, _prepare_flex, _time_cuda
from .masks import MaskSpec


def dense_attention(q, k, v, spec, documents=None):
    """Independent differentiable oracle; never reads the custom schedule."""
    import torch
    n = q.shape[-2]
    positions = torch.arange(n, device=q.device)
    distance = positions[:, None] - positions[None, :]
    mask = torch.ones((n, n), dtype=torch.bool, device=q.device)
    if spec.causal:
        mask = mask & (distance >= 0)
    if spec.window_left is not None:
        mask = mask & (distance <= spec.window_left)
    if spec.window_right is not None:
        mask = mask & (distance >= -spec.window_right)
    if spec.same_document:
        labels = torch.tensor(documents, device=q.device)
        mask = mask & (labels[:, None] == labels[None, :])
    scores = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
    return scores.masked_fill(~mask, -float("inf")).softmax(-1) @ v


def comparison(actual, expected, dtype):
    import torch
    atol, rtol = (0.025, 0.035) if dtype == torch.bfloat16 else (0.004, 0.006)
    difference = actual.detach().float() - expected.detach().float()
    finite = bool(torch.isfinite(difference).all())
    return {"correct": finite and bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)),
            "max_abs_error": float(difference.abs().max()) if finite else None,
            "rms_error": float(difference.square().mean().sqrt()) if finite else None,
            "relative_rms_error": float(difference.square().mean().sqrt() /
                                        expected.detach().float().square().mean().sqrt().clamp_min(1e-12)) if finite else None,
            "atol": atol, "rtol": rtol}


def values_for(n, d, dtype, seed, heads=2):
    import torch
    torch.manual_seed(seed)
    values = tuple(torch.randn(1, heads, n, d, device="cuda", dtype=dtype, requires_grad=True)
                   for _ in range(3))
    return values, torch.randn_like(values[0])


def scale_aware_comparison(actual, expected, relative_tolerance=0.05, absolute_floor=1e-8):
    """Compare small gradients/updates without a large absolute tolerance.

    In particular, zero gradients fail whenever the reference RMS exceeds the
    tiny numerical floor. Updated weights are compared as deltas, not against
    the much larger pre-existing parameter values.
    """
    import torch
    value, reference = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
    value_rms = float(value.square().mean().sqrt())
    reference_rms = float(reference.square().mean().sqrt())
    error_rms = float((value - reference).square().mean().sqrt())
    return {"correct": finite and error_rms <= relative_tolerance * reference_rms + absolute_floor,
            "actual_rms": value_rms, "reference_rms": reference_rms, "rms_error": error_rms,
            "relative_rms_error": error_rms / max(reference_rms, 1e-30),
            "actual_norm": float(value.norm()), "reference_norm": float(reference.norm()),
            "relative_rms_tolerance": relative_tolerance, "absolute_rms_floor": absolute_floor}


def numerical_case(n, d, dtype, spec, docs, seed=17, tile_q=64, tile_k=64):
    import torch
    result = {"kind": "correctness", "n": n, "d": d, "dtype": str(dtype),
              "spec": asdict(spec), "seed": seed, "block_q": tile_q, "block_k": tile_k,
              "document_pattern": "repeated labels" if docs else None}
    values, upstream = values_for(n, d, dtype, seed)
    reference_values = tuple(value.detach().float().requires_grad_() for value in values)
    actual = training_attention(*values, spec, docs, block_q=tile_q, block_k=tile_k)
    reference = dense_attention(*reference_values, spec, docs)
    actual_grads = torch.autograd.grad(actual, values, upstream)
    expected_grads = torch.autograd.grad(reference, reference_values, upstream.float())
    result["errors"] = {key: comparison(value, target, dtype) for key, value, target in
                        zip(("output", "dq", "dk", "dv"), (actual, *actual_grads), (reference, *expected_grads))}
    result["status"] = "passed" if all(v["correct"] for v in result["errors"].values()) else "failed_correctness"
    return result


def _timing_and_memory(function, repeats):
    import torch
    function()
    torch.cuda.synchronize()
    gc.collect()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    timing = _time_cuda(function, device="cuda:0", warmup=2, repeats=repeats,
                        mode="cuda_graph", inner_repeats=3)
    return {"timing": timing, "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak, "peak_increment_bytes": max(0, peak - baseline),
            "memory_scope": "one eager invocation after warmup; live inputs, retained graph and cached masks counted in baseline; CUDA graph pool excluded"}


def _variants(n, spec, docs):
    import torch.nn.functional as F
    variants = [("structured_triton", lambda q, k, v: training_attention(q, k, v, spec, docs))]
    # Compile masks outside measured paths. The Flex predicate is independently
    # constructed; CSR scheduling is never reused by the reference.
    flex, block_mask, setup_ms = _prepare_flex(n, spec, docs or (), "cuda:0")
    variants.append(("flex_attention_triton", lambda q, k, v: flex(
        q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "TRITON"})))
    if not spec.same_document and spec.window_left is None and spec.window_right is None:
        variants.append(("pytorch_sdpa_native_causal" if spec.causal else "pytorch_sdpa_native_dense",
                         lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=spec.causal)))
    return variants, setup_ms


def performance_case(n, d, dtype, spec, docs, seed=19, repeats=15):
    import torch
    result = {"kind": "attention_training", "n": n, "d": d, "batch": 1, "heads": 2,
              "dtype": str(dtype), "spec": asdict(spec), "seed": seed, "variants": {},
              "scope": "GPU graph replay, no Python dispatch; no dropout; identical Q/K/V, mask and upstream gradient"}
    values, upstream = values_for(n, d, dtype, seed)
    reference_values = tuple(value.detach().float().requires_grad_() for value in values)
    reference = dense_attention(*reference_values, spec, docs)
    reference_grads = torch.autograd.grad(reference, reference_values, upstream.float())
    variants, result["flex_mask_setup_ms"] = _variants(n, spec, docs)
    for name, function in variants:
        record = result["variants"][name] = {}
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = function(*values)
            grads = torch.autograd.grad(output, values, upstream, retain_graph=True)
            torch.cuda.synchronize()
            record["first_forward_backward_ms"] = (time.perf_counter() - start) * 1000
            record["first_call_scope"] = "compile/cache lookup, allocations, forward and backward, synchronization; caches not cleared"
            record["errors"] = {key: comparison(value, target, dtype) for key, value, target in
                                zip(("output", "dq", "dk", "dv"), (output, *grads), (reference, *reference_grads))}
            if not all(error["correct"] for error in record["errors"].values()):
                record["status"] = "failed_correctness"
                continue
            def backward_only():
                return torch.autograd.grad(output, values, upstream, retain_graph=True)
            def forward_backward():
                return torch.autograd.grad(function(*values), values, upstream)
            record["backward_only"] = _timing_and_memory(backward_only, repeats)
            record["forward_backward"] = _timing_and_memory(forward_backward, repeats)
            record["status"] = "passed"
            del output, grads
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    custom = result["variants"].get("structured_triton", {})
    if custom.get("status") == "passed":
        for name, variant in result["variants"].items():
            if name != "structured_triton" and variant.get("status") == "passed":
                for operation in ("backward_only", "forward_backward"):
                    variant[operation]["baseline_over_custom_speedup"] = (variant[operation]["timing"]["median_ms"] /
                                                                            custom[operation]["timing"]["median_ms"])
    result["status"] = "passed" if all(v.get("status") == "passed" for v in result["variants"].values()) else "failed"
    return result


def transformer_step_case(n, d, dtype, spec, docs, repeats=12):
    """A whole pre-norm block, loss, backward and manual SGD update.

    Restoring identical weights on every call is included in all timings. This
    makes every measured optimizer step use the same workload and loss, and
    prevents warmups from changing either numerics or comparison conditions.
    """
    import torch
    import torch.nn.functional as F
    heads, width = 2, 2 * d
    torch.manual_seed(73)
    x = torch.randn(1, n, width, device="cuda", dtype=dtype)
    shapes = [(3 * width, width), (width, width), (4 * width, width), (width, 4 * width)]
    # FP32 master weights preserve small SGD updates. The linear operators
    # explicitly cast to the requested activation dtype inside each step.
    initial = [torch.randn(shape, device="cuda", dtype=torch.float32) * shape[1] ** -0.5 for shape in shapes]
    variants, setup_ms = _variants(n, spec, docs)
    result = {"kind": "transformer_optimizer_step", "n": n, "d": d, "heads": heads,
              "dtype": str(dtype), "spec": asdict(spec), "variants": {}, "flex_mask_setup_ms": setup_ms,
              "architecture": "one pre-norm Transformer block, 2 heads, GELU MLP width 4C, mean squared output loss, FP32 master weights and SGD lr=0.01, weights cast to activation dtype for linear operators; no dropout",
              "scope": "CUDA graph replay of parameter restoration, forward, loss, backward and SGD; identical reset weights each invocation; excludes Python, compilation and setup"}
    baseline = None
    # SDPA/Flex is the first optimizer-step numerical comparator, rather than
    # making the custom implementation its own oracle.
    for name, attention_function in list(reversed(variants)):
        record = result["variants"][name] = {}
        try:
            weights = [torch.nn.Parameter(value.clone()) for value in initial]
            def step():
                with torch.no_grad():
                    for weight, source in zip(weights, initial):
                        weight.copy_(source)
                        weight.grad = None
                normalized = F.layer_norm(x, (width,))
                compute_weights = [weight.to(dtype) for weight in weights]
                qkv = F.linear(normalized, compute_weights[0]).view(1, n, 3, heads, d).permute(2, 0, 3, 1, 4)
                q, k, v = (part.contiguous() for part in qkv.unbind(0))
                attended = attention_function(q, k, v).transpose(1, 2).reshape(1, n, width)
                hidden = x + F.linear(attended, compute_weights[1])
                output = hidden + F.linear(F.gelu(F.linear(F.layer_norm(hidden, (width,)), compute_weights[2])), compute_weights[3])
                loss = output.float().square().mean()
                loss.backward()
                with torch.no_grad():
                    for weight in weights:
                        weight.add_(weight.grad, alpha=-0.01)
                return loss, output
            start = time.perf_counter()
            loss, output = step()
            torch.cuda.synchronize()
            record["first_step_ms"] = (time.perf_counter() - start) * 1000
            observed = (loss.detach(), output.detach(), *(w.grad.detach().clone() for w in weights),
                        *((w.detach() - source).clone() for w, source in zip(weights, initial)))
            if baseline is None:
                baseline = observed
                result["optimizer_correctness_reference"] = name
            labels = ("loss", "output", "qkv_gradient", "projection_gradient", "up_gradient", "down_gradient",
                      "qkv_update_delta", "projection_update_delta", "up_update_delta", "down_update_delta")
            record["errors"] = {label: scale_aware_comparison(value, reference)
                                for label, value, reference in zip(labels, observed, baseline)}
            record["optimizer_observability"] = {
                label: {"changed_fraction": float((update != 0).float().mean()),
                        "update_norm": float(update.norm()),
                        "gradient_norm": float(gradient.norm()),
                        "nonzero_gradient_and_update": bool(gradient.norm() > 0 and update.norm() > 0)}
                for label, gradient, update in zip(("qkv", "projection", "up", "down"), observed[2:6], observed[6:10])}
            record["loss"] = float(loss.detach())
            if (all(error["correct"] for error in record["errors"].values())
                    and all(item["nonzero_gradient_and_update"] for item in record["optimizer_observability"].values())):
                record["step"] = _timing_and_memory(step, repeats)
                record["status"] = "passed"
            else:
                record["status"] = "failed_correctness"
        except Exception as exc:
            record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    custom = result["variants"].get("structured_triton", {})
    if custom.get("status") == "passed":
        for name, variant in result["variants"].items():
            if name != "structured_triton" and variant.get("status") == "passed":
                variant["baseline_over_custom_speedup"] = variant["step"]["timing"]["median_ms"] / custom["step"]["timing"]["median_ms"]
    result["status"] = "passed" if all(v.get("status") == "passed" for v in result["variants"].values()) else "failed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-budget-seconds", type=float, default=480)
    args = parser.parse_args(argv)
    if args.time_budget_seconds <= 0:
        parser.error("time budget must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. This command does not provision GPUs.")
    started = time.perf_counter()
    _configure_dynamo_for_study()
    props = torch.cuda.get_device_properties(0)
    report = {"stage": args.stage, "budget_seconds": args.time_budget_seconds,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "triton": importlib.metadata.version("triton"), "cuda": torch.version.cuda,
                              "gpu": props.name, "gpu_memory_bytes": props.total_memory},
              "implementation": "online-softmax forward saves natural-log LSE; query-owned dQ; transposed-CSR key-owned dK/dV; FP32 accumulation; no atomics or dense attention matrix in custom implementation",
              "limitations": ["first derivatives only", "no dropout", "same sequence length for Q/K/V", "one shared document-label sequence across batch and heads", "GPU graph timings exclude Python dispatch", "correctness uses dense FP32 oracle outside timing", "time budget checked between cases; one compilation can exceed the remaining budget"],
              "cases": []}
    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        target = args.output_dir / "summary.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(target)
    jobs = []
    cases = [(17, 64, torch.float16, MaskSpec(), None, 32, 64),
             (65, 128, torch.bfloat16, MaskSpec(causal=True), None, 64, 32),
             (129, 64, torch.float16, MaskSpec(window_left=11, window_right=5), None, 32, 64),
             (129, 128, torch.bfloat16, MaskSpec(causal=True, same_document=True), tuple((i // 7) % 3 - 1 for i in range(129)), 64, 64),
             (65, 64, torch.bfloat16, MaskSpec(window_left=0, window_right=0), None, 64, 64)]
    if args.stage == "study":
        cases += [(257, 128, torch.float16, MaskSpec(causal=True, window_left=63, same_document=True), tuple((i // 31) % 4 for i in range(257)), 32, 64),
                  (129, 128, torch.float16, MaskSpec(same_document=True), tuple(i % 5 for i in range(129)), 64, 32),
                  (65, 64, torch.bfloat16, MaskSpec(window_left=1, window_right=3), None, 64, 64)]
    for case in cases:
        n, d, dtype, spec, docs, bq, bk = case
        jobs.append(("correctness", lambda n=n, d=d, dtype=dtype, spec=spec, docs=docs, bq=bq, bk=bk:
                     numerical_case(n, d, dtype, spec, docs, tile_q=bq, tile_k=bk)))
    # Prioritize the whole optimizer step before the longer sweep, ensuring a
    # bounded run obtains an integrated training result even on a cold cache.
    jobs.append(("transformer", lambda: transformer_step_case(512, 64, torch.bfloat16, MaskSpec(causal=True), None)))
    for n, d, spec, docs in [(1024, 64, MaskSpec(causal=True), None),
                             (1024, 64, MaskSpec(causal=True, window_left=127), None)]:
        jobs.append(("performance", lambda n=n, d=d, spec=spec, docs=docs:
                     performance_case(n, d, torch.bfloat16, spec, docs)))
    if args.stage == "study":
        for n, d, spec, docs in [(4096, 64, MaskSpec(causal=True, window_left=255), None),
                                 (2048, 128, MaskSpec(causal=True, same_document=True), tuple((i // 256) % 3 for i in range(2048)))]:
            jobs.append(("performance", lambda n=n, d=d, spec=spec, docs=docs:
                         performance_case(n, d, torch.bfloat16, spec, docs, seed=29)))
        jobs.append(("transformer", lambda: transformer_step_case(1024, 128, torch.bfloat16,
                                                                  MaskSpec(causal=True, window_left=127), None)))
    report["planned_cases"] = len(jobs)
    save()
    for index, (kind, job) in enumerate(jobs):
        if time.perf_counter() - started >= args.time_budget_seconds:
            report["stopped_for_budget"] = True
            break
        try:
            item = job()
        except Exception as exc:
            item = {"kind": kind, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        item["case_index"] = index
        report["cases"].append(item)
        (args.output_dir / f"case_{index:03d}.json").write_text(json.dumps(item, indent=2))
        save()
        print(json.dumps({"case": index, "kind": kind, "status": item["status"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        gc.collect()
    report["completed_cases"] = len(report["cases"])
    report["passed_cases"] = sum(case["status"] == "passed" for case in report["cases"])
    report["status"] = ("failed" if any(case["status"] != "passed" for case in report["cases"])
                        else "partial_budget" if len(report["cases"]) < len(jobs) else "passed")
    save()
    print(json.dumps({"status": report["status"], "output": str(args.output_dir / "summary.json")}), flush=True)
    return 0 if report["status"] == "passed" else 2 if report["status"] == "partial_budget" else 1


if __name__ == "__main__":
    raise SystemExit(main())
