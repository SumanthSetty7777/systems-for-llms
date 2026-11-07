"""CUDA-only correctness and timing harness for the attention compiler prototype.

Run ``python -m attention_lab.benchmark --help`` from the repository root.
No cloud resources are launched. GPU results exist only after an explicit run.
"""

import argparse
from dataclasses import asdict
from functools import lru_cache
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time


def _dense_reference_mask(length, spec, document_ids, device):
    """Independent token-level oracle: deliberately does not read the schedule."""
    import torch

    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    allowed = torch.ones((length, length), dtype=torch.bool, device=device)
    if spec.causal:
        allowed &= key <= query
    if spec.window_left is not None:
        allowed &= key >= query - spec.window_left
    if spec.window_right is not None:
        allowed &= key <= query + spec.window_right
    if spec.same_document:
        if document_ids is None:
            raise ValueError("same_document requires document_ids")
        docs = torch.as_tensor(document_ids, dtype=torch.int64, device=device)
        if docs.ndim != 1 or docs.numel() != length:
            raise ValueError("document_ids must have length N")
        allowed &= docs[:, None] == docs[None, :]
    return allowed


def _time_cuda(function, *, device, warmup, repeats, mode="cuda_graph", inner_repeats=20):
    """Time warmed device execution, or explicitly requested eager dispatch.

    Graph mode captures a batch of calls before timing and divides replay time
    by batch size. It excludes Python validation/launch overhead and therefore
    must not be presented as application end-to-end latency. Eager mode includes
    CPU submission gaps observable on the stream. No silent mode fallback.
    """
    import torch

    if mode not in ("cuda_graph", "eager") or inner_repeats < 1:
        raise ValueError("mode must be cuda_graph/eager and inner_repeats positive")
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    if mode == "cuda_graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner_repeats):
                function()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)
        launch = graph.replay
    else:
        def launch():
            for _ in range(inner_repeats):
                function()
    events = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) / inner_repeats for start, end in events]
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise RuntimeError("CUDA timing returned nonfinite or nonpositive samples")
    return {"median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples,
            "mode": mode, "calls_per_sample": inner_repeats}


def _sdpa_mode(spec):
    if not spec.same_document and spec.window_left is None and spec.window_right is None:
        return "native_causal" if spec.causal else "native_dense"
    return "explicit_boolean_mask"


def _flex_options(spec, contiguous=False):
    """Interval masks may use the hint after validating actual block lists."""
    if type(contiguous) is not bool:
        raise TypeError("flex_contiguous must be bool")
    if contiguous and spec.same_document:
        raise ValueError("flex_contiguous requires an interval mask without document constraints")
    options = {"BACKEND": "TRITON"}
    if contiguous:
        options["BLOCKS_ARE_CONTIGUOUS"] = True
    return options


def _flex_block_contiguity(block_mask):
    """Validate both lists: full-block partitioning can split boundary blocks."""
    import torch
    result = {}
    for prefix, key in (("kv", "partial_blocks_contiguous"), ("full_kv", "full_blocks_contiguous")):
        counts = getattr(block_mask, prefix + "_num_blocks")
        indices = getattr(block_mask, prefix + "_indices")
        if counts is None or indices is None:
            result[key] = True
            continue
        counts, indices = counts.detach().cpu(), indices.detach().cpu()
        transitions = torch.arange(max(indices.shape[-1] - 1, 0)) < (counts[..., None] - 1)
        wrong_step = (indices[..., 1:] - indices[..., :-1]) != 1
        result[key] = not bool((transitions & wrong_step).any().item())
    return result


def _make_flex_mask(length, spec, documents):
    """Token-level FlexAttention semantics, independent of custom CSR lowering."""
    def mask_mod(batch, head, query, key):
        valid = (query >= 0) & (query < length) & (key >= 0) & (key < length)
        if spec.causal:
            valid = valid & (key <= query)
        if spec.window_left is not None:
            valid = valid & (key >= query - spec.window_left)
        if spec.window_right is not None:
            valid = valid & (key <= query + spec.window_right)
        if spec.same_document:
            # Block-mask construction can evaluate padded indices. Bounds still
            # mask them out; safe indexing prevents an out-of-range gather.
            valid = valid & (documents[query.clamp(0, length - 1)]
                             == documents[key.clamp(0, length - 1)])
        return valid
    return mask_mod


def _configure_dynamo_for_study():
    """Allow a bounded sweep of distinct FlexAttention mask specializations.

    This benchmark runs in a dedicated study process. The defaults (often eight
    recompilations per frame) are intended to catch application recompilation
    loops, but a deliberately varied mask sweep legitimately exceeds them.
    Raise, never lower, these process-global limits without clearing compiler
    caches or permitting eager fallback. Each Flex result records the settings.
    """
    import torch

    config = torch._dynamo.config
    config.recompile_limit = max(config.recompile_limit, 128)
    config.accumulated_recompile_limit = max(config.accumulated_recompile_limit, 512)
    return {"recompile_limit": config.recompile_limit,
            "accumulated_recompile_limit": config.accumulated_recompile_limit,
            "scope": "process-global limits raised for the dedicated benchmark sweep; caches preserved; fullgraph remains enabled"}


@lru_cache(maxsize=16)
def _prepare_flex(length, spec, documents, device):
    import torch
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    docs = torch.tensor(documents, dtype=torch.int64, device=device) if spec.same_document else None
    mask_mod = _make_flex_mask(length, spec, docs)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=length,
                                  KV_LEN=length, device=device, BLOCK_SIZE=128,
                                  _compile=True)
    torch.cuda.synchronize(device)
    setup_ms = (time.perf_counter() - start) * 1000
    compiled = torch.compile(flex_attention, fullgraph=True, dynamic=True)
    return compiled, block_mask, setup_ms


def _comparison(output, reference, *, atol, rtol):
    import torch
    finite = bool(torch.isfinite(output).all().item())
    error = (output.float() - reference.float()).abs()
    max_error, mean_error = float(error.max().item()), float(error.mean().item())
    return {"correct": finite and bool(torch.isfinite(reference).all().item())
            and bool(torch.allclose(output, reference, atol=atol, rtol=rtol)),
            "output_finite": finite,
            "max_abs_error": max_error if math.isfinite(max_error) else None,
            "mean_abs_error": mean_error if math.isfinite(mean_error) else None}


def run_case(*, length, head_dim, batch=1, heads=4, spec=None,
             document_ids=None, block_q=64, block_k=64, num_warps=4,
             num_stages=2, device="cuda:0", dtype="float16", seed=0,
             warmup=10, repeats=30, atol=None, rtol=None, implementation="csr",
             use_full_blocks=True, include_flex=False, flex_contiguous=False, timing_mode="cuda_graph",
             inner_repeats=20, reference_chunk_size=512):
    """Evaluate one candidate, returning correctness, first-call and warm timing.

    ``cold_call_ms`` is a compatibility alias for ``first_call_ms``. It includes
    Python validation, uploads, allocation, JIT/cache lookup, launch, and sync;
    compiler caches are not reset and this is not pure compiler time.
    """
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .masks import MaskSpec, compile_mask
    from .triton_attention import attention

    if spec is None:
        spec = MaskSpec(causal=True)
    flex_options = _flex_options(spec, flex_contiguous)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("A CUDA GPU is required; no benchmark or cloud job was run")
    if warmup < 0 or repeats < 1 or inner_repeats < 1 or reference_chunk_size < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    if implementation not in ("csr", "interval") or type(use_full_blocks) is not bool:
        raise ValueError("implementation must be csr/interval and use_full_blocks must be bool")
    if timing_mode not in ("cuda_graph", "eager"):
        raise ValueError("timing_mode must be cuda_graph or eager")
    dtype_aliases = {"float16": torch.float16, "fp16": torch.float16,
                     "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if isinstance(dtype, str):
        if dtype not in dtype_aliases:
            raise ValueError("dtype must be float16/fp16 or bfloat16/bf16")
        dtype = dtype_aliases[dtype]
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Only FP16/BF16 candidates are supported")
    atol = (0.005 if dtype == torch.float16 else 0.02) if atol is None else atol
    rtol = (0.01 if dtype == torch.float16 else 0.02) if rtol is None else rtol
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("numerical tolerances must be nonnegative")
    if isinstance(document_ids, torch.Tensor):
        document_ids = document_ids.detach().cpu().tolist()
    if spec.same_document and document_ids is None:
        raise ValueError("Supply document_ids for a same-document benchmark")

    device = torch.device(device)
    with torch.cuda.device(device), torch.inference_mode():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (batch, heads, length, head_dim)
        q, k, v = [torch.randn(shape, generator=generator, dtype=torch.float32)
                   .to(device=device, dtype=dtype).contiguous() for _ in range(3)]
        originals = [tensor.clone() for tensor in (q, k, v)]
        dense_mask = _dense_reference_mask(length, spec, document_ids, device)
        schedule_start = time.perf_counter()
        schedule = compile_mask(length, spec, block_q=block_q, block_k=block_k,
                                document_ids=document_ids)
        schedule_compile_ms = (time.perf_counter() - schedule_start) * 1000.0
        props = torch.cuda.get_device_properties(device)
        result = {
            "status": "running", "correct": False, "latency_ms": None,
            "cold_call_ms": None, "first_call_ms": None, "compile_time_ms": None,
            "first_call_scope": "validation + metadata upload + allocation + JIT/cache lookup + launch + synchronize; caches not reset",
            "timing_scope": ("CUDA event timing of batched graph replay, per forward call; excludes Python dispatch, mask construction and compile/setup"
                             if timing_mode == "cuda_graph" else
                             "CUDA events around batched Python wrapper calls; includes stream submission gaps; excludes first-call setup"),
            "shape": list(shape), "dtype": str(dtype), "seed": seed,
            "mask": asdict(spec), "document_ids": document_ids if spec.same_document else None,
            "schedule_compile_ms": schedule_compile_ms,
            "candidate": {"block_q": block_q, "block_k": block_k,
                          "num_warps": num_warps, "num_stages": num_stages,
                          "implementation": implementation, "use_full_blocks": use_full_blocks},
            "tolerances": {"atol": atol, "rtol": rtol},
            "schedule": {"query_blocks": len(schedule.row_ptr) - 1,
                         "retained_blocks": len(schedule.col_idx),
                         "full_blocks": sum(schedule.full_blocks)},
            "hardware": {"device": str(device), "name": props.name,
                         "compute_capability": [props.major, props.minor],
                         "total_memory_bytes": props.total_memory,
                         "multiprocessor_count": props.multi_processor_count},
            "software": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda},
            "warmup": warmup, "repeats": repeats, "inner_repeats": inner_repeats,
            "reference": "PyTorch SDPA math backend, same input dtype, explicit boolean token mask; query chunks limit temporary memory",
            "reference_chunk_size": reference_chunk_size,
            "baseline": "PyTorch SDPA automatic backend selection; selected backend not inspected",
            "baseline_mask_mode": _sdpa_mode(spec),
            "flex": {"status": "not_requested", "backend_requested": "TRITON",
                     "kernel_options": flex_options},
            "forward_only": True,
        }

        def baseline():
            mode = _sdpa_mode(spec)
            return functional.scaled_dot_product_attention(
                q, k, v, attn_mask=dense_mask if mode == "explicit_boolean_mask" else None,
                dropout_p=0.0, is_causal=mode == "native_causal")

        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.cat([
                functional.scaled_dot_product_attention(
                    q[:, :, start:start + reference_chunk_size], k, v,
                    attn_mask=dense_mask[start:start + reference_chunk_size],
                    dropout_p=0.0, is_causal=False)
                for start in range(0, length, reference_chunk_size)
            ], dim=2)
        baseline_output = baseline()
        baseline_correct = bool(torch.isfinite(baseline_output).all().item()
                                and torch.allclose(baseline_output, reference, atol=atol, rtol=rtol))
        result["baseline_correct"] = baseline_correct

        def candidate(launch_metadata=None):
            return attention(q, k, v, schedule, spec, document_ids=document_ids,
                             num_warps=num_warps, num_stages=num_stages,
                             implementation=implementation, use_full_blocks=use_full_blocks,
                             launch_metadata=launch_metadata)

        try:
            torch.cuda.synchronize(device)
            first_start = time.perf_counter()
            launch_metadata = {}
            output = candidate(launch_metadata)
            torch.cuda.synchronize(device)
            result["first_call_ms"] = (time.perf_counter() - first_start) * 1000.0
            result["cold_call_ms"] = result["first_call_ms"]
            result["kernel_metadata"] = launch_metadata
            try:
                result["software"]["triton"] = importlib.metadata.version("triton")
            except importlib.metadata.PackageNotFoundError:
                result["software"]["triton"] = "unknown (module import succeeded)"
            finite = bool(torch.isfinite(output).all().item())
            unchanged = all(torch.equal(t, original) for t, original in zip((q, k, v), originals))
            close = bool(torch.allclose(output, reference, atol=atol, rtol=rtol))
            error = (output.float() - reference.float()).abs()
            max_error, mean_error = float(error.max().item()), float(error.mean().item())
            result.update({"output_finite": finite, "reference_finite": bool(torch.isfinite(reference).all().item()),
                           "inputs_unchanged": unchanged, "close_to_reference": close,
                           "max_abs_error": max_error if math.isfinite(max_error) else None,
                           "mean_abs_error": mean_error if math.isfinite(mean_error) else None})
            result["correct"] = finite and unchanged and close and result["reference_finite"]
            if result["correct"] and baseline_correct:
                result["candidate_timing"] = _time_cuda(candidate, device=device, warmup=warmup,
                    repeats=repeats, mode=timing_mode, inner_repeats=inner_repeats)
                result["baseline_timing"] = _time_cuda(baseline, device=device, warmup=warmup,
                    repeats=repeats, mode=timing_mode, inner_repeats=inner_repeats)
                result["inputs_unchanged"] = all(torch.equal(t, original) for t, original in zip((q, k, v), originals))
                result["correct"] &= result["inputs_unchanged"]
                if result["correct"]:
                    result["latency_ms"] = result["candidate_timing"]["median_ms"]
                    result["baseline_latency_ms"] = result["baseline_timing"]["median_ms"]
                    result["speedup"] = result["baseline_latency_ms"] / result["latency_ms"]
            result["status"] = "passed" if result["correct"] and baseline_correct else "failed_correctness"
        except Exception as exc:
            result.update({"status": "failed", "correct": False,
                           "error_type": type(exc).__name__, "error": str(exc)})
        if include_flex:
            flex_result = {"status": "running", "backend_requested": "TRITON",
                           "block_size": 128, "compile_dynamic": True,
                           "kernel_options": flex_options,
                           "baseline_variant": "triton_contiguous_blocks" if flex_contiguous else "triton_default",
                           "latency_ms": None, "correct": False}
            result["flex"] = flex_result
            try:
                flex_result["dynamo_settings"] = _configure_dynamo_for_study()
                cache_hits = _prepare_flex.cache_info().hits
                flex_function, flex_mask, setup_ms = _prepare_flex(
                    length, spec, tuple(document_ids) if spec.same_document else (), str(device))
                reused = _prepare_flex.cache_info().hits > cache_hits
                flex_result.update({"mask_setup_reused": reused,
                                    "initial_mask_setup_ms": setup_ms,
                                    "mask_setup_ms_this_call": 0.0 if reused else setup_ms})
                if flex_contiguous:
                    contiguity = _flex_block_contiguity(flex_mask)
                    flex_result["block_contiguity"] = contiguity
                    if not all(contiguity.values()):
                        raise ValueError("BLOCKS_ARE_CONTIGUOUS is invalid for this emitted BlockMask: partial and full block lists must each be contiguous")
                def flex_call():
                    return flex_function(q, k, v, block_mask=flex_mask,
                                         kernel_options=flex_options)
                torch.cuda.synchronize(device)
                first_start = time.perf_counter()
                flex_output = flex_call()
                torch.cuda.synchronize(device)
                flex_result["first_call_ms"] = (time.perf_counter() - first_start) * 1000
                flex_result["first_call_scope"] = "compile/cache lookup + execution + synchronization; excludes mask setup; compiler caches not reset"
                flex_result.update(_comparison(flex_output, reference, atol=atol, rtol=rtol))
                flex_result["inputs_unchanged"] = all(torch.equal(t, original) for t, original in zip((q, k, v), originals))
                flex_result["correct"] &= flex_result["inputs_unchanged"]
                if flex_result["correct"]:
                    flex_result["timing"] = _time_cuda(flex_call, device=device, warmup=warmup,
                        repeats=repeats, mode=timing_mode, inner_repeats=inner_repeats)
                    flex_result["inputs_unchanged"] = all(torch.equal(t, original) for t, original in zip((q, k, v), originals))
                    flex_result["correct"] &= flex_result["inputs_unchanged"]
                    if flex_result["correct"]:
                        flex_result["latency_ms"] = flex_result["timing"]["median_ms"]
                        if result["status"] == "passed" and result["latency_ms"]:
                            result["speedup_vs_flex"] = flex_result["latency_ms"] / result["latency_ms"]
                flex_result["status"] = "passed" if flex_result["correct"] else "failed_correctness"
            except (ImportError, ModuleNotFoundError) as exc:
                flex_result.update({"status": "unavailable", "error": str(exc), "error_type": type(exc).__name__})
            except Exception as exc:
                flex_result.update({"status": "failed", "correct": False,
                                    "error": str(exc), "error_type": type(exc).__name__})
        return result


def _integer_list(value):
    values = [int(piece) for piece in value.split(",")]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=_integer_list, default=[128, 257])
    parser.add_argument("--head-dims", type=_integer_list, default=[64])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block-q", type=_integer_list, default=[64])
    parser.add_argument("--block-k", type=_integer_list, default=[64])
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mask", choices=["dense", "causal", "sliding", "document", "combined"], default="causal")
    parser.add_argument("--window-left", type=int, default=128)
    parser.add_argument("--window-right", type=int, default=0)
    parser.add_argument("--document-length", type=int, default=97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--implementation", choices=["csr", "interval"], default="csr")
    parser.add_argument("--disable-full-blocks", action="store_true")
    parser.add_argument("--include-flex", action="store_true")
    parser.add_argument("--flex-contiguous", action="store_true",
                        help="Give FlexAttention the contiguous-block guarantee (no document masks; requires --include-flex)")
    parser.add_argument("--timing-mode", choices=["cuda_graph", "eager"], default="cuda_graph")
    parser.add_argument("--inner-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .masks import MaskSpec

    if args.document_length < 1:
        parser.error("--document-length must be positive")
    if args.flex_contiguous and (not args.include_flex or args.mask in ("document", "combined")):
        parser.error("--flex-contiguous requires --include-flex and a mask without document constraints")
    spec = MaskSpec(causal=args.mask in ("causal", "combined"),
                    window_left=args.window_left if args.mask in ("sliding", "combined") else None,
                    window_right=args.window_right if args.mask in ("sliding", "combined") else None,
                    same_document=args.mask in ("document", "combined"))
    report = {"schema_version": 1, "status": "not_run", "cases": [],
              "note": "Forward-only prototype; results apply only to the recorded device, inputs and timing protocol."}
    try:
        for length in args.lengths:
            documents = [i // args.document_length for i in range(length)] if spec.same_document else None
            for dimension in args.head_dims:
                for block_q in args.block_q:
                    for block_k in args.block_k:
                        report["cases"].append(run_case(
                            length=length, head_dim=dimension, batch=args.batch, heads=args.heads,
                            spec=spec, document_ids=documents, block_q=block_q, block_k=block_k,
                            num_warps=args.num_warps, num_stages=args.num_stages,
                            device=args.device, dtype=args.dtype, seed=args.seed,
                            warmup=args.warmup, repeats=args.repeats,
                            implementation=args.implementation, use_full_blocks=not args.disable_full_blocks,
                            include_flex=args.include_flex, flex_contiguous=args.flex_contiguous,
                            timing_mode=args.timing_mode,
                            inner_repeats=args.inner_repeats))
        report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
    except Exception as exc:
        report.update({"error_type": type(exc).__name__, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
