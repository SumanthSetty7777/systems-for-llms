"""Measure attention substitutions inside one inference Transformer block.

All variants execute identical surrounding operators and weights. This is an
inference block study, not a whole-model, training, or application benchmark.
CUDA graph replay excludes Python launch overhead and compile/mask setup.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

from .masks import MaskSpec, compile_mask
from .search import KernelConfig, default_configs
from .study import DEFAULT, _write_json, load_winners


def build_block_plan(winners=None, *, seed=211):
    """Freeze all configurations before block evaluation; never select on it."""
    winners = winners or {}
    cases = []
    for length in (1024, 4096):
        for dimension in (64, 128):
            for pattern in ("causal", "window"):
                family = "tune-causal" if pattern == "causal" else "tune-window128"
                winner = winners.get(family)
                config = KernelConfig(**winner["config"]) if winner else DEFAULT
                if config not in default_configs():
                    raise ValueError("selected config must belong to the declared tuning space")
                cases.append({
                    "case_id": f"block-n{length}-d{dimension}-{pattern}-s{seed}",
                    "length": length, "head_dim": dimension, "heads": 8, "batch": 1,
                    "dtype": "bfloat16", "seed": seed, "pattern": pattern,
                    "mask": asdict(MaskSpec(causal=True, window_left=127 if pattern == "window" else None)),
                    "config": asdict(config),
                    "selection": {"policy": "training_family_transfer" if winner else "fixed_default",
                                  "source_workload": family if winner else None,
                                  "source_case_id": winner.get("case_id") if winner else None,
                                  "block_results_used_for_selection": False},
                })
    return {"stage": "block", "cases": cases,
            "scope": "One BF16 inference Transformer block; same weights and operators for each attention variant.",
            "selection_policy": "Configurations frozen from training-family kernel winners or fixed defaults; block measurements never select configurations."}


def _training_winners(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("stage") != "tuning":
        raise ValueError("--tuning-dir must contain a tuning-stage plan, not held-out measurements")
    return load_winners(directory)


def _passed_measurement(measurement):
    if (measurement.get("status") != "passed" or measurement.get("correct") is not True
            or measurement.get("reference_finite") is not True
            or measurement.get("inputs_weights_unchanged") is not True):
        return False
    for name in ("custom_interval_split", "sdpa", "flex_triton"):
        variant = measurement.get("variants", {}).get(name, {})
        latency = variant.get("block_latency_ms")
        if (variant.get("status") != "passed" or variant.get("correct") is not True
                or isinstance(latency, bool) or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0):
            return False
    return True


def _custom_attention(q, k, v, schedule, spec, config):
    """Tiles belong to the compiled schedule; only launch controls go to attention."""
    from .triton_attention import attention

    return attention(q, k, v, schedule, spec,
                     num_warps=config["num_warps"], num_stages=config["num_stages"],
                     implementation="interval", use_full_blocks=True)


def run_block_case(case, *, device="cuda:0", warmup=5, repeats=20, inner_repeats=5):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .benchmark import (_comparison, _configure_dynamo_for_study, _dense_reference_mask,
                            _prepare_flex, _sdpa_mode, _time_cuda)

    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("A CUDA GPU is required; no Transformer block was run")
    device = torch.device(device)
    spec = MaskSpec(**case["mask"])
    length, dimension, heads, batch = (case[k] for k in ("length", "head_dim", "heads", "batch"))
    width = heads * dimension
    props = torch.cuda.get_device_properties(device)
    result = {
        "status": "running", "correct": False,
        "hardware": {"device": str(device), "name": props.name,
                     "compute_capability": [props.major, props.minor]},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda,
                     "triton": importlib.metadata.version("triton")},
        "architecture": {"width": width, "heads": heads, "head_dim": dimension,
                         "mlp_width": 4 * width, "layer_norm_epsilon": 1e-5,
                         "activation": "GELU exact", "bias": False, "dropout": 0.0,
                         "order": "LN -> QKV -> attention -> output projection + residual -> LN -> MLP + residual"},
        "timing_scope": "Batched CUDA graph replay of one full inference block, per block call; excludes Python dispatch, compilation, masks and graph capture. Not whole-model or application latency.",
        "reference": "Identical BF16 block with explicit-mask PyTorch math SDPA attention in query chunks of 256",
        "baseline_mask_mode": _sdpa_mode(spec),
        "tolerances": {"atol": 0.03, "rtol": 0.03},
        "variants": {},
    }
    with torch.cuda.device(device), torch.inference_mode():
        generator = torch.Generator(device="cpu").manual_seed(case["seed"])

        def normal(shape, scale=1.0):
            return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(device=device, dtype=torch.bfloat16)

        x = normal((batch, length, width))
        weights = {
            "qkv": normal((3 * width, width), width ** -0.5),
            "output": normal((width, width), width ** -0.5),
            "up": normal((4 * width, width), width ** -0.5),
            "down": normal((width, 4 * width), (4 * width) ** -0.5),
            "ln1": torch.ones(width, dtype=torch.bfloat16, device=device),
            "ln2": torch.ones(width, dtype=torch.bfloat16, device=device),
        }
        originals = [tensor.clone() for tensor in [x, *weights.values()]]
        dense_mask = _dense_reference_mask(length, spec, None, device)
        config = case["config"]
        schedule = compile_mask(length, spec, block_q=config["block_q"], block_k=config["block_k"])

        def project(value):
            normalized = F.layer_norm(value, (width,), weight=weights["ln1"], eps=1e-5)
            qkv = F.linear(normalized, weights["qkv"]).reshape(batch, length, 3, heads, dimension)
            return tuple(t.contiguous() for t in qkv.permute(2, 0, 3, 1, 4).unbind(0))

        def block(attend):
            q, k, v = project(x)
            y = attend(q, k, v).transpose(1, 2).contiguous().reshape(batch, length, width)
            residual = x + F.linear(y, weights["output"])
            normalized = F.layer_norm(residual, (width,), weight=weights["ln2"], eps=1e-5)
            return residual + F.linear(F.gelu(F.linear(normalized, weights["up"])), weights["down"])

        def reference_attention(q, k, v):
            with sdpa_kernel(SDPBackend.MATH):
                return torch.cat([
                    F.scaled_dot_product_attention(q[:, :, start:start + 256], k, v,
                        attn_mask=dense_mask[start:start + 256], dropout_p=0.0, is_causal=False)
                    for start in range(0, length, 256)
                ], dim=2)

        reference = block(reference_attention)
        reference_finite = bool(torch.isfinite(reference).all().item())
        result["reference_finite"] = reference_finite
        if not reference_finite:
            raise RuntimeError("Independent reference block produced nonfinite values")
        q_static, k_static, v_static = project(x)

        def custom(q, k, v):
            return _custom_attention(q, k, v, schedule, spec, config)

        def sdpa(q, k, v):
            native_causal = _sdpa_mode(spec) == "native_causal"
            return F.scaled_dot_product_attention(q, k, v,
                attn_mask=None if native_causal else dense_mask,
                dropout_p=0.0, is_causal=native_causal)

        variants = {"custom_interval_split": custom, "sdpa": sdpa}
        try:
            result["flex_dynamo_settings"] = _configure_dynamo_for_study()
            flex_function, flex_mask, setup_ms = _prepare_flex(length, spec, (), str(device))
            def flex(q, k, v):
                return flex_function(q, k, v, block_mask=flex_mask, kernel_options={"BACKEND": "TRITON"})
            variants["flex_triton"] = flex
            result["flex_mask_initial_setup_ms"] = setup_ms
        except Exception as exc:
            result["variants"]["flex_triton"] = {"status": "failed", "correct": False,
                                                     "error": str(exc), "error_type": type(exc).__name__}

        # Rotate order deterministically; no implementation always runs first.
        order = list(variants)
        random.Random(case["seed"] + length + dimension).shuffle(order)
        result["measurement_order"] = order
        for name in order:
            attend = variants[name]
            measured = {"status": "running", "correct": False}
            result["variants"][name] = measured
            try:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = block(attend)
                torch.cuda.synchronize(device)
                measured["first_block_call_ms"] = (time.perf_counter() - started) * 1000
                measured.update(_comparison(output, reference, atol=0.03, rtol=0.03))
                if measured["correct"]:
                    measured["block_timing"] = _time_cuda(lambda: block(attend), device=device,
                        warmup=warmup, repeats=repeats, inner_repeats=inner_repeats, mode="cuda_graph")
                    measured["block_latency_ms"] = measured["block_timing"]["median_ms"]
                    measured["attention_timing"] = _time_cuda(lambda: attend(q_static, k_static, v_static),
                        device=device, warmup=warmup, repeats=repeats,
                        inner_repeats=inner_repeats, mode="cuda_graph")
                    measured["attention_latency_ms"] = measured["attention_timing"]["median_ms"]
                measured["status"] = "passed" if measured["correct"] else "failed_correctness"
            except Exception as exc:
                measured.update({"status": "failed", "correct": False,
                                 "error": str(exc), "error_type": type(exc).__name__})

        result["inputs_weights_unchanged"] = all(torch.equal(tensor, original)
            for tensor, original in zip([x, *weights.values()], originals))
        all_passed = (result["inputs_weights_unchanged"] and
                      all(result["variants"].get(name, {}).get("status") == "passed"
                          for name in ("custom_interval_split", "sdpa", "flex_triton")))
        result["correct"] = all_passed
        result["status"] = "passed" if all_passed else "failed"
        candidate = result["variants"].get("custom_interval_split", {})
        if candidate.get("status") == "passed" and result["inputs_weights_unchanged"]:
            for name in ("sdpa", "flex_triton"):
                baseline = result["variants"].get(name, {})
                if baseline.get("status") != "passed":
                    continue
                speedup = baseline["block_latency_ms"] / candidate["block_latency_ms"]
                attention_speedup = baseline["attention_latency_ms"] / candidate["attention_latency_ms"]
                fraction = baseline["attention_latency_ms"] / baseline["block_latency_ms"]
                result[f"speedup_vs_{name}"] = speedup
                result[f"attention_speedup_vs_{name}"] = attention_speedup
                result[f"isolated_attention_fraction_{name}"] = fraction
                if 0 <= fraction <= 1:
                    result[f"amdahl_estimate_vs_{name}"] = 1 / (1 - fraction + fraction / attention_speedup)
        result["amdahl_scope"] = "Illustrative estimate from separately timed attention and block. It does not assume identical cache behavior or predict whole-model speedup."
    return result


def run_block_study(output_dir, *, tuning_dir=None, winners=None, device="cuda:0",
                    seed=211, warmup=5, repeats=20, inner_repeats=5,
                    time_budget_seconds=360, evaluate=None):
    if tuning_dir is not None and winners is not None:
        raise ValueError("Supply either tuning_dir or winners, not both")
    if not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
        raise ValueError("time budget must be finite and positive")
    if warmup < 0 or repeats < 1 or inner_repeats < 1:
        raise ValueError("warmup must be nonnegative and timing repetitions positive")
    winners = _training_winners(tuning_dir) if tuning_dir is not None else winners
    plan = build_block_plan(winners, seed=seed)
    plan.update({"warmup": warmup, "repeats": repeats, "inner_repeats": inner_repeats})
    output_dir = Path(output_dir)
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    if any(case_dir.iterdir()):
        raise FileExistsError("Block cases directory must be empty; use a new run directory")
    _write_json(output_dir / "plan.json", plan)
    summary = {"schema_version": 1, "stage": "block", "status": "running",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "planned_cases": len(plan["cases"]), "completed_cases": 0,
               "passed_cases": 0, "failed_cases": 0, "not_run_cases": len(plan["cases"]),
               "flex_failed_cases": 0, "flex_requested_cases": len(plan["cases"]),
               "flex_passed_cases": 0, "source_sha256": {
                   p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob("*.py"))},
               "scope": plan["scope"], "time_budget_seconds": time_budget_seconds}
    if evaluate is None:
        import torch
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            summary.update({"status": "not_run", "error": "A CUDA GPU is required; no block experiment was run"})
            _write_json(output_dir / "summary.json", summary)
            return summary
        evaluate = run_block_case
    started = time.monotonic()
    for case in plan["cases"]:
        if time.monotonic() - started >= time_budget_seconds:
            summary["status"] = "time_budget_exhausted"
            break
        case_start = time.monotonic()
        try:
            measurement = evaluate(case, device=device, warmup=warmup,
                                   repeats=repeats, inner_repeats=inner_repeats)
        except Exception as exc:
            measurement = {"status": "failed", "correct": False,
                           "error_type": type(exc).__name__, "error": str(exc)}
        record = {"case": case, "measurement": measurement,
                  "wall_seconds": time.monotonic() - case_start}
        _write_json(case_dir / (case["case_id"] + ".json"), record)
        summary["completed_cases"] += 1
        passed = _passed_measurement(measurement)
        summary["passed_cases" if passed else "failed_cases"] += 1
        flex_passed = measurement.get("variants", {}).get("flex_triton", {}).get("status") == "passed"
        summary["flex_passed_cases" if flex_passed else "flex_failed_cases"] += 1
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["not_run_cases"] = summary["planned_cases"] - summary["completed_cases"]
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps({"case": case["case_id"], "status": measurement.get("status"),
                          "progress": f"{summary['completed_cases']}/{summary['planned_cases']}"}), flush=True)
        errors = json.dumps(measurement).lower()
        if "illegal memory access" in errors or "device-side assert" in errors:
            summary["status"] = "stopped_after_failure"
            break
    else:
        summary["status"] = "passed" if not summary["failed_cases"] and not summary["flex_failed_cases"] else "completed_with_failures"
    summary["elapsed_seconds"] = time.monotonic() - started
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-dir", type=Path)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inner-repeats", type=int, default=5)
    parser.add_argument("--time-budget-seconds", type=float, default=360)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        winners = _training_winners(args.tuning_dir) if args.tuning_dir else None
        plan = build_block_plan(winners, seed=args.seed)
        _write_json(args.output_dir / "plan.json", plan)
        print(json.dumps({"status": "dry_run", "stage": "block", "cases": len(plan["cases"])}))
        return 0
    result = run_block_study(args.output_dir, tuning_dir=args.tuning_dir, seed=args.seed,
        warmup=args.warmup, repeats=args.repeats, inner_repeats=args.inner_repeats,
        time_budget_seconds=args.time_budget_seconds)
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
