"""Matched-global-batch training under torchrun; isolated from the single-GPU study."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from llmlab.config import TrainConfig, load_config
from llmlab.data import batch_at, load_data
from llmlab.model.transformer import TransformerLM
from llmlab.train import autocast, check_precision, learning_rate, synchronize
from llmlab.training.loss import cross_entropy


MODES = ("single", "naive", "ddp", "zero1", "fsdp", "tp")


def validate_layout(mode, world_size, global_batch_size, microbatch_size, sync_policy):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if sync_policy not in {"last", "every"}:
        raise ValueError("sync_policy must be last or every")
    if mode == "single" and world_size != 1:
        raise ValueError("single mode requires one rank")
    if mode != "single" and world_size < 2:
        raise ValueError("Distributed modes require at least two ranks")
    if any(type(n) is not int or n <= 0 for n in (world_size, global_batch_size, microbatch_size)):
        raise ValueError("World, global batch and microbatch sizes must be positive integers")
    data_world = 1 if mode == "tp" else world_size
    if global_batch_size % (data_world * microbatch_size):
        raise ValueError("Global batch must be divisible by data-parallel ranks times microbatch")
    if sync_policy == "every" and mode not in {"ddp", "zero1"}:
        raise ValueError("sync-policy every applies to DDP and zero1 only")
    return global_batch_size // data_world // microbatch_size


def source_identity():
    root = Path(__file__).resolve().parents[2]
    files = {}
    for folder in (root / "src/llmlab", root / "distributed/llmdist"):
        for path in sorted(folder.rglob("*.py")):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def state_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = _local(parameter.detach()).to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        local = _local(value)
        return local.numel() * local.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def _cpu_tree(value):
    """Export ordinary CPU tensors, including local DTensor optimizer shards."""
    if isinstance(value, torch.Tensor):
        return _local(value.detach()).cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _all_reduce(value, op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(value, op=op)
    return value


def fsdp_clip_grad_norm(model, max_norm):
    """Count each global gradient element once on a one-dimensional FSDP mesh."""
    from torch.distributed.tensor import DTensor, Replicate, Shard
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError("FSDP model has no gradients to clip")
    squared = torch.zeros((), device=_local(parameters[0]).device, dtype=torch.float32)
    world = dist.get_world_size()
    for parameter in parameters:
        gradient = parameter.grad
        if isinstance(gradient, DTensor):
            if len(gradient.placements) != 1 or not isinstance(gradient.placements[0], (Shard, Replicate)):
                raise ValueError("Only a one-dimensional fully sharded mesh is supported")
            divisor = world if isinstance(gradient.placements[0], Replicate) else 1
        else:
            divisor = world
        # A vector-norm reduction avoids allocating a full-sized squared copy.
        squared += torch.linalg.vector_norm(_local(gradient), 2, dtype=torch.float32).square() / divisor
    norm = _all_reduce(squared).sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite global gradient norm")
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        _local(parameter.grad).mul_(coefficient.to(dtype=parameter.grad.dtype))
    return norm


def train_step(model, base_model, optimizer, data, config, sampler, device, *, mode,
               rank, world_size, global_batch_size, microbatch_size, sync_policy):
    accumulation = validate_layout(mode, world_size, global_batch_size, microbatch_size, sync_policy)
    starts = torch.randint(len(data) - config.model.context_length,
                           (global_batch_size,), generator=sampler)
    batch_hash = hashlib.sha256(starts.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    local_batch = global_batch_size if mode == "tp" else global_batch_size // world_size
    local_starts = starts if mode == "tp" else starts[rank * local_batch:(rank + 1) * local_batch]
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for index, offset in enumerate(range(0, local_batch, microbatch_size)):
        last = index == accumulation - 1
        context = (model.no_sync() if mode in {"ddp", "zero1"} and sync_policy == "last" and not last
                   else nullcontext())
        if mode == "fsdp":
            model.set_requires_gradient_sync(last)
        with context:
            x, y = batch_at(data, local_starts[offset:offset + microbatch_size].tolist(),
                            config.model.context_length, device)
            with autocast(config, device):
                loss = cross_entropy(model(x).float(), y) / accumulation
            loss_sum += loss.detach()
            loss.backward()
            del x, y, loss
    if mode == "naive":
        for parameter in model.parameters():
            if parameter.grad is None:
                raise RuntimeError("The naive reducer requires every parameter to have a gradient")
            _all_reduce(parameter.grad).div_(world_size)
    finite = torch.isfinite(loss_sum).to(dtype=torch.int32)
    _all_reduce(finite, dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("Non-finite training loss on a rank")
    if mode == "tp":
        from .tensor_parallel import clip_grad_norm_tp
        norm = clip_grad_norm_tp(base_model, config.grad_clip, error_if_nonfinite=True)
    elif mode == "fsdp":
        norm = fsdp_clip_grad_norm(base_model, config.grad_clip)
    else:
        norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    mean_loss = _all_reduce(loss_sum).div_(world_size)
    return {"loss": mean_loss.item(), "gradient_norm": norm.item(), "batch_start_sha256": batch_hash}


@torch.no_grad()
def evaluate(model, data, config, device, *, rank, world_size, mode):
    windows = config.eval_batches * config.eval_batch_size
    # Equal forward counts avoid FSDP collective mismatches; the same windows as llmlab.
    if mode != "tp" and config.eval_batches % world_size:
        raise ValueError("eval_batches must be divisible by the number of data-parallel ranks")
    starts = np.linspace(0, len(data) - config.model.context_length - 1, windows, dtype=np.int64)
    if mode != "tp":
        starts = starts[rank * windows // world_size:(rank + 1) * windows // world_size]
    was_training = model.training
    model.eval()
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    try:
        for offset in range(0, len(starts), config.eval_batch_size):
            x, y = batch_at(data, starts[offset:offset + config.eval_batch_size].tolist(),
                            config.model.context_length, device)
            with autocast(config, device):
                loss = cross_entropy(model(x).float(), y)
            totals[0] += loss.double() * y.numel()
            totals[1] += y.numel()
    finally:
        model.train(was_training)
    _all_reduce(totals)
    return (totals[0] / totals[1]).item()


def run(config: TrainConfig, output_dir, *, mode="single", global_batch_size=8,
        microbatch_size=1, warmup_steps=5, sync_policy="last", bucket_cap_mb=25.,
        save_final_state=False, profile_steps=0):
    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in
                               (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    accumulation = validate_layout(mode, world, global_batch_size, microbatch_size, sync_policy)
    config.validate()
    if not 0 <= warmup_steps < config.max_steps:
        raise ValueError("Timing warmup must be smaller than the update count")
    if bucket_cap_mb <= 0 or not np.isfinite(bucket_cap_mb) or profile_steps < 0:
        raise ValueError("Bucket capacity must be finite and positive; profile steps nonnegative")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("Distributed runner supports explicit cpu or cuda devices")
    if mode == "fsdp" and config.device != "cuda":
        raise ValueError("FSDP2 validation requires CUDA; CPU fallback is not supported")
    if mode != "tp" and config.eval_batches % world:
        raise ValueError("eval_batches must be divisible by data-parallel ranks")
    torch.set_num_threads(config.num_threads)
    if config.device == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    check_precision(config, device)
    torch.set_float32_matmul_precision("highest")
    output = Path(output_dir)
    initialized_here = False
    output_ready = False
    started = time.perf_counter()
    try:
        if world > 1:
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                    init_method="env://", timeout=timedelta(seconds=120))
            initialized_here = True
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            output_ready = True
        if world > 1:
            dist.barrier()
        output_ready = True
        torch.manual_seed(config.seed)
        train_data, valid_data, data_identity = load_data(config)
        base_model = TransformerLM(**asdict(config.model), device=device, dtype=torch.float32)
        parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
        initial_hash = state_hash(base_model)
        if world > 1:
            initial_hashes = [None for _ in range(world)]
            dist.all_gather_object(initial_hashes, initial_hash)
            if len(set(initial_hashes)) != 1:
                raise RuntimeError("Initial full-model parameters differ across ranks")
        sharding = None
        if mode == "tp":
            from .tensor_parallel import parallelize_model, describe_sharding
            parallelize_model(base_model)
            sharding = describe_sharding(base_model)
        elif mode == "fsdp":
            from torch.distributed.fsdp import fully_shard
            for block in base_model.layers:
                fully_shard(block)
            fully_shard(base_model)
        model = (DistributedDataParallel(base_model, device_ids=[local_rank] if device.type == "cuda" else None,
                                         broadcast_buffers=False, bucket_cap_mb=bucket_cap_mb)
                 if mode in {"ddp", "zero1"} else base_model)
        optimizer_kwargs = dict(lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
        if mode == "zero1":
            from .optim_shard import ShardedAdamW
            optimizer = ShardedAdamW(base_model.parameters(), **optimizer_kwargs)
        else:
            optimizer = torch.optim.AdamW(base_model.parameters(), **optimizer_kwargs)
        sampler = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        metadata = {"format_version": 1, "config": config.to_dict(), "mode": mode,
                    "rank": rank, "local_rank": local_rank, "world_size": world,
                    "data_parallel_size": 1 if mode == "tp" else world,
                    "global_batch_size": global_batch_size, "microbatch_size": microbatch_size,
                    "accumulation_steps": accumulation, "sync_policy": sync_policy,
                    "bucket_cap_mb": bucket_cap_mb, "timing_warmup_steps": warmup_steps,
                    "profile_steps": profile_steps, "parameter_count": parameter_count,
                    "local_parameter_bytes": sum(_local(p).numel() * p.element_size() for p in base_model.parameters()),
                    "local_gradient_bytes_before_training": sum(tensor_bytes(p.grad) for p in base_model.parameters()),
                    "optimizer_state_bytes_before_training": optimizer.local_state_bytes() if mode == "zero1" else tensor_bytes(optimizer.state),
                    "cuda_allocated_bytes_before_training": torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
                    "cuda_reserved_bytes_before_training": torch.cuda.memory_reserved(device) if device.type == "cuda" else None,
                    "initial_parameter_sha256": initial_hash, "data_identity": data_identity,
                    "source": source_identity(), "sharding": sharding,
                    "optimizer_ownership": optimizer.ownership_metadata() if mode == "zero1" else None,
                    "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "einops")},
                    "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "cuda_version": torch.version.cuda,
                    "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                    "timing_scope": "max rank synchronized optimizer-step time after an inter-rank barrier; includes data, backward synchronization, clipping, update and loss reduction; excludes timing-max reduction, evaluation, setup and checkpoint writes",
                    "peak_memory_scope": "training and evaluation after model/wrapper setup; final state export excluded",
                    "attention_dispatch": "automatic PyTorch SDPA; kernel not identified" if config.model.attention_backend == "sdpa" else "explicit eager",
                    "checkpoint_scope": "optional final model/optimizer state for correctness checks; resume unsupported"}
        (output / f"rank-{rank}-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        synchronize(device)
        if world > 1:
            dist.barrier()
        setup_seconds = time.perf_counter() - started
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = []
        measured = []
        first_update_memory = None
        last_valid_loss = None
        tokens_per_step = global_batch_size * config.model.context_length
        model.train()
        profile = nullcontext()
        if profile_steps:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile = torch.profiler.profile(activities=activities,
                schedule=torch.profiler.schedule(wait=warmup_steps, warmup=1, active=profile_steps, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(output / f"rank-{rank}-trace.json")),
                record_shapes=True, profile_memory=True)
        with (output / f"rank-{rank}-metrics.jsonl").open("w") as stream, profile as profiler:
            def record(event, step, **values):
                row = {"event": event, "step": step, "wall_seconds": time.perf_counter() - started, **values}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                metrics.append(row)

            for step in range(1, config.max_steps + 1):
                if world > 1:
                    dist.barrier()
                synchronize(device)
                tick = time.perf_counter()
                lr = learning_rate(step - 1, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                values = train_step(model, base_model, optimizer, train_data, config, sampler, device,
                                    mode=mode, rank=rank, world_size=world, global_batch_size=global_batch_size,
                                    microbatch_size=microbatch_size, sync_policy=sync_policy)
                synchronize(device)
                local_seconds = time.perf_counter() - tick
                maximum = _all_reduce(torch.tensor(local_seconds, device=device, dtype=torch.float64), dist.ReduceOp.MAX).item()
                record("train", step, **values, learning_rate=lr, tokens_seen=step * tokens_per_step,
                       step_seconds=maximum, local_step_seconds=local_seconds)
                if step == 1:
                    first_update_memory = {
                        "optimizer_state_bytes": optimizer.local_state_bytes() if mode == "zero1" else tensor_bytes(optimizer.state),
                        "local_gradient_bytes": sum(tensor_bytes(p.grad) for p in base_model.parameters()),
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
                        "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else None,
                    }
                if step > warmup_steps:
                    measured.append(maximum)
                if step % config.eval_interval == 0 or step == config.max_steps:
                    synchronize(device)
                    tick = time.perf_counter()
                    # Unwrapped DDP avoids per-forward synchronization; FSDP/TP remain collective.
                    last_valid_loss = evaluate(base_model, valid_data, config, device,
                                               rank=rank, world_size=world, mode=mode)
                    synchronize(device)
                    record("eval", step, valid_loss=last_valid_loss, eval_seconds=time.perf_counter() - tick,
                           valid_tokens=config.eval_batches * config.eval_batch_size * config.model.context_length)
                if profiler is not None:
                    profiler.step()
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        optimizer_bytes = optimizer.local_state_bytes() if mode == "zero1" else tensor_bytes(optimizer.state)
        final_hash = state_hash(base_model)
        batch_stream_hash = hashlib.sha256("".join(row["batch_start_sha256"] for row in metrics
                                                  if row["event"] == "train").encode()).hexdigest()
        rank_result = {"rank": rank, "peak_cuda_allocated_bytes": peak_allocated,
                       "peak_cuda_reserved_bytes": peak_reserved, "optimizer_state_bytes": optimizer_bytes,
                       "local_parameter_bytes": sum(_local(p).numel() * p.element_size() for p in base_model.parameters()),
                       "local_gradient_bytes": sum(tensor_bytes(p.grad) for p in base_model.parameters() if p.grad is not None),
                       "first_update_memory": first_update_memory,
                       "batch_stream_sha256": batch_stream_hash,
                       "final_parameter_sha256": final_hash,
                       "parameter_hash_scope": "rank-local shard" if mode in {"fsdp", "tp"} else "full replicated model"}
        ranks = [None for _ in range(world)]
        if world > 1:
            dist.all_gather_object(ranks, rank_result)
        else:
            ranks = [rank_result]
        if len({item["batch_stream_sha256"] for item in ranks}) != 1:
            raise RuntimeError("Global batch streams differ across ranks")
        if mode in {"naive", "ddp", "zero1"} and len({item["final_parameter_sha256"] for item in ranks}) != 1:
            raise RuntimeError("Final replicated parameters differ across ranks")
        if save_final_state:
            if mode == "tp":
                from .tensor_parallel import reconstruct_state_dict
                state = reconstruct_state_dict(base_model)
            elif mode == "fsdp":
                state = {name: value.full_tensor() if hasattr(value, "full_tensor") else value
                         for name, value in base_model.state_dict().items()}
            else:
                state = base_model.state_dict()
            if rank == 0:
                torch.save({"model": {name: value.detach().cpu() for name, value in state.items()},
                            "optimizer": _cpu_tree(optimizer.state_dict()), "sampler_rng_state": sampler.get_state()},
                           output / "final-state.pt")
        summary = {"status": "completed", "mode": mode, "world_size": world,
                   "global_batch_size": global_batch_size, "microbatch_size": microbatch_size,
                   "accumulation_steps": accumulation, "sync_policy": sync_policy,
                   "step": config.max_steps, "tokens_seen": config.max_steps * tokens_per_step,
                   "tokens_per_step": tokens_per_step, "measured_steps": len(measured),
                   "step_seconds_median": statistics.median(measured),
                   "step_seconds_p90": float(np.percentile(measured, 90)),
                   "tokens_per_second": len(measured) * tokens_per_step / sum(measured),
                   "measured_train_seconds": sum(measured), "setup_seconds": setup_seconds,
                   "valid_loss": last_valid_loss, "profiled": bool(profile_steps), "ranks": ranks,
                   "peak_cuda_allocated_bytes": max(item["peak_cuda_allocated_bytes"] for item in ranks) if device.type == "cuda" else None,
                   "peak_cuda_reserved_bytes": max(item["peak_cuda_reserved_bytes"] for item in ranks) if device.type == "cuda" else None,
                   "optimizer_state_bytes_max_rank": max(item["optimizer_state_bytes"] for item in ranks),
                   "optimizer_state_bytes_sum": sum(item["optimizer_state_bytes"] for item in ranks),
                   "batch_stream_sha256": batch_stream_hash,
                   "wall_seconds": time.perf_counter() - started, "timing_scope": metadata["timing_scope"]}
        (output / f"rank-{rank}-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if rank == 0:
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    except Exception as error:
        if output_ready:
            failure = {"status": "failed", "rank": rank, "mode": mode, "error": str(error),
                       "wall_seconds": time.perf_counter() - started}
            (output / f"rank-{rank}-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            if rank == 0:
                (output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        if initialized_here:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=MODES, default="single")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int, default=5, help="training updates excluded from timing")
    parser.add_argument("--sync-policy", choices=["last", "every"], default="last")
    parser.add_argument("--bucket-cap-mb", type=float, default=25.)
    parser.add_argument("--checkpointing", choices=["none", "block"])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--save-final-state", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {key: value for key, value in (("seed", args.seed), ("max_steps", args.steps),
                 ("device", args.device), ("precision", args.precision)) if value is not None}
    config = replace(config, **overrides)
    if args.checkpointing is not None:
        config = replace(config, model=replace(config.model, activation_checkpointing=args.checkpointing))
    result = run(config, args.output_dir, mode=args.mode, global_batch_size=args.global_batch_size,
                 microbatch_size=args.microbatch_size, warmup_steps=args.warmup_steps,
                 sync_policy=args.sync_policy, bucket_cap_mb=args.bucket_cap_mb,
                 save_final_state=args.save_final_state, profile_steps=args.profile_steps)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
