"""Single-device training with controlled batching, validation and resumption."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, replace
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig, ModelConfig, TrainConfig, load_config
from .data import batch_at, file_sha256, load_data
from .model.transformer import TransformerLM
from .training.loss import cross_entropy


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def check_precision(config: TrainConfig, device: torch.device) -> None:
    if config.precision == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 training requires a CUDA device")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("bf16 requires native hardware support on the selected CUDA device")


def autocast(config: TrainConfig, device: torch.device):
    return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if config.precision == "bf16" else nullcontext())


def train_step(model: TransformerLM, optimizer: torch.optim.Optimizer,
               data: np.ndarray, config: TrainConfig, sampler: torch.Generator,
               device: torch.device) -> dict:
    """One update; sample the effective batch once, then partition it."""
    effective_batch = config.batch_size * config.grad_accum_steps
    starts = torch.randint(len(data) - config.model.context_length,
                           (effective_batch,), generator=sampler)
    batch_hash = hashlib.sha256(starts.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    for offset in range(0, effective_batch, config.batch_size):
        x, y = batch_at(data, starts[offset:offset + config.batch_size].tolist(),
                        config.model.context_length, device)
        with autocast(config, device):
            logits = model(x)
            # The custom loss needs explicit FP32 accumulation under autocast.
            loss = cross_entropy(logits.float(), y)
        total_loss += loss.detach() / config.grad_accum_steps
        (loss / config.grad_accum_steps).backward()
        del logits, loss, x, y
    if not torch.isfinite(total_loss):
        raise FloatingPointError("non-finite training loss")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": total_loss.item(), "gradient_norm": gradient_norm.item(),
            "batch_start_sha256": batch_hash}


def learning_rate(step: int, config: TrainConfig) -> float:
    # step is the number of optimizer updates already completed.
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    span = max(1, config.max_steps - config.warmup_steps - 1)
    progress = (step - config.warmup_steps) / span
    return config.min_learning_rate + 0.5 * (config.learning_rate - config.min_learning_rate) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: TransformerLM, data: np.ndarray, config: TrainConfig,
             device: torch.device) -> float:
    was_training = model.training
    model.eval()
    # Fixed windows spread over the validation split; no RNG is consumed.
    starts = np.linspace(0, len(data) - config.model.context_length - 1,
                         config.eval_batch_size * config.eval_batches, dtype=np.int64)
    total_loss = 0.0
    total_tokens = 0
    try:
        for i in range(0, len(starts), config.eval_batch_size):
            x, y = batch_at(data, starts[i:i + config.eval_batch_size].tolist(),
                            config.model.context_length, device)
            with autocast(config, device):
                loss = cross_entropy(model(x).float(), y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    finally:
        model.train(was_training)
    return total_loss / total_tokens


def source_identity() -> dict:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    repo = package.parent.parent
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=repo,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"python_source_sha256": digest.hexdigest(), "git_head": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain")}


def resume_config(config: dict) -> dict:
    # Cadence changes are safe because evaluation has no training RNG side effects.
    ignored = {"eval_interval", "checkpoint_interval", "data"}
    return {key: value for key, value in config.items() if key not in ignored}


def run(config: TrainConfig, output_dir: str | Path, resume: str | Path | None = None,
        stop_after: int | None = None) -> dict:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _run(config, output_dir, resume, stop_after, started)
    except Exception as error:
        if not (output_dir / "summary.json").exists():
            failure = {"status": "failed", "phase": "setup", "step": 0,
                       "error": str(error), "requested_config": repr(config),
                       "wall_seconds": time.perf_counter() - started}
            (output_dir / "summary.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def _run(config: TrainConfig, output_dir: Path, resume: str | Path | None,
         stop_after: int | None, started: float) -> dict:
    config.validate()
    target_step = config.max_steps if stop_after is None else stop_after
    if type(target_step) is not int or not 0 < target_step <= config.max_steps:
        raise ValueError("stop_after must be an integer in [1, max_steps]")
    torch.set_num_threads(config.num_threads)
    device = get_device(config.device)
    check_precision(config, device)
    # Fix FP32 matmul policy rather than inherit a process-wide performance knob.
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(config.seed)
    train_data, valid_data, data_identity = load_data(config)
    model = TransformerLM(**asdict(config.model), device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=config.weight_decay)
    sampler = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    source = source_identity()
    config_dict = config.to_dict()
    step = 0
    parent = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if resume_config(checkpoint["config"]) != resume_config(config_dict):
            raise ValueError("resume configuration does not match checkpoint")
        if checkpoint["data_identity"] != data_identity:
            raise ValueError("resume data/tokenizer identity does not match checkpoint")
        if checkpoint["source_sha256"] != source["python_source_sha256"]:
            raise ValueError("resume source differs from checkpoint")
        if checkpoint["device_type"] != device.type:
            raise ValueError("resume device type differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.set_state(checkpoint["sampler_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        elif device.type == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
        step = checkpoint["step"]
        if step >= target_step:
            raise ValueError("checkpoint is already at or beyond the requested stop step")
        parent = {"path": str(Path(resume).resolve()), "sha256": file_sha256(resume), "step": step}
    start_step = step
    metadata = {"config": config_dict,
                "config_sha256": hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest(),
                "data_identity": data_identity, "source": source, "parent_checkpoint": parent,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "device": str(device), "gpu_count": int(device.type == "cuda"),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "platform": platform.platform(), "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "einops")},
                "precision": config.precision,
                "parameter_dtype": "float32",
                "effective_batch_size": config.batch_size * config.grad_accum_steps,
                "tokens_per_optimizer_step": config.batch_size * config.grad_accum_steps * config.model.context_length,
                "attention_dispatch": ("automatic PyTorch SDPA; kernel not identified"
                                       if config.model.attention_backend == "sdpa" else "explicit eager attention"),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_version": torch.version.cuda,
                "cuda_device_capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                "timing_scope": "run() entry to summary construction; excludes interpreter/import startup, final output writes, and parent invocations",
                "peak_memory_scope": "training and evaluation after model initialization",
                "workload": "synthetic pipeline check" if config.data.kind == "synthetic" else "tokenized language modeling"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    train_seconds = eval_seconds = checkpoint_seconds = 0.0
    last_valid_loss = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokens_per_step = config.batch_size * config.grad_accum_steps * config.model.context_length
    metrics = (output_dir / "metrics.jsonl").open("w")

    def record(event: str, **fields) -> None:
        metrics.write(json.dumps({"event": event, "step": step,
                                  "wall_seconds": time.perf_counter() - started, **fields}, allow_nan=False) + "\n")
        metrics.flush()

    try:
        model.train()
        record("start", start_step=step)
        while step < target_step:
            synchronize(device)
            tick = time.perf_counter()
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_metrics = train_step(model, optimizer, train_data, config, sampler, device)
            synchronize(device)
            duration = time.perf_counter() - tick
            train_seconds += duration
            step += 1
            record("train", **step_metrics, learning_rate=lr, tokens_seen=step * tokens_per_step,
                   step_seconds=duration, train_seconds=train_seconds,
                   tokens_per_second=tokens_per_step / duration)
            if step % config.eval_interval == 0 or step == target_step:
                synchronize(device)
                tick = time.perf_counter()
                last_valid_loss = evaluate(model, valid_data, config, device)
                synchronize(device)
                duration = time.perf_counter() - tick
                eval_seconds += duration
                record("eval", valid_loss=last_valid_loss, eval_seconds=duration,
                       valid_tokens=config.eval_batches * config.eval_batch_size * config.model.context_length)
            if step % config.checkpoint_interval == 0 or step == target_step:
                synchronize(device)
                tick = time.perf_counter()
                state = {"format_version": 1, "config": config_dict, "data_identity": data_identity,
                         "source_sha256": source["python_source_sha256"], "device_type": device.type,
                         "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "sampler_rng_state": sampler.get_state(), "torch_rng_state": torch.get_rng_state()}
                if device.type == "cuda":
                    state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
                elif device.type == "mps":
                    state["mps_rng_state"] = torch.mps.get_rng_state()
                temporary = output_dir / "checkpoint.pt.tmp"
                torch.save(state, temporary)
                os.replace(temporary, output_dir / "checkpoint.pt")
                synchronize(device)
                duration = time.perf_counter() - tick
                checkpoint_seconds += duration
                record("checkpoint", checkpoint_seconds=duration, path="checkpoint.pt")
        summary = {"status": "completed" if step == config.max_steps else "paused",
                   "step": step, "tokens_seen": step * tokens_per_step,
                   "invocation_tokens": (step - start_step) * tokens_per_step,
                   "valid_loss": last_valid_loss, "setup_seconds": setup_seconds,
                   "train_seconds": train_seconds, "eval_seconds": eval_seconds,
                   "checkpoint_seconds": checkpoint_seconds,
                   "wall_seconds": time.perf_counter() - started,
                   "timing_scope": metadata["timing_scope"],
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        record("finish", status=summary["status"])
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    except Exception as error:
        failure = {"status": "failed", "step": step, "error": str(error),
                   "wall_seconds": time.perf_counter() - started}
        (output_dir / "summary.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        metrics.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--attention-backend", choices=["eager", "sdpa"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--batch-size", type=int, help="sequences per microbatch")
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--activation-checkpointing", choices=["none", "block"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.attention_backend is not None:
        config = replace(config, model=replace(config.model, attention_backend=args.attention_backend))
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.activation_checkpointing is not None:
        config = replace(config, model=replace(config.model, activation_checkpointing=args.activation_checkpointing))
    overrides = {name: getattr(args, name) for name in ("precision", "batch_size", "grad_accum_steps")
                 if getattr(args, name) is not None}
    config = replace(config, **overrides)
    summary = run(config, args.output_dir, args.resume, args.stop_after)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
