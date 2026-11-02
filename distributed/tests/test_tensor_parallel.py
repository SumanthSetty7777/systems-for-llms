"""Two-rank TP correctness, using Gloo locally or NCCL in the GPU gate.

Set LLMDIST_TEST_DEVICE=cuda to exercise the same checks on two actual GPUs.
These tests use small FP32 models; the training campaign separately exercises BF16.
"""

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from llmlab.model.transformer import TransformerLM
from llmdist.tensor_parallel import (clip_grad_norm_tp, describe_sharding,
                                     parallelize_model, reconstruct_state_dict)

_CPU_GLOO_UNAVAILABLE = False


def _model(backend="sdpa", checkpointing="none", device="cpu"):
    return TransformerLM(vocab_size=32, context_length=8, d_model=16,
                         num_layers=2, num_heads=4, d_ff=32, rope_theta=10000.,
                         device=torch.device(device), dtype=torch.float32,
                         attention_backend=backend, activation_checkpointing=checkpointing)


def _worker(rank, world_size, rendezvous, output, attention_backend, checkpointing, device_type):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo",
                            init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(1729)
        reference = _model(attention_backend, checkpointing, device)
        parallel = copy.deepcopy(reference)
        before_rng = torch.get_rng_state().clone()
        parallelize_model(parallel)
        assert torch.equal(before_rng, torch.get_rng_state())
        description = describe_sharding(parallel)
        assert description["logical_parameter_count"] == sum(p.numel() for p in reference.parameters())
        assert description["local_sharded_parameters"] + description["local_replicated_parameters"] < description["logical_parameter_count"]
        # Identity reconstruction, before training, verifies the shard boundaries.
        initial = reconstruct_state_dict(parallel)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(initial[name], value, rtol=0, atol=0)

        # Check gradients into a replicated block input, including q/k/v summation.
        features = torch.linspace(-1, 1, 2 * 8 * 16, device=device).reshape(2, 8, 16)
        reference_x, parallel_x = features.clone().requires_grad_(), features.clone().requires_grad_()
        reference_attention = reference.layers[0].attn(reference_x)
        parallel_attention = parallel.layers[0].attn(parallel_x)
        torch.testing.assert_close(parallel_attention, reference_attention, rtol=2e-5, atol=2e-6)
        reference_attention.square().mean().backward()
        parallel_attention.square().mean().backward()
        torch.testing.assert_close(parallel_x.grad, reference_x.grad, rtol=3e-5, atol=2e-6)

        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
        parallel_optimizer = torch.optim.AdamW(parallel.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
        largest_gradient_error, largest_parameter_error = 0., 0.
        for step in range(3):
            reference_optimizer.zero_grad(set_to_none=True)
            parallel_optimizer.zero_grad(set_to_none=True)
            # Two microbatches exercise accumulation with replicated TP examples.
            for microbatch in range(2):
                tokens = ((torch.arange(16, device=device).reshape(2, 8) + step * 3 + microbatch) % 32).long()
                labels = (tokens + 1) % 32
                full_logits, local_logits = reference(tokens), parallel(tokens)
                torch.testing.assert_close(local_logits, full_logits, rtol=5e-5, atol=3e-6)
                full_loss = F.cross_entropy(full_logits.reshape(-1, 32), labels.reshape(-1)) / 2
                local_loss = F.cross_entropy(local_logits.reshape(-1, 32), labels.reshape(-1)) / 2
                full_loss.backward()
                local_loss.backward()
            full_parameters = dict(reference.named_parameters())
            for name, parameter in parallel.named_parameters():
                expected = full_parameters[name].grad
                dimension = getattr(parameter, "_tp_shard_dim", None)
                if dimension is not None:
                    width = expected.shape[dimension] // world_size
                    expected = expected.narrow(dimension, rank * width, width)
                largest_gradient_error = max(largest_gradient_error, (parameter.grad - expected).abs().max().item())
                torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=3e-6)
            # Low threshold makes clipping active and catches duplicate counting
            # of replicated embedding/norm/output gradients in the global norm.
            full_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .1, error_if_nonfinite=True)
            parallel_norm = clip_grad_norm_tp(parallel, .1)
            assert full_norm > .1
            torch.testing.assert_close(parallel_norm, full_norm, rtol=3e-5, atol=2e-6)
            reference_optimizer.step()
            parallel_optimizer.step()
            restored = reconstruct_state_dict(parallel)
            for name, value in reference.state_dict().items():
                largest_parameter_error = max(largest_parameter_error, (restored[name] - value).abs().max().item())
                torch.testing.assert_close(restored[name], value, rtol=2e-4, atol=5e-6)
        Path(output, f"rank-{rank}.json").write_text(json.dumps({
            "device": str(device), "backend": attention_backend, "checkpointing": checkpointing,
            "updates": 3, "max_gradient_error": largest_gradient_error,
            "max_parameter_error": largest_parameter_error, "sharding": description}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("attention_backend,checkpointing", [("sdpa", "none"), ("sdpa", "block"), ("eager", "block")])
def test_two_rank_tp_matches_full_model_and_clipped_adamw(tmp_path, attention_backend, checkpointing):
    global _CPU_GLOO_UNAVAILABLE
    device_type = os.environ.get("LLMDIST_TEST_DEVICE", "cpu")
    if device_type not in {"cpu", "cuda"}:
        pytest.fail("LLMDIST_TEST_DEVICE must be cpu or cuda")
    if os.environ.get("LLMDIST_SKIP_NETWORK_TESTS") == "1":
        if device_type == "cuda":
            pytest.fail("LLMDIST_SKIP_NETWORK_TESTS cannot disable the CUDA correctness gate")
        pytest.skip("Local network tests explicitly disabled; NCCL GPU correctness gate remains required")
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        pytest.fail("CUDA TP correctness gate requires two GPUs")
    if device_type == "cpu" and _CPU_GLOO_UNAVAILABLE:
        pytest.skip("Local execution environment blocks Gloo loopback sockets; NCCL GPU gate remains required")
    rendezvous = tmp_path / "rendezvous"
    context = mp.spawn(_worker, args=(2, str(rendezvous), str(tmp_path), attention_backend,
                                     checkpointing, device_type), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Two-rank TP correctness test exceeded 60 seconds")
    except mp.ProcessRaisedException as error:
        if device_type == "cpu" and "Cannot resolve 127.0.0.1 to a (local) address" in str(error):
            _CPU_GLOO_UNAVAILABLE = True
            pytest.skip("Local execution environment blocks Gloo loopback sockets; NCCL GPU gate remains required")
        raise
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(result["updates"] == 3 for result in results)
    assert all(result["sharding"]["world_size"] == 2 for result in results)


def test_parallelization_rejects_uninitialized_group():
    with pytest.raises(RuntimeError, match="initialized process group"):
        parallelize_model(_model())


def test_parallelization_validates_all_layers_before_modifying(monkeypatch):
    import llmdist.tensor_parallel as module
    monkeypatch.setattr(module, "_world", lambda group: (3, 0))
    model = _model()
    original = model.layers[0].attn
    with pytest.raises(ValueError, match="head count"):
        parallelize_model(model)
    assert model.layers[0].attn is original
