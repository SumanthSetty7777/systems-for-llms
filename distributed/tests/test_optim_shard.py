"""State sharding must preserve AdamW updates while removing replica state copies."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llmdist.optim_shard import ShardedAdamW, _tensor_bytes


def test_rejects_duplicate_and_empty_parameters():
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="Duplicate"):
        ShardedAdamW([parameter, parameter])
    with pytest.raises(ValueError, match="at least one"):
        ShardedAdamW([])


def test_single_rank_shard_round_trip_and_zero_grad():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = ShardedAdamW([parameter], lr=0.01)
    parameter.grad = torch.tensor([0.5, -0.2], dtype=torch.float64)
    optimizer.step()
    state = optimizer.state_dict()
    assert optimizer.local_state_bytes() > 0
    optimizer.zero_grad(set_to_none=False)
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))
    optimizer.zero_grad()
    assert parameter.grad is None
    restored = ShardedAdamW([parameter], lr=0.01)
    restored.load_state_dict(state)
    assert restored.local_state_bytes() == optimizer.local_state_bytes()
    state["sharding"]["rank"] = 1
    with pytest.raises(ValueError, match="does not match"):
        restored.load_state_dict(state)


def test_zero_grad_accepts_ddp_bucket_views():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = ShardedAdamW([parameter])
    bucket = torch.ones(4)
    parameter.grad = bucket[:2]
    pointer = parameter.grad.data_ptr()
    optimizer.zero_grad(set_to_none=False)
    assert parameter.grad.data_ptr() == pointer
    assert torch.equal(bucket, torch.tensor([0.0, 0.0, 1.0, 1.0]))


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_deterministic_ownership_including_empty_ranks(monkeypatch, world_size):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world_size)
    monkeypatch.setattr(dist, "all_gather_object", lambda result, value, group=None: result.__setitem__(slice(None), [value] * world_size))
    layouts = []
    for rank in range(world_size):
        monkeypatch.setattr(dist, "get_rank", lambda group=None, rank=rank: rank)
        parameters = [torch.nn.Parameter(torch.ones(size)) for size in (7, 5, 3)]
        optimizer = ShardedAdamW(parameters)
        layout = optimizer.ownership_metadata()
        layouts.append(layout)
        assert sum(len(group["params"]) for group in optimizer.param_groups) == len(layout["owned_parameter_indices"])
        assert optimizer.local_state_bytes() == 0
    assert all(layout["owners"] == layouts[0]["owners"] for layout in layouts)
    assert sum(layouts[0]["parameter_numel_per_rank"]) == 15
    assert sorted(index for layout in layouts for index in layout["owned_parameter_indices"]) == [0, 1, 2]
    if world_size > 3:
        assert any(not layout["owned_parameter_indices"] for layout in layouts)


def _worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    device_kind = os.environ.get("LLMDIST_TEST_DEVICE", "cpu")
    device = torch.device("cuda", rank) if device_kind == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend, init_method=(root / "rendezvous").resolve().as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).double().to(device)
        reference = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).double().to(device)
        reference.load_state_dict(model.state_dict())
        groups = [{"params": list(model[0].parameters()), "weight_decay": 0.03},
                  {"params": list(model[2].parameters()), "lr": 0.005, "weight_decay": 0.2}]
        ref_groups = [{"params": list(reference[0].parameters()), "weight_decay": 0.03},
                      {"params": list(reference[2].parameters()), "lr": 0.005, "weight_decay": 0.2}]
        optimizer = ShardedAdamW(groups, lr=0.01, betas=(0.9, 0.95), foreach=False)
        ref_optimizer = torch.optim.AdamW(ref_groups, lr=0.01, betas=(0.9, 0.95), weight_decay=0.1,
                                         foreach=False)
        assert optimizer.local_state_bytes() == 0
        for step in range(5):
            torch.manual_seed(100 + step)
            x, y = torch.randn(8, 4, dtype=torch.float64, device=device), torch.randn(8, 3, dtype=torch.float64, device=device)
            optimizer.zero_grad()
            ref_optimizer.zero_grad(set_to_none=True)
            for index, (group, ref_group) in enumerate(zip(optimizer.param_groups, ref_optimizer.param_groups)):
                group["lr"] = ref_group["lr"] = 0.02 / ((step + 1) * (index + 1))
            local_loss = (model(x[rank * 4:(rank + 1) * 4]) - y[rank * 4:(rank + 1) * 4]).square().mean()
            local_loss.backward()
            for parameter in model.parameters():
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(2)
            (reference(x) - y).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.2, error_if_nonfinite=True)
            optimizer.step()
            ref_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        layout = optimizer.ownership_metadata()
        model_parameters = list(model.parameters())
        assert len(optimizer.state) == len(layout["owned_parameter_indices"])
        assert all(model_parameters[index] in optimizer.state for index in layout["owned_parameter_indices"])
        assert optimizer.local_state_bytes() < _tensor_bytes(ref_optimizer.state)
        total = torch.tensor(optimizer.local_state_bytes(), dtype=torch.int64, device=device)
        dist.all_reduce(total)
        assert total.item() == _tensor_bytes(ref_optimizer.state)
        flattened = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        replicas = [torch.empty_like(flattened) for _ in range(2)]
        dist.all_gather(replicas, flattened)
        assert torch.equal(replicas[0], replicas[1])
        optimizer.zero_grad(set_to_none=False)
        assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters())
        optimizer.zero_grad()
        assert all(parameter.grad is None for parameter in model.parameters())

        # One complete parameter leaves rank one with no optimizer-owned state.
        single = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64, device=device))
        single_reference = torch.nn.Parameter(single.detach().clone())
        single_ref_optimizer = torch.optim.AdamW([single_reference], lr=0.01, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
        empty_owner = ShardedAdamW([single], lr=0.01, foreach=False)
        for _ in range(3):
            single.grad = torch.tensor([0.2, -0.4], dtype=torch.float64, device=device)
            single_reference.grad = single.grad.detach().clone()
            empty_owner.step()
            single_ref_optimizer.step()
        torch.testing.assert_close(single, single_reference, rtol=1e-11, atol=1e-12)
        assert empty_owner.local_state_bytes() == (_tensor_bytes(single_ref_optimizer.state) if rank == 0 else 0)
        replicas = [torch.empty_like(single) for _ in range(2)]
        dist.all_gather(replicas, single.detach())
        assert torch.equal(replicas[0], replicas[1])

        # Shape differences must be rejected by every rank before any update.
        mismatched = torch.nn.Parameter(torch.ones(2 + rank, device=device))
        try:
            ShardedAdamW([mismatched], lr=0.01)
        except ValueError as error:
            assert "differ across ranks" in str(error)
        else:
            raise AssertionError("Mismatched optimizer layouts were accepted")
        (root / f"rank-{rank}.json").write_text(json.dumps({"passed": True, "state_bytes": layout["local_optimizer_state_bytes"],
                                                          "owners": layout["owners"]}))
    finally:
        dist.destroy_process_group()


def test_two_rank_adamw_equivalence_and_state_sharding(tmp_path):
    device = os.environ.get("LLMDIST_TEST_DEVICE", "cpu")
    assert device in {"cpu", "cuda"}
    if device == "cuda":
        assert dist.is_nccl_available() and torch.cuda.device_count() >= 2, "CUDA correctness gate requires two NCCL GPUs"
    elif not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is required for the two-rank optimizer correctness test")
    env = dict(os.environ)
    repo = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "distributed"), str(repo / "src"), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--distributed-worker", str(tmp_path)],
                            capture_output=True, text=True, timeout=90, env=env)
    if (device == "cpu" and os.environ.get("LLMDIST_ALLOW_LOCAL_NETWORK_SKIP") == "1"
            and "Cannot resolve 127.0.0.1 to a (local) address" in result.stderr):
        pytest.skip("Local sandbox cannot initialize Gloo loopback; GPU correctness gate is still required")
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(report["passed"] for report in reports)
    assert reports[0]["owners"] == reports[1]["owners"]


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--distributed-worker":
        raise SystemExit("This entry point only runs the bounded optimizer correctness workers")
    mp.spawn(_worker, args=(sys.argv[2],), nprocs=2, join=True)
