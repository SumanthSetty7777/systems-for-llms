"""Global-batch/update checks; the cloud gate sets LLMDIST_TEST_DEVICE=cuda."""

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from llmlab.config import DataConfig, ModelConfig, TrainConfig
from llmdist.train import run, validate_layout


REPO = Path(__file__).resolve().parents[2]


def tiny_config(checkpointing="none", grad_clip=0.15):
    return TrainConfig(model=ModelConfig(vocab_size=32, context_length=8, d_model=16,
                       num_layers=2, num_heads=4, d_ff=32, attention_backend="sdpa",
                       activation_checkpointing=checkpointing),
                       data=DataConfig(kind="synthetic", synthetic_tokens=256),
                       max_steps=4, warmup_steps=1, eval_interval=4,
                       eval_batches=4, eval_batch_size=1, num_threads=1,
                       batch_size=4, grad_accum_steps=1, grad_clip=grad_clip,
                       learning_rate=0.001, min_learning_rate=0.0001)


@pytest.mark.parametrize("checkpointing", ["none", "block"])
def test_single_accumulation_preserves_updates_and_ordered_global_batch(tmp_path, checkpointing):
    config = tiny_config(checkpointing)
    summaries, states = [], []
    for micro in (1, 4):
        output = tmp_path / str(micro)
        summaries.append(run(config, output, mode="single", global_batch_size=4,
                             microbatch_size=micro, warmup_steps=1, save_final_state=True))
        states.append(torch.load(output / "final-state.pt", weights_only=True)["model"])
    assert summaries[0]["batch_stream_sha256"] == summaries[1]["batch_stream_sha256"]
    assert summaries[0]["tokens_seen"] == 4 * 4 * 8
    assert summaries[0]["measured_steps"] == 3
    assert summaries[0]["valid_loss"] == pytest.approx(summaries[1]["valid_loss"], abs=2e-6)
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=2e-5, atol=2e-6)


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    original = '{"existing": true}\n'
    (output / "summary.json").write_text(original)
    with pytest.raises(FileExistsError):
        run(tiny_config(), output, warmup_steps=1)
    assert (output / "summary.json").read_text() == original


@pytest.mark.parametrize("arguments", [
    ("single", 2, 8, 1, "last"), ("ddp", 1, 8, 1, "last"),
    ("ddp", 2, 7, 1, "last"), ("ddp", 2, 8, 3, "last"),
    ("tp", 2, 8, 3, "last"), ("naive", 2, 8, 1, "every"),
])
def test_invalid_layout_is_rejected(arguments):
    with pytest.raises(ValueError):
        validate_layout(*arguments)


def test_tensor_parallel_replicates_global_batch():
    assert validate_layout("tp", 2, 8, 1, "last") == 8
    assert validate_layout("ddp", 2, 8, 1, "last") == 4


def _invoke(config_path, output, *, mode, device, micro=1, sync="last", precision="fp32"):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO / "distributed")])
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    command = [sys.executable]
    if mode != "single":
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2"]
    command += ["-m", "llmdist.train", "--config", str(config_path), "--output-dir", str(output),
                "--mode", mode, "--global-batch-size", "4", "--microbatch-size", str(micro),
                "--warmup-steps", "1", "--sync-policy", sync,
                "--device", device, "--precision", precision, "--save-final-state"]
    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
        pytest.fail("Distributed training exceeded the 90-second correctness deadline:\n" + stdout[-5000:])
    if process.returncode:
        recognized = ("Cannot resolve 127.0.0.1", "Operation not permitted", "failed to listen",
                      "The IPv6 network addresses of (localhost", "Cannot resolve localhost")
        if device == "cpu" and any(message in stdout for message in recognized):
            pytest.skip("Local process-group networking is restricted: " + stdout[-1000:])
        pytest.fail("Distributed training subprocess failed:\n" + stdout[-8000:])
    return json.loads((output / "summary.json").read_text()), torch.load(output / "final-state.pt", weights_only=True)


@pytest.fixture(scope="module")
def distributed_device():
    device = os.environ.get("LLMDIST_TEST_DEVICE", "cpu")
    assert device in {"cpu", "cuda"}
    skip_network = os.environ.get("LLMDIST_SKIP_NETWORK_TESTS") == "1"
    if device == "cuda":
        assert not skip_network, "The CUDA correctness gate must not skip distributed tests"
        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "The gate requires two real GPUs"
    elif skip_network:
        pytest.skip("Local networking was observed to be denied; two-rank checks are required by the CUDA gate")
    return device


@pytest.mark.parametrize("mode,sync,checkpointing,grad_clip,seed", [
    ("naive", "last", "none", 0.15, 1337),
    ("ddp", "last", "none", 0.15, 1337),
    ("ddp", "every", "none", 0.15, 1337),
    ("ddp", "last", "block", 1000.0, 2027),
    ("zero1", "last", "none", 0.15, 1337),
    ("zero1", "last", "block", 1000.0, 2027),
    ("tp", "last", "none", 0.15, 1337),
    ("tp", "last", "block", 1000.0, 2027),
    ("fsdp", "last", "none", 0.15, 1337),
    ("fsdp", "last", "block", 1000.0, 2027),
])
def test_two_rank_updates_match_single_global_batch(tmp_path, distributed_device,
                                                  mode, sync, checkpointing, grad_clip, seed):
    if mode == "fsdp" and distributed_device == "cpu":
        pytest.skip("FSDP2 requires CUDA; mandatory in the GPU gate")
    config = replace(tiny_config(checkpointing, grad_clip), seed=seed)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    baseline, expected = _invoke(path, tmp_path / "single", mode="single", device=distributed_device, micro=4)
    actual, result = _invoke(path, tmp_path / "distributed", mode=mode, device=distributed_device, sync=sync)
    assert actual["batch_stream_sha256"] == baseline["batch_stream_sha256"]
    assert actual["tokens_seen"] == baseline["tokens_seen"]
    assert actual["valid_loss"] == pytest.approx(baseline["valid_loss"], abs=2e-5)
    expected_metrics = [json.loads(line) for line in
                        (tmp_path / "single/rank-0-metrics.jsonl").read_text().splitlines()]
    actual_metrics = [json.loads(line) for line in
                      (tmp_path / "distributed/rank-0-metrics.jsonl").read_text().splitlines()]
    expected_norms = [row["gradient_norm"] for row in expected_metrics if row["event"] == "train"]
    actual_norms = [row["gradient_norm"] for row in actual_metrics if row["event"] == "train"]
    assert actual_norms == pytest.approx(expected_norms, rel=2e-5, abs=2e-6)
    if grad_clip < 1:
        assert all(norm > grad_clip for norm in expected_norms)
    else:
        assert all(norm < grad_clip for norm in expected_norms)
    torch.testing.assert_close(result["sampler_rng_state"], expected["sampler_rng_state"], rtol=0, atol=0)
    assert result["model"].keys() == expected["model"].keys()
    for name in result["model"]:
        torch.testing.assert_close(result["model"][name], expected["model"][name], rtol=2e-5, atol=2e-6,
                                   msg=lambda message: name + ": " + message)
    expected_optimizer = expected["optimizer"]["state"]
    if mode in {"naive", "ddp"}:
        actual_optimizer = result["optimizer"]["state"]
        assert actual_optimizer.keys() == expected_optimizer.keys()
        for index, state in actual_optimizer.items():
            for key, value in state.items():
                torch.testing.assert_close(value, expected_optimizer[index][key], rtol=2e-5, atol=2e-6)
    if mode == "zero1":
        assert actual["optimizer_state_bytes_sum"] == baseline["optimizer_state_bytes_sum"]
        assert actual["optimizer_state_bytes_max_rank"] < baseline["optimizer_state_bytes_max_rank"]
        owned = result["optimizer"]["sharding"]["owned_parameter_indices"]
        for local_index, state in result["optimizer"]["optimizer"]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(value, expected_optimizer[owned[local_index]][key], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("mode", ["ddp", "zero1", "fsdp", "tp"])
def test_two_rank_native_bf16_is_finite(tmp_path, distributed_device, mode):
    if distributed_device != "cuda":
        pytest.skip("Native BF16 GPU check")
    assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 is required"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_config("block").to_dict()))
    summary, state = _invoke(path, tmp_path / "bf16", mode=mode, device="cuda", precision="bf16")
    assert summary["status"] == "completed" and np_isfinite(summary["valid_loss"])
    assert all(torch.isfinite(value).all().item() for value in state["model"].values())


def np_isfinite(value):
    # Avoid a second numeric library solely for the assertion.
    return value == value and abs(value) < float("inf")
