from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
import torch

from llmlab.model.transformer import TransformerLM
from llmlab.train import DataConfig, ModelConfig, TrainConfig, autocast, evaluate, run, train_step
from llmlab.training.loss import cross_entropy


def tiny_config():
    return TrainConfig(
        model=ModelConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            rope_theta=10000.0,
            attention_backend="eager",
        ),
        data=DataConfig(kind="synthetic", synthetic_tokens=256),
        seed=7,
        device="cpu",
        num_threads=1,
        batch_size=2,
        max_steps=6,
        learning_rate=0.003,
        min_learning_rate=0.0003,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=2,
        eval_batches=2,
        eval_batch_size=2,
        checkpoint_interval=3,
    )


def checkpoint(directory):
    return torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)


def events(directory, kind):
    path = directory / "metrics.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return [record for record in records if record["event"] == kind]


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "grad_accum_steps,activation_checkpointing", [(1, "none"), (3, "block")]
)
def test_resume_matches_uninterrupted_training(
    tmp_path, grad_accum_steps, activation_checkpointing
):
    config = tiny_config()
    config = replace(
        config,
        grad_accum_steps=grad_accum_steps,
        model=replace(
            config.model,
            num_layers=2,
            activation_checkpointing=activation_checkpointing,
        ),
    )
    full_dir, paused_dir, resumed_dir = [
        tmp_path / name for name in ("full", "paused", "resumed")
    ]
    full = run(config, full_dir)
    paused = run(config, paused_dir, stop_after=3)
    resumed = run(config, resumed_dir, resume=paused_dir / "checkpoint.pt")

    assert paused["status"] == "paused"
    assert paused["step"] == 3
    tokens_per_step = config.batch_size * grad_accum_steps * config.model.context_length
    assert paused["tokens_seen"] == 3 * tokens_per_step
    assert full["status"] == resumed["status"] == "completed"
    assert full["step"] == resumed["step"] == config.max_steps
    assert full["tokens_seen"] == resumed["tokens_seen"] == config.max_steps * tokens_per_step
    assert resumed["invocation_tokens"] == (config.max_steps - 3) * tokens_per_step

    expected, actual = checkpoint(full_dir), checkpoint(resumed_dir)
    for key in ("model", "optimizer", "sampler_rng_state", "torch_rng_state"):
        assert_identical(expected[key], actual[key])

    remaining = [row for row in events(full_dir, "train") if row["step"] > 3]
    continued = events(resumed_dir, "train")
    assert [row["step"] for row in remaining] == [row["step"] for row in continued]
    assert [row["loss"] for row in remaining] == [row["loss"] for row in continued]
    assert [row["batch_start_sha256"] for row in remaining] == [
        row["batch_start_sha256"] for row in continued
    ]
    assert [row["tokens_seen"] for row in remaining] == [
        row["tokens_seen"] for row in continued
    ]


def test_evaluation_schedule_does_not_change_training(tmp_path):
    config = tiny_config()
    frequent, sparse = tmp_path / "frequent", tmp_path / "sparse"
    run(replace(config, eval_interval=1), frequent)
    run(replace(config, eval_interval=4), sparse)

    assert len(events(frequent, "eval")) > len(events(sparse, "eval"))
    assert [row["loss"] for row in events(frequent, "train")] == [
        row["loss"] for row in events(sparse, "train")
    ]
    left, right = checkpoint(frequent), checkpoint(sparse)
    for key in ("model", "optimizer", "sampler_rng_state", "torch_rng_state"):
        assert_identical(left[key], right[key])


def test_validation_examples_do_not_depend_on_training_batch_size():
    config = tiny_config()
    model = TransformerLM(**asdict(config.model))
    data = np.arange(257, dtype=np.int64) % config.model.vocab_size
    device = torch.device("cpu")
    rng_before = torch.get_rng_state().clone()

    expected = evaluate(model, data, config, device)
    actual = evaluate(model, data, replace(config, batch_size=3), device)

    assert expected == actual
    assert model.training
    assert_identical(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "change", ["batch_size", "attention_backend", "eval_batches", "grad_accum_steps"]
)
def test_resume_rejects_incompatible_training_config(tmp_path, change):
    config = tiny_config()
    paused, resumed = tmp_path / "paused", tmp_path / "resumed"
    run(config, paused, stop_after=3)
    if change == "batch_size":
        changed = replace(config, batch_size=3)
    elif change == "eval_batches":
        changed = replace(config, eval_batches=3)
    elif change == "grad_accum_steps":
        changed = replace(config, grad_accum_steps=3)
    else:
        changed = replace(config, model=replace(config.model, attention_backend="sdpa"))

    with pytest.raises(ValueError):
        run(changed, resumed, resume=paused / "checkpoint.pt")
    assert not events(resumed, "train")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_steps", 2.5),
        ("max_steps", math.inf),
        ("weight_decay", math.nan),
        ("weight_decay", math.inf),
        ("grad_clip", math.nan),
        ("grad_clip", math.inf),
        ("grad_accum_steps", 0),
        ("grad_accum_steps", -1),
        ("grad_accum_steps", 1.5),
        ("grad_accum_steps", True),
    ],
)
def test_invalid_numeric_config_is_rejected_before_training(tmp_path, field, value):
    config = replace(tiny_config(), **{field: value})
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        run(config, output)
    assert not events(output, "train")


def test_fractional_stop_step_is_rejected_before_training(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        run(tiny_config(), output, stop_after=2.5)
    assert not events(output, "train")


def test_missing_data_records_failed_setup(tmp_path):
    config = replace(
        tiny_config(),
        data=DataConfig(
            kind="npy",
            train_path=str(tmp_path / "missing-train.npy"),
            valid_path=str(tmp_path / "missing-valid.npy"),
        ),
    )
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run(config, output)

    failure = json.loads((output / "summary.json").read_text())
    assert failure["status"] == "failed"
    assert failure["step"] == 0
    assert "missing-train.npy" in failure["error"]
    assert not events(output, "train")


@pytest.mark.parametrize("changed_split", ["train", "valid"])
def test_resume_rejects_changed_data_at_same_path(tmp_path, changed_split):
    train_path, valid_path = tmp_path / "train.npy", tmp_path / "valid.npy"
    np.save(train_path, np.arange(256, dtype=np.uint16) % 32)
    np.save(valid_path, (np.arange(256, dtype=np.uint16) * 3) % 32)
    config = replace(
        tiny_config(),
        data=DataConfig(kind="npy", train_path=str(train_path), valid_path=str(valid_path)),
    )
    paused, resumed = tmp_path / "paused", tmp_path / "resumed"
    run(config, paused, stop_after=3)

    changed_path = train_path if changed_split == "train" else valid_path
    data = np.load(changed_path, allow_pickle=False)
    data[0] = (int(data[0]) + 1) % config.model.vocab_size
    np.save(changed_path, data)

    with pytest.raises(ValueError):
        run(config, resumed, resume=paused / "checkpoint.pt")
    assert not events(resumed, "train")


@pytest.mark.parametrize("grad_accum_steps", [1, 3])
def test_run_records_steps_tokens_and_validation(tmp_path, grad_accum_steps):
    config = replace(tiny_config(), grad_accum_steps=grad_accum_steps)
    output = tmp_path / "run"
    summary = run(config, output)
    train_rows = events(output, "train")
    tokens_per_step = config.batch_size * grad_accum_steps * config.model.context_length

    assert [row["step"] for row in train_rows] == list(range(1, config.max_steps + 1))
    assert [row["tokens_seen"] for row in train_rows] == [
        step * tokens_per_step for step in range(1, config.max_steps + 1)
    ]
    assert summary["tokens_seen"] == config.max_steps * tokens_per_step
    assert summary["invocation_tokens"] == summary["tokens_seen"]
    assert checkpoint(output)["step"] == config.max_steps
    assert json.loads((output / "summary.json").read_text()) == summary
    assert (output / "metadata.json").is_file()
    assert events(output, "checkpoint")
    validation = events(output, "eval")
    assert validation
    assert all(math.isfinite(row["valid_loss"]) for row in validation)
    assert all(
        row["valid_tokens"]
        == config.eval_batch_size * config.eval_batches * config.model.context_length
        for row in validation
    )


@pytest.mark.parametrize("activation_checkpointing", ["none", "block"])
def test_accumulation_matches_full_batch_gradients_and_adamw(
    monkeypatch, activation_checkpointing
):
    config = tiny_config()
    config = replace(
        config,
        model=replace(config.model, num_layers=2),
        batch_size=6,
        grad_clip=0.1,
    )
    accumulated_config = replace(
        config,
        batch_size=2,
        grad_accum_steps=3,
        model=replace(config.model, activation_checkpointing=activation_checkpointing),
    )
    torch.manual_seed(config.seed)
    full_model = TransformerLM(**asdict(config.model))
    accumulated_model = TransformerLM(**asdict(accumulated_config.model))
    accumulated_model.load_state_dict(full_model.state_dict())
    full_optimizer, accumulated_optimizer = [
        torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.95), weight_decay=config.weight_decay,
        )
        for model in (full_model, accumulated_model)
    ]
    full_sampler, accumulated_sampler = [
        torch.Generator().manual_seed(config.seed + 1) for _ in range(2)
    ]
    data = np.random.default_rng(21).integers(0, config.model.vocab_size, size=256)
    device = torch.device("cpu")
    gradients_before_clipping = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture_gradients(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients_before_clipping.append(
            torch.cat([parameter.grad.detach().flatten().clone() for parameter in parameters])
        )
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_gradients)
    for _ in range(3):
        full_before = torch.cat([p.detach().flatten().clone() for p in full_model.parameters()])
        accumulated_before = torch.cat([
            p.detach().flatten().clone() for p in accumulated_model.parameters()
        ])
        full = train_step(full_model, full_optimizer, data, config, full_sampler, device)
        accumulated = train_step(
            accumulated_model, accumulated_optimizer, data,
            accumulated_config, accumulated_sampler, device,
        )

        assert full["batch_start_sha256"] == accumulated["batch_start_sha256"]
        assert_identical(full_sampler.get_state(), accumulated_sampler.get_state())
        assert full["loss"] == pytest.approx(accumulated["loss"], rel=1e-6, abs=1e-6)
        # Clipping is active, so this also checks accumulation precedes clipping.
        assert full["gradient_norm"] > config.grad_clip
        assert full["gradient_norm"] == pytest.approx(accumulated["gradient_norm"], rel=2e-5)
        assert len(gradients_before_clipping) == 2
        torch.testing.assert_close(
            gradients_before_clipping[0], gradients_before_clipping[1],
            rtol=2e-5, atol=2e-7,
        )
        gradients_before_clipping.clear()

        full_after = torch.cat([p.detach().flatten() for p in full_model.parameters()])
        accumulated_after = torch.cat([
            p.detach().flatten() for p in accumulated_model.parameters()
        ])
        full_update = full_after - full_before
        accumulated_update = accumulated_after - accumulated_before
        # Compare update vectors by norm, avoiding unstable relative checks near zero.
        relative_update_error = (
            torch.linalg.vector_norm(full_update - accumulated_update)
            / torch.linalg.vector_norm(full_update)
        )
        assert relative_update_error.item() < 5e-4
        torch.testing.assert_close(full_after, accumulated_after, rtol=2e-5, atol=2e-6)
        for full_state, accumulated_state in zip(
            full_optimizer.state.values(), accumulated_optimizer.state.values()
        ):
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    full_state[key], accumulated_state[key], rtol=3e-5, atol=2e-8
                )


def test_bf16_on_cpu_records_failed_setup(tmp_path):
    config = replace(tiny_config(), precision="bf16")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="CUDA"):
        run(config, output)

    failure = json.loads((output / "summary.json").read_text())
    assert failure["status"] == "failed"
    assert failure["step"] == 0
    assert not events(output, "train")


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.is_bf16_supported(including_emulation=False),
    reason="requires a CUDA GPU with native BF16 support",
)
@pytest.mark.parametrize("activation_checkpointing", ["none", "block"])
def test_cuda_bf16_accumulation_keeps_fp32_optimizer_state(
    tmp_path, activation_checkpointing
):
    config = tiny_config()
    config = replace(
        config,
        device="cuda",
        precision="bf16",
        grad_accum_steps=3,
        max_steps=3,
        model=replace(
            config.model, num_layers=2, activation_checkpointing=activation_checkpointing,
        ),
    )
    output = tmp_path / "run"
    summary = run(config, output)
    state = checkpoint(output)

    assert summary["status"] == "completed"
    assert summary["tokens_seen"] == 3 * 2 * 3 * config.model.context_length
    assert all(value.dtype == torch.float32 for value in state["model"].values())
    for parameter_state in state["optimizer"]["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            assert parameter_state[key].dtype == torch.float32
            assert torch.isfinite(parameter_state[key]).all()
    assert all(math.isfinite(row["loss"]) for row in events(output, "train"))
    assert all(math.isfinite(row["valid_loss"]) for row in events(output, "eval"))

    # Observe real autocast output in both training and validation, not just metadata.
    model = TransformerLM(**asdict(config.model)).to("cuda")
    model.load_state_dict(state["model"])
    output_dtypes = []
    hook = model.register_forward_hook(lambda _model, _args, result: output_dtypes.append(result.dtype))
    data = np.arange(256, dtype=np.int64) % config.model.vocab_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    try:
        train_step(
            model, optimizer, data, config,
            torch.Generator().manual_seed(42), torch.device("cuda"),
        )
        assert output_dtypes == [torch.bfloat16] * config.grad_accum_steps
        output_dtypes.clear()
        assert math.isfinite(evaluate(model, data, config, torch.device("cuda")))
        assert output_dtypes == [torch.bfloat16] * config.eval_batches
    finally:
        hook.remove()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.is_bf16_supported(including_emulation=False),
    reason="requires a CUDA GPU with native BF16 support",
)
@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_cuda_bf16_loss_and_gradients_pass_small_numerical_gate(attention_backend):
    config = tiny_config()
    config = replace(
        config,
        device="cuda",
        model=replace(config.model, num_layers=2, attention_backend=attention_backend),
    )
    torch.manual_seed(config.seed)
    fp32_model = TransformerLM(**asdict(config.model)).to("cuda")
    bf16_model = TransformerLM(**asdict(config.model)).to("cuda")
    bf16_model.load_state_dict(fp32_model.state_dict())
    tokens = torch.randint(
        config.model.vocab_size,
        (4, config.model.context_length + 1),
        generator=torch.Generator().manual_seed(113),
    ).to("cuda")
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    losses, gradients = [], []
    for model, precision in ((fp32_model, "fp32"), (bf16_model, "bf16")):
        activation_dtypes = []
        # Residual streams may remain FP32; observe actual attention/FFN projections.
        observed_modules = [
            module for layer in model.layers for module in (layer.attn, layer.ffn)
        ] + [model.lm_head]
        hooks = [
            module.register_forward_hook(
                lambda _module, _args, output: activation_dtypes.append(output.dtype)
            )
            for module in observed_modules
        ]
        try:
            with autocast(replace(config, precision=precision), torch.device("cuda")):
                logits = model(inputs)
                loss = cross_entropy(logits.float(), targets)
            assert torch.isfinite(loss)
            loss.backward()
        finally:
            for hook in hooks:
                hook.remove()
        expected_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        assert activation_dtypes == [expected_dtype] * len(observed_modules)
        assert logits.dtype == expected_dtype
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        assert all(parameter.grad is not None for parameter in model.parameters())
        gradient = torch.cat([parameter.grad.detach().flatten() for parameter in model.parameters()])
        assert torch.isfinite(gradient).all()
        losses.append(loss.item())
        gradients.append(gradient)

    # This is a small numerical gate, not evidence of matching training quality.
    assert losses[1] == pytest.approx(losses[0], rel=0.01, abs=0.01)
    fp32_norm = torch.linalg.vector_norm(gradients[0])
    assert fp32_norm.item() > 0
    relative_gradient_error = torch.linalg.vector_norm(gradients[1] - gradients[0]) / fp32_norm
    assert relative_gradient_error.item() < 0.05
