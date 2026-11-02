import pytest
import torch
from torch.nn import functional as F

from llmlab.model import transformer
from llmlab.model.attention import MultiHeadSelfAttention
from llmlab.model.transformer import TransformerLM
from llmlab.training.loss import cross_entropy


def make_model(attention_backend="eager", **overrides):
    config = dict(
        vocab_size=64,
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000.0,
        attention_backend=attention_backend,
    )
    config.update(overrides)
    return TransformerLM(**config)


@pytest.mark.parametrize("seq_len", [1, 7, 16])
def test_attention_backends_match_outputs_and_gradients(seq_len):
    torch.manual_seed(7)
    eager = MultiHeadSelfAttention(32, 4, theta=10_000.0, max_seq_len=16)
    sdpa = MultiHeadSelfAttention(
        32, 4, theta=10_000.0, max_seq_len=16, attention_backend="sdpa"
    )
    sdpa.load_state_dict(eager.state_dict())
    x = torch.randn(2, seq_len, 32, requires_grad=True)
    x_sdpa = x.detach().clone().requires_grad_()

    eager_output = eager(x)
    sdpa_output = sdpa(x_sdpa)
    torch.testing.assert_close(eager_output, sdpa_output, atol=2e-6, rtol=2e-5)

    upstream = torch.randn_like(eager_output)
    eager_output.backward(upstream)
    sdpa_output.backward(upstream)
    torch.testing.assert_close(x.grad, x_sdpa.grad, atol=2e-6, rtol=2e-5)
    for name, parameter in eager.named_parameters():
        torch.testing.assert_close(
            parameter.grad, sdpa.get_parameter(name).grad, atol=5e-6, rtol=2e-5
        )


def test_model_backends_match_loss_and_parameter_gradients():
    torch.manual_seed(19)
    eager = make_model()
    sdpa = make_model("sdpa")
    sdpa.load_state_dict(eager.state_dict())
    tokens = torch.randint(0, 64, (2, 11))
    targets = torch.randint(0, 64, (2, 11))

    eager_logits = eager(tokens)
    sdpa_logits = sdpa(tokens)
    torch.testing.assert_close(eager_logits, sdpa_logits, atol=2e-6, rtol=2e-5)
    eager_loss = cross_entropy(eager_logits, targets)
    sdpa_loss = cross_entropy(sdpa_logits, targets)
    torch.testing.assert_close(eager_loss, sdpa_loss)
    eager_loss.backward()
    sdpa_loss.backward()
    for name, parameter in eager.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, sdpa.get_parameter(name).grad, atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_activation_checkpointing_preserves_initialization_and_default(backend):
    models = []
    for overrides in ({}, {"activation_checkpointing": "none"}, {"activation_checkpointing": "block"}):
        torch.manual_seed(31)
        models.append(make_model(backend, **overrides))

    default = models[0]
    tokens = torch.randint(0, 64, (2, 11))
    expected = default(tokens)
    for model in models[1:]:
        assert model.state_dict().keys() == default.state_dict().keys()
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(expected, model(tokens), atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    )),
])
def test_activation_checkpointing_matches_loss_and_all_parameter_gradients(backend, device):
    torch.manual_seed(37)
    reference = make_model(backend, device=device)
    checkpointed = make_model(backend, device=device, activation_checkpointing="block")
    checkpointed.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 64, (2, 11), device=device)
    targets = torch.randint(0, 64, (2, 11), device=device)

    expected = reference(tokens)
    actual = checkpointed(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    reference_loss = cross_entropy(expected, targets)
    checkpointed_loss = cross_entropy(actual, targets)
    torch.testing.assert_close(checkpointed_loss, reference_loss, atol=2e-6, rtol=2e-5)
    reference_loss.backward()
    checkpointed_loss.backward()
    for name, parameter in reference.named_parameters():
        actual_grad = checkpointed.get_parameter(name).grad
        assert parameter.grad is not None and actual_grad is not None
        torch.testing.assert_close(actual_grad, parameter.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("training, grad_enabled", [
    (True, True), (False, True), (True, False), (False, False),
])
def test_activation_checkpointing_recomputes_only_in_training_with_gradients(
    backend, training, grad_enabled, monkeypatch
):
    model = make_model(backend, activation_checkpointing="block")
    model.train(training)
    counts = {layer: 0 for layer in model.layers}

    def count_forward(module, inputs):
        counts[module] += 1

    handles = [layer.register_forward_pre_hook(count_forward) for layer in model.layers]
    checkpoint_calls = []
    original_checkpoint = transformer.checkpoint

    def track_checkpoint(layer, *args, **kwargs):
        assert kwargs["use_reentrant"] is False
        checkpoint_calls.append(layer)
        return original_checkpoint(layer, *args, **kwargs)

    monkeypatch.setattr(transformer, "checkpoint", track_checkpoint)
    try:
        with torch.set_grad_enabled(grad_enabled):
            logits = model(torch.randint(0, 64, (2, 11)))
            if grad_enabled:
                logits.square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()

    active = training and grad_enabled
    assert checkpoint_calls == (list(model.layers) if active else [])
    assert list(counts.values()) == [2 if active else 1] * len(model.layers)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_future_tokens_do_not_change_prefix(backend):
    torch.manual_seed(23)
    model = make_model(backend)
    tokens = torch.randint(0, 64, (2, 11))
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 64
    with torch.no_grad():
        prefix = model(tokens)[:, :5]
        torch.testing.assert_close(prefix, model(changed)[:, :5], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(prefix, model(tokens[:, :5]), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_cross_entropy_matches_pytorch_value_and_gradient(scale):
    torch.manual_seed(29)
    logits = (torch.randn(2, 3, 11) * scale).requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.randint(0, 11, (2, 3))
    loss = cross_entropy(logits, targets)
    reference = F.cross_entropy(reference_logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(loss, reference)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_checkpoint_keys_preserve_legacy_layout():
    model = make_model()
    expected_keys = {
        "token_embeddings.weight", "ln_final.weight", "lm_head.weight"
    }
    block_keys = [
        "attn_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
        "attn.v_proj.weight", "attn.output_proj.weight", "ffn_norm.weight",
        "ffn.w1.weight", "ffn.w2.weight", "ffn.w3.weight",
    ]
    expected_keys.update(
        f"layers.{layer}.{key}" for layer in range(2) for key in block_keys
    )
    assert set(model.state_dict()) == expected_keys
    make_model("sdpa").load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("context_length", [0, -1])
def test_invalid_context_configuration(context_length):
    with pytest.raises(ValueError, match="context_length"):
        make_model(context_length=context_length)


@pytest.mark.parametrize("seq_len", [0, 17])
def test_sequence_outside_context_is_rejected(seq_len):
    with pytest.raises(ValueError, match="sequence length"):
        make_model()(torch.zeros(2, seq_len, dtype=torch.long))


@pytest.mark.parametrize("overrides, message", [
    ({"num_heads": 0}, "num_heads"),
    ({"num_heads": 3}, "divisible"),
    ({"d_model": 20}, "even"),
    ({"rope_theta": 0.0}, "theta"),
    ({"attention_backend": "unknown"}, "attention_backend"),
    ({"activation_checkpointing": "unknown"}, "activation_checkpointing"),
])
def test_invalid_model_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)
