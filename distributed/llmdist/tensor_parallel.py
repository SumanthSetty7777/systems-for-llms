"""Head and feed-forward tensor parallelism for the project's dense Transformer.

Each rank receives the same token batch. Attention heads and SwiGLU hidden units
are partitioned; embeddings, RMSNorm and the vocabulary projection are replicated.
This is pure tensor parallelism, with no data-parallel gradient averaging.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from llmlab.model.attention import scaled_dot_product_attention
from llmlab.model.layers import silu


def _world(group):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Tensor parallelism requires an initialized process group")
    return dist.get_world_size(group), dist.get_rank(group)


class _CopyToParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient):
        # Input was replicated; accumulate contributions from the local shards.
        result = gradient.contiguous().clone()
        if dist.get_world_size(ctx.group) > 1:
            dist.all_reduce(result, group=ctx.group)
        return result, None


class _ReduceFromParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.contiguous().clone()
        if dist.get_world_size(group) > 1:
            dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        # Output and its loss are replicated, so do not sum this gradient again.
        return gradient, None


class _ShardLinear(nn.Module):
    def __init__(self, original, dimension, rank, world_size):
        super().__init__()
        width = original.weight.shape[dimension] // world_size
        value = original.weight.detach().narrow(dimension, rank * width, width).clone().contiguous()
        self.weight = nn.Parameter(value, requires_grad=original.weight.requires_grad)
        self.weight._tp_shard_dim = dimension
        self.weight._tp_full_shape = tuple(original.weight.shape)
        self.in_features, self.out_features = value.shape[1], value.shape[0]

    def forward(self, value):
        return F.linear(value, self.weight)


class _ParallelAttention(nn.Module):
    def __init__(self, original, group):
        super().__init__()
        world_size, rank = _world(group)
        self.group = group
        self.d_model = original.d_model
        self.d_k = original.d_k
        self.num_heads = original.num_heads // world_size
        self.max_seq_len = original.max_seq_len
        self.attention_backend = original.attention_backend
        self.q_proj = _ShardLinear(original.q_proj, 0, rank, world_size)
        self.k_proj = _ShardLinear(original.k_proj, 0, rank, world_size)
        self.v_proj = _ShardLinear(original.v_proj, 0, rank, world_size)
        self.output_proj = _ShardLinear(original.output_proj, 1, rank, world_size)
        self.rope = original.rope

    def forward(self, value, token_positions=None):
        if value.ndim < 2 or value.shape[-1] != self.d_model:
            raise ValueError("Attention input must end in (sequence_length, d_model)")
        length = value.shape[-2]
        if length <= 0 or (self.max_seq_len is not None and length > self.max_seq_len):
            raise ValueError("Sequence length outside the configured context")
        if token_positions is None:
            token_positions = torch.arange(length, device=value.device)
        # Share one copy operation across q/k/v, summing their partial input
        # gradients locally before the single backward collective.
        copied = _CopyToParallel.apply(value, self.group)

        def heads(projected):
            return projected.reshape(*projected.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)

        query, key, content = (heads(projection(copied)) for projection in
                               (self.q_proj, self.k_proj, self.v_proj))
        if self.rope is not None:
            query, key = self.rope(query, token_positions), self.rope(key, token_positions)
        if self.attention_backend == "sdpa":
            attended = F.scaled_dot_product_attention(query, key, content, is_causal=True, dropout_p=0.0)
        else:
            causal_mask = torch.ones(length, length, device=value.device, dtype=torch.bool).tril()
            attended = scaled_dot_product_attention(query, key, content, mask=causal_mask)
        attended = attended.transpose(-3, -2).reshape(*value.shape[:-1], self.num_heads * self.d_k)
        return _ReduceFromParallel.apply(self.output_proj(attended), self.group)


class _ParallelSwiGLU(nn.Module):
    def __init__(self, original, group):
        super().__init__()
        world_size, rank = _world(group)
        self.group = group
        self.d_model, self.d_ff = original.d_model, original.d_ff // world_size
        self.w1 = _ShardLinear(original.w1, 0, rank, world_size)
        self.w2 = _ShardLinear(original.w2, 1, rank, world_size)
        self.w3 = _ShardLinear(original.w3, 0, rank, world_size)

    def forward(self, value):
        copied = _CopyToParallel.apply(value, self.group)
        partial = self.w2(silu(self.w1(copied)) * self.w3(copied))
        return _ReduceFromParallel.apply(partial, self.group)


def parallelize_model(model, group=None):
    """Replace attention/MLP matrices in-place using slices of identical weights.

    Initialize every rank from the same full model before calling this function.
    No random numbers are consumed. Call before constructing the optimizer.
    """
    world_size, _ = _world(group)
    if getattr(model, "_tp_parallelized", False):
        raise ValueError("Model is already tensor parallel")
    for layer in model.layers:
        if layer.attn.num_heads % world_size or layer.ffn.d_ff % world_size:
            raise ValueError("Attention head count and feed-forward width must divide TP world size")
    for layer in model.layers:
        layer.attn = _ParallelAttention(layer.attn, group)
        layer.ffn = _ParallelSwiGLU(layer.ffn, group)
    model._tp_parallelized, model._tp_group = True, group
    return model


@torch.no_grad()
def clip_grad_norm_tp(model, max_norm, error_if_nonfinite=True):
    """Clip by the full-model L2 norm, counting replicated gradients only once."""
    if not getattr(model, "_tp_parallelized", False):
        raise ValueError("Expected a tensor-parallel model")
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    _, rank = _world(model._tp_group)
    parameters = list(model.parameters())
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    # FP32 for low-precision gradients, FP64 when checking a double model.
    dtype = torch.float64 if any(gradient.dtype == torch.float64 for gradient in gradients) else torch.float32
    terms = [torch.linalg.vector_norm(parameter.grad.detach(), 2, dtype=dtype).square()
             for parameter in parameters if parameter.grad is not None
             and (hasattr(parameter, "_tp_shard_dim") or rank == 0)]
    squared = (torch.stack(terms).sum() if terms else
               torch.zeros((), device=parameters[0].device, dtype=dtype))
    dist.all_reduce(squared, group=model._tp_group)
    norm = squared.sqrt()
    if error_if_nonfinite and not torch.isfinite(norm):
        raise RuntimeError("Non-finite tensor-parallel gradient norm")
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return norm


def describe_sharding(model):
    world_size, rank = _world(model._tp_group)
    local_sharded = sum(parameter.numel() for parameter in model.parameters()
                        if hasattr(parameter, "_tp_shard_dim"))
    local_replicated = sum(parameter.numel() for parameter in model.parameters()
                           if not hasattr(parameter, "_tp_shard_dim"))
    return {"world_size": world_size, "rank": rank,
            "local_sharded_parameters": local_sharded,
            "local_replicated_parameters": local_replicated,
            "logical_parameter_count": local_sharded * world_size + local_replicated,
            "replicated_components": ["token_embeddings", "RMSNorm", "lm_head"],
            "shard_dimensions": {name: getattr(parameter, "_tp_shard_dim", None)
                                 for name, parameter in model.named_parameters()}}


@torch.no_grad()
def reconstruct_state_dict(model):
    """Collect full parameter tensors on every rank; intended for small audits.

    Every TP rank must call in the same order. Full tensors remain on the model's
    device; callers can move the returned values to CPU before saving.
    """
    world_size, _ = _world(model._tp_group)
    parameters = dict(model.named_parameters())
    result = {}
    for name, tensor in model.state_dict().items():
        dimension = getattr(parameters.get(name), "_tp_shard_dim", None)
        if dimension is None:
            result[name] = tensor.detach().clone()
        else:
            pieces = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(pieces, tensor.contiguous(), group=model._tp_group)
            result[name] = torch.cat(pieces, dim=dimension)
    return result
