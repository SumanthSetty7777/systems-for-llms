import math

import torch
from torch.nn import functional as F
from einops import einsum, rearrange

from .layers import Linear


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be positive and even because RoPE rotates dimension pairs")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device)
        dimension_pairs = torch.arange(0, d_k, 2, device=device)
        inverse_frequencies = 1.0 / (theta ** (dimension_pairs / d_k))

        angles = einsum(
            positions,
            inverse_frequencies,
            "seq_len, d_pair -> seq_len d_pair",
        )

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos
        return output


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.shape[-1]

    attention_scores = einsum(
        query,
        key,
        "... query_position d_k, ... key_position d_k"
        " -> ... query_position key_position",
    ) / math.sqrt(d_k)

    if mask is not None:
        # True means this query is allowed to attend to this key.
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    # Keep normalization stable when matmuls run in BF16 under autocast.
    normalization_scores = (attention_scores.float()
                            if attention_scores.dtype in {torch.bfloat16, torch.float16}
                            else attention_scores)
    attention_weights = softmax(normalization_scores, dim=-1).to(attention_scores.dtype)

    return einsum(
        attention_weights,
        value,
        "... query_position key_position, ... key_position d_v"
        " -> ... query_position d_v",
    )


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = "eager",
    ):
        super().__init__()

        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if attention_backend not in {"eager", "sdpa"}:
            raise ValueError("attention_backend must be 'eager' or 'sdpa'")
        if (theta is None) != (max_seq_len is None):
            raise ValueError("theta and max_seq_len must be provided together")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_backend = attention_backend
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape (..., sequence_length, d_model)")
        seq_len = x.shape[-2]
        if seq_len == 0:
            raise ValueError("sequence length must be positive")
        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = rearrange(
            query,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        key = rearrange(
            key,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        value = rearrange(
            value,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )

        if self.rope is not None:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        if self.attention_backend == "sdpa":
            attention_output = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, dropout_p=0.0
            )
        else:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )
            attention_output = scaled_dot_product_attention(
                query, key, value, mask=causal_mask
            )

        attention_output = rearrange(
            attention_output,
            "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",
        )
        return self.output_proj(attention_output)
