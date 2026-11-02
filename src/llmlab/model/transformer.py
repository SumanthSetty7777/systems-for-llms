import torch
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadSelfAttention
from .layers import Embedding, Linear, RMSNorm, SwiGLU


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = "eager",
    ):
        super().__init__()

        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
        )

        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attention_backend: str = "eager",
        activation_checkpointing: str = "none",
    ):
        super().__init__()

        dimensions = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if activation_checkpointing not in {"none", "block"}:
            raise ValueError("activation_checkpointing must be 'none' or 'block'")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta
        self.attention_backend = attention_backend
        self.activation_checkpointing = activation_checkpointing

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
                attention_backend=attention_backend,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim < 1:
            raise ValueError("input_ids must have a sequence dimension")
        if not 0 < input_ids.shape[-1] <= self.context_length:
            raise ValueError("sequence length must be between 1 and context_length")
        x = self.token_embeddings(input_ids)
        token_positions = torch.arange(
            input_ids.shape[-1], device=input_ids.device
        )

        for layer in self.layers:
            if self.activation_checkpointing == "block" and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, token_positions, use_reentrant=False)
            else:
                x = layer(x, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
