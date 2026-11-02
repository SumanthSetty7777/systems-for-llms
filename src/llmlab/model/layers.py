import torch
import math
from einops import einsum

# Linear layer
class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2 / (in_features + out_features))
        weight = torch.empty(
            out_features,
            in_features,
            device = device,
            dtype = dtype,
        )
        torch.nn.init.trunc_normal_(
            weight,
            mean = 0.0,
            std = std,
            a = -3 * std,
            b = 3 * std,
        )
        self.weight = torch.nn.Parameter(weight) # learnable weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


# Embedding layer
class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # self.weight: (vocab_size, d_model / embedding_dim)
        weight = torch.empty(
            num_embeddings,
            embedding_dim,
            device = device,
            dtype = dtype,
        )
        torch.nn.init.trunc_normal_(
            weight,
            mean = 0.0,
            std = 1.0,
            a = -3.0,
            b = 3.0,
        )
        self.weight = torch.nn.Parameter(weight) # learnable weights

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        #token_ids: (B, T)
        return self.weight[token_ids] # embeddings: (B, T, D) (D = d_model (or) embedding_dim)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# feed-forward network
class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # GPUs like dimensions that align nicely with hardware tiles.
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * math.ceil(d_ff / 64) # should we use only torch functions?
        
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

# pre-norm layer
class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        # Accumulate the squared values in FP32.
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight

        return result.to(in_dtype)
