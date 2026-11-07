"""Single-token attention over a contiguous KV cache.

The prepared operation owns its scratch buffers and input references. It supports
CUDA graph capture after construction. Concurrent invocations must use separate
prepared operations because their scratch/output storage is mutable.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    return load(
        name="split_kv_decode_cuda",
        sources=[str(Path(__file__).with_name("decode.cu"))],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _validate(q, k, v, lengths, splits):
    import torch

    if type(splits) is not int or not 1 <= splits <= 128:
        raise ValueError("splits must be an integer in [1, 128]")
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected q[B,Hq,D], k/v[B,Hkv,N,D]")
    batch, query_heads, dim = q.shape
    if min(q.shape) <= 0 or k.shape[0] != batch or k.shape[3] != dim:
        raise ValueError("Q/K/V batch and head dimensions must match and be nonempty")
    kv_heads, capacity = k.shape[1:3]
    if kv_heads <= 0 or capacity <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if dim not in (64, 128) or query_heads // kv_heads not in (1, 2, 4, 8):
        raise ValueError("supported head dimensions are 64/128 and GQA ratios 1/2/4/8")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q/K/V must have the same FP16 or BF16 dtype")
    if not q.is_cuda or any(x.device != q.device for x in (k, v, lengths)):
        raise ValueError("all tensors must be on the same CUDA device")
    if any(not x.is_contiguous() for x in (q, k, v, lengths)):
        raise ValueError("all tensors must be contiguous")
    if lengths.dtype != torch.int32 or lengths.shape != (batch,):
        raise ValueError("lengths must be an int32 tensor of shape [B]")
    low, high = torch.aminmax(lengths)
    if low.item() < 0 or high.item() > capacity:
        raise ValueError("each length must be in [0, cache capacity]")
    return batch, query_heads, dim


@dataclass
class PreparedDecode:
    q: object
    k: object
    v: object
    lengths: object
    partial_max: object
    partial_sum: object
    partial_output: object
    output: object
    splits: int
    scale: float
    variant: str
    extension: object

    def __call__(self):
        self.extension.run(
            self.q, self.k, self.v, self.lengths,
            self.partial_max, self.partial_sum, self.partial_output,
            self.output, self.splits, self.scale, self.variant == "tile",
        )
        return self.output

    @property
    def scratch_bytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.partial_max, self.partial_sum, self.partial_output))


def prepare(q, k, v, lengths=None, *, splits=1, scale=None, variant="token"):
    """Validate and allocate outside capture, then return a reusable operation.

Lengths may be updated in place between calls but must remain within [0, N].
Zero-length rows return zeros. The kernel defensively clamps lengths to the
allocated cache. This is forward inference only; returned output has no autograd
relationship to inputs. Single-token Q is the final query, so every valid cache
position is visible.
    """
    import torch

    if variant not in ("token", "tile"):
        raise ValueError("variant must be 'token' or 'tile'")
    if lengths is None:
        if q.ndim != 3 or k.ndim != 4:
            raise ValueError("expected q[B,Hq,D], k/v[B,Hkv,N,D]")
        lengths = torch.full((q.shape[0],), k.shape[2], device=q.device, dtype=torch.int32)
    batch, heads, dim = _validate(q, k, v, lengths, splits)
    if scale is None:
        scale = dim ** -0.5
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    extension = _extension()
    maximum = torch.empty((batch, heads, splits), device=q.device, dtype=torch.float32)
    normalizer = torch.empty_like(maximum)
    numerator = torch.empty((batch, heads, splits, dim), device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    return PreparedDecode(q, k, v, lengths, maximum, normalizer, numerator,
                          output, splits, float(scale), variant, extension)


def decode(q, k, v, lengths=None, *, splits=1, scale=None, variant="token"):
    """Allocate scratch and compute a single decode step on the current stream."""
    return prepare(q, k, v, lengths, splits=splits, scale=scale, variant=variant)()
