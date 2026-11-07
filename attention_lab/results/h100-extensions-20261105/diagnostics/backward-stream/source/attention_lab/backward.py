"""Trainable structured attention with query-owned and key-owned gradients.

The forward is shared with triton_attention. Backward reconstructs probabilities
from row log-sum-exp and traverses the schedule and its transpose. Each output
gradient tile has one owner, so no atomic gradient accumulation is necessary.
Only first derivatives are supported. Mask labels are shared across batch/heads.
"""

from functools import lru_cache

from .masks import BlockSchedule, MaskSpec, _documents, compile_mask
from .triton_attention import _upload_documents, _upload_schedule, _validated_schedule, attention


def transpose_schedule(schedule):
    """Transpose CSR tile ownership while retaining exact full/partial flags."""
    block_q, block_k, rows, columns, full, _ = _validated_schedule(schedule)
    key_blocks = (schedule.length + block_k - 1) // block_k
    incoming = [[] for _ in range(key_blocks)]
    for query in range(len(rows) - 1):
        for entry in range(rows[query], rows[query + 1]):
            incoming[columns[entry]].append((query, full[entry]))
    row_ptr, col_idx, full_blocks = [0], [], []
    for entries in incoming:
        col_idx.extend(query for query, _ in entries)
        full_blocks.extend(flag for _, flag in entries)
        row_ptr.append(len(col_idx))
    return BlockSchedule(schedule.length, block_k, block_q, row_ptr, col_idx, full_blocks)


@lru_cache(maxsize=64)
def _plan(length, spec, documents, block_q, block_k, implementation, num_warps, num_stages):
    schedule = compile_mask(length, spec, block_q, block_k, documents)
    reverse = transpose_schedule(schedule)
    return schedule, reverse, spec, documents, implementation, num_warps, num_stages


@lru_cache(maxsize=1)
def _kernels():
    global triton, tl, _mask_tile
    import triton
    import triton.language as tl

    @triton.jit
    def _mask_tile(rows, columns, DOCS, full,
                   N: tl.constexpr, CAUSAL: tl.constexpr,
                   HAS_LEFT: tl.constexpr, LEFT: tl.constexpr,
                   HAS_RIGHT: tl.constexpr, RIGHT: tl.constexpr,
                   SAME_DOCUMENT: tl.constexpr):
        allowed = (rows[:, None] < N) & (columns[None, :] < N)
        if full == 0:
            if CAUSAL:
                allowed &= columns[None, :] <= rows[:, None]
            if HAS_LEFT:
                allowed &= columns[None, :] >= rows[:, None] - LEFT
            if HAS_RIGHT:
                allowed &= columns[None, :] <= rows[:, None] + RIGHT
            if SAME_DOCUMENT:
                qdocs = tl.load(DOCS + rows, mask=rows < N, other=-1)
                kdocs = tl.load(DOCS + columns, mask=columns < N, other=-2)
                allowed &= qdocs[:, None] == kdocs[None, :]
        return allowed

    @triton.jit
    def delta_kernel(O, DO, DELTA, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        bh = tl.program_id(1)
        dims = tl.arange(0, D)
        offsets = bh * N * D + rows[:, None] * D + dims[None, :]
        output = tl.load(O + offsets, mask=rows[:, None] < N, other=0).to(tl.float32)
        upstream = tl.load(DO + offsets, mask=rows[:, None] < N, other=0).to(tl.float32)
        delta = tl.sum(output * upstream, axis=1)
        tl.store(DELTA + bh * N + rows, delta, mask=rows < N)

    @triton.jit
    def dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, PTR, IDX, FULL, DOCS,
                  N: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
                  SCALE: tl.constexpr, CAUSAL: tl.constexpr,
                  HAS_LEFT: tl.constexpr, LEFT: tl.constexpr,
                  HAS_RIGHT: tl.constexpr, RIGHT: tl.constexpr,
                  SAME_DOCUMENT: tl.constexpr):
        qb, bh = tl.program_id(0), tl.program_id(1)
        rows = qb * BQ + tl.arange(0, BQ)
        dims = tl.arange(0, D)
        base = bh * N * D
        offsets = base + rows[:, None] * D + dims[None, :]
        q = tl.load(Q + offsets, mask=rows[:, None] < N, other=0)
        upstream = tl.load(DO + offsets, mask=rows[:, None] < N, other=0)
        lse = tl.load(LSE + bh * N + rows, mask=rows < N, other=float("inf"))
        delta = tl.load(DELTA + bh * N + rows, mask=rows < N, other=0)
        dq = tl.zeros((BQ, D), tl.float32)
        entry, end = tl.load(PTR + qb), tl.load(PTR + qb + 1)
        while entry < end:
            columns = tl.load(IDX + entry) * BK + tl.arange(0, BK)
            kv_offsets = base + columns[:, None] * D + dims[None, :]
            k = tl.load(K + kv_offsets, mask=columns[:, None] < N, other=0)
            v = tl.load(V + kv_offsets, mask=columns[:, None] < N, other=0)
            allowed = _mask_tile(rows, columns, DOCS, tl.load(FULL + entry),
                                 N, CAUSAL, HAS_LEFT, LEFT, HAS_RIGHT, RIGHT, SAME_DOCUMENT)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            p = tl.where(allowed, tl.exp(scores - lse[:, None]), 0.0)
            dp = tl.dot(upstream, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k) * SCALE
            entry += 1
        tl.store(DQ + offsets, dq, mask=rows[:, None] < N)

    @triton.jit
    def dkv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, PTR, IDX, FULL, DOCS,
                   N: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
                   SCALE: tl.constexpr, CAUSAL: tl.constexpr,
                   HAS_LEFT: tl.constexpr, LEFT: tl.constexpr,
                   HAS_RIGHT: tl.constexpr, RIGHT: tl.constexpr,
                   SAME_DOCUMENT: tl.constexpr):
        kb, bh = tl.program_id(0), tl.program_id(1)
        columns = kb * BK + tl.arange(0, BK)
        dims = tl.arange(0, D)
        base = bh * N * D
        offsets = base + columns[:, None] * D + dims[None, :]
        k = tl.load(K + offsets, mask=columns[:, None] < N, other=0)
        v = tl.load(V + offsets, mask=columns[:, None] < N, other=0)
        dk = tl.zeros((BK, D), tl.float32)
        dv = tl.zeros((BK, D), tl.float32)
        entry, end = tl.load(PTR + kb), tl.load(PTR + kb + 1)
        while entry < end:
            rows = tl.load(IDX + entry) * BQ + tl.arange(0, BQ)
            q_offsets = base + rows[:, None] * D + dims[None, :]
            q = tl.load(Q + q_offsets, mask=rows[:, None] < N, other=0)
            upstream = tl.load(DO + q_offsets, mask=rows[:, None] < N, other=0)
            lse = tl.load(LSE + bh * N + rows, mask=rows < N, other=float("inf"))
            delta = tl.load(DELTA + bh * N + rows, mask=rows < N, other=0)
            allowed = _mask_tile(rows, columns, DOCS, tl.load(FULL + entry),
                                 N, CAUSAL, HAS_LEFT, LEFT, HAS_RIGHT, RIGHT, SAME_DOCUMENT)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            p = tl.where(allowed, tl.exp(scores - lse[:, None]), 0.0)
            dp = tl.dot(upstream, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * SCALE
            dv += tl.dot(tl.trans(p.to(upstream.dtype)), upstream)
            entry += 1
        tl.store(DK + offsets, dk, mask=columns[:, None] < N)
        tl.store(DV + offsets, dv, mask=columns[:, None] < N)

    return delta_kernel, dq_kernel, dkv_kernel


def _run_backward(q, k, v, output, lse, upstream, plan):
    import torch

    schedule, reverse, spec, documents, _, num_warps, num_stages = plan
    # Autograd may provide a broadcast/strided gradient (for example out.sum()).
    upstream = upstream.contiguous()
    batch, heads, length, dimension = q.shape
    delta = torch.empty_like(lse)
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    delta_kernel, dq_kernel, dkv_kernel = _kernels()
    delta_kernel[(triton.cdiv(length, 128), batch * heads)](
        output, upstream, delta, N=length, D=dimension, BLOCK=128)
    device = str(q.device)
    forward_gpu = _upload_schedule(device, tuple(schedule.row_ptr), tuple(schedule.col_idx), tuple(schedule.full_blocks))
    reverse_gpu = _upload_schedule(device, tuple(reverse.row_ptr), tuple(reverse.col_idx), tuple(reverse.full_blocks))
    docs = _upload_documents(device, documents) if spec.same_document else forward_gpu[0]
    options = dict(N=length, D=dimension, BQ=schedule.block_q, BK=schedule.block_k,
                   SCALE=dimension ** -0.5, CAUSAL=spec.causal,
                   HAS_LEFT=spec.window_left is not None, LEFT=spec.window_left or 0,
                   HAS_RIGHT=spec.window_right is not None, RIGHT=spec.window_right or 0,
                   SAME_DOCUMENT=spec.same_document, num_warps=num_warps, num_stages=num_stages)
    dq_kernel[(triton.cdiv(length, schedule.block_q), batch * heads)](
        q, k, v, upstream, lse, delta, dq, *forward_gpu, docs, **options)
    dkv_kernel[(triton.cdiv(length, schedule.block_k), batch * heads)](
        q, k, v, upstream, lse, delta, dk, dv, *reverse_gpu, docs, **options)
    return dq, dk, dv


@lru_cache(maxsize=1)
def _autograd_function():
    import torch
    from torch.autograd.function import once_differentiable

    class StructuredAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, plan):
            schedule, _, spec, documents, implementation, num_warps, num_stages = plan
            output, lse = attention(q.detach(), k.detach(), v.detach(), schedule, spec, documents,
                                    implementation=implementation, num_warps=num_warps,
                                    num_stages=num_stages, return_lse=True)
            ctx.save_for_backward(q, k, v, output, lse)
            ctx.plan = plan
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, upstream):
            q, k, v, output, lse = ctx.saved_tensors
            dq, dk, dv = _run_backward(q, k, v, output, lse, upstream, ctx.plan)
            return dq, dk, dv, None

    return StructuredAttention


def training_attention(q, k, v, spec=MaskSpec(), document_ids=None, *, block_q=64,
                       block_k=64, implementation="auto", num_warps=4, num_stages=2):
    """Masked attention for contiguous FP16/BF16 CUDA [B,H,N,64|128] tensors.

    Uses analytic interval forward for masks without document constraints and
    CSR otherwise. Backward traverses exact forward and transposed CSR schedules.
    Sequence lengths need not be tile-aligned. The mask language always permits
    each token's diagonal entry, so fully masked rows are not a supported input.
    """
    if not isinstance(spec, MaskSpec):
        raise TypeError("spec must be a MaskSpec")
    if q.ndim != 4:
        raise ValueError("q must have shape [B,H,N,D]")
    if implementation == "auto":
        implementation = "csr" if spec.same_document else "interval"
    documents = _documents(document_ids, length=q.shape[2], required=spec.same_document)
    plan = _plan(q.shape[2], spec, documents, block_q, block_k, implementation, num_warps, num_stages)
    return _autograd_function().apply(q, k, v, plan)
