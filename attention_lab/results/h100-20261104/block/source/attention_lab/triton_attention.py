"""Forward-only attention over a compiled CSR block-mask schedule.

Uses the established FlashAttention online-softmax algorithm, not a new attention
algorithm. See https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.
Triton is imported only when a CUDA invocation reaches the kernel loader.
"""

from functools import lru_cache


def interval_regions(length, spec, block_q, block_k, query_block):
    """Return retained/full key-block half-open ranges for one query tile.

    This host-side mirror is used for independent schedule tests. The interval
    GPU kernel derives the same bounds in registers and reads no CSR metadata.
    """
    from .masks import MaskSpec

    if not isinstance(spec, MaskSpec) or spec.same_document:
        raise ValueError("interval schedules require a MaskSpec without document constraints")
    for name, value in (("length", length), ("block_q", block_q), ("block_k", block_k)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(query_block) is not int or not 0 <= query_block < (length + block_q - 1) // block_q:
        raise ValueError("query_block is out of bounds")
    first_query = query_block * block_q
    last_query = min(first_query + block_q, length) - 1
    right = 0 if spec.causal else spec.window_right
    first_key = 0 if spec.window_left is None else max(0, first_query - spec.window_left)
    last_key = length - 1 if right is None else min(length - 1, last_query + right)
    begin, end = first_key // block_k, last_key // block_k + 1
    first_full_key = 0 if spec.window_left is None else max(0, last_query - spec.window_left)
    full_begin = min(end, max(begin, (first_full_key + block_k - 1) // block_k))
    full_end = end if right is None or first_query + right >= length - 1 else (first_query + right + 1) // block_k
    full_end = max(full_begin, min(end, full_end))
    return begin, full_begin, full_end, end


def _record_launch_metadata(compiled, destination):
    """Record compiler resource data, never a profiler or runtime measurement."""
    import hashlib

    metadata = getattr(compiled, "metadata", None)
    destination.update({
        "kernel_name": getattr(compiled, "name", None),
        "n_regs": getattr(compiled, "n_regs", None),
        "n_spills": getattr(compiled, "n_spills", None),
        "shared_memory_bytes": getattr(metadata, "shared", None),
        "source": "Triton CompiledKernel metadata; not Nsight profiling",
    })
    hashes = {}
    for name, source in getattr(compiled, "asm", {}).items():
        if isinstance(source, str):
            hashes[name] = hashlib.sha256(source.encode()).hexdigest()
        elif isinstance(source, bytes):
            hashes[name] = hashlib.sha256(source).hexdigest()
    destination["asm_sha256"] = hashes


def _validated_schedule(schedule):
    """Validate metadata representation and bounds, not semantic provenance."""
    int32_max = (1 << 31) - 1

    def integer(name, value, minimum=0):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer (bool is not accepted)")
        if value < minimum or value > int32_max:
            raise ValueError(f"{name} must be in [{minimum}, {int32_max}]")

    integer("schedule.length", schedule.length, 1)
    integer("schedule.block_q", schedule.block_q, 1)
    integer("schedule.block_k", schedule.block_k, 1)
    block_q, block_k = schedule.block_q, schedule.block_k
    if block_q not in (16, 32, 64, 128) or block_k not in (16, 32, 64, 128):
        raise ValueError("block_q and block_k must each be one of 16, 32, 64, 128")
    row_ptr = tuple(schedule.row_ptr)
    col_idx = tuple(schedule.col_idx)
    full_blocks = tuple(schedule.full_blocks)
    for value in row_ptr:
        integer("row_ptr entry", value)
    for value in col_idx:
        integer("col_idx entry", value)
    if any(not isinstance(flag, bool) for flag in full_blocks):
        raise TypeError("full_blocks entries must be bool values")

    query_blocks = (schedule.length + block_q - 1) // block_q
    key_blocks = (schedule.length + block_k - 1) // block_k
    if (len(row_ptr) != query_blocks + 1 or not row_ptr or row_ptr[0] != 0
            or row_ptr[-1] != len(col_idx) or len(full_blocks) != len(col_idx)
            or any(a > b for a, b in zip(row_ptr, row_ptr[1:]))
            or any(c >= key_blocks for c in col_idx)):
        raise ValueError("invalid CSR block schedule")
    for start, end in zip(row_ptr, row_ptr[1:]):
        columns = col_idx[start:end]
        if tuple(sorted(set(columns))) != columns:
            raise ValueError("CSR columns must be sorted and unique in each row")
    return block_q, block_k, row_ptr, col_idx, full_blocks, query_blocks


@lru_cache(maxsize=32)
def _upload_schedule(device, row_ptr, col_idx, full_blocks):
    import torch

    return (
        torch.tensor(row_ptr, dtype=torch.int32, device=device),
        torch.tensor(col_idx, dtype=torch.int32, device=device),
        torch.tensor(full_blocks, dtype=torch.uint8, device=device),
    )


@lru_cache(maxsize=32)
def _upload_documents(device, document_ids):
    import torch

    return torch.tensor(document_ids, dtype=torch.int64, device=device)


@lru_cache(maxsize=1)
def _get_kernel():
    # JIT source resolution requires these modules in the function's globals.
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def csr_attention_forward(
        Q, K, V, O, ROW_PTR, COL_IDX, FULL_BLOCKS, DOCUMENT_IDS,
        N: tl.constexpr, D: tl.constexpr, BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr, SCALE: tl.constexpr,
        CAUSAL: tl.constexpr, HAS_LEFT: tl.constexpr, LEFT: tl.constexpr,
        HAS_RIGHT: tl.constexpr, RIGHT: tl.constexpr, SAME_DOCUMENT: tl.constexpr,
        USE_FULL_BLOCKS: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        rows = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        dimensions = tl.arange(0, D)
        base = batch_head * N * D
        q = tl.load(Q + base + rows[:, None] * D + dimensions[None, :],
                    mask=rows[:, None] < N, other=0)
        if SAME_DOCUMENT:
            query_documents = tl.load(DOCUMENT_IDS + rows, mask=rows < N, other=-1)

        maximum = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_Q,), tl.float32)
        accumulator = tl.zeros((BLOCK_Q, D), tl.float32)
        entry = tl.load(ROW_PTR + query_block)
        end = tl.load(ROW_PTR + query_block + 1)
        while entry < end:
            key_block = tl.load(COL_IDX + entry)
            columns = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k = tl.load(K + base + columns[:, None] * D + dimensions[None, :],
                        mask=columns[:, None] < N, other=0)
            v = tl.load(V + base + columns[:, None] * D + dimensions[None, :],
                        mask=columns[:, None] < N, other=0)
            allowed = (rows[:, None] < N) & (columns[None, :] < N)
            if USE_FULL_BLOCKS:
                full = tl.load(FULL_BLOCKS + entry)
            else:
                full = 0
            if full == 0:
                if CAUSAL:
                    allowed = allowed & (columns[None, :] <= rows[:, None])
                if HAS_LEFT:
                    allowed = allowed & (columns[None, :] >= rows[:, None] - LEFT)
                if HAS_RIGHT:
                    allowed = allowed & (columns[None, :] <= rows[:, None] + RIGHT)
                if SAME_DOCUMENT:
                    key_documents = tl.load(DOCUMENT_IDS + columns,
                                             mask=columns < N, other=-2)
                    allowed = allowed & (query_documents[:, None] == key_documents[None, :])

            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(allowed, scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            # An empty/fully masked row must never evaluate exp(-inf - -inf)
            # into a value used by either the normalizer or accumulator.
            safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
            alpha = tl.where(maximum == -float("inf"), 0.0,
                             tl.exp(maximum - safe_maximum))
            probabilities = tl.where(allowed,
                                     tl.exp(scores - safe_maximum[:, None]), 0.0)
            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
            accumulator = accumulator * alpha[:, None]
            # Dot products accumulate in FP32; probabilities use the input
            # dtype for the tensor-core multiplication, as in common FA kernels.
            accumulator += tl.dot(probabilities.to(v.dtype), v)
            maximum = next_maximum
            entry += 1

        denominator = tl.where(normalizer > 0.0, normalizer, 1.0)
        result = accumulator / denominator[:, None]
        tl.store(O + base + rows[:, None] * D + dimensions[None, :], result,
                 mask=rows[:, None] < N)

    return csr_attention_forward


@lru_cache(maxsize=1)
def _get_interval_kernel():
    global triton, tl, _interval_tile_step
    import triton
    import triton.language as tl

    @triton.jit
    def _interval_tile_step(
        q, K, V, base, rows, dimensions, key_block,
        maximum, normalizer, accumulator,
        N: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
        SCALE: tl.constexpr, FULL: tl.constexpr,
        HAS_LEFT: tl.constexpr, LEFT: tl.constexpr,
        HAS_RIGHT: tl.constexpr, RIGHT: tl.constexpr,
    ):
        columns = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k = tl.load(K + base + columns[:, None] * D + dimensions[None, :],
                    mask=columns[:, None] < N, other=0)
        v = tl.load(V + base + columns[:, None] * D + dimensions[None, :],
                    mask=columns[:, None] < N, other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        # Full interior tiles are a distinct specialization: no token-level
        # logical mask. Sequence bounds remain necessary for a short last tile.
        if FULL:
            if N % BLOCK_K != 0:
                scores = tl.where(columns[None, :] < N, scores, -float("inf"))
        else:
            allowed = columns[None, :] < N
            if HAS_LEFT:
                allowed = allowed & (columns[None, :] >= rows[:, None] - LEFT)
            if HAS_RIGHT:
                allowed = allowed & (columns[None, :] <= rows[:, None] + RIGHT)
            scores = tl.where(allowed, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        alpha = tl.where(maximum == -float("inf"), 0.0,
                         tl.exp(maximum - safe_maximum))
        probabilities = tl.exp(scores - safe_maximum[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
        return next_maximum, normalizer, accumulator

    @triton.jit
    def interval_attention_forward(
        Q, K, V, O,
        N: tl.constexpr, D: tl.constexpr, BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr, SCALE: tl.constexpr,
        HAS_LEFT: tl.constexpr, LEFT: tl.constexpr,
        HAS_RIGHT: tl.constexpr, RIGHT: tl.constexpr,
        USE_FULL_BLOCKS: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        first_query = query_block * BLOCK_Q
        last_query = tl.minimum(first_query + BLOCK_Q, N) - 1
        rows = first_query + tl.arange(0, BLOCK_Q)
        dimensions = tl.arange(0, D)
        base = batch_head * N * D
        q = tl.load(Q + base + rows[:, None] * D + dimensions[None, :],
                    mask=rows[:, None] < N, other=0)
        maximum = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_Q,), tl.float32)
        accumulator = tl.zeros((BLOCK_Q, D), tl.float32)

        # Union of admissible key intervals in this query tile. Compile-time
        # constraints become arithmetic bounds; no schedule loads or branches.
        if HAS_LEFT:
            first_key = tl.maximum(0, first_query - LEFT)
            first_full_key = tl.maximum(0, last_query - LEFT)
        else:
            first_key = 0
            first_full_key = 0
        if HAS_RIGHT:
            last_key = tl.minimum(N - 1, last_query + RIGHT)
        else:
            last_key = N - 1
        begin = first_key // BLOCK_K
        end = last_key // BLOCK_K + 1

        if USE_FULL_BLOCKS:
            full_begin = tl.minimum(end, tl.maximum(begin, (first_full_key + BLOCK_K - 1) // BLOCK_K))
            if HAS_RIGHT:
                full_end = tl.where(first_query + RIGHT >= N - 1, end,
                                    (first_query + RIGHT + 1) // BLOCK_K)
                full_end = tl.maximum(full_begin, tl.minimum(end, full_end))
            else:
                full_end = end

            for key_block in range(begin, full_begin):
                maximum, normalizer, accumulator = _interval_tile_step(
                    q, K, V, base, rows, dimensions, key_block,
                    maximum, normalizer, accumulator,
                    N, BLOCK_K, D, SCALE, False, HAS_LEFT, LEFT, HAS_RIGHT, RIGHT,
                )
            for key_block in range(full_begin, full_end):
                maximum, normalizer, accumulator = _interval_tile_step(
                    q, K, V, base, rows, dimensions, key_block,
                    maximum, normalizer, accumulator,
                    N, BLOCK_K, D, SCALE, True, HAS_LEFT, LEFT, HAS_RIGHT, RIGHT,
                )
            for key_block in range(full_end, end):
                maximum, normalizer, accumulator = _interval_tile_step(
                    q, K, V, base, rows, dimensions, key_block,
                    maximum, normalizer, accumulator,
                    N, BLOCK_K, D, SCALE, False, HAS_LEFT, LEFT, HAS_RIGHT, RIGHT,
                )
        else:
            for key_block in range(begin, end):
                maximum, normalizer, accumulator = _interval_tile_step(
                    q, K, V, base, rows, dimensions, key_block,
                    maximum, normalizer, accumulator,
                    N, BLOCK_K, D, SCALE, False, HAS_LEFT, LEFT, HAS_RIGHT, RIGHT,
                )
        denominator = tl.where(normalizer > 0.0, normalizer, 1.0)
        result = accumulator / denominator[:, None]
        tl.store(O + base + rows[:, None] * D + dimensions[None, :], result,
                 mask=rows[:, None] < N)

    return interval_attention_forward


def attention(q, k, v, schedule, spec, document_ids=None,
              num_warps=4, num_stages=2, implementation="csr", use_full_blocks=True,
              launch_metadata=None):
    """Return masked self-attention for contiguous CUDA ``[B,H,N,D]`` tensors.

    Supports FP16/BF16, head dimensions 64/128, and forward execution only.
    ``schedule`` is trusted compiler output: compile it from the exact same
    ``spec`` and document IDs, then leave all metadata unchanged. Changing
    metadata or document IDs after compilation is unsupported. Structural
    validation does not verify that full/partial flags match mask semantics.
    Empty schedule rows produce zero output. CSR metadata uploads are cached.
    The ``interval`` variant supports masks without document constraints and
    computes bounds directly, without device-side CSR metadata. Both variants
    support ``use_full_blocks=False`` for a semantic-mask ablation.
    An optional ``launch_metadata`` dict receives compiler resource fields and
    assembly hashes. Omit it in timing loops to avoid metadata processing.
    No backward implementation or numerical/GPU performance results are implied.
    """
    import torch
    from .masks import MaskSpec

    if not isinstance(spec, MaskSpec):
        raise TypeError("spec must be a MaskSpec")
    if implementation not in ("csr", "interval"):
        raise ValueError("implementation must be 'csr' or 'interval'")
    if type(use_full_blocks) is not bool:
        raise TypeError("use_full_blocks must be a bool")
    if launch_metadata is not None and not isinstance(launch_metadata, dict):
        raise TypeError("launch_metadata must be a dict or None")
    if implementation == "interval" and spec.same_document:
        raise ValueError("interval implementation does not support document constraints")
    block_q, block_k, row_ptr, col_idx, full_blocks, query_blocks = _validated_schedule(schedule)

    if any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("attention_lab attention is forward-only; requires_grad inputs are unsupported")
    if q.device.type != "cuda" or any(t.device != q.device for t in (k, v)):
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.ndim != 4 or any(t.shape != q.shape for t in (k, v)):
        raise ValueError("q, k, and v must have identical [B,H,N,D] shapes")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in (k, v)):
        raise ValueError("q, k, and v must have the same FP16 or BF16 dtype")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous")
    batch, heads, length, dimension = q.shape
    if min(batch, heads, length) < 1 or dimension not in (64, 128):
        raise ValueError("positive B/H/N and head dimension 64 or 128 are required")
    if schedule.length != length:
        raise ValueError("schedule length must match the attention sequence length")
    if (isinstance(num_warps, bool) or not isinstance(num_warps, int)
            or isinstance(num_stages, bool) or not isinstance(num_stages, int)
            or num_warps not in (1, 2, 4, 8) or num_stages not in (1, 2, 3, 4)):
        raise ValueError("num_warps must be 1/2/4/8 and num_stages must be 1/2/3/4")
    for name in ("window_left", "window_right"):
        value = getattr(spec, name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")

    output = torch.empty_like(q)
    if implementation == "interval":
        right = 0 if spec.causal else spec.window_right
        kernel = _get_interval_kernel()
        compiled = kernel[(query_blocks, batch * heads)](
            q, k, v, output,
            N=length, D=dimension, BLOCK_Q=block_q, BLOCK_K=block_k,
            SCALE=dimension ** -0.5,
            HAS_LEFT=spec.window_left is not None, LEFT=spec.window_left or 0,
            HAS_RIGHT=right is not None, RIGHT=right or 0,
            USE_FULL_BLOCKS=use_full_blocks,
            num_warps=num_warps, num_stages=num_stages,
        )
        if launch_metadata is not None:
            _record_launch_metadata(compiled, launch_metadata)
        return output

    device = str(q.device)
    rows_gpu, columns_gpu, full_gpu = _upload_schedule(device, row_ptr, col_idx, full_blocks)
    if spec.same_document:
        if document_ids is None:
            raise ValueError("same_document attention requires document_ids")
        if isinstance(document_ids, torch.Tensor):
            if document_ids.ndim != 1 or document_ids.numel() != length:
                raise ValueError("document_ids must be a one-dimensional sequence of length N")
            if document_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("document_ids must contain integers")
            docs_gpu = document_ids.to(device=q.device, dtype=torch.int64).contiguous()
        else:
            documents = tuple(document_ids)
            if len(documents) != length or any(not isinstance(d, int) or isinstance(d, bool) for d in documents):
                raise ValueError("document_ids must contain N integers")
            docs_gpu = _upload_documents(device, documents)
    else:
        # Never dereferenced by a specialization without document masking.
        docs_gpu = rows_gpu

    kernel = _get_kernel()
    compiled = kernel[(query_blocks, batch * heads)](
        q, k, v, output, rows_gpu, columns_gpu, full_gpu, docs_gpu,
        N=length, D=dimension, BLOCK_Q=block_q, BLOCK_K=block_k,
        SCALE=dimension ** -0.5, CAUSAL=bool(spec.causal),
        HAS_LEFT=spec.window_left is not None, LEFT=spec.window_left or 0,
        HAS_RIGHT=spec.window_right is not None, RIGHT=spec.window_right or 0,
        SAME_DOCUMENT=bool(spec.same_document),
        USE_FULL_BLOCKS=use_full_blocks,
        num_warps=num_warps, num_stages=num_stages,
    )
    if launch_metadata is not None:
        _record_launch_metadata(compiled, launch_metadata)
    return output
