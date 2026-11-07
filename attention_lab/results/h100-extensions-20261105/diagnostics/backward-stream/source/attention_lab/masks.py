"""Compile a small attention-mask language into an exact block-sparse schedule.

The compiler targets square self-attention, but query and key tile sizes may
differ. Causal and window constraints become interval bounds: no dense N-by-N
mask is constructed. Document labels may repeat in noncontiguous positions.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass


def _integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True)
class MaskSpec:
    """Conjunctive constraints; windows count positions to either side of q.

    A missing window is unbounded. Causal attention additionally requires k <= q.
    Document labels express equality, not consecutive document boundaries.
    """

    causal: bool = False
    window_left: int | None = None
    window_right: int | None = None
    same_document: bool = False

    def __post_init__(self) -> None:
        for name in ("causal", "same_document"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("window_left", "window_right"):
            value = getattr(self, name)
            if value is not None:
                _integer(name, value, 0)


@dataclass
class BlockSchedule:
    """CSR over query tiles; full_blocks[i] describes col_idx[i].

    Columns are sorted and unique within every row. A stored block contains at
    least one permitted element. A full block needs no elementwise mask, even
    at the final (short) query/key tiles. Partial blocks still use MaskSpec.
    """

    length: int
    block_q: int
    block_k: int
    row_ptr: list[int]
    col_idx: list[int]
    full_blocks: list[bool]


def _documents(
    document_ids: Sequence[int] | None, *, length: int | None, required: bool
) -> tuple[int, ...] | None:
    if document_ids is None:
        if required:
            raise ValueError("same_document requires document_ids")
        return None
    if not isinstance(document_ids, Sequence) or isinstance(document_ids, (str, bytes)):
        raise TypeError("document_ids must be a sequence of integer labels")
    labels = tuple(document_ids)
    if length is not None and len(labels) != length:
        raise ValueError("document_ids must have exactly length elements")
    if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
        raise TypeError("document_ids must contain integer labels")
    return labels


def _spec(spec: MaskSpec) -> None:
    if not isinstance(spec, MaskSpec):
        raise TypeError("spec must be a MaskSpec")


def allows(
    q: int, k: int, spec: MaskSpec, document_ids: Sequence[int] | None = None
) -> bool:
    """Scalar mask semantics, excluding out-of-range indices.

    Without document_ids there is no sequence length to enforce; q and k must
    simply be nonnegative. With labels, both indices must fall inside them.
    Validation is deliberately explicit; this oracle is not a GPU hot path.
    """

    _spec(spec)
    _integer("q", q, 0)
    _integer("k", k, 0)
    labels = _documents(document_ids, length=None, required=spec.same_document)
    if labels is not None and (q >= len(labels) or k >= len(labels)):
        raise ValueError("q and k must be within document_ids")
    if spec.causal and k > q:
        return False
    if spec.window_left is not None and k < q - spec.window_left:
        return False
    if spec.window_right is not None and k > q + spec.window_right:
        return False
    return not spec.same_document or labels[q] == labels[k]


def compile_mask(
    length: int,
    spec: MaskSpec,
    block_q: int = 64,
    block_k: int = 64,
    document_ids: Sequence[int] | None = None,
) -> BlockSchedule:
    """Compile exact empty/full/partial tile classifications without dense masks.

    For interval masks, work is O(query_tiles + stored_blocks), independent of
    how many elements each tile contains. Document masking builds an O(N)
    positional index and tile label sets. Each candidate uses an intersection
    check costing O(min(block_q, block_k)); only geometrically partial,
    mixed-label candidates require up to block_q binary searches (O(log N)
    each). Thus no per-element tile scan or dense mask is needed. Dense output
    still requires quadratically many tile entries for fixed tile sizes.
    """

    _spec(spec)
    for name, value in (("length", length), ("block_q", block_q), ("block_k", block_k)):
        _integer(name, value, 1)
    labels = _documents(document_ids, length=length, required=spec.same_document)
    right = 0 if spec.causal else spec.window_right

    positions: dict[int, list[int]] = {}
    q_labels: list[frozenset[int]] = []
    k_labels: list[frozenset[int]] = []
    if spec.same_document:
        for index, label in enumerate(labels):
            positions.setdefault(label, []).append(index)
        q_labels = [
            frozenset(labels[start : start + block_q])
            for start in range(0, length, block_q)
        ]
        k_labels = [
            frozenset(labels[start : start + block_k])
            for start in range(0, length, block_k)
        ]

    row_ptr = [0]
    col_idx: list[int] = []
    full_blocks: list[bool] = []
    for row, q_start in enumerate(range(0, length, block_q)):
        q_last = min(q_start + block_q, length) - 1
        # Union of all row intervals in this query tile. Since each row allows
        # its own position, this union has no gaps for this mask language.
        first_key = 0 if spec.window_left is None else max(0, q_start - spec.window_left)
        last_key = length - 1 if right is None else min(length - 1, q_last + right)

        for col in range(first_key // block_k, last_key // block_k + 1):
            k_start = col * block_k
            k_last = min(k_start + block_k, length) - 1
            full = (
                (spec.window_left is None or k_start >= q_last - spec.window_left)
                and (right is None or k_last <= q_start + right)
            )
            if spec.same_document:
                q_set, k_set = q_labels[row], k_labels[col]
                if q_set.isdisjoint(k_set):
                    continue
                same_label = len(q_set) == len(k_set) == 1 and q_set == k_set
                # Shared labels prove existence immediately in a geometrically
                # full tile. Boundary tiles need a joint label/interval check.
                if not same_label and not full:
                    nonempty = False
                    for q in range(q_start, q_last + 1):
                        lower = k_start if spec.window_left is None else max(k_start, q - spec.window_left)
                        upper = k_last if right is None else min(k_last, q + right)
                        if lower > upper:
                            continue
                        candidates = positions[labels[q]]
                        index = bisect_left(candidates, lower)
                        if index < len(candidates) and candidates[index] <= upper:
                            nonempty = True
                            break
                    if not nonempty:
                        continue
                full = full and same_label
            col_idx.append(col)
            full_blocks.append(full)
        row_ptr.append(len(col_idx))

    return BlockSchedule(length, block_q, block_k, row_ptr, col_idx, full_blocks)
