"""CPU semantic checks for compiled schedules; never a GPU performance report."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from .masks import MaskSpec, compile_mask


def dense_mask(n, spec, documents):
    q = torch.arange(n)[:, None]
    k = torch.arange(n)[None, :]
    valid = torch.ones((n, n), dtype=torch.bool)
    if spec.causal:
        valid &= k <= q
    if spec.window_left is not None:
        valid &= k >= q - spec.window_left
    if spec.window_right is not None:
        valid &= k <= q + spec.window_right
    if spec.same_document:
        ids = torch.tensor(documents)
        valid &= ids[:, None] == ids[None, :]
    return valid


def scheduled_reference(q, k, v, schedule, spec, documents):
    """Differentiable CPU interpreter for block schedules, not the Triton kernel."""
    allowed = dense_mask(schedule.length, spec, documents)
    rows = []
    for r in range(len(schedule.row_ptr) - 1):
        start = r * schedule.block_q
        stop = min(start + schedule.block_q, schedule.length)
        selected = []
        masks = []
        for j in range(schedule.row_ptr[r], schedule.row_ptr[r + 1]):
            ks = schedule.col_idx[j] * schedule.block_k
            ke = min(ks + schedule.block_k, schedule.length)
            selected.extend(range(ks, ke))
            masks.append(torch.ones((stop-start, ke-ks), dtype=torch.bool)
                         if schedule.full_blocks[j] else allowed[start:stop, ks:ke])
        if not selected:
            rows.append(q[..., start:stop, :] * 0)
            continue
        scores = q[..., start:stop, :] @ k[..., selected, :].transpose(-1, -2)
        scores = scores / q.shape[-1] ** 0.5
        mask = torch.cat(masks, dim=-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        rows.append(weights @ v[..., selected, :])
    return torch.cat(rows, dim=-2)


def validate():
    torch.set_num_threads(1)
    specs = [
        MaskSpec(), MaskSpec(causal=True),
        MaskSpec(window_left=0, window_right=0),
        MaskSpec(window_left=7, window_right=3),
        MaskSpec(causal=True, window_left=31),
        MaskSpec(same_document=True),
        MaskSpec(causal=True, same_document=True),
        MaskSpec(causal=True, window_left=7, same_document=True),
    ]
    cases = 0
    for n in [1, 7, 16, 17, 31, 64, 65, 97, 128, 129, 257]:
        documents = [(i // 11) % 3 for i in range(n)]
        for bq, bk in [(16, 16), (16, 32), (32, 16), (32, 64), (64, 64)]:
            for spec in specs:
                schedule = compile_mask(n, spec, bq, bk, documents)
                expected = dense_mask(n, spec, documents)
                actual = torch.zeros_like(expected)
                for r in range(len(schedule.row_ptr) - 1):
                    qs, qe = r*bq, min((r+1)*bq, n)
                    columns = schedule.col_idx[schedule.row_ptr[r]:schedule.row_ptr[r+1]]
                    assert len(set(columns)) == len(columns), 'duplicate tile'
                    for j in range(schedule.row_ptr[r], schedule.row_ptr[r+1]):
                        ks = schedule.col_idx[j]*bk
                        ke = min(ks+bk, n)
                        assert 0 <= ks < ke <= n
                        actual[qs:qe, ks:ke] = (True if schedule.full_blocks[j]
                                                else expected[qs:qe, ks:ke])
                assert torch.equal(actual, expected), (n, bq, bk, spec)
                cases += 1
    numeric_cases, max_output_error, max_gradient_error = 0, 0.0, 0.0
    for n in [7, 33, 65]:
        documents = [i // 9 for i in range(n)]
        for spec in specs:
            generator = torch.Generator().manual_seed(2026 + n)
            q, k, v = [torch.randn(1, 2, n, 8, dtype=torch.float64,
                                   generator=generator, requires_grad=True) for _ in range(3)]
            schedule = compile_mask(n, spec, 16, 16, documents)
            mask = dense_mask(n, spec, documents)
            reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            candidate = scheduled_reference(q, k, v, schedule, spec, documents)
            torch.testing.assert_close(candidate, reference, atol=1e-10, rtol=1e-10)
            max_output_error = max(max_output_error, (candidate-reference).abs().max().item())
            grad = torch.randn(candidate.shape, dtype=torch.float64, generator=generator)
            expected_grads = torch.autograd.grad(reference, (q, k, v), grad, retain_graph=True)
            actual_grads = torch.autograd.grad(candidate, (q, k, v), grad)
            for a, b in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
                max_gradient_error = max(max_gradient_error, (a-b).abs().max().item())
            numeric_cases += 1
    example = compile_mask(4096, MaskSpec(causal=True, window_left=127), 64, 64)
    dense_tiles = 64 * 64
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
        'semantic_cases': cases, 'numerical_cases': numeric_cases,
        'numeric_precision': 'float64', 'max_output_abs_error': max_output_error,
        'max_gradient_abs_error': max_gradient_error,
        'example': {'length': 4096, 'window_left': 127, 'causal': True,
                    'block_q': 64, 'block_k': 64, 'dense_tiles': dense_tiles,
                    'scheduled_tiles': len(example.col_idx),
                    'skipped_tile_fraction': 1-len(example.col_idx)/dense_tiles},
        'scope': 'CPU schedule semantics and differentiable reference interpreter only; '
                 'Triton execution, GPU gradients and GPU performance are not validated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate()
    payload = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
