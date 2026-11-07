"""CPU ownership checks and CUDA numerical tests for training attention."""

import unittest

from attention_lab.backward import training_attention, transpose_schedule
from attention_lab.backward_study import dense_attention, scale_aware_comparison
from attention_lab.masks import MaskSpec, compile_mask

try:
    import torch
except ImportError:
    torch = None


class OwnershipTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_small_gradients_and_update_deltas_cannot_pass_when_zero(self):
        reference = torch.tensor([0.0003, -0.0007, 0.0017, -0.0034])
        for values in (reference, reference * 0.01):
            with self.subTest(scale=float(values.norm())):
                self.assertFalse(scale_aware_comparison(torch.zeros_like(values), values)["correct"])
                self.assertFalse(scale_aware_comparison(values * 0.8, values)["correct"])
                self.assertFalse(scale_aware_comparison(-values, values)["correct"])
                self.assertTrue(scale_aware_comparison(values * 1.01, values)["correct"])

    def test_transpose_preserves_edges_flags_and_is_involution(self):
        for n in (1, 17, 65, 129):
            docs = tuple((i // 3) % 4 - 2 for i in range(n))
            for spec in (MaskSpec(), MaskSpec(causal=True),
                         MaskSpec(window_left=0, window_right=0),
                         MaskSpec(window_left=11, window_right=5),
                         MaskSpec(causal=True, same_document=True)):
                for bq, bk in ((16, 32), (64, 64), (128, 32)):
                    with self.subTest(n=n, spec=spec, bq=bq, bk=bk):
                        original = compile_mask(n, spec, bq, bk, docs)
                        reverse = transpose_schedule(original)
                        self.assertEqual(transpose_schedule(reverse), original)
                        edges = {(q, original.col_idx[e], original.full_blocks[e])
                                 for q in range(len(original.row_ptr) - 1)
                                 for e in range(original.row_ptr[q], original.row_ptr[q + 1])}
                        reverse_edges = {(reverse.col_idx[e], k, reverse.full_blocks[e])
                                         for k in range(len(reverse.row_ptr) - 1)
                                         for e in range(reverse.row_ptr[k], reverse.row_ptr[k + 1])}
                        self.assertEqual(edges, reverse_edges)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_independent_dense_oracle_scalar_gradcheck(self):
        torch.manual_seed(7)
        values = tuple(torch.randn(1, 1, 4, 3, dtype=torch.float64, requires_grad=True)
                       for _ in range(3))
        spec = MaskSpec(causal=True, window_left=2, same_document=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda q, k, v: dense_attention(q, k, v, spec, (2, -1, 2, 2)),
            values, eps=1e-6, atol=1e-5, rtol=1e-4))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class CUDABackwardTests(unittest.TestCase):
    def test_outputs_and_all_three_gradients(self):
        # Different Q/K tile sizes exercise transpose geometry; N crosses tiles.
        cases = [(17, 64, MaskSpec(), None),
                 (65, 128, MaskSpec(causal=True), None),
                 (129, 64, MaskSpec(window_left=11, window_right=5), None),
                 (65, 64, MaskSpec(window_left=0, window_right=0), None),
                 (129, 128, MaskSpec(causal=True, same_document=True),
                  tuple((i // 7) % 3 - 1 for i in range(129)))]
        for dtype in (torch.float16, torch.bfloat16):
            for n, d, spec, docs in cases:
                with self.subTest(dtype=dtype, n=n, d=d, spec=spec):
                    torch.manual_seed(n + d)
                    values = tuple(torch.randn(1, 2, n, d, device="cuda", dtype=dtype,
                                               requires_grad=True) for _ in range(3))
                    reference_values = tuple(value.detach().float().requires_grad_() for value in values)
                    upstream = torch.randn_like(values[0])
                    actual = training_attention(*values, spec, docs, block_q=32, block_k=64)
                    expected = dense_attention(*reference_values, spec, docs)
                    actual_grad = torch.autograd.grad(actual, values, upstream)
                    expected_grad = torch.autograd.grad(expected, reference_values, upstream.float())
                    atol, rtol = ((0.025, 0.035) if dtype == torch.bfloat16 else (0.004, 0.006))
                    for observed, reference in zip((actual, *actual_grad), (expected, *expected_grad)):
                        torch.testing.assert_close(observed.float(), reference, atol=atol, rtol=rtol)

    def test_broadcast_upstream_gradient(self):
        q, k, v = (torch.randn(1, 1, 17, 64, device="cuda", dtype=torch.float16,
                               requires_grad=True) for _ in range(3))
        output = training_attention(q, k, v, MaskSpec(window_left=0, window_right=0))
        grads = torch.autograd.grad(output.sum(), (q, k, v))
        torch.testing.assert_close(grads[0], torch.zeros_like(q), atol=0.004, rtol=0)
        torch.testing.assert_close(grads[1], torch.zeros_like(k), atol=0.004, rtol=0)
        torch.testing.assert_close(grads[2], torch.ones_like(v), atol=0.004, rtol=0)


if __name__ == "__main__":
    unittest.main()
