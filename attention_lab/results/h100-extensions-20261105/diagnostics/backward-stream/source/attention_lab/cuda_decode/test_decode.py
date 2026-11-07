"""CPU checks for reference semantics and rejected input contracts."""

import unittest

import torch

from .kernel import prepare
from .study import reference


class DecodeContractTests(unittest.TestCase):
    def test_grouped_reference_matches_explicit_repetition(self):
        torch.manual_seed(7)
        q = torch.randn(3, 8, 64)
        k = torch.randn(3, 2, 17, 64)
        v = torch.randn_like(k)
        lengths = torch.tensor([0, 3, 17], dtype=torch.int32)
        actual = reference(q, k, v, lengths)
        allowed = torch.arange(17)[None, :] < lengths[:, None]
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2), k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1),
            attn_mask=allowed[:, None, None, :]).squeeze(2)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))

    def test_cpu_input_rejected_before_jit(self):
        q = torch.zeros(1, 8, 64, dtype=torch.float16)
        k = torch.zeros(1, 2, 17, 64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "same CUDA"):
            prepare(q, k, k)

    def test_custom_scale_reference(self):
        torch.manual_seed(9)
        q = torch.randn(2, 4, 64)
        k = torch.randn(2, 2, 19, 64)
        v = torch.randn_like(k)
        lengths = torch.tensor([1, 19], dtype=torch.int32)
        scale = 0.37
        actual = reference(q, k, v, lengths, scale=scale)
        allowed = torch.arange(19)[None, :] < lengths[:, None]
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2), k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1),
            attn_mask=allowed[:, None, None, :], scale=scale).squeeze(2)
        torch.testing.assert_close(actual, expected)

    def test_invalid_split_count_rejected(self):
        q = torch.zeros(1, 8, 64, dtype=torch.float16)
        k = torch.zeros(1, 2, 17, 64, dtype=torch.float16)
        for splits in (0, 129, True, 2.5):
            with self.subTest(splits=splits), self.assertRaisesRegex(ValueError, "splits"):
                prepare(q, k, k, splits=splits)

    def test_unknown_variant_rejected(self):
        q = torch.zeros(1, 8, 64, dtype=torch.float16)
        k = torch.zeros(1, 2, 17, 64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "variant"):
            prepare(q, k, k, variant="approximate")

    def test_invalid_head_ratio_rejected(self):
        q = torch.zeros(1, 7, 64, dtype=torch.float16)
        k = torch.zeros(1, 2, 17, 64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            prepare(q, k, k)


if __name__ == "__main__":
    unittest.main()
