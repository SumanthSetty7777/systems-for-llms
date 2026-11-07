"""CPU semantic checks for independently constructed benchmark baselines."""

import unittest

import torch

from attention_lab.benchmark import _dense_reference_mask, _flex_block_contiguity, _flex_options, _make_flex_mask, _sdpa_mode
from attention_lab.masks import MaskSpec, allows


class BenchmarkSemanticsTests(unittest.TestCase):
    def test_flex_mask_matches_scalar_semantics_including_padded_indices(self):
        for length in (1, 17, 129):
            labels = [i % 7 for i in range(length)]
            for spec in (MaskSpec(), MaskSpec(causal=True),
                         MaskSpec(window_left=3, window_right=2),
                         MaskSpec(same_document=True),
                         MaskSpec(causal=True, window_left=11, same_document=True)):
                with self.subTest(length=length, spec=spec):
                    documents = torch.tensor(labels)
                    mask = _make_flex_mask(length, spec, documents)
                    indices = torch.arange(length + 7)
                    actual = mask(torch.tensor(0), torch.tensor(0), indices[:, None], indices[None, :])
                    expected = torch.tensor([
                        [q < length and k < length and allows(q, k, spec, labels)
                         for k in range(length + 7)] for q in range(length + 7)
                    ])
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(actual[:length, :length],
                                                _dense_reference_mask(length, spec, labels, "cpu")))

    def test_native_sdpa_only_for_equivalent_unrestricted_dense_or_causal_masks(self):
        self.assertEqual(_sdpa_mode(MaskSpec()), "native_dense")
        self.assertEqual(_sdpa_mode(MaskSpec(causal=True)), "native_causal")
        for spec in (MaskSpec(window_left=0), MaskSpec(window_right=0),
                     MaskSpec(same_document=True), MaskSpec(causal=True, window_left=127)):
            self.assertEqual(_sdpa_mode(spec), "explicit_boolean_mask")

    def test_flex_contiguous_hint_is_explicit_and_restricted_to_interval_masks(self):
        for spec in (MaskSpec(), MaskSpec(causal=True), MaskSpec(window_left=127, window_right=0)):
            self.assertEqual(_flex_options(spec), {"BACKEND": "TRITON"})
            self.assertEqual(_flex_options(spec, True),
                             {"BACKEND": "TRITON", "BLOCKS_ARE_CONTIGUOUS": True})
        self.assertEqual(_flex_options(MaskSpec(same_document=True)), {"BACKEND": "TRITON"})
        with self.assertRaises(ValueError):
            _flex_options(MaskSpec(same_document=True), True)
        with self.assertRaises(TypeError):
            _flex_options(MaskSpec(), 1)

    def test_wide_window_partial_blocks_are_not_assumed_contiguous(self):
        from torch.nn.attention.flex_attention import create_block_mask
        for left, partial_contiguous in ((127, True), (511, False)):
            mask = create_block_mask(
                _make_flex_mask(1024, MaskSpec(window_left=left, window_right=0), None),
                None, None, 1024, 1024, device="cpu", BLOCK_SIZE=128)
            actual = _flex_block_contiguity(mask)
            self.assertEqual(actual["partial_blocks_contiguous"], partial_contiguous)
            self.assertTrue(actual["full_blocks_contiguous"])


if __name__ == "__main__":
    unittest.main()
