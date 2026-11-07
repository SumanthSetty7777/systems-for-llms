"""Validate interval lowering on CPU; GPU math requires CUDA tests separately."""

import unittest

from attention_lab.masks import MaskSpec, allows, compile_mask
from attention_lab.triton_attention import attention, interval_regions


class IntervalLoweringTests(unittest.TestCase):
    def test_ranges_match_validated_csr_compiler(self):
        specs = [
            MaskSpec(), MaskSpec(causal=True),
            MaskSpec(window_left=0), MaskSpec(window_right=0),
            MaskSpec(window_left=0, window_right=0),
            MaskSpec(window_left=1, window_right=7),
            MaskSpec(window_left=63, window_right=127),
            MaskSpec(causal=True, window_left=31),
            MaskSpec(causal=True, window_left=127),
            MaskSpec(causal=True, window_left=256, window_right=63),
        ]
        cases = 0
        for length in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257):
            for block_q, block_k in ((16, 16), (64, 32), (64, 64), (128, 32), (128, 64)):
                for spec in specs:
                    schedule = compile_mask(length, spec, block_q, block_k)
                    for row, (start, stop) in enumerate(zip(schedule.row_ptr, schedule.row_ptr[1:])):
                        begin, full_begin, full_end, end = interval_regions(length, spec, block_q, block_k, row)
                        self.assertEqual(list(range(begin, end)), schedule.col_idx[start:stop])
                        self.assertEqual([full_begin <= col < full_end for col in range(begin, end)],
                                         schedule.full_blocks[start:stop])
                        self.assertLessEqual(begin, full_begin)
                        self.assertLessEqual(full_begin, full_end)
                        self.assertLessEqual(full_end, end)
                    cases += 1
        self.assertEqual(cases, 550)

    def test_small_ranges_against_independent_scalar_semantics(self):
        for length in (1, 5, 11, 17):
            for block_q, block_k in ((3, 4), (8, 3), (16, 8)):
                for causal in (False, True):
                    for left in (None, 0, 2, 8):
                        for right in (None, 0, 3, 10):
                            spec = MaskSpec(causal=causal, window_left=left, window_right=right)
                            for row in range((length + block_q - 1) // block_q):
                                begin, full_begin, full_end, end = interval_regions(length, spec, block_q, block_k, row)
                                for col in range((length + block_k - 1) // block_k):
                                    values = [allows(q, k, spec)
                                              for q in range(row * block_q, min((row + 1) * block_q, length))
                                              for k in range(col * block_k, min((col + 1) * block_k, length))]
                                    self.assertEqual(begin <= col < end, any(values))
                                    self.assertEqual(full_begin <= col < full_end, all(values))

    def test_unsupported_dispatch_is_rejected_before_launch(self):
        doc_spec = MaskSpec(same_document=True)
        with self.assertRaisesRegex(ValueError, "document"):
            interval_regions(16, doc_spec, 16, 16, 0)
        with self.assertRaisesRegex(ValueError, "document"):
            attention(None, None, None, None, doc_spec, implementation="interval")
        with self.assertRaisesRegex(ValueError, "implementation"):
            attention(None, None, None, None, MaskSpec(), implementation="other")
        with self.assertRaisesRegex(TypeError, "use_full_blocks"):
            attention(None, None, None, None, MaskSpec(), use_full_blocks=1)


if __name__ == "__main__":
    unittest.main()
