"""Dense semantic checks for the standalone structured-mask compiler."""

import itertools
import random
import unittest

try:
    from .masks import MaskSpec, allows, compile_mask
except ImportError:  # Also support unittest discovery with this directory as -s.
    from masks import MaskSpec, allows, compile_mask


def dense_oracle(q, k, spec, docs):
    """Independent definition using signed distances, not compiler intervals."""
    distance = q - k
    return (
        (not spec.causal or distance >= 0)
        and (spec.window_left is None or distance <= spec.window_left)
        and (spec.window_right is None or -distance <= spec.window_right)
        and (not spec.same_document or docs[q] == docs[k])
    )


class MaskCompilerTests(unittest.TestCase):
    def check_schedule(self, n, spec, bq, bk, docs=None):
        schedule = compile_mask(n, spec, bq, bk, docs)
        self.assertEqual((schedule.length, schedule.block_q, schedule.block_k), (n, bq, bk))
        self.assertEqual(len(schedule.row_ptr), (n + bq - 1) // bq + 1)
        self.assertEqual(schedule.row_ptr[0], 0)
        self.assertEqual(schedule.row_ptr[-1], len(schedule.col_idx))
        self.assertEqual(len(schedule.col_idx), len(schedule.full_blocks))
        expanded = [[False] * n for _ in range(n)]
        for row, q0 in enumerate(range(0, n, bq)):
            lo, hi = schedule.row_ptr[row : row + 2]
            self.assertLessEqual(lo, hi)
            columns = schedule.col_idx[lo:hi]
            self.assertEqual(columns, sorted(set(columns)))
            expected_columns = []
            for col, k0 in enumerate(range(0, n, bk)):
                expected = [
                    dense_oracle(q, k, spec, docs)
                    for q in range(q0, min(q0 + bq, n))
                    for k in range(k0, min(k0 + bk, n))
                ]
                if not any(expected):
                    continue
                expected_columns.append(col)
                entry = lo + len(expected_columns) - 1
                self.assertLess(entry, hi)
                self.assertEqual(schedule.col_idx[entry], col)
                self.assertEqual(schedule.full_blocks[entry], all(expected))
                for q in range(q0, min(q0 + bq, n)):
                    for k in range(k0, min(k0 + bk, n)):
                        expanded[q][k] = schedule.full_blocks[entry] or allows(q, k, spec, docs)
            self.assertEqual(columns, expected_columns)
        self.assertEqual(
            expanded,
            [[dense_oracle(q, k, spec, docs) for k in range(n)] for q in range(n)],
        )

    def test_known_causal_schedule_with_short_final_tile(self):
        result = compile_mask(5, MaskSpec(causal=True), 2, 2)
        self.assertEqual(result.row_ptr, [0, 1, 3, 6])
        self.assertEqual(result.col_idx, [0, 0, 1, 0, 1, 2])
        self.assertEqual(result.full_blocks, [False, True, False, True, True, True])

    def test_dense_and_composed_windows_at_irregular_boundaries(self):
        specs = [
            MaskSpec(), MaskSpec(causal=True),
            MaskSpec(window_left=0), MaskSpec(window_right=0),
            MaskSpec(window_left=0, window_right=0),
            MaskSpec(window_left=2, window_right=1),
            MaskSpec(causal=True, window_left=3, window_right=100),
            MaskSpec(window_left=100, window_right=100),
        ]
        for n, spec, (bq, bk) in itertools.product(
            [1, 3, 7, 17, 65], specs, [(1, 1), (2, 3), (5, 2), (8, 16), (64, 64), (128, 64)]
        ):
            with self.subTest(n=n, spec=spec, bq=bq, bk=bk):
                self.check_schedule(n, spec, bq, bk)

    def test_document_labels_need_not_be_contiguous(self):
        docs = [3, -2, 3, 5, -2, 5, 3, 8, 8, -2, 5]
        for causal, left, right, (bq, bk) in itertools.product(
            [False, True], [None, 0, 2], [None, 0, 3], [(1, 3), (3, 2), (4, 8), (16, 16)]
        ):
            spec = MaskSpec(causal, left, right, same_document=True)
            with self.subTest(spec=spec, bq=bq, bk=bk):
                self.check_schedule(len(docs), spec, bq, bk, docs)

    def test_seeded_mixed_documents_and_compositions(self):
        rng = random.Random(1701)
        for _ in range(100):
            n = rng.randint(1, 35)
            docs = [rng.randrange(4) for _ in range(n)]
            spec = MaskSpec(
                causal=rng.choice([False, True]),
                window_left=rng.choice([None, 0, 1, 7, 40]),
                window_right=rng.choice([None, 0, 2, 9]),
                same_document=True,
            )
            self.check_schedule(n, spec, rng.randint(1, 12), rng.randint(1, 13), docs)

    def test_homogeneous_and_unique_document_extremes(self):
        for docs in ([1] * 17, list(range(17))):
            self.check_schedule(17, MaskSpec(same_document=True), 4, 7, docs)

    def test_narrow_window_prunes_large_geometry_without_dense_mask(self):
        n = 100_000
        result = compile_mask(n, MaskSpec(window_left=0, window_right=0), 64, 64)
        tiles = (n + 63) // 64
        self.assertEqual(result.col_idx, list(range(tiles)))
        self.assertEqual(result.row_ptr, list(range(tiles + 1)))
        self.assertFalse(any(result.full_blocks))

    def test_spec_validation(self):
        for name in ("window_left", "window_right"):
            for bad in (True, 1.5, "2"):
                with self.subTest(name=name, bad=bad), self.assertRaises(TypeError):
                    MaskSpec(**{name: bad})
            with self.assertRaises(ValueError):
                MaskSpec(**{name: -1})
        for name in ("causal", "same_document"):
            with self.assertRaises(TypeError):
                MaskSpec(**{name: 1})

    def test_compile_validation(self):
        for name in ("length", "block_q", "block_k"):
            for bad in (0, -1, False, 2.5):
                args = dict(length=5, spec=MaskSpec(), block_q=2, block_k=2)
                args[name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises((TypeError, ValueError)):
                    compile_mask(**args)
        with self.assertRaises(TypeError):
            compile_mask(3, {"causal": True})
        with self.assertRaises(ValueError):
            compile_mask(3, MaskSpec(same_document=True))
        for docs in ([1, 2], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                compile_mask(3, MaskSpec(), document_ids=docs)
        for docs in ("123", iter([1, 2, 3]), [1, True, 3], [1, 2.0, 3]):
            with self.assertRaises(TypeError):
                compile_mask(3, MaskSpec(same_document=True), document_ids=docs)

    def test_scalar_oracle_validation_and_document_semantics(self):
        self.assertTrue(allows(2, 0, MaskSpec(causal=True)))
        self.assertFalse(allows(0, 2, MaskSpec(causal=True)))
        self.assertTrue(allows(0, 2, MaskSpec(same_document=True), [7, 8, 7]))
        self.assertFalse(allows(0, 1, MaskSpec(same_document=True), [7, 8, 7]))
        for q, k in ((-1, 0), (0, -1), (3, 0), (0, 3)):
            with self.assertRaises(ValueError):
                allows(q, k, MaskSpec(), [1, 1, 1])
        with self.assertRaises(TypeError):
            allows(True, 0, MaskSpec())
        with self.assertRaises(ValueError):
            allows(0, 0, MaskSpec(same_document=True))


if __name__ == "__main__":
    unittest.main()
