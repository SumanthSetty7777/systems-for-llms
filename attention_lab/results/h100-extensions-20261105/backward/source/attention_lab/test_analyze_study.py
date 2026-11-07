"""Synthetic arithmetic/provenance tests, not GPU performance evidence."""

import json
from pathlib import Path
import tempfile
import unittest

from attention_lab.analyze_study import _baselines, _comparisons, _regrets, analyze_directory, markdown


def record(variant, latency, seed=0, config=None):
    return {"case": {"case_id": f"{variant}-{seed}", "workload_id": "sample",
                     "variant": variant, "length": 128, "head_dim": 64, "batch": 1,
                     "heads": 2, "dtype": "bfloat16", "spec": {"causal": True},
                     "seed": seed, "config": config or {"block_q": 64, "block_k": 64, "num_warps": 4, "num_stages": 2}},
            "measurement": {"status": "passed", "correct": True, "latency_ms": latency,
                            "hardware": {"name": "synthetic"}, "software": {}, "timing_scope": "test"}}


class AnalysisTests(unittest.TestCase):
    def test_geomean_includes_regression_and_counts_failed_pair(self):
        records = [record("csr_masked", 2, 0), record("csr_full", 1, 0),
                   record("csr_masked", 1, 1), record("csr_full", 2, 1),
                   record("csr_masked", 1, 2), record("csr_full", None, 2)]
        records[-1]["measurement"] = {"status": "failed", "correct": False}
        result = _comparisons(records)[0]
        self.assertEqual(result["paired_cases"], 2)
        self.assertAlmostEqual(result["geomean_speedup"], 1)
        self.assertEqual(result["regressions"], 1)
        self.assertEqual(result["failed_groups"], 1)

    def test_environment_mismatch_is_not_paired(self):
        records = [record("csr_masked", 2), record("csr_full", 1)]
        records[1]["measurement"]["hardware"]["name"] = "other"
        result = _comparisons(records)[0]
        self.assertEqual(result["paired_cases"], 0)
        self.assertEqual(result["incompatible_environment_groups"], 1)

    def test_flex_source_cannot_cross_input_seeds(self):
        records = [record("csr_full", 1, 0), record("interval_split", 1, 1)]
        records[0]["measurement"]["flex"] = {"status": "passed", "correct": True, "latency_ms": 2}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "baseline_pairs.json").write_text(json.dumps([
                {"case_id": "interval_split-1", "baseline_source_case_id": "csr_full-0"}]))
            result = _baselines(records, path)
        item = next(row for row in result if row["candidate"].startswith("interval_split") and row["baseline"] == "flex")
        self.assertEqual(item["paired_cases"], 0)
        self.assertEqual(item["baseline_unavailable"], 1)

    def test_confirmation_pairs_distinct_configs_without_selecting_on_holdout(self):
        records = [record("interval_split", 2), record("interval_split", 1, config={"block_q": 128, "block_k": 64, "num_warps": 4, "num_stages": 2})]
        records[0]["case"]["selection"] = "fixed_default"
        records[1]["case"]["selection"] = {"policy": "fastest_correct_training_configuration", "source_workload": "training"}
        records[1]["case"]["case_id"] += "-transferred"
        result = _comparisons(records, confirmation=True)[0]
        self.assertEqual(result["paired_cases"], 1)
        self.assertEqual(result["geomean_speedup"], 2)

    def test_replay_failed_trials_remain_counted(self):
        result = _regrets(("random", 4), [0.0, 0.2, None])
        self.assertEqual(result["trials"], 3)
        self.assertEqual(result["failed_trials"], 1)
        self.assertAlmostEqual(result["mean_relative_regret"], 0.1)

    def test_flex_hints_never_pool_with_default_options(self):
        records = [record("interval_split", 1, seed) for seed in (0, 1)]
        for index, value in enumerate(records):
            value["measurement"]["flex"] = {
                "status": "passed", "correct": True, "latency_ms": 2 + index,
                "backend_requested": "TRITON", "baseline_variant": "triton_default" if index == 0 else "triton_contiguous_blocks",
                "kernel_options": {"BACKEND": "TRITON", **({"BLOCKS_ARE_CONTIGUOUS": True} if index else {})}}
        with tempfile.TemporaryDirectory() as temp:
            results = [row for row in _baselines(records, Path(temp)) if row["baseline"] == "flex"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(row["paired_cases"] == 1 for row in results))
        self.assertEqual({row["baseline_variant"] for row in results}, {"triton_default", "triton_contiguous_blocks"})

    def test_block_schema_has_separate_full_block_and_attention_ratios(self):
        value = {"case": {"case_id": "block-example", "length": 128, "head_dim": 64,
                          "pattern": "causal", "seed": 211, "config": {"block_q": 64}},
                 "measurement": {"status": "passed", "correct": True, "reference_finite": True,
                                 "inputs_weights_unchanged": True, "baseline_mask_mode": "native_causal",
                                 "variants": {name: {"status": "passed", "correct": True,
                                                      "block_latency_ms": block, "attention_latency_ms": attention}
                                              for name, block, attention in (("custom_interval_split", 1.0, 0.1),
                                                                             ("sdpa", 1.2, 0.2),
                                                                             ("flex_triton", 1.1, 0.3))}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cases").mkdir()
            (root / "cases" / "example.json").write_text(json.dumps(value))
            (root / "plan.json").write_text(json.dumps({"stage": "block", "cases": [value["case"]]}))
            result = analyze_directory(root)
        self.assertEqual(result["passed_cases"], 1)
        self.assertEqual(result["failed_cases"], 0)
        self.assertEqual(result["flex_requested"], 1)
        self.assertEqual(result["flex_failed_or_unavailable"], 0)
        self.assertEqual(len(result["block_comparisons"]), 4)
        ratios = {(row["baseline"], row["scope"]): row["geomean_speedup"] for row in result["block_comparisons"]}
        self.assertAlmostEqual(ratios[("sdpa", "single_block_forward")], 1.2)
        self.assertAlmostEqual(ratios[("sdpa", "isolated_attention_forward")], 2.0)
        self.assertIn("one inference block", markdown({"scope": "test", "studies": [result]}))


if __name__ == "__main__":
    unittest.main()
