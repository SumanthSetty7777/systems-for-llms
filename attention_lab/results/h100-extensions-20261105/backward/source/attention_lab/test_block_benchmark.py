"""CPU tests of experiment selection and evidence gates, not GPU measurements."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel

from attention_lab.block_benchmark import build_block_plan, run_block_study, _custom_attention, _training_winners


class BlockStudyTests(unittest.TestCase):
    def test_custom_launch_respects_attention_api_and_keeps_tiles_in_schedule(self):
        config = {"block_q": 128, "block_k": 64, "num_warps": 4, "num_stages": 3}
        # autospec rejects unsupported kwargs, including forwarding tile sizes
        # to attention() instead of consuming them during mask compilation.
        with patch("attention_lab.triton_attention.attention", autospec=True) as launch:
            launch.return_value = sentinel.output
            output = _custom_attention(sentinel.q, sentinel.k, sentinel.v,
                                       sentinel.schedule, sentinel.spec, config)
            self.assertIs(output, sentinel.output)
            launch.assert_called_once_with(sentinel.q, sentinel.k, sentinel.v,
                sentinel.schedule, sentinel.spec, num_warps=4, num_stages=3,
                implementation="interval", use_full_blocks=True)

    def test_transfer_uses_only_predeclared_training_families(self):
        winners = {
            "tune-causal": {"config": {"block_q": 128, "block_k": 64, "num_warps": 4, "num_stages": 3},
                            "case_id": "training-causal"},
            "holdout-window": {"config": {"block_q": 128, "block_k": 32, "num_warps": 2, "num_stages": 2},
                               "case_id": "forbidden-heldout"},
        }
        plan = build_block_plan(winners)
        self.assertEqual(len(plan["cases"]), 8)
        for case in plan["cases"]:
            self.assertFalse(case["selection"]["block_results_used_for_selection"])
            if case["pattern"] == "causal":
                self.assertEqual(case["selection"]["source_case_id"], "training-causal")
                self.assertEqual(case["config"]["block_q"], 128)
            else:
                self.assertEqual(case["selection"]["policy"], "fixed_default")
                self.assertIsNone(case["selection"]["source_case_id"])

    def test_heldout_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "plan.json").write_text(json.dumps({"stage": "confirmation"}))
            with self.assertRaisesRegex(ValueError, "held-out"):
                _training_winners(temporary)

    def test_failed_baseline_prevents_success_and_results_are_checkpointed(self):
        calls = []
        def evaluate(case, **kwargs):
            calls.append(case)
            variants = {name: {"status": "passed", "correct": True, "block_latency_ms": 1.0}
                        for name in ("custom_interval_split", "sdpa", "flex_triton")}
            if len(calls) == 3:
                variants["sdpa"]["correct"] = False
            return {"status": "passed", "correct": True, "reference_finite": True,
                    "inputs_weights_unchanged": True, "variants": variants}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            summary = run_block_study(temporary, evaluate=evaluate)
            self.assertEqual(summary["status"], "completed_with_failures")
            self.assertEqual(summary["completed_cases"], 8)
            self.assertEqual(summary["passed_cases"], 7)
            self.assertEqual(summary["failed_cases"], 1)
            self.assertEqual(summary["not_run_cases"], 0)
            self.assertEqual(len(list((Path(temporary) / "cases").glob("*.json"))), 8)


if __name__ == "__main__":
    unittest.main()
