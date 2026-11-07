"""CPU tests of plan separation and evidence accounting, using synthetic timings."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from attention_lab.study import build_plan, load_winners, replay_search, run_study


class StudyTests(unittest.TestCase):
    def test_plan_sizes_and_no_interval_document_dispatch(self):
        for stage, count in (("smoke", 20), ("ablation", 120), ("flex_hints", 24), ("tuning", 96), ("confirmation", 12)):
            plan = build_plan(stage)
            self.assertEqual(len(plan["cases"]), count)
            self.assertEqual(len({case["case_id"] for case in plan["cases"]}), count)
            self.assertTrue(all(not case["spec"]["same_document"] or case["implementation"] == "csr"
                                for case in plan["cases"]))
        self.assertEqual(sum(case["include_flex"] for case in build_plan("smoke")["cases"]), 2)

    def test_confirmation_transfers_without_using_heldout_measurements(self):
        winner = {"config": {"block_q": 128, "block_k": 32, "num_warps": 2, "num_stages": 3},
                  "case_id": "training-case", "latency_ms": 1.0}
        winners = {name: winner for name in ("tune-window128", "tune-window512", "tune-causal", "tune-document")}
        cases = build_plan("confirmation", winners=winners)["cases"]
        self.assertEqual(len(cases), 24)
        transferred = [case for case in cases if case["case_id"].endswith("-transferred")]
        self.assertEqual(len(transferred), 12)
        self.assertTrue(all(case["config"] == winner["config"] for case in transferred))
        self.assertTrue(all(case["seed"] in (101, 102, 103) for case in cases))
        self.assertTrue(all(case["length"] != 4096 for case in cases))

    def test_smoke_failure_stops_and_preserves_raw_evidence(self):
        def failed(**kwargs):
            return {"status": "failed_correctness", "correct": False, "latency_ms": None}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            summary = run_study(build_plan("smoke"), directory, evaluate=failed)
            self.assertEqual(summary["status"], "stopped_after_failure")
            self.assertEqual(summary["completed_cases"], 1)
            self.assertEqual(summary["not_run_cases"], 19)
            files = list((Path(directory) / "cases").glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertFalse(json.loads(files[0].read_text())["measurement"]["correct"])

    def test_tuning_replay_rejects_incomplete_tables_and_incorrect_winners(self):
        cases = build_plan("tuning")["cases"][:24]
        records = [{"case": case, "measurement": {"status": "passed", "correct": True, "latency_ms": 1 + i}}
                   for i, case in enumerate(cases)]
        records[0]["measurement"] = {"status": "failed_correctness", "correct": False, "latency_ms": 0.001}
        incomplete = replay_search(records[:-1])["workloads"][0]
        self.assertEqual(incomplete["status"], "incomplete_table")
        report = replay_search(records)["workloads"][0]
        self.assertEqual(report["oracle_latency_ms"], 2)
        self.assertEqual(len(report["trials"]), 120)
        self.assertTrue(all(trial["best_latency_ms"] is None or trial["best_latency_ms"] >= 2
                            for trial in report["trials"]))

    def test_smoke_does_not_hide_missing_flex_baseline(self):
        def missing_flex(**kwargs):
            return {"status": "passed", "correct": True, "latency_ms": 1.0,
                    "flex": {"status": "unavailable"}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            summary = run_study(build_plan("smoke"), directory, evaluate=missing_flex)
        self.assertEqual(summary["status"], "stopped_after_baseline_failure")
        self.assertEqual(summary["passed_cases"], 1)
        self.assertEqual(summary["flex_failed_cases"], 1)

    def test_flex_hints_passed_to_benchmark_and_failed_baselines_reported(self):
        seen = []
        def missing_flex(**kwargs):
            seen.append(kwargs)
            return {"status": "passed", "correct": True, "latency_ms": 1.0,
                    "flex": {"status": "failed", "correct": False}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            summary = run_study(build_plan("flex_hints"), directory, evaluate=missing_flex)
        self.assertEqual(len(seen), 24)
        self.assertTrue(all(case["flex_contiguous"] and case["include_flex"] for case in seen))
        self.assertTrue(all(case["implementation"] == "interval" and case["use_full_blocks"] for case in seen))
        self.assertTrue(all(case["spec"].causal and not case["spec"].same_document for case in seen))
        self.assertEqual({case["spec"].window_left for case in seen}, {None, 127})
        self.assertEqual(summary["status"], "completed_with_baseline_failures")
        self.assertEqual(summary["flex_failed_cases"], 24)
        self.assertEqual(summary["passed_cases"], 24)


if __name__ == "__main__":
    unittest.main()
