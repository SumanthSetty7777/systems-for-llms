"""CPU orchestration tests; simulated timings are not performance evidence."""

import unittest
from unittest.mock import patch

from attention_lab.gpu_search import run_experiment


class SearchIntegrationTests(unittest.TestCase):
    def test_incorrect_fast_candidate_cannot_win_and_winner_is_rechecked(self):
        calls = []

        def measured_case(**case):
            calls.append(case)
            index = len(calls)
            return {"status": "failed_correctness" if index == 1 else "passed",
                    "correct": index != 1, "latency_ms": 0.01 if index == 1 else 1.0,
                    "first_call_ms": 50.0, "compile_time_ms": None}

        with patch("attention_lab.gpu_search.run_case", side_effect=measured_case):
            result = run_experiment(budget=3, seed=7, length=257, head_dim=64)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["search"]["evaluations"], 3)
        self.assertEqual(result["search"]["eligible_evaluations"], 2)
        self.assertEqual([call["seed"] for call in calls], [7, 7, 7, 8])
        winner = result["search"]["best"]["config"]
        self.assertEqual({name: calls[-1][name] for name in winner}, winner)
        self.assertFalse(result["confirmation"]["counts_toward_search_budget"])
        self.assertIsNone(result["search"]["best"]["result"]["compile_ms"])

    def test_failed_fresh_input_confirmation_does_not_pass(self):
        def measured_case(**case):
            if case["seed"] == 1:
                raise RuntimeError("confirmation failed")
            return {"status": "passed", "correct": True, "latency_ms": 1.0}

        with patch("attention_lab.gpu_search.run_case", side_effect=measured_case):
            result = run_experiment(budget=1, seed=0, length=128, head_dim=64)
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result["search"]["best"])
        self.assertEqual(result["confirmation"]["measurement"]["error"], "confirmation failed")


if __name__ == "__main__":
    unittest.main()
