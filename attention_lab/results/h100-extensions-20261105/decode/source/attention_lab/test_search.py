"""Search-policy tests use synthetic evaluators; these are not GPU benchmarks."""

from dataclasses import FrozenInstanceError
import json
import math
import unittest

from attention_lab.search import CandidateResult, KernelConfig, default_configs, search


class SearchTests(unittest.TestCase):
    def test_default_space_has_24_unique_configs(self):
        configs = default_configs()
        self.assertEqual(len(configs), 24)
        self.assertEqual(len(set(configs)), 24)

    def test_duplicate_inputs_do_not_spend_budget(self):
        configs = default_configs()[:3]
        calls = []

        def evaluate(config):
            calls.append(config)
            return CandidateResult(True, 1.0)

        result = search(evaluate, configs + configs, budget=100)
        self.assertEqual(len(calls), 3)
        self.assertEqual(set(calls), set(configs))
        self.assertEqual(result.unique_configurations, 3)

    def test_budget_counts_failed_candidates(self):
        calls = []

        def failing(config):
            calls.append(config)
            raise RuntimeError("compiler rejected configuration")

        result = search(failing, budget=7)
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(result.history), 7)
        self.assertIsNone(result.best)
        self.assertTrue(all("RuntimeError" in item.result.error for item in result.history))

    def test_invalid_and_incorrect_candidates_never_win(self):
        configs = default_configs()[:9]
        outcomes = [
            CandidateResult(False, 0.0001),
            CandidateResult(True, float("nan")),
            CandidateResult(True, float("inf")),
            CandidateResult(True, 0.0),
            CandidateResult(True, -1.0),
            CandidateResult(True, 0.0001, error="input mutation"),
            CandidateResult(True, 0.0001, compile_ms=float("inf")),
            CandidateResult(True, None),
            CandidateResult(True, 3.0, compile_ms=2.0),
        ]
        mapping = dict(zip(configs, outcomes))
        result = search(mapping.__getitem__, configs, budget=9, method="random")
        self.assertEqual(result.best.config, configs[-1])
        payload = json.loads(result.to_json())
        self.assertEqual(payload["eligible_evaluations"], 1)
        self.assertEqual(payload["evaluations"], 9)

    def test_evaluator_return_contract_is_checked(self):
        configs = default_configs()[:3]
        outcomes = [None, CandidateResult("yes", 1.0), CandidateResult(True, True)]
        mapping = dict(zip(configs, outcomes))
        result = search(mapping.__getitem__, configs, budget=3)
        self.assertIsNone(result.best)
        self.assertTrue(all(item.result.error for item in result.history))

    def test_deterministic_evaluation_order(self):
        def objective(config):
            return CandidateResult(True, 100.0 / config.block_q + config.num_stages)

        for method in ("random", "evolution"):
            first = search(objective, budget=14, seed=42, method=method)
            second = search(objective, budget=14, seed=42, method=method)
            self.assertEqual(first, second)

    def test_mutations_change_one_parameter_and_have_valid_parents(self):
        result = search(lambda _: CandidateResult(True, 1.0), budget=24, restart_probability=0)
        known = set()
        mutations = 0
        for item in result.history:
            if item.origin == "mutation":
                mutations += 1
                self.assertIn(item.parent, known)
                left = (item.config.block_q, item.config.block_k, item.config.num_warps, item.config.num_stages)
                right = (item.parent.block_q, item.parent.block_k, item.parent.num_warps, item.parent.num_stages)
                self.assertEqual(sum(a != b for a, b in zip(left, right)), 1)
            known.add(item.config)
        self.assertGreater(mutations, 0)

    def test_disconnected_candidate_space_restarts(self):
        configs = [KernelConfig(64, 32, 2, 2), KernelConfig(128, 64, 4, 4)]
        result = search(lambda _: CandidateResult(True, 1.0), configs, budget=2, restart_probability=0)
        self.assertEqual([item.origin for item in result.history], ["restart", "restart"])

    def test_full_budget_finds_measured_minimum(self):
        target = KernelConfig(128, 64, 4, 4)

        def objective(config):
            distance = sum(a != b for a, b in zip(config.__dict__.values(), target.__dict__.values()))
            return CandidateResult(True, 1.0 + distance)

        for method in ("evolution", "random"):
            result = search(objective, budget=24, method=method)
            self.assertEqual(result.best.config, target)
            self.assertEqual(result.best.result.latency_ms, 1.0)

    def test_history_is_immutable(self):
        result = search(lambda _: CandidateResult(True, 1.0), budget=1)
        self.assertIsInstance(result.history, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.history[0].result.latency_ms = 0.1

    def test_empty_and_zero_budget_do_not_call_evaluator(self):
        def forbidden(_):
            self.fail("evaluator should not be called")

        self.assertEqual(search(forbidden, budget=0).history, ())
        self.assertEqual(search(forbidden, [], budget=20).history, ())

    def test_bad_search_parameters_are_rejected(self):
        for kwargs in ({"budget": -1}, {"budget": True}, {"budget": 1, "method": "greedy"},
                       {"budget": 1, "population_size": 0}, {"budget": 1, "restart_probability": math.nan}):
            with self.assertRaises(ValueError):
                search(lambda _: CandidateResult(True, 1.0), **kwargs)


if __name__ == "__main__":
    unittest.main()
