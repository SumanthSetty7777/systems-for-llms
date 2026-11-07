"""Bounded search over a declared attention-kernel configuration space.

The evaluator owns compilation, correctness checks, and hardware timing. Search
never changes those rules or manufactures performance measurements. This is an
evolutionary configuration selector, not an LLM or an RL-trained policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Callable, Iterable


@dataclass(frozen=True)
class KernelConfig:
    block_q: int
    block_k: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def default_configs() -> tuple[KernelConfig, ...]:
    """Return 24 candidates; GPU feasibility is still evaluator-dependent."""
    return tuple(
        KernelConfig(q, k, warps, stages)
        for q in (64, 128)
        for k in (32, 64)
        for warps in (2, 4)
        for stages in (2, 3, 4)
    )


@dataclass(frozen=True)
class CandidateResult:
    correct: bool
    latency_ms: float | None = None
    compile_ms: float | None = None
    error: str | None = None

    @property
    def eligible(self) -> bool:
        return (
            self.correct is True
            and self.error is None
            and _finite_number(self.latency_ms)
            and self.latency_ms > 0
            and (
                self.compile_ms is None
                or (_finite_number(self.compile_ms) and self.compile_ms >= 0)
            )
        )


@dataclass(frozen=True)
class Observation:
    evaluation: int
    config: KernelConfig
    result: CandidateResult
    origin: str
    parent: KernelConfig | None = None


@dataclass(frozen=True)
class SearchResult:
    method: str
    seed: int
    requested_budget: int
    unique_configurations: int
    history: tuple[Observation, ...]

    @property
    def best(self) -> Observation | None:
        valid = (item for item in self.history if item.result.eligible)
        return min(valid, key=lambda item: item.result.latency_ms, default=None)

    def to_dict(self) -> dict:
        best = self.best
        return {
            "schema_version": 1,
            "method": self.method,
            "seed": self.seed,
            "requested_budget": self.requested_budget,
            "unique_configurations": self.unique_configurations,
            "evaluations": len(self.history),
            "eligible_evaluations": sum(item.result.eligible for item in self.history),
            "best": asdict(best) if best is not None else None,
            "history": [asdict(item) for item in self.history],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize(value: object) -> CandidateResult:
    if not isinstance(value, CandidateResult):
        return CandidateResult(False, error="invalid_result: expected CandidateResult")
    errors = [str(value.error)] if value.error is not None else []
    correct = value.correct
    if type(correct) is not bool:
        correct = False
        errors.append("invalid_correctness: expected bool")

    latency = value.latency_ms
    if latency is not None and (not _finite_number(latency) or latency <= 0):
        errors.append(f"invalid_latency_ms: {latency!r}")
        latency = None
    if correct and latency is None:
        errors.append("missing_valid_latency")

    compile_ms = value.compile_ms
    if compile_ms is not None and (not _finite_number(compile_ms) or compile_ms < 0):
        errors.append(f"invalid_compile_ms: {compile_ms!r}")
        compile_ms = None
    return CandidateResult(
        correct,
        float(latency) if latency is not None else None,
        float(compile_ms) if compile_ms is not None else None,
        "; ".join(errors) if errors else None,
    )


def _distance(left: KernelConfig, right: KernelConfig) -> int:
    return sum(a != b for a, b in zip(asdict(left).values(), asdict(right).values()))


def _parents(history: list[Observation], size: int) -> list[KernelConfig]:
    """Keep the fastest elite, then retain distinct schedules for exploration."""
    valid = sorted(
        (item for item in history if item.result.eligible),
        key=lambda item: (item.result.latency_ms, item.evaluation),
    )
    if not valid:
        return []
    selected = [valid.pop(0)]
    while valid and len(selected) < size:
        choice = max(
            valid,
            key=lambda item: (
                min(_distance(item.config, parent.config) for parent in selected),
                -item.result.latency_ms,
                -item.evaluation,
            ),
        )
        selected.append(choice)
        valid.remove(choice)
    return [item.config for item in selected]


def search(
    evaluate: Callable[[KernelConfig], CandidateResult],
    configs: Iterable[KernelConfig] | None = None,
    *,
    budget: int,
    method: str = "evolution",
    seed: int = 0,
    population_size: int = 4,
    restart_probability: float = 0.2,
) -> SearchResult:
    """Evaluate at most ``budget`` unique configurations, including failures.

    Evolution mutates exactly one parameter of a numerically correct measured
    parent. Restarts avoid disconnected regions and preserve exploration. Random
    search uses the same declared candidate space and evaluation budget. An
    evaluation-count budget does not imply equal GPU time; callers should report
    compilation and total experiment time separately.

    Results are cached by configuration within one call. A fresh call intentionally
    reevaluates its candidates so independent confirmation runs remain possible.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if method not in {"evolution", "random"}:
        raise ValueError("method must be 'evolution' or 'random'")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer")
    if not _finite_number(restart_probability) or not 0 <= restart_probability <= 1:
        raise ValueError("restart_probability must be between zero and one")

    supplied = default_configs() if configs is None else tuple(configs)
    if any(not isinstance(config, KernelConfig) for config in supplied):
        raise TypeError("configs must contain KernelConfig values")
    candidates = list(dict.fromkeys(supplied))
    unseen = candidates.copy()
    history: list[Observation] = []
    cache: dict[KernelConfig, CandidateResult] = {}
    rng = random.Random(seed)

    while unseen and len(cache) < budget:
        parent = None
        origin = "random" if method == "random" else "restart"
        config = None
        if method == "evolution":
            parents = _parents(history, population_size)
            if parents and rng.random() >= restart_probability:
                rng.shuffle(parents)
                for possible_parent in parents:
                    neighbors = [item for item in unseen if _distance(item, possible_parent) == 1]
                    if neighbors:
                        config = rng.choice(neighbors)
                        parent = possible_parent
                        origin = "mutation"
                        break
        if config is None:
            config = rng.choice(unseen)
        unseen.remove(config)
        try:
            result = _normalize(evaluate(config))
        except Exception as exc:
            # Compilation/resource/verification failures spend budget and remain
            # evidence. KeyboardInterrupt and SystemExit still interrupt a run.
            result = CandidateResult(False, error=f"{type(exc).__name__}: {exc}")
        cache[config] = result
        history.append(Observation(len(history) + 1, config, result, origin, parent))

    return SearchResult(method, seed, budget, len(candidates), tuple(history))
