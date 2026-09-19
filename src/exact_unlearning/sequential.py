from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import DeletionStage, SequentialExperimentSpec


@dataclass(frozen=True)
class SequentialStageResult:
    """One approximate state and its independently retrained, stage-matched controls."""

    stage: DeletionStage
    approximate_state: Any
    exact_states: tuple[Any, ...]
    metrics: Mapping[str, Any]


def build_random_deletion_stages(
    training_count: int, spec: SequentialExperimentSpec
) -> list[DeletionStage]:
    """Build disjoint requests and cumulative retained sets for stage-matched oracles."""
    if training_count < 1:
        raise ValueError("training_count must be positive")
    if any(fraction <= 0 or fraction >= 1 for fraction in spec.stage_fractions):
        raise ValueError("each stage fraction must be in (0, 1)")
    rng = np.random.default_rng(spec.request_seed)
    available = np.arange(training_count, dtype=np.int64)
    cumulative = np.array([], dtype=np.int64)
    stages: list[DeletionStage] = []
    for stage_number, fraction in enumerate(spec.stage_fractions, start=1):
        count = max(1, int(round(training_count * fraction)))
        if count >= len(available):
            raise ValueError("sequential request would exhaust retained data")
        request = np.sort(rng.choice(available, size=count, replace=False))
        cumulative = np.union1d(cumulative, request)
        retained = np.setdiff1d(np.arange(training_count), cumulative, assume_unique=True)
        if np.intersect1d(cumulative, retained).size:
            raise AssertionError("cumulative forgotten and retained sets overlap")
        stages.append(DeletionStage(stage_number, request, cumulative.copy(), retained))
        available = retained
    return stages


def run_sequential_experiment(
    training_count: int,
    spec: SequentialExperimentSpec,
    original_state: Any,
    *,
    unlearn: Callable[[Any, DeletionStage], Any],
    exact_retrain: Callable[[np.ndarray, int, DeletionStage], Any],
    evaluate: Callable[[Any, tuple[Any, ...], DeletionStage], Mapping[str, Any]],
) -> list[SequentialStageResult]:
    """Execute chained deletion states with a fresh exact ensemble at every stage.

    The callbacks keep the orchestration independent of dataset modality and model
    architecture. Exact controls receive only the cumulative retained positions; the
    approximate callback receives the preceding approximate state and current request.
    """
    if not spec.oracle_seeds:
        raise ValueError("sequential experiments require at least one exact-retraining seed")
    stages = build_random_deletion_stages(training_count, spec)
    previous_state = original_state
    results: list[SequentialStageResult] = []
    for stage in stages:
        if np.intersect1d(
            stage.cumulative_forget_positions, stage.cumulative_retain_positions
        ).size:
            raise AssertionError("deleted positions reached a stage-matched exact dataset")
        approximate_state = unlearn(previous_state, stage)
        exact_states = tuple(
            exact_retrain(stage.cumulative_retain_positions.copy(), seed, stage)
            for seed in spec.oracle_seeds
        )
        metrics = dict(evaluate(approximate_state, exact_states, stage))
        results.append(SequentialStageResult(stage, approximate_state, exact_states, metrics))
        previous_state = approximate_state
    return results
