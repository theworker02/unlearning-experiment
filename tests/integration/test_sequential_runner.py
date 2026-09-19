from __future__ import annotations

import numpy as np

from exact_unlearning.contracts import DeletionStage, SequentialExperimentSpec
from exact_unlearning.sequential import run_sequential_experiment


def test_runner_chains_approximate_states_and_builds_stage_matched_oracles() -> None:
    spec = SequentialExperimentSpec((0.1, 0.1, 0.1), 19, (101, 102, 103))
    exact_calls: list[tuple[int, int, np.ndarray]] = []

    def unlearn(previous: tuple[int, ...], stage: DeletionStage) -> tuple[int, ...]:
        return (*previous, stage.stage)

    def exact_retrain(retained: np.ndarray, seed: int, stage: DeletionStage) -> tuple[int, int]:
        exact_calls.append((stage.stage, seed, retained.copy()))
        assert not np.intersect1d(retained, stage.cumulative_forget_positions).size
        return stage.stage, seed

    def evaluate(
        approximate: tuple[int, ...],
        exact: tuple[tuple[int, int], ...],
        stage: DeletionStage,
    ) -> dict[str, int]:
        assert approximate == tuple(range(1, stage.stage + 1))
        assert tuple(seed for _, seed in exact) == spec.oracle_seeds
        return {"cumulative_forget_count": len(stage.cumulative_forget_positions)}

    results = run_sequential_experiment(
        100,
        spec,
        (),
        unlearn=unlearn,
        exact_retrain=exact_retrain,
        evaluate=evaluate,
    )

    assert [result.metrics["cumulative_forget_count"] for result in results] == [10, 20, 30]
    assert len(exact_calls) == len(spec.stage_fractions) * len(spec.oracle_seeds)
    for stage_number, _, retained in exact_calls:
        expected = results[stage_number - 1].stage.cumulative_retain_positions
        assert np.array_equal(retained, expected)


def test_runner_rejects_missing_exact_controls() -> None:
    spec = SequentialExperimentSpec((0.1,), 19, ())
    try:
        run_sequential_experiment(
            20,
            spec,
            object(),
            unlearn=lambda previous, stage: previous,
            exact_retrain=lambda retained, seed, stage: object(),
            evaluate=lambda approximate, exact, stage: {},
        )
    except ValueError as error:
        assert "at least one exact-retraining seed" in str(error)
    else:
        raise AssertionError("runner accepted a sequential study without exact controls")
