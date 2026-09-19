import numpy as np

from exact_unlearning.contracts import SequentialExperimentSpec
from exact_unlearning.sequential import build_random_deletion_stages


def test_sequential_stages_are_disjoint_and_cumulative() -> None:
    spec = SequentialExperimentSpec((0.1, 0.1, 0.1), 8, (1, 2, 3))
    stages = build_random_deletion_stages(100, spec)
    assert len(stages) == 3
    assert len(stages[-1].cumulative_forget_positions) == 30
    for previous, current in zip(stages, stages[1:], strict=False):
        assert not np.intersect1d(
            previous.cumulative_forget_positions, current.new_forget_positions
        ).size
        assert set(previous.cumulative_forget_positions).issubset(
            current.cumulative_forget_positions
        )
    assert not np.intersect1d(
        stages[-1].cumulative_forget_positions, stages[-1].cumulative_retain_positions
    ).size
