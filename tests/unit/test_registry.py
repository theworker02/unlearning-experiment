from pathlib import Path

import pytest

from exact_unlearning.registry import ImmutableResultError, reserve_experiment, write_raw_result


def test_raw_results_are_immutable(tmp_path: Path) -> None:
    reserve_experiment(tmp_path, "run-1")
    path = write_raw_result(tmp_path, "run-1", {"value": 1})
    assert path.exists()
    with pytest.raises(ImmutableResultError):
        write_raw_result(tmp_path, "run-1", {"value": 2})
    assert '"value": 1' in path.read_text(encoding="utf-8")


def test_experiment_ids_are_reserved_atomically(tmp_path: Path) -> None:
    reserve_experiment(tmp_path, "reserved")
    with pytest.raises(ImmutableResultError):
        reserve_experiment(tmp_path, "reserved")
