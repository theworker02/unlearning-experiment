from pathlib import Path

import pytest

from scripts.run_ssd_fisher_sensitivity import _analytic_control, _conditions, main


def test_ssd_fisher_sensitivity_runner_refuses_draft_before_reservation() -> None:
    with pytest.raises(SystemExit, match="LOCKED and authorized"):
        main()
    assert not Path("results/raw/ssd-fisher-estimator-sensitivity-v1").exists()


def test_ssd_fisher_sensitivity_condition_grid_and_analytic_control() -> None:
    protocol = {
        "estimators": {
            "paper_aligned_per_example": {"loader_batch_sizes": [1, 32, 128]},
            "pinned_official_style_minibatch": {"batch_sizes": [1, 32, 128]},
        }
    }
    conditions = _conditions(protocol)
    assert [condition["condition_id"] for condition in conditions] == [
        "per-example-batch-1",
        "per-example-batch-32",
        "per-example-batch-128",
        "official-minibatch-1",
        "official-minibatch-32",
        "official-minibatch-128",
    ]
    assert _analytic_control()["passed"] is True
