from pathlib import Path

import pytest

from scripts.run_ssd_reproduction import _candidates, main
from scripts.ssd_rules import select_candidate


def test_ssd_runner_refuses_draft_before_reservation() -> None:
    with pytest.raises(SystemExit, match="LOCKED and authorized"):
        main()
    assert not Path("results/raw/ssd-fmnist-sanity-v1").exists()
    assert not Path("results/raw/ssd-fmnist-tuning-v1").exists()


def test_ssd_candidate_grid_matches_frozen_order() -> None:
    protocol = {
        "stage_2_validation_only_tuning": {
            "grid": {"alpha": [5.0, 10.0], "dampening_constant": [0.5, 1.0]}
        }
    }
    candidates = _candidates(protocol)
    assert candidates == [
        {
            "candidate_id": "candidate-01",
            "hyperparameters": {"alpha": 5.0, "dampening_constant": 0.5},
        },
        {
            "candidate_id": "candidate-02",
            "hyperparameters": {"alpha": 5.0, "dampening_constant": 1.0},
        },
        {
            "candidate_id": "candidate-03",
            "hyperparameters": {"alpha": 10.0, "dampening_constant": 0.5},
        },
        {
            "candidate_id": "candidate-04",
            "hyperparameters": {"alpha": 10.0, "dampening_constant": 1.0},
        },
    ]


def test_ssd_selection_applies_constraint_and_tie_breaks() -> None:
    runs = []
    for candidate_id, alpha, dampening, loss, fraction, drop in (
        ("candidate-01", 5.0, 1.0, 2.0, 0.20, 0.04),
        ("candidate-02", 10.0, 1.0, 2.0, 0.10, 0.04),
        ("candidate-03", 25.0, 0.5, 9.0, 0.01, 0.06),
    ):
        for seed in (1, 2, 3):
            runs.append(
                {
                    "candidate_id": candidate_id,
                    "hyperparameters": {
                        "alpha": alpha,
                        "dampening_constant": dampening,
                    },
                    "validation_accuracy_drop": drop,
                    "forget_training_cross_entropy": loss,
                    "selected_parameter_fraction": fraction,
                    "original_seed": seed,
                }
            )
    assert select_candidate(runs)["selected"]["candidate_id"] == "candidate-02"
