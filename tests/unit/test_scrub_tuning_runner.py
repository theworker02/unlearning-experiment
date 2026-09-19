from pathlib import Path

import pytest

from scripts.run_scrub_tuning import _candidates, main
from scripts.scrub_tuning_rules import select_candidate


def test_scrub_tuning_runner_refuses_draft_before_prerequisite_or_reservation() -> None:
    with pytest.raises(SystemExit, match="LOCKED and authorized"):
        main()
    assert not Path("results/raw/scrub-fmnist-tuning-v1").exists()


def test_scrub_candidate_grid_and_mechanism_constrained_selection() -> None:
    protocol = {
        "validation_only_tuning": {
            "grid": {
                "learning_rate": [0.0001],
                "alpha": [0.1, 0.5],
                "gamma": [1.0],
                "temperature": [1.0],
                "max_steps": [1],
                "steps": [3],
                "optimizer": ["adam"],
                "weight_decay": [0.0005],
                "forget_batch_size": [512],
                "retain_batch_size": [128],
                "batch_size": [128],
            }
        }
    }
    assert [item["candidate_id"] for item in _candidates(protocol)] == [
        "candidate-01",
        "candidate-02",
    ]
    runs = []
    for candidate_id, alpha, loss, valid in (
        ("candidate-01", 0.1, 9.0, False),
        ("candidate-02", 0.5, 2.0, True),
    ):
        for _seed in (1, 2, 3):
            runs.append(
                {
                    "candidate_id": candidate_id,
                    "hyperparameters": {
                        "learning_rate": 0.0001,
                        "alpha": alpha,
                        "max_steps": 1,
                    },
                    "validation_accuracy_drop": 0.01,
                    "forget_training_cross_entropy": loss,
                    "mechanism_valid": valid,
                }
            )
    assert select_candidate(runs)["selected"]["candidate_id"] == "candidate-02"
