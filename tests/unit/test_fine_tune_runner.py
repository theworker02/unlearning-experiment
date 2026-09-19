from pathlib import Path

import pytest

from scripts.fine_tune_rules import select_candidate
from scripts.run_fine_tune_tuning import _candidates, main


def test_fine_tune_runner_refuses_draft_before_reservation() -> None:
    with pytest.raises(SystemExit, match="LOCKED and authorized"):
        main()
    assert not Path("results/raw/fine-tune-fmnist-tuning-v1").exists()


def test_fine_tune_candidate_grid_and_selection_rule() -> None:
    protocol = {
        "validation_only_tuning": {
            "grid": {
                "epochs": [1, 3],
                "learning_rate": [0.0001],
                "batch_size": [128],
                "optimizer": ["adamw"],
                "weight_decay": [0.0],
            }
        }
    }
    candidates = _candidates(protocol)
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "candidate-01",
        "candidate-02",
    ]
    runs = []
    for candidate_id, epochs, loss, drop in (
        ("candidate-01", 1, 2.0, 0.04),
        ("candidate-02", 3, 2.0, 0.04),
        ("candidate-03", 5, 9.0, 0.06),
    ):
        for _seed in (1, 2, 3):
            runs.append(
                {
                    "candidate_id": candidate_id,
                    "hyperparameters": {"epochs": epochs, "learning_rate": 0.0001},
                    "validation_accuracy_drop": drop,
                    "forget_training_cross_entropy": loss,
                }
            )
    assert select_candidate(runs)["selected"]["candidate_id"] == "candidate-01"
