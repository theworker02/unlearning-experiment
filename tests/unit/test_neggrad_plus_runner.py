import pytest

from scripts.neggrad_plus_rules import select_candidate
from scripts.run_neggrad_plus_reproduction import validate_execution_plan


def test_neggrad_plus_runner_preflight_accepts_locked_protocol_without_reservation() -> None:
    plan = validate_execution_plan()
    assert plan["status"] == "VALID_LOCKED_PROTOCOL"
    assert plan["experiment_family"] == "neggrad-plus-fmnist-reproduction-v1"


def test_candidate_selection_applies_constraint_and_tie_breaks() -> None:
    runs = []
    for candidate_id, beta, epochs, loss, drop in (
        ("candidate-01", 0.95, 10, 2.0, 0.04),
        ("candidate-02", 0.99, 5, 2.0, 0.04),
        ("candidate-03", 0.999, 5, 9.0, 0.06),
    ):
        for seed in (1, 2, 3):
            runs.append(
                {"candidate_id": candidate_id, "hyperparameters": {"beta": beta, "epochs": epochs},
                 "validation_accuracy_drop": drop,
                 "target_class_validation_cross_entropy": loss, "original_seed": seed}
            )
    selected = select_candidate(runs)["selected"]
    assert selected["candidate_id"] == "candidate-02"


def test_candidate_selection_fails_when_none_meet_utility_constraint() -> None:
    runs = [{"candidate_id": "candidate-01", "hyperparameters": {"beta": 0.95, "epochs": 5},
             "validation_accuracy_drop": 0.051,
             "target_class_validation_cross_entropy": 3.0, "original_seed": seed}
            for seed in (1, 2, 3)]
    with pytest.raises(ValueError, match=r"no NegGrad\+ candidate"):
        select_candidate(runs)
