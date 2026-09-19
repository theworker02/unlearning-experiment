import numpy as np
import torch

from exact_unlearning.metrics import (
    Predictions,
    capture_activations,
    membership_audit,
    prediction_reference_deltas,
    squared_mmd,
)
from exact_unlearning.models import MLPClassifier


def test_mmd_is_zero_for_identical_samples() -> None:
    values = np.arange(12, dtype=float).reshape(4, 3)
    assert squared_mmd(values, values) == 0.0


def test_mmd_detects_shift() -> None:
    left = np.zeros((6, 2))
    right = np.ones((6, 2)) * 10
    assert squared_mmd(left, right, bandwidth=1.0) > 1.9


def test_membership_audit_reports_all_predeclared_signals() -> None:
    members = np.array([[0.99, 0.01], [0.98, 0.02]])
    nonmembers = np.array([[0.51, 0.49], [0.55, 0.45]])
    audit = membership_audit(members, np.array([0, 0]), nonmembers, np.array([0, 0]))
    assert set(audit) == {"loss", "confidence", "entropy", "margin"}
    assert all(item["roc_auc"] == 1.0 for item in audit.values())


def test_reference_deltas_are_signed_and_zero_when_identical() -> None:
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])
    prediction = Predictions(probabilities, np.array([0, 1]), np.eye(2))
    assert all(
        abs(value) < 1e-12
        for value in prediction_reference_deltas(prediction, prediction, np.array([0, 1])).values()
    )


def test_activation_capture_preserves_layer_order() -> None:
    model = MLPClassifier((4,), 3, 2)
    activations = capture_activations(model, torch.ones((2, 4)))
    assert list(activations) == ["hidden", "activation", "output"]
    assert activations["hidden"].shape == (2, 3)
