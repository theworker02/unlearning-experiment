import numpy as np
import pytest
import torch

from scripts.ssd_fisher_sensitivity_rules import (
    fisher_distances,
    prediction_sensitivity,
    selected_mask,
    selected_set_metrics,
    state_l2,
)


def test_fisher_and_selection_metrics_are_recomputable_on_known_case() -> None:
    reference = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0, 0.0])}
    candidate = {"a": torch.tensor([2.0, 0.0]), "b": torch.tensor([2.0])}
    distances = fisher_distances(candidate, reference)
    assert distances["relative_l1_difference"] == pytest.approx(1 / 3)
    assert distances["relative_l2_difference"] == pytest.approx(1 / np.sqrt(5))
    mask = selected_mask(reference, candidate, alpha=1.5)
    metrics = selected_set_metrics(mask, torch.tensor([False, False, False]))
    assert metrics == {
        "selected_count": 1,
        "selected_fraction": pytest.approx(1 / 3),
        "selected_set_jaccard": 0.0,
    }


def test_state_and_prediction_sensitivity_known_cases() -> None:
    assert state_l2({"w": torch.tensor([3.0, 4.0])}, {"w": torch.zeros(2)}) == 5.0
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    metrics = prediction_sensitivity(probabilities, probabilities, np.array([0, 1]))
    assert metrics["prediction_agreement"] == 1.0
    assert metrics["mean_jensen_shannon_divergence"] == pytest.approx(0.0)
    assert metrics["accuracy_difference"] == 0.0
