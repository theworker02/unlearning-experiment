import importlib.util
from pathlib import Path

import numpy as np

from exact_unlearning.metrics import Predictions, model_discrepancy, predictive_metrics


def _reference():
    path = Path("validation/reference_checks/metrics_reference.py")
    spec = importlib.util.spec_from_file_location("metric_reference_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_metrics_match_independent_known_answers() -> None:
    reference = _reference()
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [0.75, 0.25], [0.25, 0.75]])
    labels = np.array([0, 1, 0, 1])
    prediction = Predictions(probabilities, probabilities.argmax(1), np.eye(4))
    metrics = predictive_metrics(prediction, labels, 4)
    accuracy, f1 = reference.reference_accuracy_f1(probabilities, labels)
    nll, brier = reference.reference_nll_brier(probabilities, labels)
    assert metrics["accuracy"] == accuracy == 1.0
    assert metrics["macro_f1"] == f1 == 1.0
    assert np.isclose(metrics["nll"], nll)
    assert np.isclose(metrics["brier"], brier)


def test_production_js_matches_independent_scipy_reference() -> None:
    reference = _reference()
    left_p = np.array([[0.9, 0.1], [0.4, 0.6]])
    right_p = np.array([[0.1, 0.9], [0.8, 0.2]])
    features = np.eye(2)
    left = Predictions(left_p, left_p.argmax(1), features)
    right = Predictions(right_p, right_p.argmax(1), features)
    actual = model_discrepancy(left, right)
    assert np.isclose(actual["js_divergence"], reference.reference_js(left_p, right_p))
    assert np.isclose(
        actual["total_variation"], reference.reference_total_variation(left_p, right_p)
    )
