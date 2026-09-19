from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from exact_unlearning.metrics import Predictions, model_discrepancy, predictive_metrics


def _load_reference():
    path = Path("validation/reference_checks/metrics_reference.py")
    spec = importlib.util.spec_from_file_location("independent_metric_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load independent reference checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    reference = _load_reference()
    rng = np.random.default_rng(70419)
    for _ in range(50):
        labels = rng.integers(0, 5, size=37)
        left_p = rng.dirichlet(np.ones(5), size=37)
        right_p = rng.dirichlet(np.ones(5), size=37)
        left_features = rng.normal(size=(37, 7))
        right_features = rng.normal(size=(37, 11))
        left = Predictions(left_p, left_p.argmax(1), left_features)
        right = Predictions(right_p, right_p.argmax(1), right_features)
        production = predictive_metrics(left, labels, 10)
        accuracy, macro_f1 = reference.reference_accuracy_f1(left_p, labels)
        nll, brier = reference.reference_nll_brier(left_p, labels)
        comparison = model_discrepancy(left, right)
        expected = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "nll": nll,
            "brier": brier,
        }
        for name, value in expected.items():
            if not np.isclose(production[name], value, atol=1e-12):
                raise AssertionError(
                    f"reference mismatch for {name}: {production[name]} != {value}"
                )
        if not np.isclose(comparison["js_divergence"], reference.reference_js(left_p, right_p)):
            raise AssertionError("reference mismatch for Jensen-Shannon divergence")
        if not np.isclose(
            comparison["total_variation"], reference.reference_total_variation(left_p, right_p)
        ):
            raise AssertionError("reference mismatch for total variation")
        expected_cka_distance = 1.0 - reference.reference_linear_cka(left_features, right_features)
        if not np.isclose(comparison["linear_cka_distance"], expected_cka_distance, atol=1e-12):
            raise AssertionError("reference mismatch for linear CKA distance")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "fixture_count": 50,
        "seed": 70419,
        "production_module": "src/exact_unlearning/metrics.py",
        "reference_module": "validation/reference_checks/metrics_reference.py",
        "metrics": [
            "accuracy",
            "macro_f1",
            "nll",
            "brier",
            "js_divergence",
            "total_variation",
            "linear_cka_distance",
        ],
    }
    Path("validation/reference_checks/report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("independent metric reference checks passed: 50 randomized fixtures")


if __name__ == "__main__":
    main()
