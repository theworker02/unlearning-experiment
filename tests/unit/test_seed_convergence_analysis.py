from __future__ import annotations

import numpy as np

from scripts.analyze_exact_seed_convergence import bootstrap_replica_mean, distances


def test_convergence_distances_have_known_extremes() -> None:
    left = {
        "probabilities": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "labels": np.array([0, 1]),
    }
    identical = distances(left, left)
    assert identical["js_divergence"] == 0.0
    assert identical["prediction_disagreement"] == 0.0
    opposite = {
        "probabilities": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "labels": np.array([1, 0]),
    }
    changed = distances(left, opposite)
    assert np.isclose(changed["js_divergence"], np.log(2.0))
    assert changed["prediction_disagreement"] == 1.0


def test_replica_bootstrap_is_deterministic_and_finite() -> None:
    matrix = np.array([[np.nan, 0.1, 0.2], [0.1, np.nan, 0.3], [0.2, 0.3, np.nan]])
    left = bootstrap_replica_mean(matrix, 50, np.random.default_rng(7))
    right = bootstrap_replica_mean(matrix, 50, np.random.default_rng(7))
    assert np.array_equal(left, right)
    assert np.isfinite(left).all()
