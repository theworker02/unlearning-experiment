from __future__ import annotations

import numpy as np

from scripts.analyze_phase3_class_conditioned_assay import distances


def prediction(probabilities: np.ndarray, features: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "probabilities": probabilities,
        "labels": probabilities.argmax(axis=1),
        "features": features,
        "ids": np.arange(len(probabilities)),
    }


def test_distances_are_zero_for_identical_artifacts() -> None:
    item = prediction(
        np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )
    measured = distances(item, item)
    assert measured["js_divergence"] == 0.0
    assert measured["prediction_disagreement"] == 0.0
    assert abs(measured["final_feature_cka_distance"]) < 1e-12
    assert abs(measured["final_feature_mmd"]) < 1e-12


def test_distances_reject_mismatched_sample_lineage() -> None:
    left = prediction(np.asarray([[0.8, 0.2]]), np.asarray([[1.0, 0.0]]))
    right = prediction(np.asarray([[0.2, 0.8]]), np.asarray([[0.0, 1.0]]))
    right["ids"] = np.asarray([9])
    try:
        distances(left, right)
    except ValueError as error:
        assert "sample IDs" in str(error)
    else:
        raise AssertionError("mismatched prediction lineage was accepted")
