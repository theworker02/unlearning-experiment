import numpy as np

from exact_unlearning.metrics import (
    Predictions,
    class_conditioned_disagreement,
    linear_cka,
    local_behavior_profile,
    loss_membership_auc,
    model_discrepancy,
    reliability_curve,
)


def test_identical_predictions_have_zero_functional_discrepancy() -> None:
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    prediction = Predictions(probabilities, np.array([0, 1]), np.eye(2))
    discrepancy = model_discrepancy(prediction, prediction)
    assert discrepancy["top1_disagreement"] == 0.0
    assert abs(discrepancy["js_divergence"]) < 1e-12
    assert abs(discrepancy["total_variation"]) < 1e-12
    assert abs(discrepancy["linear_cka_distance"]) < 1e-12


def test_linear_cka_is_invariant_to_isotropic_scaling() -> None:
    features = np.arange(24, dtype=float).reshape(6, 4)
    assert abs(linear_cka(features, features * 3.0) - 1.0) < 1e-12


def test_known_opposite_distributions_are_detected() -> None:
    left = Predictions(np.array([[0.99, 0.01]]), np.array([0]), np.array([[1.0, 0.0]]))
    right = Predictions(np.array([[0.01, 0.99]]), np.array([1]), np.array([[0.0, 1.0]]))
    discrepancy = model_discrepancy(left, right)
    assert discrepancy["top1_disagreement"] == 1.0
    assert discrepancy["js_divergence"] > 0.5
    assert discrepancy["total_variation"] > 0.9


def test_loss_membership_auc_detects_separated_confidence() -> None:
    members = np.array([[0.99, 0.01], [0.98, 0.02]])
    nonmembers = np.array([[0.55, 0.45], [0.51, 0.49]])
    assert loss_membership_auc(members, np.array([0, 0]), nonmembers, np.array([0, 0])) == 1.0


def test_local_behavior_profile_orders_probe_bins_by_forgotten_distance() -> None:
    forgotten = np.array([[0.0]])
    probes = np.arange(8, dtype=float)[:, None]
    left = np.tile(np.array([[0.9, 0.1]]), (8, 1))
    right = left.copy()
    right[:2] = np.array([0.1, 0.9])
    profile = local_behavior_profile(forgotten, probes, left, right, distance_bins=2)
    assert profile[0]["maximum_distance"] < profile[1]["minimum_distance"]
    assert profile[0]["top1_change_rate"] == 0.5
    assert profile[1]["top1_change_rate"] == 0.0
    assert profile[0]["mean_js_divergence"] > profile[1]["mean_js_divergence"]


def test_local_behavior_profile_rejects_nonprobability_rows() -> None:
    with np.testing.assert_raises_regex(ValueError, "sum to one"):
        local_behavior_profile(
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[0.8, 0.8]]),
            np.array([[0.5, 0.5]]),
            distance_bins=1,
        )


def test_class_conditioned_disagreement_preserves_hidden_class_failure() -> None:
    rates = class_conditioned_disagreement(
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
    )
    assert rates == {0: 0.0, 1: 0.5}


def test_reliability_curve_preserves_empty_bins_and_weighted_ece() -> None:
    probabilities = np.array([[0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
    labels = np.array([0, 1, 1])
    curve = reliability_curve(probabilities, labels, bins=5)
    assert len(curve) == 5
    assert sum(item["count"] for item in curve) == 3
    assert any(item["count"] == 0 and item["calibration_gap"] is None for item in curve)
    weighted_gap = sum(item["count"] * (item["calibration_gap"] or 0.0) for item in curve) / len(
        labels
    )
    from exact_unlearning.metrics import expected_calibration_error

    assert np.isclose(weighted_gap, expected_calibration_error(probabilities, labels, 5))
