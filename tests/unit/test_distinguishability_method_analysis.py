from __future__ import annotations

import numpy as np
import yaml

from scripts.analyze_distinguishability_method_pilot import (
    PLAN,
    distances,
    seed_bootstrap_mean,
    smoothed_percentile,
    validate_method_grid,
    validate_only,
)


def prediction() -> dict[str, np.ndarray]:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    return {
        "ids": np.asarray([4, 8]),
        "true_labels": np.asarray([0, 1]),
        "probabilities": probabilities,
        "predicted_labels": probabilities.argmax(axis=1),
        "features": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    }


def test_identical_predictions_have_zero_distances() -> None:
    measured = distances(prediction(), prediction())
    assert measured["js_divergence"] == 0.0
    assert measured["prediction_disagreement"] == 0.0
    assert abs(measured["final_feature_cka_distance"]) < 1e-12
    assert abs(measured["final_feature_mmd"]) < 1e-12


def test_seed_bootstrap_is_deterministic_and_seed_level() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    left = seed_bootstrap_mean(values, repetitions=1000, seed=71)
    right = seed_bootstrap_mean(values, repetitions=1000, seed=71)
    assert left == right
    assert left["seed_count"] == 5
    assert left["resampling_unit"] == "original_model_seed"
    assert left["ci_low"] < left["estimate"] < left["ci_high"]


def test_smoothed_percentile_never_returns_zero() -> None:
    assert smoothed_percentile(-1.0, [0.1, 0.2, 0.3]) == 0.25


def test_validate_only_forbids_equivalence_and_confirmatory_labels() -> None:
    result = validate_only()
    assert result["equivalence_claim_permitted"] is False
    assert result["confirmatory_pass_fail_permitted"] is False


def test_method_grid_rejects_duplicate_seed_rows() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    conditions = []
    for condition in plan["conditions"]:
        method_runs = [
            {"method": method["name"], "original_seed": seed}
            for method in plan["methods"]
            for seed in plan["source_ensemble"]["original_seeds"]
        ]
        conditions.append(
            {
                "condition_id": condition["condition_id"],
                "method_runs": method_runs,
                "exact_runs": [
                    {"seed": seed} for seed in plan["source_ensemble"]["exact_seeds"]
                ],
            }
        )
    validate_method_grid({"conditions": conditions}, plan)
    conditions[0]["method_runs"].append(conditions[0]["method_runs"][0])
    try:
        validate_method_grid({"conditions": conditions}, plan)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate method-seed row was accepted")
