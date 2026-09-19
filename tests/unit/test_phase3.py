from __future__ import annotations

import json

import numpy as np
import pytest

from exact_unlearning.phase3 import (
    Phase3Outcome,
    bootstrap_equivalence,
    bootstrap_seed_pair_mean,
    calibrate_distance_equivalence_margin,
    classify_outcome,
    minimum_detectable_separation,
    seed_count_precision_curve,
    seed_level_pair_matrix,
    separability_ratio,
    validate_v3_envelope,
    write_v3_envelope,
)
from exact_unlearning.registry import ImmutableResultError


def test_seed_pair_bootstrap_is_reproducible() -> None:
    matrix = seed_level_pair_matrix(np.array([[0.0], [1.0], [3.0], [4.0]]))
    first = bootstrap_seed_pair_mean(matrix, 200, 7)
    second = bootstrap_seed_pair_mean(matrix, 200, 7)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_separability_ratio_uses_median_and_reference_q95() -> None:
    ratio = separability_ratio(np.array([2.0, 4.0, 3.0]), np.array([1.0] * 20))
    assert ratio == 3.0


def test_bootstrap_equivalence_requires_interval_inside_bounds() -> None:
    result = bootstrap_equivalence(
        np.zeros(20), lower_bound=-0.1, upper_bound=0.1, repetitions=200, seed=2
    )
    assert result.status == "equivalent"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"assay_sensitive": False}, Phase3Outcome.ASSAY_INSENSITIVE),
        ({"functional": "not_equivalent"}, Phase3Outcome.FORGETTING_FAILURE),
        ({"privacy": "not_equivalent"}, Phase3Outcome.FUNCTIONAL_ONLY),
        ({}, Phase3Outcome.STRONG_EMPIRICAL_EQUIVALENCE),
        ({"representation": "inconclusive"}, Phase3Outcome.INCONCLUSIVE),
    ],
)
def test_phase3_decision_table(arguments: dict[str, object], expected: Phase3Outcome) -> None:
    inputs = {
        "assay_sensitive": True,
        "functional": "equivalent",
        "privacy": "equivalent",
        "representation": "equivalent",
        "utility": "equivalent",
    }
    inputs.update(arguments)
    assert classify_outcome(**inputs) is expected


def test_power_simulation_returns_frozen_candidate_ratio() -> None:
    result = minimum_detectable_separation(
        np.linspace(0.5, 1.5, 30),
        alpha=0.05,
        target_power=0.8,
        repetitions=300,
        seed=4,
        candidate_ratios=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert result["minimum_ratio"] in {2.0, 3.0, 4.0}


def test_seed_count_precision_resamples_models_and_is_reproducible() -> None:
    values = np.arange(10, dtype=float)[:, None]
    matrix = seed_level_pair_matrix(values)
    first = seed_count_precision_curve(
        matrix,
        candidate_counts=(3, 5, 10),
        repetitions=200,
        seed=73,
        maximum_relative_half_width=1.0,
    )
    second = seed_count_precision_curve(
        matrix,
        candidate_counts=(3, 5, 10),
        repetitions=200,
        seed=73,
        maximum_relative_half_width=1.0,
    )
    assert first == second
    assert first["resampling_unit"] == "independently_trained_model_seed"
    assert [row["seed_count"] for row in first["curve"]] == [3, 5, 10]
    assert first["curve"][-1]["ci_low"] == first["curve"][-1]["ci_high"]


def test_seed_count_precision_rejects_pairwise_nan_diagonal() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite non-negative"):
        seed_count_precision_curve(matrix, candidate_counts=(3,), repetitions=10, seed=1)


def test_equivalence_margin_requires_controlled_perturbation_sensitivity() -> None:
    admissible = calibrate_distance_equivalence_margin(
        np.array([0.01, 0.02, 0.03, 0.02]),
        np.array([0.10, 0.11, 0.12, 0.13]),
        reference_quantile=0.95,
        minimum_detection_probability=0.80,
    )
    insensitive = calibrate_distance_equivalence_margin(
        np.array([0.01, 0.02, 0.03, 0.02]),
        np.array([0.01, 0.02, 0.01, 0.02]),
        reference_quantile=0.95,
        minimum_detection_probability=0.80,
    )
    assert admissible["admissible"] is True
    assert insensitive["admissible"] is False
    assert admissible["lower_bound"] == 0.0


def test_equivalence_margin_rejects_invalid_distance_inputs() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        calibrate_distance_equivalence_margin(np.array([0.0, -0.1, 0.2]), np.array([0.1, 0.2, 0.3]))


def valid_envelope() -> dict[str, object]:
    return {
        "schema_version": 3,
        "parent": {},
        "experiment": {},
        "reference_ensemble": {},
        "separability": {},
        "functional": {},
        "privacy": {},
        "representation": {},
        "relearning": {},
        "utility": {},
        "statistics": {},
        "decision": {"overall_class": "U0_ASSAY_INSENSITIVE"},
        "provenance": {},
    }


def test_v3_writer_is_append_only(tmp_path) -> None:
    destination = tmp_path / "result-v3.json"
    write_v3_envelope(str(destination), valid_envelope())
    assert json.loads(destination.read_text())["schema_version"] == 3
    with pytest.raises(ImmutableResultError):
        write_v3_envelope(str(destination), valid_envelope())


def test_v3_rejects_missing_sections() -> None:
    envelope = valid_envelope()
    del envelope["privacy"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_v3_envelope(envelope)
