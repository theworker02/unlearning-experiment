from __future__ import annotations

import numpy as np
import pytest

from exact_unlearning.equivalence_controls import (
    calibrate_excess_js_margin,
    deterministic_top2_swap,
    leave_one_out_excess_js,
)
from scripts.analyze_equivalence_margin_v2 import analyze_manifest


def _probabilities(samples: int = 200) -> np.ndarray:
    values = np.tile(np.array([0.70, 0.20, 0.10]), (samples, 1))
    values[:, 0] -= np.arange(samples) / 100000
    values[:, 1] += np.arange(samples) / 100000
    return values


def test_top2_control_is_deterministic_and_changes_exact_fraction() -> None:
    probabilities = _probabilities()
    sample_ids = np.arange(len(probabilities)) + 5000
    first, first_indices = deterministic_top2_swap(
        probabilities, sample_ids, fraction=0.01, selection_seed=91631
    )
    second, second_indices = deterministic_top2_swap(
        probabilities, sample_ids, fraction=0.01, selection_seed=91631
    )
    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first, second)
    assert len(first_indices) == 2
    changed = np.argmax(first, axis=1) != np.argmax(probabilities, axis=1)
    assert np.flatnonzero(changed).tolist() == sorted(first_indices.tolist())
    assert np.allclose(first.sum(axis=1), 1.0)


def test_top2_control_rejects_probability_ties() -> None:
    probabilities = np.tile(np.array([0.45, 0.45, 0.10]), (20, 1))
    with pytest.raises(ValueError, match="tied"):
        deterministic_top2_swap(probabilities, np.arange(20), fraction=0.1, selection_seed=1)


def test_leave_one_out_effect_has_one_observation_per_model() -> None:
    base = _probabilities()
    models = []
    for offset in (-0.01, 0.0, 0.01):
        model = base.copy()
        model[:, 0] += offset
        model[:, 1] -= offset
        models.append(model)
    effects = leave_one_out_excess_js(
        np.asarray(models), np.arange(len(base)), fraction=0.01, selection_seed=91631
    )
    assert effects.shape == (3,)
    assert np.isfinite(effects).all()


def test_margin_uses_disjoint_calibration_and_holdout_seeds() -> None:
    seed_offsets = np.linspace(0, 0.001, 25)[:, None]
    effects = seed_offsets + np.array([0.01, 0.03, 0.06])[None, :]
    result = calibrate_excess_js_margin(
        effects,
        primary_index=1,
        calibration_indices=np.arange(15),
        holdout_indices=np.arange(15, 25),
        repetitions=500,
        seed=91632,
    )
    assert result["candidate_margin"] > 0
    assert result["heldout_primary_exceedance_probability"] == 1.0
    assert result["strictly_monotone"] is True
    assert result["admissible"] is True

    with pytest.raises(ValueError, match="disjoint"):
        calibrate_excess_js_margin(
            effects,
            primary_index=1,
            calibration_indices=np.arange(15),
            holdout_indices=np.arange(14, 25),
            repetitions=10,
            seed=1,
        )


def test_stratified_control_allocates_primary_one_per_balanced_class() -> None:
    probabilities = _probabilities(1000)
    sample_ids = np.arange(1000) + 10000
    labels = np.repeat(np.arange(10), 100)
    _, primary = deterministic_top2_swap(
        probabilities,
        sample_ids,
        fraction=0.01,
        selection_seed=91631,
        strata=labels,
    )
    assert np.bincount(labels[primary], minlength=10).tolist() == [1] * 10

    _, lower = deterministic_top2_swap(
        probabilities,
        sample_ids,
        fraction=0.005,
        selection_seed=91631,
        strata=labels,
    )
    _, upper = deterministic_top2_swap(
        probabilities,
        sample_ids,
        fraction=0.02,
        selection_seed=91631,
        strata=labels,
    )
    assert len(lower) == 5
    assert max(np.bincount(labels[lower], minlength=10)) == 1
    assert np.bincount(labels[upper], minlength=10).tolist() == [2] * 10


def test_v2_analyzer_requires_hash_verified_validation_inputs(tmp_path) -> None:
    import hashlib

    import yaml

    protocol = yaml.safe_load(
        open("study/preregistration/equivalence_margin_calibration_v2.yaml", encoding="utf-8")
    )
    protocol["scope"]["initial_conditions"] = [{"condition_id": "synthetic-condition"}]
    protocol["probe"]["minimum_samples"] = 100
    protocol["margin_estimation"]["calibration_seed_count"] = 3
    protocol["margin_estimation"]["heldout_seed_count"] = 2
    protocol["margin_estimation"]["bootstrap_repetitions"] = 100
    sample_ids = np.arange(200)
    canonical_ids = __import__("json").dumps(sample_ids.tolist(), separators=(",", ":"))
    protocol["probe"]["expected_split_hash"] = hashlib.sha256(canonical_ids.encode()).hexdigest()
    base = _probabilities(200)
    models = []
    for offset in (-0.02, -0.01, 0.0, 0.01, 0.02):
        model = base.copy()
        model[:, 0] += offset
        model[:, 1] -= offset
        models.append(model)
    input_path = tmp_path / "predictions.npz"
    np.savez(
        input_path,
        probabilities=np.asarray(models),
        sample_ids=sample_ids,
        model_seeds=np.arange(1001, 1006),
    )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "publication_evidence": False,
        "split": "validation",
        "validation_split_sha256": protocol["probe"]["expected_split_hash"],
        "conditions": [
            {
                "condition_id": "synthetic-condition",
                "path": str(input_path),
                "sha256": digest,
                "calibration_seeds": [1001, 1002, 1003],
                "holdout_seeds": [1004, 1005],
            }
        ],
    }
    result = analyze_manifest(protocol, manifest)
    assert result["analysis_status"] == "VALIDATION_ONLY_CALIBRATION"
    assert result["publication_evidence"] is False
    assert len(result["conditions"]) == 1

    manifest["conditions"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze_manifest(protocol, manifest)
