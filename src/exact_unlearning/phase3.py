from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

import numpy as np

from .registry import ImmutableResultError


class Phase3Outcome(StrEnum):
    ASSAY_INSENSITIVE = "U0_ASSAY_INSENSITIVE"
    FORGETTING_FAILURE = "U1_FORGETTING_FAILURE"
    FUNCTIONAL_ONLY = "U2_FUNCTIONAL_ONLY"
    STRONG_EMPIRICAL_EQUIVALENCE = "U3_STRONG_EMPIRICAL_EQUIVALENCE"
    INCONCLUSIVE = "U4_INCONCLUSIVE"


@dataclass(frozen=True)
class EquivalenceResult:
    status: str
    estimate: float
    ci_low: float
    ci_high: float
    lower_bound: float
    upper_bound: float


def seed_level_pair_matrix(values: np.ndarray) -> np.ndarray:
    """Create a symmetric seed-pair matrix without claiming pairwise independence."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or len(array) < 2:
        raise ValueError("values must be a 2-D array containing at least two seeds")
    matrix = np.zeros((len(array), len(array)), dtype=float)
    for left, right in combinations(range(len(array)), 2):
        distance = float(np.mean(np.abs(array[left] - array[right])))
        matrix[left, right] = matrix[right, left] = distance
    return matrix


def bootstrap_seed_pair_mean(
    pair_matrix: np.ndarray, repetitions: int, seed: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap models/seeds, rather than treating dependent pairs as observations."""
    matrix = np.asarray(pair_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("pair_matrix must be square with at least two seeds")
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(matrix), size=len(matrix))
        distances = [
            matrix[left, right] for left, right in combinations(sampled, 2) if left != right
        ]
        if distances:
            estimates.append(float(np.mean(distances)))
    if not estimates:
        raise ValueError("bootstrap produced no distinct seed pairs")
    alpha = (1 - confidence) / 2
    observed = matrix[np.triu_indices(len(matrix), k=1)]
    return (
        float(np.mean(observed)),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    )


def separability_ratio(original_exact: np.ndarray, exact_exact: np.ndarray) -> float:
    original = np.asarray(original_exact, dtype=float)
    reference = np.asarray(exact_exact, dtype=float)
    if not len(original) or not len(reference):
        raise ValueError("separability requires both original/exact and exact/exact distances")
    denominator = float(np.quantile(reference, 0.95))
    return float(np.median(original) / denominator) if denominator > 0 else float("inf")


def bootstrap_equivalence(
    differences: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    repetitions: int,
    seed: int,
    confidence: float = 0.90,
) -> EquivalenceResult:
    """Two-one-sided bootstrap analogue: equivalence only when the CI is inside bounds."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("differences must be a finite non-empty vector")
    if lower_bound >= upper_bound:
        raise ValueError("equivalence lower bound must be below upper bound")
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(repetitions)]
    )
    alpha = (1 - confidence) / 2
    low = float(np.quantile(estimates, alpha))
    high = float(np.quantile(estimates, 1 - alpha))
    if low >= lower_bound and high <= upper_bound:
        status = "equivalent"
    elif high < lower_bound or low > upper_bound:
        status = "not_equivalent"
    else:
        status = "inconclusive"
    return EquivalenceResult(status, float(np.mean(values)), low, high, lower_bound, upper_bound)


def classify_outcome(
    *,
    assay_sensitive: bool,
    functional: str,
    privacy: str,
    representation: str,
    utility: str,
    converged: bool = True,
) -> Phase3Outcome:
    valid = {"equivalent", "not_equivalent", "inconclusive"}
    statuses = {functional, privacy, representation, utility}
    if not statuses <= valid:
        raise ValueError("invalid equivalence status")
    if not assay_sensitive:
        return Phase3Outcome.ASSAY_INSENSITIVE
    if not converged or "inconclusive" in statuses:
        return Phase3Outcome.INCONCLUSIVE
    if functional == "not_equivalent":
        return Phase3Outcome.FORGETTING_FAILURE
    if functional == "equivalent" and (
        privacy == "not_equivalent" or representation == "not_equivalent"
    ):
        return Phase3Outcome.FUNCTIONAL_ONLY
    if statuses == {"equivalent"}:
        return Phase3Outcome.STRONG_EMPIRICAL_EQUIVALENCE
    return Phase3Outcome.INCONCLUSIVE


def minimum_detectable_separation(
    null_distances: np.ndarray,
    *,
    alpha: float,
    target_power: float,
    repetitions: int,
    seed: int,
    candidate_ratios: np.ndarray,
) -> dict[str, object]:
    """Pre-result power simulation using scaled draws from the exact/exact null."""
    null = np.asarray(null_distances, dtype=float)
    ratios = np.asarray(candidate_ratios, dtype=float)
    if len(null) < 3 or not np.isfinite(null).all() or np.any(null < 0):
        raise ValueError("null_distances require at least three finite non-negative values")
    if not 0 < alpha < 1 or not 0 < target_power < 1 or repetitions < 1:
        raise ValueError("invalid power simulation configuration")
    if not len(ratios) or np.any(ratios < 1) or np.any(np.diff(ratios) < 0):
        raise ValueError("candidate_ratios must be sorted and at least one")
    rng = np.random.default_rng(seed)
    critical = float(np.quantile(null, 1 - alpha))
    sample_size = len(null)
    curve = []
    for ratio in ratios:
        rejections = 0
        for _ in range(repetitions):
            synthetic = rng.choice(null, size=sample_size, replace=True) * ratio
            rejections += float(np.median(synthetic)) > critical
        power = rejections / repetitions
        curve.append({"separation_ratio": float(ratio), "estimated_power": power})
    eligible = [row for row in curve if row["estimated_power"] >= target_power]
    return {
        "alpha": alpha,
        "target_power": target_power,
        "critical_value": critical,
        "minimum_ratio": eligible[0]["separation_ratio"] if eligible else None,
        "curve": curve,
        "simulation_model": "bootstrap exact/exact distances scaled multiplicatively",
    }


def seed_count_precision_curve(
    pair_matrix: np.ndarray,
    *,
    candidate_counts: tuple[int, ...],
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
    maximum_relative_half_width: float = 0.20,
) -> dict[str, object]:
    """Estimate reference-summary precision while resampling whole model seeds.

    Each repetition draws distinct model indices and computes the mean of their
    induced pairwise distances. Pairwise distances are never resampled as though
    independent. The selected count is the first candidate meeting the frozen
    relative interval half-width rule.
    """
    matrix = np.asarray(pair_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 3:
        raise ValueError("pair_matrix must be square with at least three seeds")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("pair_matrix must contain finite non-negative distances")
    counts = tuple(int(value) for value in candidate_counts)
    if (
        not counts
        or any(value < 3 or value > len(matrix) for value in counts)
        or any(right <= left for left, right in zip(counts, counts[1:], strict=False))
    ):
        raise ValueError("candidate_counts must be strictly increasing and within [3, k]")
    if repetitions < 1 or not 0 < confidence < 1 or maximum_relative_half_width <= 0:
        raise ValueError("invalid precision configuration")
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    curve: list[dict[str, float | int | bool]] = []
    selected: int | None = None
    for count in counts:
        estimates = []
        for _ in range(repetitions):
            indices = rng.choice(len(matrix), size=count, replace=False)
            subset = matrix[np.ix_(indices, indices)]
            estimates.append(float(np.mean(subset[np.triu_indices(count, k=1)])))
        low, median, high = np.quantile(estimates, [alpha, 0.5, 1 - alpha])
        relative_half_width = float((high - low) / (2 * median)) if median > 0 else float("inf")
        meets_rule = relative_half_width <= maximum_relative_half_width
        curve.append(
            {
                "seed_count": count,
                "median_pair_mean": float(median),
                "ci_low": float(low),
                "ci_high": float(high),
                "relative_half_width": relative_half_width,
                "meets_precision_rule": meets_rule,
            }
        )
        if selected is None and meets_rule:
            selected = count
    return {
        "confidence": confidence,
        "maximum_relative_half_width": maximum_relative_half_width,
        "repetitions": repetitions,
        "seed": seed,
        "selected_seed_count": selected,
        "curve": curve,
        "resampling_unit": "independently_trained_model_seed",
    }


def calibrate_distance_equivalence_margin(
    exact_exact_distances: np.ndarray,
    smallest_meaningful_perturbation_distances: np.ndarray,
    *,
    reference_quantile: float = 0.95,
    minimum_detection_probability: float = 0.80,
) -> dict[str, object]:
    """Calibrate a non-negative distance margin without approximate-method outcomes.

    The candidate upper margin is a prespecified quantile of exact/exact variability.
    It is admissible only if the smallest scientifically meaningful controlled
    perturbation exceeds that margin with the required probability.
    """
    reference = np.asarray(exact_exact_distances, dtype=float)
    perturbation = np.asarray(smallest_meaningful_perturbation_distances, dtype=float)
    if (
        reference.ndim != 1
        or perturbation.ndim != 1
        or len(reference) < 3
        or len(perturbation) < 3
        or not np.isfinite(reference).all()
        or not np.isfinite(perturbation).all()
        or np.any(reference < 0)
        or np.any(perturbation < 0)
    ):
        raise ValueError("margin calibration requires finite non-negative distance vectors")
    if not 0 < reference_quantile < 1 or not 0 < minimum_detection_probability <= 1:
        raise ValueError("invalid margin calibration configuration")
    upper_margin = float(np.quantile(reference, reference_quantile))
    detection_probability = float(np.mean(perturbation > upper_margin))
    admissible = detection_probability >= minimum_detection_probability
    return {
        "lower_bound": 0.0,
        "candidate_upper_bound": upper_margin,
        "reference_quantile": reference_quantile,
        "detection_probability": detection_probability,
        "minimum_detection_probability": minimum_detection_probability,
        "admissible": admissible,
        "interpretation": (
            "candidate_margin_detects_prespecified_smallest_meaningful_perturbation"
            if admissible
            else "metric_or_margin_is_insufficiently_sensitive"
        ),
    }


def validate_v3_envelope(envelope: dict[str, object]) -> None:
    required = {
        "schema_version",
        "parent",
        "experiment",
        "reference_ensemble",
        "separability",
        "functional",
        "privacy",
        "representation",
        "relearning",
        "utility",
        "statistics",
        "decision",
        "provenance",
    }
    missing = sorted(required - envelope.keys())
    if missing:
        raise ValueError(f"schema-v3 envelope missing fields: {missing}")
    if envelope["schema_version"] != 3:
        raise ValueError("schema-v3 envelope must declare schema_version 3")
    decision = envelope["decision"]
    try:
        Phase3Outcome(decision.get("overall_class") if isinstance(decision, dict) else None)
    except ValueError as error:
        raise ValueError("schema-v3 envelope has invalid overall decision") from error


def write_v3_envelope(path: str, envelope: dict[str, object]) -> None:
    """Write a validated analysis envelope once; raw schema-v2 evidence remains untouched."""
    import json
    import os
    from pathlib import Path

    validate_v3_envelope(envelope)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ImmutableResultError(f"refusing to overwrite schema-v3 envelope: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
