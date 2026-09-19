from __future__ import annotations

import hashlib

import numpy as np


def _mean_js(left: np.ndarray, right: np.ndarray) -> float:
    p = np.clip(left, 1e-12, 1.0)
    q = np.clip(right, 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    per_sample = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(
        q * np.log(q / midpoint), axis=1
    )
    return float(np.mean(per_sample))


def deterministic_top2_swap(
    probabilities: np.ndarray,
    sample_ids: np.ndarray,
    *,
    fraction: float,
    selection_seed: int,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap each selected sample's two largest probabilities using stable ID hashing."""
    values = np.asarray(probabilities, dtype=float)
    identifiers = np.asarray(sample_ids)
    if values.ndim != 2 or len(values) != len(identifiers) or len(values) < 2:
        raise ValueError("probabilities and sample_ids must describe at least two samples")
    if values.shape[1] < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite, non-negative, and multiclass")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("probability rows must sum to one")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("sample_ids must be unique")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    count = max(1, int(np.ceil(len(values) * fraction)))
    ranks = [
        hashlib.sha256(f"{selection_seed}:{identifier}".encode()).digest()
        for identifier in identifiers
    ]
    if strata is None:
        selected_values = sorted(range(len(values)), key=ranks.__getitem__)[:count]
    else:
        groups = np.asarray(strata)
        if groups.ndim != 1 or len(groups) != len(values):
            raise ValueError("strata must align one-to-one with sample IDs")
        unique_groups = sorted(set(groups.tolist()))
        members = {group: np.flatnonzero(groups == group).tolist() for group in unique_groups}
        raw_quotas = {group: len(members[group]) * count / len(values) for group in unique_groups}
        quotas = {group: int(np.floor(raw_quotas[group])) for group in unique_groups}
        remaining = count - sum(quotas.values())
        priority = sorted(
            unique_groups,
            key=lambda group: (
                -(raw_quotas[group] - quotas[group]),
                hashlib.sha256(f"{selection_seed}:stratum:{group}".encode()).digest(),
            ),
        )
        for group in priority[:remaining]:
            quotas[group] += 1
        selected_values = []
        for group in unique_groups:
            ranked = sorted(members[group], key=ranks.__getitem__)
            selected_values.extend(ranked[: quotas[group]])
        if len(selected_values) != count:
            raise AssertionError("stratified selection failed to allocate the requested count")
    selected = np.asarray(sorted(selected_values), dtype=int)
    order = np.argsort(values[selected], axis=1)
    runner_up = order[:, -2]
    top = order[:, -1]
    if np.any(values[selected, top] == values[selected, runner_up]):
        raise ValueError("selected samples contain tied top-two probabilities")
    transformed = values.copy()
    top_values = values[selected, top].copy()
    transformed[selected, top] = values[selected, runner_up]
    transformed[selected, runner_up] = top_values
    if not np.all(np.argmax(transformed[selected], axis=1) == runner_up):
        raise AssertionError("top-two control failed to change every selected prediction")
    return transformed, selected


def leave_one_out_excess_js(
    exact_probabilities: np.ndarray,
    sample_ids: np.ndarray,
    *,
    fraction: float,
    selection_seed: int,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Return one excess-distance observation per independently trained exact model."""
    models = np.asarray(exact_probabilities, dtype=float)
    if models.ndim != 3 or len(models) < 3:
        raise ValueError("exact_probabilities must contain at least three model replicas")
    excess = np.empty(len(models), dtype=float)
    for index, probabilities in enumerate(models):
        perturbed, _ = deterministic_top2_swap(
            probabilities,
            sample_ids,
            fraction=fraction,
            selection_seed=selection_seed,
            strata=strata,
        )
        references = np.delete(models, index, axis=0)
        baseline = np.median([_mean_js(probabilities, item) for item in references])
        controlled = np.median([_mean_js(perturbed, item) for item in references])
        excess[index] = controlled - baseline
    if not np.isfinite(excess).all():
        raise ValueError("controlled excess JS must be finite")
    return excess


def calibrate_excess_js_margin(
    excess_by_severity: np.ndarray,
    *,
    primary_index: int,
    calibration_indices: np.ndarray,
    holdout_indices: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
    minimum_exceedance_probability: float = 0.80,
) -> dict[str, object]:
    """Estimate a conservative margin and validate it on disjoint exact-model seeds."""
    effects = np.asarray(excess_by_severity, dtype=float)
    calibration = np.asarray(calibration_indices, dtype=int)
    holdout = np.asarray(holdout_indices, dtype=int)
    if effects.ndim != 2 or not np.isfinite(effects).all() or len(effects) < 3:
        raise ValueError("effects must be a finite seed-by-severity matrix")
    if not 0 <= primary_index < effects.shape[1] or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("invalid calibration configuration")
    if set(calibration.tolist()) & set(holdout.tolist()):
        raise ValueError("calibration and holdout seed partitions must be disjoint")
    if not len(calibration) or not len(holdout):
        raise ValueError("both seed partitions must be nonempty")
    if (
        np.any(calibration < 0)
        or np.any(holdout < 0)
        or max(*calibration, *holdout) >= len(effects)
    ):
        raise ValueError("seed partition index is out of range")
    rng = np.random.default_rng(seed)
    primary = effects[calibration, primary_index]
    bootstrap_means = np.asarray(
        [np.mean(rng.choice(primary, size=len(primary), replace=True)) for _ in range(repetitions)]
    )
    margin = float(np.quantile(bootstrap_means, 1 - confidence))
    holdout_primary = effects[holdout, primary_index]
    exceedance = float(np.mean(holdout_primary > margin))
    medians = np.median(effects[holdout], axis=0)
    monotone = bool(np.all(np.diff(medians) > 0))
    admissible = bool(
        np.isfinite(margin)
        and margin > 0
        and exceedance >= minimum_exceedance_probability
        and monotone
    )
    return {
        "candidate_margin": margin,
        "heldout_primary_exceedance_probability": exceedance,
        "minimum_exceedance_probability": minimum_exceedance_probability,
        "heldout_median_by_severity": medians.tolist(),
        "strictly_monotone": monotone,
        "calibration_seed_count": len(calibration),
        "holdout_seed_count": len(holdout),
        "admissible": admissible,
    }
