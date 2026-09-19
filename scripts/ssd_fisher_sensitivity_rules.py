from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def conditions(protocol: dict[str, object]) -> list[dict[str, object]]:
    estimators = protocol["estimators"]
    result = [
        {
            "condition_id": f"per-example-batch-{batch_size}",
            "estimator": "paper_aligned_per_example",
            "batch_size": int(batch_size),
        }
        for batch_size in estimators["paper_aligned_per_example"]["loader_batch_sizes"]
    ]
    result.extend(
        {
            "condition_id": f"official-minibatch-{batch_size}",
            "estimator": "pinned_official_style_minibatch",
            "batch_size": int(batch_size),
        }
        for batch_size in estimators["pinned_official_style_minibatch"]["batch_sizes"]
    )
    return result


def flatten_tensors(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("tensor mapping must be non-empty")
    return torch.cat([values[name].detach().cpu().reshape(-1).double() for name in sorted(values)])


def fisher_distances(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    candidate_flat = flatten_tensors(candidate)
    reference_flat = flatten_tensors(reference)
    if candidate_flat.shape != reference_flat.shape:
        raise ValueError("Fisher tensors must have identical flattened shapes")
    difference = candidate_flat - reference_flat
    epsilon = torch.finfo(torch.float64).eps
    cosine_denominator = torch.linalg.vector_norm(candidate_flat) * torch.linalg.vector_norm(
        reference_flat
    )
    cosine = (
        float(torch.dot(candidate_flat, reference_flat) / cosine_denominator)
        if cosine_denominator > 0
        else float(candidate_flat.equal(reference_flat))
    )
    return {
        "cosine_similarity": cosine,
        "relative_l1_difference": float(
            torch.linalg.vector_norm(difference, ord=1)
            / max(torch.linalg.vector_norm(reference_flat, ord=1), epsilon)
        ),
        "relative_l2_difference": float(
            torch.linalg.vector_norm(difference)
            / max(torch.linalg.vector_norm(reference_flat), epsilon)
        ),
    }


def selected_mask(
    full_fisher: Mapping[str, torch.Tensor],
    forget_fisher: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> torch.Tensor:
    full = flatten_tensors(full_fisher)
    forget = flatten_tensors(forget_fisher)
    if full.shape != forget.shape:
        raise ValueError("full and forget Fisher tensors must have identical flattened shapes")
    return forget > alpha * full


def selected_set_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | int]:
    if candidate.dtype != torch.bool or reference.dtype != torch.bool:
        raise ValueError("selection masks must be Boolean")
    if candidate.shape != reference.shape:
        raise ValueError("selection masks must have identical shapes")
    intersection = int(torch.logical_and(candidate, reference).sum())
    union = int(torch.logical_or(candidate, reference).sum())
    return {
        "selected_count": int(candidate.sum()),
        "selected_fraction": float(candidate.double().mean()),
        "selected_set_jaccard": intersection / union if union else 1.0,
    }


def state_l2(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> float:
    if set(candidate) != set(reference):
        raise ValueError("state mappings must contain identical keys")
    squared = sum(
        float(
            (
                candidate[name].detach().cpu().double()
                - reference[name].detach().cpu().double()
            )
            .square()
            .sum()
        )
        for name in candidate
    )
    return float(np.sqrt(squared))


def prediction_sensitivity(
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    candidate = np.asarray(candidate_probabilities, dtype=float)
    reference = np.asarray(reference_probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("probability matrices must have identical two-dimensional shapes")
    if len(labels) != len(candidate):
        raise ValueError("labels must align with probability rows")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("probabilities must be finite")
    midpoint = 0.5 * (candidate + reference)
    epsilon = np.finfo(float).tiny
    candidate_safe = np.clip(candidate, epsilon, 1.0)
    reference_safe = np.clip(reference, epsilon, 1.0)
    midpoint_safe = np.clip(midpoint, epsilon, 1.0)
    js = 0.5 * np.sum(candidate_safe * np.log(candidate_safe / midpoint_safe), axis=1)
    js += 0.5 * np.sum(reference_safe * np.log(reference_safe / midpoint_safe), axis=1)
    candidate_labels = candidate.argmax(axis=1)
    reference_labels = reference.argmax(axis=1)
    return {
        "prediction_agreement": float(np.mean(candidate_labels == reference_labels)),
        "mean_jensen_shannon_divergence": float(np.mean(js)),
        "accuracy_difference": float(
            np.mean(candidate_labels == labels) - np.mean(reference_labels == labels)
        ),
    }
