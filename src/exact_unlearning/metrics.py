from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .models import Classifier


@dataclass(frozen=True)
class Predictions:
    probabilities: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    logits: np.ndarray | None = None


@torch.inference_mode()
def predict(model: Classifier, x: torch.Tensor) -> Predictions:
    model.eval()
    device = next(model.parameters()).device
    device_x = x.to(device)
    logits = model(device_x)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return Predictions(
        probabilities=probabilities,
        labels=probabilities.argmax(axis=1),
        features=model.features(device_x).cpu().numpy(),
        logits=logits.cpu().numpy(),
    )


def expected_calibration_error(probabilities: np.ndarray, y: np.ndarray, bins: int) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if upper_inclusive else confidence < edges[index + 1]
        )
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def reliability_curve(
    probabilities: np.ndarray, y: np.ndarray, bins: int
) -> list[dict[str, float | int | None]]:
    """Fixed-bin reliability data, including explicit empty bins."""
    values = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y, dtype=int)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels) or not len(labels):
        raise ValueError("reliability curve requires aligned non-empty probabilities and labels")
    if bins < 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("invalid reliability curve configuration")
    if not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("probability rows must sum to one")
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[dict[str, float | int | None]] = []
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if upper_inclusive else confidence < edges[index + 1]
        )
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        result.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "calibration_gap": abs(accuracy - mean_confidence) if count else None,
            }
        )
    return result


def predictive_metrics(prediction: Predictions, y: np.ndarray, bins: int) -> dict[str, float]:
    probabilities = np.clip(prediction.probabilities, 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1])[y]
    return {
        "accuracy": float(accuracy_score(y, prediction.labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction.labels)),
        "macro_f1": float(f1_score(y, prediction.labels, average="macro", zero_division=0)),
        "nll": float(-np.log(probabilities[np.arange(len(y)), y]).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(probabilities, y, bins),
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        raise ValueError("CKA inputs must contain the same observations")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    return float(cross / denominator) if denominator else float("nan")


def squared_mmd(left: np.ndarray, right: np.ndarray, bandwidth: float | None = None) -> float:
    """Biased Gaussian-kernel MMD²; stable and non-negative for small audit sets."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must be 2-D with equal feature dimensions")
    combined = np.concatenate([left, right])
    distances = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=2)
    positive = distances[distances > 0]
    scale = bandwidth if bandwidth is not None else float(np.sqrt(np.median(positive) / 2))
    if scale <= 0 or not np.isfinite(scale):
        return 0.0
    gamma = 1 / (2 * scale**2)
    xx = np.exp(-gamma * np.sum((left[:, None] - left[None, :]) ** 2, axis=2))
    yy = np.exp(-gamma * np.sum((right[:, None] - right[None, :]) ** 2, axis=2))
    xy = np.exp(-gamma * np.sum((left[:, None] - right[None, :]) ** 2, axis=2))
    return float(max(0.0, xx.mean() + yy.mean() - 2 * xy.mean()))


@torch.inference_mode()
def capture_activations(model: Classifier, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Capture leaf-module activations in deterministic module order."""
    activations: dict[str, np.ndarray] = {}
    hooks = []
    for name, module in model.named_modules():
        if name and not any(module.children()):
            hooks.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, layer=name: activations.__setitem__(
                        layer, output.detach().flatten(start_dim=1).cpu().numpy()
                    )
                )
            )
    try:
        model.eval()
        model(x.to(next(model.parameters()).device))
    finally:
        for hook in hooks:
            hook.remove()
    return activations


def model_discrepancy(
    candidate: Predictions,
    oracle: Predictions,
    y: np.ndarray | None = None,
    bins: int = 10,
) -> dict[str, float]:
    p = np.clip(candidate.probabilities, 1e-12, 1.0)
    q = np.clip(oracle.probabilities, 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(
        q * np.log(q / midpoint), axis=1
    )
    result = {
        "top1_disagreement": float(np.mean(candidate.labels != oracle.labels)),
        "js_divergence": float(js.mean()),
        "total_variation": float((0.5 * np.abs(p - q).sum(axis=1)).mean()),
        "linear_cka_distance": float(1.0 - linear_cka(candidate.features, oracle.features)),
    }
    if y is not None:
        candidate_metrics = predictive_metrics(candidate, y, bins)
        oracle_metrics = predictive_metrics(oracle, y, bins)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier", "ece"):
            result[f"absolute_{metric}_difference"] = abs(
                candidate_metrics[metric] - oracle_metrics[metric]
            )
    return result


def parameter_distance(candidate: Classifier, oracle: Classifier) -> dict[str, float]:
    left = torch.cat([parameter.detach().flatten() for parameter in candidate.parameters()])
    right = torch.cat([parameter.detach().flatten() for parameter in oracle.parameters()])
    if left.shape != right.shape:
        return {"parameter_l2": float("nan"), "parameter_cosine_distance": float("nan")}
    return {
        "parameter_l2": float(torch.linalg.vector_norm(left - right)),
        "parameter_cosine_distance": float(
            1.0 - torch.nn.functional.cosine_similarity(left, right, dim=0)
        ),
    }


def loss_membership_auc(
    member_probabilities: np.ndarray,
    member_y: np.ndarray,
    nonmember_probabilities: np.ndarray,
    nonmember_y: np.ndarray,
) -> float:
    member = -np.log(np.clip(member_probabilities[np.arange(len(member_y)), member_y], 1e-12, 1.0))
    nonmember = -np.log(
        np.clip(nonmember_probabilities[np.arange(len(nonmember_y)), nonmember_y], 1e-12, 1.0)
    )
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    scores = -np.concatenate([member, nonmember])
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else math.nan


def membership_scores(probabilities: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Established black-box signals; higher scores mean more member-like."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    labels = np.asarray(y, dtype=int)
    true_confidence = probabilities[np.arange(len(labels)), labels]
    sorted_probabilities = np.sort(probabilities, axis=1)
    return {
        "loss": np.log(true_confidence),
        "confidence": true_confidence,
        "entropy": np.sum(probabilities * np.log(probabilities), axis=1),
        "margin": sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
    }


def membership_audit(
    member_probabilities: np.ndarray,
    member_y: np.ndarray,
    nonmember_probabilities: np.ndarray,
    nonmember_y: np.ndarray,
    *,
    low_fpr: float = 0.01,
) -> dict[str, dict[str, float]]:
    if not 0 < low_fpr < 1:
        raise ValueError("low_fpr must be between zero and one")
    member = membership_scores(member_probabilities, member_y)
    nonmember = membership_scores(nonmember_probabilities, nonmember_y)
    labels = np.concatenate([np.ones(len(member_y)), np.zeros(len(nonmember_y))])
    result: dict[str, dict[str, float]] = {}
    for name in member:
        scores = np.concatenate([member[name], nonmember[name]])
        auc = float(roc_auc_score(labels, scores))
        nonmember_threshold = float(np.quantile(nonmember[name], 1 - low_fpr, method="higher"))
        tpr = float(np.mean(member[name] >= nonmember_threshold))
        result[name] = {"roc_auc": auc, "attack_advantage": 2 * auc - 1, "tpr_at_fpr": tpr}
    return result


def prediction_reference_deltas(
    candidate: Predictions, oracle: Predictions, y: np.ndarray, bins: int = 10
) -> dict[str, float]:
    candidate_metrics = predictive_metrics(candidate, y, bins)
    oracle_metrics = predictive_metrics(oracle, y, bins)
    candidate_confidence = candidate.probabilities.max(axis=1).mean()
    oracle_confidence = oracle.probabilities.max(axis=1).mean()
    candidate_sorted = np.sort(candidate.probabilities, axis=1)
    oracle_sorted = np.sort(oracle.probabilities, axis=1)
    return {
        "accuracy_delta": candidate_metrics["accuracy"] - oracle_metrics["accuracy"],
        "nll_delta": candidate_metrics["nll"] - oracle_metrics["nll"],
        "brier_delta": candidate_metrics["brier"] - oracle_metrics["brier"],
        "ece_delta": candidate_metrics["ece"] - oracle_metrics["ece"],
        "confidence_delta": float(candidate_confidence - oracle_confidence),
        "margin_delta": float(
            np.mean(candidate_sorted[:, -1] - candidate_sorted[:, -2])
            - np.mean(oracle_sorted[:, -1] - oracle_sorted[:, -2])
        ),
    }


def local_behavior_profile(
    forgotten_features: np.ndarray,
    probe_features: np.ndarray,
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
    *,
    distance_bins: int = 4,
) -> list[dict[str, float | int]]:
    """Summarize model discrepancy by distance from the nearest forgotten example.

    Bins contain contiguous, deterministically sorted probe observations with near-equal
    counts. The function operates on frozen features supplied by the caller; selecting a
    representation after inspecting unlearning outcomes would invalidate the analysis.
    """
    forgotten = np.asarray(forgotten_features, dtype=float)
    probes = np.asarray(probe_features, dtype=float)
    left = np.asarray(left_probabilities, dtype=float)
    right = np.asarray(right_probabilities, dtype=float)
    if (
        forgotten.ndim != 2
        or probes.ndim != 2
        or forgotten.shape[1] != probes.shape[1]
        or len(forgotten) == 0
        or len(probes) == 0
    ):
        raise ValueError("local profile requires non-empty 2-D features of equal dimension")
    if (
        left.shape != right.shape
        or left.ndim != 2
        or len(left) != len(probes)
        or distance_bins < 1
        or distance_bins > len(probes)
    ):
        raise ValueError("invalid local profile probabilities or bin count")
    if not all(np.isfinite(value).all() for value in (forgotten, probes, left, right)):
        raise ValueError("local profile inputs must be finite")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("probabilities must be non-negative")
    left_sums = left.sum(axis=1)
    right_sums = right.sum(axis=1)
    if not np.allclose(left_sums, 1.0) or not np.allclose(right_sums, 1.0):
        raise ValueError("probability rows must sum to one")

    nearest_distance = np.sqrt(
        np.min(np.sum((probes[:, None, :] - forgotten[None, :, :]) ** 2, axis=2), axis=1)
    )
    midpoint = np.clip(0.5 * (left + right), 1e-12, 1.0)
    safe_left = np.clip(left, 1e-12, 1.0)
    safe_right = np.clip(right, 1e-12, 1.0)
    js = 0.5 * np.sum(safe_left * np.log(safe_left / midpoint), axis=1) + 0.5 * np.sum(
        safe_right * np.log(safe_right / midpoint), axis=1
    )
    changed = left.argmax(axis=1) != right.argmax(axis=1)
    order = np.argsort(nearest_distance, kind="stable")
    profile = []
    for bin_number, indices in enumerate(np.array_split(order, distance_bins), start=1):
        profile.append(
            {
                "distance_bin": bin_number,
                "count": len(indices),
                "minimum_distance": float(nearest_distance[indices].min()),
                "maximum_distance": float(nearest_distance[indices].max()),
                "mean_distance": float(nearest_distance[indices].mean()),
                "mean_js_divergence": float(js[indices].mean()),
                "top1_change_rate": float(changed[indices].mean()),
            }
        )
    return profile


def class_conditioned_disagreement(
    left_labels: np.ndarray, right_labels: np.ndarray, true_labels: np.ndarray
) -> dict[int, float]:
    """Top-1 disagreement within each observed ground-truth class."""
    left = np.asarray(left_labels)
    right = np.asarray(right_labels)
    truth = np.asarray(true_labels)
    if (
        left.ndim != 1
        or right.ndim != 1
        or truth.ndim != 1
        or not (len(left) == len(right) == len(truth))
    ):
        raise ValueError("class-conditioned disagreement requires aligned label vectors")
    if len(truth) == 0:
        raise ValueError("class-conditioned disagreement requires observations")
    return {
        int(label): float(np.mean(left[truth == label] != right[truth == label]))
        for label in np.unique(truth)
    }
