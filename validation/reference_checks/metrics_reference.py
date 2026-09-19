"""Independent metric formulas. This module must not import production metric code."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score


def reference_accuracy_f1(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predicted = probabilities.argmax(axis=1)
    return float(accuracy_score(labels, predicted)), float(
        f1_score(labels, predicted, average="macro", zero_division=0)
    )


def reference_js(left: np.ndarray, right: np.ndarray) -> float:
    # SciPy returns the square root of the divergence for each row.
    distances = np.array([jensenshannon(p, q, base=np.e) for p, q in zip(left, right, strict=True)])
    return float(np.square(distances).mean())


def reference_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(left - right).sum(axis=1)))


def reference_nll_brier(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1])[labels]
    nll = -np.log(clipped[np.arange(len(labels)), labels]).mean()
    brier = np.square(clipped - one_hot).sum(axis=1).mean()
    return float(nll), float(brier)


def reference_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear CKA computed independently through centered Gram matrices."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must be 2-D with the same observation count")
    count = left.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    left_gram = centering @ (left @ left.T) @ centering
    right_gram = centering @ (right @ right.T) @ centering
    numerator = float(np.sum(left_gram * right_gram))
    denominator = float(np.sqrt(np.sum(left_gram * left_gram) * np.sum(right_gram * right_gram)))
    return numerator / denominator if denominator else float("nan")
