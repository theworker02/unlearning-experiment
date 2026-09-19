from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanarySet:
    source_positions: np.ndarray
    images: np.ndarray
    labels: np.ndarray
    sha256: str


def stable_top_k(scores: np.ndarray, sample_ids: np.ndarray, count: int) -> np.ndarray:
    """Rank descending by a precomputed score, breaking ties by ascending sample ID."""
    scores = np.asarray(scores, dtype=float)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if scores.shape != sample_ids.shape or not 0 <= count <= len(scores):
        raise ValueError("invalid ranking inputs")
    order = np.lexsort((sample_ids, -scores))
    return np.sort(order[:count].astype(np.int64))


def generate_memorization_canaries(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    seed: int,
    class_count: int,
) -> CanarySet:
    """Deterministic stress probes with random corner patches and non-semantic labels."""
    x = np.asarray(images).copy()
    y = np.asarray(labels).copy()
    if x.ndim not in {3, 4} or not 0 < count < len(x) or class_count < 2:
        raise ValueError("invalid canary configuration")
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(x), size=count, replace=False))
    for position in positions:
        image = x[position]
        patch = rng.uniform(float(image.min()), float(image.max()) or 1.0, size=(3, 3))
        if x.ndim == 4:
            image[:, -3:, -3:] = patch
        else:
            image[-3:, -3:] = patch
        offset = int(rng.integers(1, class_count))
        y[position] = (int(y[position]) + offset) % class_count
    payload = positions.astype("<i8").tobytes() + x[positions].tobytes() + y[positions].tobytes()
    return CanarySet(positions, x[positions], y[positions], hashlib.sha256(payload).hexdigest())
