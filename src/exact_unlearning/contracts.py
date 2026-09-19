from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from .models import Classifier


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    version: str
    source: str
    license: str
    sha256: str


class DatasetAdapter(Protocol):
    """Stable boundary for vision, tabular, and text datasets."""

    def load(self, config: dict[str, Any]) -> Any: ...


class ModelFactory(Protocol):
    """Constructs models without coupling experiments to one architecture family."""

    def __call__(
        self, config: dict[str, Any], input_shape: tuple[int, ...], classes: int
    ) -> Classifier: ...


@dataclass(frozen=True)
class UnlearningContext:
    original: Classifier
    x_retain: torch.Tensor
    y_retain: torch.Tensor
    x_forget: torch.Tensor
    y_forget: torch.Tensor
    seed: int
    config: dict[str, Any]
    device: torch.device


class UnlearningMethod(Protocol):
    """Extension point for validated methods such as SISA or SCRUB."""

    name: str
    literature_reference: str | None
    simple_baseline: bool

    def run(self, context: UnlearningContext) -> Any: ...


@dataclass(frozen=True)
class DeletionStage:
    stage: int
    new_forget_positions: np.ndarray
    cumulative_forget_positions: np.ndarray
    cumulative_retain_positions: np.ndarray


class DeletionStrategy(Protocol):
    """Boundary for random, class, rare, cluster, error, and influence regimes."""

    name: str

    def select(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SequentialExperimentSpec:
    """Repeated deletion with a stage-matched retraining distribution."""

    stage_fractions: tuple[float, ...]
    request_seed: int
    oracle_seeds: tuple[int, ...]
