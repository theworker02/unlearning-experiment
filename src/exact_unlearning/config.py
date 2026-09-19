from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .quality import Maturity, validate_launch_preconditions


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the public contract."""


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source: Path
    sha256: str

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment_id"])

    @property
    def maturity(self) -> Maturity:
        return Maturity(str(self.raw["maturity"]).upper())


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    payload = source.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    required = {
        "experiment_id",
        "maturity",
        "dataset",
        "model",
        "training",
        "original_seed",
        "oracle_seeds",
        "deletion",
        "unlearning",
        "evaluation",
        "output_root",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ConfigError(f"missing required configuration keys: {missing}")
    validate_launch_preconditions(raw)
    if not raw["oracle_seeds"] or len(set(raw["oracle_seeds"])) != len(raw["oracle_seeds"]):
        raise ConfigError("oracle_seeds must contain distinct values")
    fraction = float(raw["deletion"]["fraction"])
    if not 0.0 <= fraction < 1.0:
        raise ConfigError("deletion.fraction must be in [0, 1)")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return ExperimentConfig(raw=raw, source=source, sha256=hashlib.sha256(canonical).hexdigest())
