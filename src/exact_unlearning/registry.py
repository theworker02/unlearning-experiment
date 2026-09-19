from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ImmutableResultError(FileExistsError):
    """Raised rather than silently replacing an experiment artifact."""


def reserve_experiment(output_root: str | Path, experiment_id: str) -> Path:
    """Atomically reserve an ID before any expensive training begins."""
    safe = _safe_experiment_id(experiment_id)
    directory = Path(output_root) / safe
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as error:
        raise ImmutableResultError(f"experiment ID already exists: {directory}") from error
    return directory


def _safe_experiment_id(experiment_id: str) -> str:
    safe = "".join(
        character for character in experiment_id if character.isalnum() or character in "-_"
    )
    if safe != experiment_id or not safe:
        raise ValueError("experiment_id may contain only letters, numbers, '-' and '_'")
    return safe


def write_checkpoint(
    output_root: str | Path, experiment_id: str, name: str, state_dict: dict[str, Any]
) -> dict[str, str]:
    safe = _safe_experiment_id(experiment_id)
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid checkpoint name")
    path = Path(output_root) / safe / "checkpoints" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ImmutableResultError(f"refusing to overwrite checkpoint: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state_dict, handle)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_raw_result(output_root: str | Path, experiment_id: str, result: dict[str, Any]) -> Path:
    safe = _safe_experiment_id(experiment_id)
    directory = Path(output_root) / safe
    path = directory / "result.json"
    directory.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as error:
        raise ImmutableResultError(f"refusing to overwrite raw result: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def write_prediction_artifact(
    output_root: str | Path,
    experiment_id: str,
    name: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, str]:
    """Write compressed model outputs once so metrics can be independently reconstructed."""
    safe = _safe_experiment_id(experiment_id)
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid prediction artifact name")
    if not arrays or any(not key.replace("_", "").isalnum() for key in arrays):
        raise ValueError("prediction arrays require safe, non-empty keys")
    path = Path(output_root) / safe / "predictions" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ImmutableResultError(f"refusing to overwrite predictions: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_failure_record(
    output_root: str | Path,
    experiment_id: str,
    record: dict[str, Any],
) -> Path:
    """Append a failure as an immutable individual record; failures are never observations."""
    safe = _safe_experiment_id(experiment_id)
    failure_id = str(record["failure_id"])
    if not failure_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid failure_id")
    path = Path(output_root).parent / "failures" / safe / f"{failure_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ImmutableResultError(f"refusing to overwrite failure record: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
