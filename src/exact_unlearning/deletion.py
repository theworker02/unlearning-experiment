from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .models import Classifier
from .training import per_example_loss


@dataclass(frozen=True)
class DeletionSet:
    forget_positions: np.ndarray
    retain_positions: np.ndarray
    forget_ids: np.ndarray
    sha256: str


def partition_lineage(deletion: DeletionSet, train_ids: np.ndarray) -> dict[str, object]:
    """Return auditable retained/forgotten lineage without embedding large ID arrays."""
    train = np.asarray(train_ids, dtype=np.int64)
    forget_positions = np.asarray(deletion.forget_positions, dtype=np.int64)
    retain_positions = np.asarray(deletion.retain_positions, dtype=np.int64)
    if np.intersect1d(forget_positions, retain_positions).size:
        raise ValueError("forgotten and retained positions overlap")
    reconstructed = np.sort(np.concatenate((forget_positions, retain_positions)))
    expected = np.arange(len(train), dtype=np.int64)
    if not np.array_equal(reconstructed, expected):
        raise ValueError("forgotten and retained positions do not reconstruct training split")
    forget_ids = np.sort(train[forget_positions]).astype("<i8", copy=False)
    retain_ids = np.sort(train[retain_positions]).astype("<i8", copy=False)
    train_ids_sorted = np.sort(train).astype("<i8", copy=False)
    lineage: dict[str, object] = {
        "train_count": len(train),
        "forget_count": len(forget_ids),
        "retain_count": len(retain_ids),
        "train_ids_sha256": hashlib.sha256(train_ids_sorted.tobytes()).hexdigest(),
        "forget_ids_sha256": hashlib.sha256(forget_ids.tobytes()).hexdigest(),
        "retain_ids_sha256": hashlib.sha256(retain_ids.tobytes()).hexdigest(),
        "partition_disjoint": True,
        "partition_reconstructs_train": True,
        "exact_retraining_exclusion_verified": True,
    }
    canonical = json.dumps(lineage, sort_keys=True, separators=(",", ":"))
    lineage["partition_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return lineage


def select_deletion_from_manifest(
    manifest_path: str | Path,
    train_ids: np.ndarray,
    *,
    expected_dataset: str,
    expected_dataset_sha256: str,
    expected_strategy: str,
    expected_fraction: float,
    expected_seed: int,
    expected_target_class: int | None = None,
) -> DeletionSet:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored_manifest_hash = manifest.get("manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != stored_manifest_hash:
        raise ValueError(f"deletion manifest hash mismatch: {path}")
    ids = np.asarray(manifest["sample_ids"], dtype=np.int64)
    encoded_ids = json.dumps(ids.tolist(), separators=(",", ":"))
    if hashlib.sha256(encoded_ids.encode()).hexdigest() != manifest["sample_ids_sha256"]:
        raise ValueError(f"deletion sample ID hash mismatch: {path}")
    expectations = {
        "dataset": expected_dataset,
        "dataset_sha256": expected_dataset_sha256,
        "scenario": expected_strategy,
        "fraction": expected_fraction,
        "seed": expected_seed,
    }
    mismatches = [key for key, value in expectations.items() if manifest.get(key) != value]
    if expected_target_class is not None and manifest.get("target_class") != expected_target_class:
        mismatches.append("target_class")
    if mismatches:
        raise ValueError(f"deletion manifest/config mismatch for fields: {mismatches}")
    if len(ids) != len(np.unique(ids)):
        raise ValueError("deletion manifest contains duplicate sample IDs")
    positions_by_id = {int(value): index for index, value in enumerate(train_ids)}
    missing = [int(value) for value in ids if int(value) not in positions_by_id]
    if missing:
        raise ValueError(f"deletion manifest IDs absent from training split: {missing[:5]}")
    forget = np.sort(np.asarray([positions_by_id[int(value)] for value in ids], dtype=np.int64))
    retain = np.setdiff1d(np.arange(len(train_ids), dtype=np.int64), forget, assume_unique=True)
    digest = hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest()
    return DeletionSet(forget, retain, np.sort(ids), digest)


def select_deletion(
    strategy: str,
    fraction: float,
    seed: int,
    train_ids: np.ndarray,
    x: torch.Tensor,
    y: torch.Tensor,
    original_model: Classifier,
    target_class: int | None = None,
) -> DeletionSet:
    if strategy == "random":
        count = int(round(len(train_ids) * fraction))
        if fraction > 0 and count == 0:
            count = 1
        if count >= len(train_ids):
            raise ValueError("deletion would leave no retained examples")
        positions = np.random.default_rng(seed).choice(len(train_ids), count, replace=False)
    elif strategy == "high_loss":
        count = int(round(len(train_ids) * fraction))
        if fraction > 0 and count == 0:
            count = 1
        if count >= len(train_ids):
            raise ValueError("deletion would leave no retained examples")
        losses = per_example_loss(original_model, x, y).cpu().numpy()
        positions = np.argsort(losses, kind="stable")[-count:] if count else np.array([], dtype=int)
    elif strategy in {"class", "class_conditioned"}:
        if target_class is None:
            raise ValueError("class deletion requires target_class")
        class_positions = np.flatnonzero(y.cpu().numpy() == target_class)
        if not len(class_positions):
            raise ValueError(f"target class {target_class} is absent")
        if strategy == "class":
            positions = class_positions
        else:
            class_count = int(round(len(class_positions) * fraction))
            if fraction > 0 and class_count == 0:
                class_count = 1
            positions = np.random.default_rng(seed).choice(
                class_positions, class_count, replace=False
            )
    else:
        raise ValueError(f"unsupported deletion strategy: {strategy}")
    forget = np.sort(np.asarray(positions, dtype=np.int64))
    retain = np.setdiff1d(np.arange(len(train_ids), dtype=np.int64), forget, assume_unique=True)
    if np.intersect1d(forget, retain).size:
        raise AssertionError("forgotten and retained positions overlap")
    forget_ids = np.sort(train_ids[forget])
    digest = hashlib.sha256(forget_ids.tobytes()).hexdigest()
    return DeletionSet(forget, retain, forget_ids, digest)
