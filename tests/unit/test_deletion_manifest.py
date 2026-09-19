from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import (
    DeletionSet,
    partition_lineage,
    select_deletion_from_manifest,
)

MANIFEST = Path("deletion_manifests/fashion_mnist/random-0p010-seed-1009.json")


def test_frozen_deletion_manifest_reconstructs_exact_partition() -> None:
    config = {
        "name": "fashion_mnist",
        "root": "data/raw",
        "validation_fraction": 0.1,
        "split_seed": 1729,
        "train_limit": 10000,
        "test_limit": 2500,
    }
    data = load_dataset(config)
    deletion = select_deletion_from_manifest(
        MANIFEST,
        data.train_ids,
        expected_dataset="fashion_mnist",
        expected_dataset_sha256=data.dataset_sha256,
        expected_strategy="random",
        expected_fraction=0.01,
        expected_seed=1009,
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert deletion.forget_ids.tolist() == manifest["sample_ids"]
    assert len(deletion.forget_positions) == 90
    assert not np.intersect1d(deletion.forget_positions, deletion.retain_positions).size


def test_partition_lineage_commits_to_both_sides_of_partition() -> None:
    train_ids = np.asarray([50, 10, 40, 20, 30], dtype=np.int64)
    deletion = DeletionSet(
        forget_positions=np.asarray([1, 3]),
        retain_positions=np.asarray([0, 2, 4]),
        forget_ids=np.asarray([10, 20]),
        sha256="unused-in-this-test",
    )

    lineage = partition_lineage(deletion, train_ids)

    assert lineage["train_count"] == 5
    assert lineage["forget_count"] == 2
    assert lineage["retain_count"] == 3
    assert lineage["partition_disjoint"] is True
    assert lineage["partition_reconstructs_train"] is True
    assert lineage["exact_retraining_exclusion_verified"] is True
    assert len(lineage["train_ids_sha256"]) == 64
    assert len(lineage["forget_ids_sha256"]) == 64
    assert len(lineage["retain_ids_sha256"]) == 64
    assert len(lineage["partition_sha256"]) == 64


def test_partition_lineage_rejects_incomplete_partition() -> None:
    deletion = DeletionSet(
        forget_positions=np.asarray([0]),
        retain_positions=np.asarray([2]),
        forget_ids=np.asarray([10]),
        sha256="unused-in-this-test",
    )

    with pytest.raises(ValueError, match="do not reconstruct"):
        partition_lineage(deletion, np.asarray([10, 20, 30], dtype=np.int64))


def test_manifest_configuration_mismatch_fails_closed() -> None:
    train_ids = np.asarray(json.loads(MANIFEST.read_text())["sample_ids"], dtype=np.int64)
    with pytest.raises(ValueError, match="manifest/config mismatch"):
        select_deletion_from_manifest(
            MANIFEST,
            train_ids,
            expected_dataset="fashion_mnist",
            expected_dataset_sha256="wrong",
            expected_strategy="random",
            expected_fraction=0.01,
            expected_seed=1009,
        )
