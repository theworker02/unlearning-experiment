from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_phase3_assay_pilot import load_reference


def test_reference_loader_materializes_only_prespecified_distance_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.npz"
    np.savez_compressed(
        artifact,
        test_probabilities=np.asarray([[0.75, 0.25]], dtype=np.float32),
        test_predicted_labels=np.asarray([0], dtype=np.int64),
        test_features=np.asarray([[1.0, 2.0]], dtype=np.float32),
        activation_unplanned=np.ones((4, 4, 4), dtype=np.float32),
    )

    loaded = load_reference({"path": str(artifact), "sha256": "not-used-by-loader"})

    assert set(loaded) == {
        "test_probabilities",
        "test_predicted_labels",
        "test_features",
    }


def test_class_conditioned_plan_has_unique_exact_artifact_names() -> None:
    config = yaml.safe_load(
        Path("configs/experiments/phase3_class_conditioned_assay_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    names = [
        f"exact-{condition['condition_id']}-{seed}"
        for condition in config["conditions"]
        for seed in config["exact_seeds"]
    ]

    assert len(names) == 45
    assert len(names) == len(set(names))


def test_failure_registry_is_valid_json_lines() -> None:
    records = [
        json.loads(line)
        for line in Path("results/failures/failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {record["experiment_id"] for record in records} >= {
        "phase3-fmnist-class-conditioned-assay-pilot-v1",
        "phase3-fmnist-class-conditioned-assay-pilot-v2",
    }
    assert all(record["batch_invalidated"] is True for record in records)
