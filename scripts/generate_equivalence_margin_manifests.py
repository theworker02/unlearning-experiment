from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from exact_unlearning.data import load_dataset

OUTPUT = Path("deletion_manifests/phase3/fashion_mnist")
DATASET_CONFIG = {
    "name": "fashion_mnist",
    "root": "data/raw",
    "validation_fraction": 0.1,
    "split_seed": 1729,
    "train_limit": 10000,
    "test_limit": 2500,
}


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    ids = np.sort(np.asarray(payload["sample_ids"], dtype=np.int64))
    payload["sample_ids"] = ids.tolist()
    payload["sample_ids_sha256"] = hashlib.sha256(
        json.dumps(ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite calibration manifest: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    data = load_dataset(DATASET_CONFIG)
    labels = data.y_train.numpy()
    generated: list[Path] = []
    for target_class in (0, 5):
        class_positions = np.flatnonzero(labels == target_class)
        for deletion_fraction in (0.90, 0.95, 0.99):
            count = round(len(class_positions) * deletion_fraction)
            positions = np.random.default_rng(4201).choice(
                class_positions, count, replace=False
            )
            path = OUTPUT / (
                f"margin-class{target_class}-f{deletion_fraction:.2f}-seed4201.json"
            )
            write_manifest(
                path,
                {
                    "schema_version": 1,
                    "study_phase": 3,
                    "regime": "R1_CLASS_CONDITIONED_MARGIN_CALIBRATION",
                    "dataset": data.metadata.name,
                    "dataset_sha256": data.dataset_sha256,
                    "scenario": "class_conditioned",
                    "purpose": "controlled_residual_forget_class_contamination",
                    "fraction": deletion_fraction,
                    "within_class_fraction": deletion_fraction,
                    "residual_class_fraction": 1.0 - deletion_fraction,
                    "target_class": target_class,
                    "seed": 4201,
                    "count_rule": "round_half_to_even_within_target_class",
                    "sample_ids": data.train_ids[positions].tolist(),
                },
            )
            generated.append(path)
    for path in generated:
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")


if __name__ == "__main__":
    main()
