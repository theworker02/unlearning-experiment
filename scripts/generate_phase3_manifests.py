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


def write_manifest(name: str, payload: dict[str, object]) -> Path:
    ids = np.sort(np.asarray(payload["sample_ids"], dtype=np.int64))
    payload["sample_ids"] = ids.tolist()
    payload["sample_ids_sha256"] = hashlib.sha256(
        json.dumps(ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite Phase 3 manifest: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> None:
    data = load_dataset(DATASET_CONFIG)
    generated = []
    for fraction in (0.005, 0.01, 0.02, 0.05, 0.10):
        count = max(1, round(len(data.train_ids) * fraction))
        for manifest_index, seed in enumerate(range(3101, 3106), start=1):
            positions = np.random.default_rng(seed).choice(
                len(data.train_ids), count, replace=False
            )
            generated.append(
                write_manifest(
                    f"r0-iid-f{fraction:.4f}-m{manifest_index}-seed{seed}.json",
                    {
                        "schema_version": 1,
                        "study_phase": 3,
                        "regime": "R0_IID",
                        "dataset": data.metadata.name,
                        "dataset_sha256": data.dataset_sha256,
                        "scenario": "random",
                        "fraction": fraction,
                        "seed": seed,
                        "count_rule": "max_one_round_half_to_even",
                        "sample_ids": data.train_ids[positions].tolist(),
                    },
                )
            )
    labels = data.y_train.numpy()
    for target_class in range(10):
        class_positions = np.flatnonzero(labels == target_class)
        for fraction in (0.25, 0.50, 1.0):
            count = round(len(class_positions) * fraction)
            positions = np.random.default_rng(3201).choice(class_positions, count, replace=False)
            generated.append(
                write_manifest(
                    f"r1-class{target_class}-f{fraction:.2f}-seed3201.json",
                    {
                        "schema_version": 1,
                        "study_phase": 3,
                        "regime": "R1_CLASS_CONDITIONED",
                        "dataset": data.metadata.name,
                        "dataset_sha256": data.dataset_sha256,
                        "scenario": "class_conditioned",
                        "fraction": fraction,
                        "within_class_fraction": fraction,
                        "target_class": target_class,
                        "seed": 3201,
                        "count_rule": "round_half_to_even_within_target_class",
                        "sample_ids": data.train_ids[positions].tolist(),
                    },
                )
            )
    checksum_path = Path("study/preregistration/manifests.sha256")
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(checksum_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite manifest inventory: {checksum_path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for path in generated:
            handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}\n")
    print(f"generated {len(generated)} immutable Phase 3 manifests")


if __name__ == "__main__":
    main()
