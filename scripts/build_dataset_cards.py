from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from exact_unlearning.data import DatasetSplit, load_dataset


def _class_counts(labels: Any) -> dict[int, int]:
    values, counts = np.unique(labels.cpu().numpy(), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def _raw_hashes(config: dict[str, Any], data: DatasetSplit) -> dict[str, str]:
    if config["name"] != "fashion_mnist":
        return {"canonical_loaded_arrays": data.dataset_sha256}
    raw = Path(config["root"]) / "fashion_mnist" / "raw"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw.glob("*.gz"))
    }


def _card(config: dict[str, Any], data: DatasetSplit) -> dict[str, Any]:
    is_digits = config["name"] == "sklearn_digits"
    return {
        "schema_version": 1,
        "dataset_id": data.metadata.name,
        "canonical_name": "Scikit-learn Digits" if is_digits else "Fashion-MNIST",
        "origin": (
            "UCI optical recognition digits via scikit-learn" if is_digits else "Zalando Research"
        ),
        "citation": (
            "scikit-learn dataset documentation"
            if is_digits
            else "Xiao, Rasul, and Vollgraf (2017)"
        ),
        "source": data.metadata.source,
        "version": data.metadata.version,
        "license": data.metadata.license,
        "access_date": "2026-08-30",
        "raw_checksums": _raw_hashes(config, data),
        "processed_hash": data.dataset_sha256,
        "counts": {
            "train": len(data.train_ids),
            "validation": len(data.validation_ids),
            "test": len(data.test_ids),
        },
        "dimensionality": list(data.x_train.shape[1:]),
        "class_distribution": {
            "train": _class_counts(data.y_train),
            "validation": _class_counts(data.y_validation),
            "test": _class_counts(data.y_test),
        },
        "preprocessing": "divide pixels by 16" if is_digits else "divide uint8 pixels by 255",
        "normalization": "[0, 1] fixed scaling; no fitted statistics",
        "augmentation": "none in current validation/pilot configurations",
        "exclusions": "stratified development limits may apply from the experiment configuration",
        "transformations": "deterministic tensor conversion",
        "splits": {
            "method": "stratified fixed-seed split",
            "split_seed": config["split_seed"],
            "train_ids_sha256": hashlib.sha256(
                np.asarray(data.train_ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "validation_ids_sha256": hashlib.sha256(
                np.asarray(data.validation_ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "test_ids_sha256": hashlib.sha256(
                np.asarray(data.test_ids, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "known_limitations": (
            "Development-only small grayscale dataset; forbidden as publication evidence."
            if is_digits
            else (
                "Pilot subset and modest image resolution do not establish broad vision "
                "generalization."
            )
        ),
        "ethical_considerations": (
            "Benchmark labels are not evidence of real-world deletion guarantees or legal "
            "compliance."
        ),
        "relevance": (
            "Infrastructure validation" if is_digits else "Low-cost counterfactual-fidelity pilot"
        ),
    }


def main() -> None:
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(Path("configs/experiments").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["dataset"]
        name = str(config["name"])
        if name not in selected or config.get("train_limit", 0) > selected[name].get(
            "train_limit", 0
        ):
            selected[name] = config
    output = Path("research/datasets")
    output.mkdir(parents=True, exist_ok=True)
    for name, config in selected.items():
        path = output / f"{name}.yaml"
        card = yaml.safe_dump(_card(config, load_dataset(config)), sort_keys=False)
        path.write_text(card, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
