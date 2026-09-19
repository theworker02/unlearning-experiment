from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from exact_unlearning.data import load_dataset


def digest_ids(values: list[int] | np.ndarray) -> str:
    canonical = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"immutable manifest differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def selected_configs() -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(Path("configs/experiments").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["dataset"]
        name = str(config["name"])
        if name not in selected or config.get("train_limit", 0) > selected[name].get(
            "train_limit", 0
        ):
            selected[name] = config
    return selected


def build_splits() -> None:
    for name, config in selected_configs().items():
        data = load_dataset(config)
        splits = {
            "train": [int(value) for value in data.train_ids],
            "validation": [int(value) for value in data.validation_ids],
            "test": [int(value) for value in data.test_ids],
        }
        payload = {
            "schema_version": 1,
            "benchmark_version": "0.5.0",
            "dataset": name,
            "dataset_sha256": data.dataset_sha256,
            "split_seed": config["split_seed"],
            "splits": splits,
            "split_hashes": {key: digest_ids(value) for key, value in splits.items()},
        }
        write_once(Path("datasets") / name / "split_manifest.json", payload)


def build_deletions() -> None:
    for path in sorted(Path("results/raw").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        deletion = result["deletion"]
        ids = [int(value) for value in deletion["forget_ids"]]
        config = result["config"]["deletion"]
        if config.get("manifest"):
            continue
        payload = {
            "schema_version": 1,
            "benchmark_version": "0.5.0",
            "status": "legacy_observed_not_publication_frozen",
            "dataset": result["dataset"]["name"],
            "dataset_sha256": result["dataset"].get(
                "sha256", result["dataset"].get("sha256_after_scaling")
            ),
            "source_experiment_id": result["experiment_id"],
            "scenario": config["strategy"],
            "fraction": config["fraction"],
            "seed": config["seed"],
            "sample_ids": ids,
            "sample_ids_sha256": digest_ids(ids),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        output = Path("deletion_manifests") / f"{result['experiment_id']}.json"
        write_once(output, payload)


def build_predeclared_random_deletions() -> None:
    policy = yaml.safe_load(Path("research/deletion_policy.yaml").read_text(encoding="utf-8"))
    master_seed = int(policy["master_seed"])
    for dataset_offset, split_path in enumerate(
        sorted(Path("datasets").glob("*/split_manifest.json"))
    ):
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = np.asarray(split["splits"]["train"], dtype=np.int64)
        for fraction_offset, fraction in enumerate(policy["random_fractions"]):
            seed = master_seed + dataset_offset * 100 + fraction_offset
            count = max(1, round(len(train_ids) * float(fraction)))
            rng = np.random.default_rng(seed)
            ids = np.sort(rng.choice(train_ids, size=count, replace=False)).tolist()
            payload = {
                "schema_version": 1,
                "benchmark_version": policy["benchmark_version"],
                "status": policy["status"],
                "dataset": split["dataset"],
                "dataset_sha256": split["dataset_sha256"],
                "split_hash": split["split_hashes"]["train"],
                "scenario": "random",
                "fraction": float(fraction),
                "seed": seed,
                "count_rule": policy["count_rule"],
                "sample_ids": ids,
                "sample_ids_sha256": digest_ids(ids),
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
            fraction_name = f"{float(fraction):.3f}".replace(".", "p")
            output = (
                Path("deletion_manifests")
                / split["dataset"]
                / f"random-{fraction_name}-seed-{seed}.json"
            )
            write_once(output, payload)


def main() -> None:
    build_splits()
    build_predeclared_random_deletions()
    build_deletions()
    print("built immutable split and legacy deletion manifests")


if __name__ == "__main__":
    main()
