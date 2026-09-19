from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from exact_unlearning.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    arguments = parser.parse_args()
    card_path = Path("research/datasets") / f"{arguments.dataset}.yaml"
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    configs = []
    for path in Path("configs/experiments").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["dataset"]
        if config["name"] == arguments.dataset:
            configs.append(config)
    if not configs:
        raise SystemExit(f"no configuration reconstructs dataset {arguments.dataset}")
    config = max(configs, key=lambda value: value.get("train_limit", 0))
    reconstructed = load_dataset(config)
    expected_hash = card.get("processed_hash") or card.get("acquisition", {}).get("processed_hash")
    if not expected_hash:
        raise SystemExit(f"dataset card lacks a processed hash: {card_path}")
    if reconstructed.dataset_sha256 != expected_hash:
        raise SystemExit("dataset reconstruction hash mismatch")
    print(f"verified {arguments.dataset}: {reconstructed.dataset_sha256}")


if __name__ == "__main__":
    main()
