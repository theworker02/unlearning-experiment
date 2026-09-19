from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    leakage = json.loads(
        Path("results/processed/data_leakage_audit.json").read_text(encoding="utf-8")
    )
    audit_by_name = {item["dataset"]: item for item in leakage["datasets"]}
    for card_path in sorted(Path("research/datasets").glob("*.yaml")):
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        name = card["dataset_id"]
        root = Path("datasets") / name
        root.mkdir(parents=True, exist_ok=True)
        source = {
            "schema_version": 1,
            "dataset": name,
            "source": card["source"],
            "version": card["version"],
            "access_date": card["access_date"],
            "citation": card["citation"],
            "license": card["license"],
        }
        checksums = {
            "schema_version": 1,
            "raw": card["raw_checksums"],
            "processed": card["processed_hash"],
        }
        statistics = {
            "schema_version": 1,
            "counts": card["counts"],
            "dimensionality": card["dimensionality"],
            "class_distribution": card["class_distribution"],
        }
        preprocessing = {
            key: card[key]
            for key in ("preprocessing", "normalization", "augmentation", "transformations")
        }
        report = {
            "schema_version": 1,
            "status": "PASS" if name in audit_by_name else "FAIL",
            "leakage_checks": audit_by_name.get(name),
            "process_level_test_tuning": "NOT_AUDITED",
        }
        documents = {
            "source.json": source,
            "checksum.json": checksums,
            "statistics.json": statistics,
            "validation_report.json": report,
        }
        for filename, payload in documents.items():
            (root / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        (root / "preprocessing.yaml").write_text(
            yaml.safe_dump(preprocessing, sort_keys=False), encoding="utf-8"
        )
        card_markdown = f"""# {card["canonical_name"]}

Role: **{card["relevance"]}**  
Version: `{card["version"]}`  
Source: {card["source"]}  
License: {card["license"]}  
Access date: {card["access_date"]}

## Composition and processing

Train/validation/test counts are {card["counts"]["train"]}/{card["counts"]["validation"]}/
{card["counts"]["test"]}. {card["preprocessing"]}; {card["normalization"]}.

## Split

{card["splits"]["method"]} with seed {card["splits"]["split_seed"]}. Exact IDs and hashes are
stored in `split_manifest.json`.

## Limitations and bias

{card["known_limitations"]} Benchmark class labels and sampling do not represent all real-world
populations or deletion requests.

## Ethical relevance

{card["ethical_considerations"]}
"""
        (root / "CARD.md").write_text(card_markdown, encoding="utf-8")
        (root / "license-reference.txt").write_text(
            f"{card['license']}\nSee source.json for the canonical source.\n", encoding="utf-8"
        )
        (root / "deletion_manifests").mkdir(exist_ok=True)
        manifests = sorted(
            path.as_posix() for path in Path("deletion_manifests").glob(f"**/{name}/**/*.json")
        )
        (root / "deletion_manifests" / "index.json").write_text(
            json.dumps(
                {"schema_version": 1, "canonical_manifest_paths": manifests},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(root)


if __name__ == "__main__":
    main()
