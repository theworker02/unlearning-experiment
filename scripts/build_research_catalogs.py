from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import yaml


def main() -> None:
    output = Path("reports/catalogs")
    output.mkdir(parents=True, exist_ok=True)

    dataset_lines = [
        "# Dataset catalog",
        "",
        "Generated from dataset packages and deletion manifests. Benchmark protocol 0.5.0.",
    ]
    for root in sorted(path for path in Path("datasets").iterdir() if path.is_dir()):
        statistics = json.loads((root / "statistics.json").read_text(encoding="utf-8"))
        checksums = json.loads((root / "checksum.json").read_text(encoding="utf-8"))
        split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
        manifests = list(Path("deletion_manifests", root.name).glob("*.json"))
        counts = statistics["counts"]
        validation = json.loads((root / "validation_report.json").read_text())["status"]
        sample_counts = f"{counts['train']}/{counts['validation']}/{counts['test']}"
        dataset_lines.extend(
            [
                "",
                f"## {root.name}",
                "",
                f"- Samples (train/validation/test): {sample_counts}",
                f"- Processed/source hash: `{checksums['processed']}`",
                f"- Training split hash: `{split['split_hashes']['train']}`",
                f"- Predeclared deletion scenarios: {len(manifests)}",
                f"- Validation: `{validation}`",
            ]
        )
    (output / "datasets.md").write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    with Path("model_zoo/model_manifest.csv").open(encoding="utf-8", newline="") as handle:
        models = list(csv.DictReader(handle))
    role_counts = Counter(row["role"].split(".")[0] for row in models)
    model_lines = [
        "# Model catalog",
        "",
        "Only existing checkpoints are counted. Benchmark protocol 0.5.0.",
        "",
        f"Total indexed checkpoints: {len(models)}",
        "",
        *[f"- {role}: {count}" for role, count in sorted(role_counts.items())],
        "",
        "All indexed checkpoints currently derive from legacy development records and are not",
        "publication-eligible. Reproduction commands require their source experiment "
        "configuration.",
    ]
    (output / "models.md").write_text("\n".join(model_lines) + "\n", encoding="utf-8")

    with Path("research/experiment_ledger.csv").open(encoding="utf-8", newline="") as handle:
        experiments = list(csv.DictReader(handle))
    status_counts = Counter(row["validation_status"].split(":")[0] for row in experiments)
    experiment_lines = [
        "# Experiment catalog",
        "",
        f"Registry rows: {len(experiments)}",
        "",
        *[f"- {status}: {count}" for status, count in sorted(status_counts.items())],
        "",
        "The full queryable registry is `results/processed/experiments.sqlite`.",
    ]
    (output / "experiments.md").write_text("\n".join(experiment_lines) + "\n", encoding="utf-8")

    method_registry = yaml.safe_load(Path("benchmark/methods.yaml").read_text(encoding="utf-8"))
    method_lines = ["# Method cards", "", "Generated from Benchmark 0.5.0 registry.", ""]
    for method in method_registry["methods"]:
        method_lines.extend(
            [
                f"## {method['id']}",
                "",
                f"- Category: {method['category']}",
                f"- Status: {method['status']}",
                f"- Citation key: {method.get('citation_key', 'not applicable or pending')}",
                "",
            ]
        )
    (output / "methods.md").write_text("\n".join(method_lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
