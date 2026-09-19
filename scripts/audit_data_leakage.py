from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from exact_unlearning.data import DatasetSplit, load_dataset


def _sample_hashes(x: Any, y: Any) -> set[str]:
    x_array = x.cpu().numpy()
    y_array = y.cpu().numpy()
    return {
        hashlib.sha256(sample.tobytes() + np.asarray(label, dtype="<i8").tobytes()).hexdigest()
        for sample, label in zip(x_array, y_array, strict=True)
    }


def _audit_dataset(name: str, data: DatasetSplit) -> dict[str, Any]:
    ids = {
        "train": set(map(int, data.train_ids)),
        "validation": set(map(int, data.validation_ids)),
        "test": set(map(int, data.test_ids)),
    }
    content = {
        "train": _sample_hashes(data.x_train, data.y_train),
        "validation": _sample_hashes(data.x_validation, data.y_validation),
        "test": _sample_hashes(data.x_test, data.y_test),
    }
    return {
        "dataset": name,
        "id_overlap": {
            "train_validation": len(ids["train"] & ids["validation"]),
            "train_test": len(ids["train"] & ids["test"]),
            "validation_test": len(ids["validation"] & ids["test"]),
        },
        "content_overlap": {
            "train_validation": len(content["train"] & content["validation"]),
            "train_test": len(content["train"] & content["test"]),
            "validation_test": len(content["validation"] & content["test"]),
        },
        "within_split_duplicate_content": {
            "train": len(data.train_ids) - len(content["train"]),
            "validation": len(data.validation_ids) - len(content["validation"]),
            "test": len(data.test_ids) - len(content["test"]),
        },
    }


def _audit_run(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    deletion = result.get("deletion")
    if deletion is None:
        # Schema-v3 assay records group multiple deletion conditions.  They do
        # not embed retained IDs, so the stronger partition equality check is
        # unavailable.  Audit their immutable forget manifests explicitly and
        # report the narrower scope instead of crashing or implying a pass.
        manifests = []
        for condition in result.get("conditions", []):
            reference = condition.get("manifest", {})
            manifest_path = Path(str(reference.get("path", "")))
            exists = manifest_path.is_file()
            manifest_bytes = manifest_path.read_bytes() if exists else b""
            manifest = json.loads(manifest_bytes.decode("utf-8")) if exists else {}
            sample_ids = manifest.get("sample_ids", [])
            expected_sha256 = reference.get("sha256")
            manifests.append(
                {
                    "path": str(manifest_path),
                    "exists": exists,
                    "reference_sha256_present": isinstance(expected_sha256, str),
                    "reference_sha256_matches": exists
                    and hashlib.sha256(manifest_bytes).hexdigest() == expected_sha256,
                    "unique_forget_ids": len(sample_ids) == len(set(sample_ids)),
                    "forget_count_matches": len(sample_ids) == condition.get("forget_count"),
                    "dataset_hash_matches": manifest.get("dataset_sha256")
                    == result.get("dataset", {}).get("sha256"),
                }
            )
        return {
            "experiment_id": result["experiment_id"],
            "audit_scope": "forget_manifests_only",
            "forget_retain_overlap": None,
            "partition_matches_train_ids": None,
            "exact_exclusion_assertion": None,
            "manifest_checks_pass": bool(manifests)
            and all(all(item.values()) for item in manifests),
            "manifests": manifests,
        }
    forget = set(deletion["forget_ids"])
    retain = set(deletion["retain_ids"])
    train = set(result["dataset"]["train_ids"])
    return {
        "experiment_id": result["experiment_id"],
        "audit_scope": "complete_partition",
        "forget_retain_overlap": len(forget & retain),
        "partition_matches_train_ids": forget | retain == train,
        "exact_exclusion_assertion": deletion.get("exact_retraining_exclusion_verified") is True,
    }


def main() -> None:
    dataset_configs: dict[str, dict[str, Any]] = {}
    for path in sorted(Path("configs/experiments").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = config["dataset"]
        dataset_configs.setdefault(str(dataset["name"]), dataset)
    datasets = [
        _audit_dataset(name, load_dataset(config)) for name, config in dataset_configs.items()
    ]
    runs = [_audit_run(path) for path in sorted(Path("results/raw").glob("*/result.json"))]
    machine = {
        "schema_version": 1,
        "datasets": datasets,
        "runs": runs,
        "unresolved_id_leakage": any(any(item["id_overlap"].values()) for item in datasets),
        "invalid_run_partition": any(
            (
                item["forget_retain_overlap"]
                or not item["partition_matches_train_ids"]
                or not item["exact_exclusion_assertion"]
            )
            if item["audit_scope"] == "complete_partition"
            else not item["manifest_checks_pass"]
            for item in runs
        ),
        "process_level_test_tuning_audited": False,
    }
    output = Path("results/processed/data_leakage_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Data leakage audit",
        "",
        "Generated by `scripts/audit_data_leakage.py`. Content duplicates are reported rather",
        "than automatically equated with ID leakage; they require dataset-specific review.",
        "",
        "## Dataset partitions",
        "",
        "| Dataset | ID overlap T/V | ID overlap T/Test | "
        "Content overlap T/V | Content overlap T/Test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in datasets:
        lines.append(
            f"| {item['dataset']} | {item['id_overlap']['train_validation']} | "
            f"{item['id_overlap']['train_test']} | {item['content_overlap']['train_validation']} | "
            f"{item['content_overlap']['train_test']} |"
        )
    lines.extend(
        [
            "",
            "## Experiment partitions",
            "",
            f"Audited {len(runs)} completed raw records. Invalid partitions: "
            f"{sum((not r['partition_matches_train_ids'] or r['forget_retain_overlap'] > 0) if r['audit_scope'] == 'complete_partition' else not r['manifest_checks_pass'] for r in runs)}.",  # noqa: E501
            f"Manifest-only aggregate records: "
            f"{sum(r['audit_scope'] == 'forget_manifests_only' for r in runs)}.",
            "",
            "Process-level leakage (for example, informal tuning after viewing final test results)",
            "cannot be established from arrays alone and remains **PARTIAL** until the "
            "confirmatory",
            "protocol and access procedure are frozen.",
        ]
    )
    Path("research/audits/data_leakage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
