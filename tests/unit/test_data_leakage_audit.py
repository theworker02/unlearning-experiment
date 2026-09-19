from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_data_leakage import _audit_run


def test_schema_three_assay_audits_deletion_manifests_without_claiming_full_partition(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = Path("deletion_manifests/condition.json")
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes = json.dumps(
        {
            "dataset_sha256": "dataset-hash",
            "sample_ids": [4, 8, 15],
        }
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    result_path = Path("results/raw/assay/result.json")
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "experiment_id": "assay",
                "schema_version": 3,
                "dataset": {"sha256": "dataset-hash"},
                "conditions": [
                    {
                        "forget_count": 3,
                        "manifest": {
                            "path": str(manifest_path),
                            "sha256": manifest_sha256,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = _audit_run(result_path)

    assert audit["audit_scope"] == "forget_manifests_only"
    assert audit["manifest_checks_pass"] is True
    assert audit["partition_matches_train_ids"] is None
    assert audit["exact_exclusion_assertion"] is None


def test_schema_three_assay_fails_when_manifest_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result_path = Path("result.json")
    result_path.write_text(
        json.dumps(
            {
                "experiment_id": "assay",
                "schema_version": 3,
                "dataset": {"sha256": "dataset-hash"},
                "conditions": [
                    {
                        "forget_count": 1,
                        "manifest": {"path": "missing.json", "sha256": "0" * 64},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _audit_run(result_path)["manifest_checks_pass"] is False
