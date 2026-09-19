import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from exact_unlearning.quality import evaluate_result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_result(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        test_ids=np.array([10, 11]),
        test_true_labels=np.array([0, 1]),
        test_probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]),
        test_predicted_labels=np.array([0, 1]),
        test_logits=np.log(np.array([[0.8, 0.2], [0.1, 0.9]])),
        test_features=np.array([[1.0, 0.0], [0.0, 1.0]]),
        artifact_schema_version=np.array(2, dtype=np.int64),
    )
    return {
        "schema_version": 2,
        "experiment_id": "fault-fixture",
        "maturity": "PILOT",
        "status": "completed",
        "config_sha256": "a" * 64,
        "dataset": {"train_count": 4},
        "deletion": {
            "forget_ids": [1],
            "retain_ids": [2, 3, 4],
            "exact_retraining_exclusion_verified": True,
        },
        "runtime": {
            "python": "3.12",
            "platform": "test",
            "torch": "test",
            "device": "cpu",
            "original_training_seconds": 1.0,
        },
        "exact_retraining_metrics_by_seed": [{"seed": 1}, {"seed": 2}],
        "checkpoints": {"original": {"path": str(checkpoint), "sha256": _sha(checkpoint)}},
        "predictions": {"original": {"path": str(predictions), "sha256": _sha(predictions)}},
    }


def test_valid_fixture_passes_but_pilot_is_not_publication_eligible(tmp_path: Path) -> None:
    report = evaluate_result(_valid_result(tmp_path))
    assert report.passed
    assert not report.publication_eligible


def test_leaked_forgotten_id_is_detected(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    result["deletion"]["retain_ids"].append(1)
    assert "forget_retain_disjoint" in evaluate_result(result).failures


def test_corrupted_checkpoint_is_detected(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    Path(result["checkpoints"]["original"]["path"]).write_bytes(b"tampered")
    assert "checkpoints_present_and_hashed" in evaluate_result(result).failures


def test_missing_predictions_and_nonfinite_metric_are_detected(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    result["predictions"] = {}
    result["metric"] = float("nan")
    failures = evaluate_result(result).failures
    assert "raw_predictions_and_labels_present" in failures
    assert "all_numeric_outputs_finite" in failures


def test_changed_metadata_cannot_inherit_a_valid_hash(tmp_path: Path) -> None:
    original = _valid_result(tmp_path)
    changed = copy.deepcopy(original)
    changed["maturity"] = "NOT_A_STATE"
    assert "recognized_maturity" in evaluate_result(changed).failures


def test_nonfinite_representation_artifact_is_detected(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    prediction_path = Path(result["predictions"]["original"]["path"])
    with np.load(prediction_path, allow_pickle=False) as existing:
        arrays = {name: existing[name] for name in existing.files}
    arrays["test_features"][0, 0] = np.nan
    np.savez_compressed(prediction_path, **arrays)
    result["predictions"]["original"]["sha256"] = _sha(prediction_path)
    assert "raw_predictions_and_labels_present" in evaluate_result(result).failures


def test_prediction_schema_two_requires_features(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    prediction_path = Path(result["predictions"]["original"]["path"])
    with np.load(prediction_path, allow_pickle=False) as existing:
        arrays = {name: existing[name] for name in existing.files if name != "test_features"}
    np.savez_compressed(prediction_path, **arrays)
    result["predictions"]["original"]["sha256"] = _sha(prediction_path)
    assert "raw_predictions_and_labels_present" in evaluate_result(result).failures


def test_schema_three_assay_requires_complete_partition_lineage(tmp_path: Path) -> None:
    fixture = _valid_result(tmp_path)
    manifest_path = tmp_path / "deletion.json"
    manifest = {
        "dataset_sha256": "dataset-hash",
        "sample_ids": [1],
    }
    manifest["sample_ids_sha256"] = hashlib.sha256(b"[1]").hexdigest()
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run = {
        "seed": 1,
        "checkpoint": fixture["checkpoints"]["original"],
        "predictions": fixture["predictions"]["original"],
    }
    lineage = {
        "train_count": 4,
        "forget_count": 1,
        "retain_count": 3,
        "train_ids_sha256": "1" * 64,
        "forget_ids_sha256": "2" * 64,
        "retain_ids_sha256": "3" * 64,
        "partition_disjoint": True,
        "partition_reconstructs_train": True,
        "exact_retraining_exclusion_verified": True,
    }
    canonical_lineage = json.dumps(lineage, sort_keys=True, separators=(",", ":"))
    lineage["partition_sha256"] = hashlib.sha256(canonical_lineage.encode()).hexdigest()
    prediction_path = Path(run["predictions"]["path"])
    with np.load(prediction_path, allow_pickle=False) as existing:
        arrays = {name: existing[name] for name in existing.files}
    arrays["partition_sha256"] = np.array(lineage["partition_sha256"])
    np.savez_compressed(prediction_path, **arrays)
    run["predictions"]["sha256"] = _sha(prediction_path)
    exact_run = copy.deepcopy(run)
    exact_run["partition_sha256"] = lineage["partition_sha256"]
    result = {
        "schema_version": 3,
        "maturity": "PILOT",
        "status": "completed",
        "config_sha256": "a" * 64,
        "dataset": {"train_count": 4, "sha256": "dataset-hash"},
        "original_runs": [run, copy.deepcopy(run)],
        "conditions": [
            {
                "forget_count": 1,
                "manifest": {"path": str(manifest_path), "sha256": _sha(manifest_path)},
                "partition_lineage": lineage,
                "metrics_partition_sha256": lineage["partition_sha256"],
                "exact_runs": [exact_run, copy.deepcopy(exact_run)],
            }
        ],
        "runtime": {"python": "3.13", "platform": "test", "torch": "test", "device": "cpu"},
    }

    assert evaluate_result(result).passed
    with np.load(prediction_path, allow_pickle=False) as existing:
        arrays = {name: existing[name] for name in existing.files}
    for split in ("retain", "forget"):
        for field in (
            "ids",
            "true_labels",
            "probabilities",
            "predicted_labels",
            "logits",
            "features",
        ):
            arrays[f"{split}_{field}"] = arrays[f"test_{field}"]
    np.savez_compressed(prediction_path, **arrays)
    run["predictions"]["sha256"] = _sha(prediction_path)
    exact_run["predictions"]["sha256"] = _sha(prediction_path)
    result["original_runs"] = [run, copy.deepcopy(run)]
    result["conditions"][0]["exact_runs"] = [exact_run, copy.deepcopy(exact_run)]
    method_run = copy.deepcopy(exact_run)
    method_run["method"] = "fine_tune"
    result["conditions"][0]["method_runs"] = [method_run]
    assert evaluate_result(result).passed
    with np.load(prediction_path, allow_pickle=False) as existing:
        arrays = {name: existing[name] for name in existing.files if name != "forget_features"}
    np.savez_compressed(prediction_path, **arrays)
    updated_hash = _sha(prediction_path)
    for original_run in result["original_runs"]:
        original_run["predictions"]["sha256"] = updated_hash
    for exact in result["conditions"][0]["exact_runs"]:
        exact["predictions"]["sha256"] = updated_hash
    result["conditions"][0]["method_runs"][0]["predictions"]["sha256"] = updated_hash
    assert "split_complete_method_study_predictions" in evaluate_result(result).failures
    arrays["forget_features"] = arrays["test_features"]
    np.savez_compressed(prediction_path, **arrays)
    updated_hash = _sha(prediction_path)
    for original_run in result["original_runs"]:
        original_run["predictions"]["sha256"] = updated_hash
    for exact in result["conditions"][0]["exact_runs"]:
        exact["predictions"]["sha256"] = updated_hash
    result["conditions"][0]["method_runs"][0]["predictions"]["sha256"] = updated_hash
    result["conditions"][0]["method_runs"][0]["partition_sha256"] = "f" * 64
    assert "partition_lineage_complete" in evaluate_result(result).failures
    result["conditions"][0]["method_runs"][0]["partition_sha256"] = lineage[
        "partition_sha256"
    ]
    result["conditions"][0]["method_runs"][0]["checkpoint"] = {
        "path": str(tmp_path / "missing.pt"),
        "sha256": "0" * 64,
    }
    assert "checkpoints_present_and_hashed" in evaluate_result(result).failures
    result["conditions"][0]["partition_lineage"]["retain_count"] = 2
    assert "partition_lineage_complete" in evaluate_result(result).failures


def test_schema_three_assay_rejects_tampered_deletion_manifest(tmp_path: Path) -> None:
    result = _valid_result(tmp_path)
    result["schema_version"] = 3
    result["conditions"] = []
    report = evaluate_result(result)
    assert "deletion_manifests_present_and_hashed" in report.failures
