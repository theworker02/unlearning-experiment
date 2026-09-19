from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class Maturity(StrEnum):
    EXPLORATORY = "EXPLORATORY"
    PILOT = "PILOT"
    VALIDATION = "VALIDATION"
    CONFIRMATORY = "CONFIRMATORY"
    PUBLICATION = "PUBLICATION"
    REPRODUCED = "REPRODUCED"


RUNNABLE_MATURITIES = {
    Maturity.EXPLORATORY,
    Maturity.PILOT,
    Maturity.VALIDATION,
    Maturity.CONFIRMATORY,
}


class QualityGateError(RuntimeError):
    """Raised when a fail-closed scientific precondition is not satisfied."""


@dataclass(frozen=True)
class GateReport:
    passed: bool
    publication_eligible: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "publication_eligible": self.publication_eligible,
            "checks": self.checks,
            "failures": list(self.failures),
        }


def validate_launch_preconditions(raw: dict[str, Any]) -> Maturity:
    try:
        maturity = Maturity(str(raw["maturity"]).upper())
    except (KeyError, ValueError) as error:
        raise QualityGateError("a valid experiment maturity is required") from error
    if maturity not in RUNNABLE_MATURITIES:
        raise QualityGateError(
            "PUBLICATION and REPRODUCED are promotion states, not runnable configuration states"
        )
    if maturity is Maturity.CONFIRMATORY:
        quality = raw.get("quality")
        required = {
            "dataset_manifest_frozen",
            "split_manifest_frozen",
            "deletion_manifest_frozen",
            "training_config_frozen",
            "statistical_plan_frozen",
            "seed_manifest_frozen",
            "tests_passed",
            "expected_outputs_declared",
        }
        if not isinstance(quality, dict):
            raise QualityGateError("confirmatory experiment requires a quality precondition map")
        missing = sorted(name for name in required if quality.get(name) is not True)
        if missing:
            raise QualityGateError(f"confirmatory preconditions failed: {missing}")
    return maturity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _artifact_refs_valid(group: Any, required_suffix: str) -> bool:
    references: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                references.append(value)
            else:
                for child in value.values():
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(group)
    if not references:
        return False
    for reference in references:
        path = Path(str(reference["path"]))
        invalid = (
            path.suffix != required_suffix
            or not path.is_file()
            or _sha256(path) != reference["sha256"]
        )
        if invalid:
            return False
    return True


def _predictions_valid(group: Any) -> bool:
    if not _artifact_refs_valid(group, ".npz"):
        return False
    references: list[Path] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and str(value["path"]).endswith(".npz"):
                references.append(Path(str(value["path"])))
            else:
                for child in value.values():
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(group)
    required = {
        "test_ids",
        "test_true_labels",
        "test_probabilities",
        "test_predicted_labels",
        "test_logits",
    }
    for path in references:
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if not required.issubset(arrays.files):
                    return False
                if not np.isfinite(arrays["test_probabilities"]).all():
                    return False
                if not np.allclose(arrays["test_probabilities"].sum(axis=1), 1.0, atol=1e-5):
                    return False
                if len(arrays["test_ids"]) != len(arrays["test_true_labels"]):
                    return False
                if "test_features" in arrays.files and (
                    len(arrays["test_features"]) != len(arrays["test_ids"])
                    or not np.isfinite(arrays["test_features"]).all()
                ):
                    return False
                if "artifact_schema_version" in arrays.files:
                    version = int(np.asarray(arrays["artifact_schema_version"]).item())
                    if version != 2 or "test_features" not in arrays.files:
                        return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def _split_complete_predictions_valid(group: Any) -> bool:
    """Require test, retained, and forgotten outputs for method-comparison studies."""
    references: list[Path] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and str(value["path"]).endswith(".npz"):
                references.append(Path(str(value["path"])))
            else:
                for child in value.values():
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(group)
    if not references:
        return False
    required = {
        f"{split}_{field}"
        for split in ("test", "retain", "forget")
        for field in (
            "ids",
            "true_labels",
            "probabilities",
            "predicted_labels",
            "logits",
            "features",
        )
    }
    for path in references:
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if not required.issubset(arrays.files):
                    return False
                for split in ("test", "retain", "forget"):
                    probabilities = arrays[f"{split}_probabilities"]
                    if not np.isfinite(probabilities).all() or not np.allclose(
                        probabilities.sum(axis=1), 1.0, atol=1e-5
                    ):
                        return False
                    size = len(arrays[f"{split}_ids"])
                    if any(
                        len(arrays[f"{split}_{field}"]) != size
                        for field in (
                            "true_labels",
                            "probabilities",
                            "predicted_labels",
                            "logits",
                            "features",
                        )
                    ):
                        return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def _assay_partition_lineage_valid(result: dict[str, Any]) -> bool:
    train_count = result.get("dataset", {}).get("train_count")
    conditions = result.get("conditions", [])
    if not conditions:
        return False
    for condition in conditions:
        lineage = condition.get("partition_lineage", {})
        hashes = (
            lineage.get("train_ids_sha256"),
            lineage.get("forget_ids_sha256"),
            lineage.get("retain_ids_sha256"),
            lineage.get("partition_sha256"),
        )
        if not all(isinstance(value, str) and len(value) == 64 for value in hashes):
            return False
        if lineage.get("train_count") != train_count:
            return False
        if lineage.get("forget_count") != condition.get("forget_count"):
            return False
        if lineage.get("forget_count", -1) + lineage.get("retain_count", -1) != train_count:
            return False
        if not all(
            lineage.get(key) is True
            for key in (
                "partition_disjoint",
                "partition_reconstructs_train",
                "exact_retraining_exclusion_verified",
            )
        ):
            return False
        unsigned = {key: value for key, value in lineage.items() if key != "partition_sha256"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != lineage["partition_sha256"]:
            return False
        partitioned_runs = [
            *condition.get("exact_runs", []),
            *condition.get("method_runs", []),
        ]
        if not all(
            run.get("partition_sha256") == lineage["partition_sha256"]
            for run in partitioned_runs
        ):
            return False
        if condition.get("metrics_partition_sha256") != lineage["partition_sha256"]:
            return False
        for run in partitioned_runs:
            prediction_path = Path(str(run.get("predictions", {}).get("path", "")))
            try:
                with np.load(prediction_path, allow_pickle=False) as arrays:
                    embedded = str(np.asarray(arrays["partition_sha256"]).item())
            except (OSError, ValueError, KeyError):
                return False
            if embedded != lineage["partition_sha256"]:
                return False
    return True


def _assay_deletion_manifests_valid(result: dict[str, Any]) -> bool:
    conditions = result.get("conditions", [])
    dataset_hash = result.get("dataset", {}).get("sha256")
    if not conditions or not isinstance(dataset_hash, str):
        return False
    for condition in conditions:
        reference = condition.get("manifest", {})
        path = Path(str(reference.get("path", "")))
        if path.suffix != ".json" or not path.is_file() or _sha256(path) != reference.get("sha256"):
            return False
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            stored_hash = manifest["manifest_hash"]
            sample_ids = manifest["sample_ids"]
            sample_ids_hash = manifest["sample_ids_sha256"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return False
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
        encoded_ids = json.dumps(sample_ids, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != stored_hash:
            return False
        if hashlib.sha256(encoded_ids.encode()).hexdigest() != sample_ids_hash:
            return False
        if manifest.get("dataset_sha256") != dataset_hash:
            return False
        if len(sample_ids) != condition.get("forget_count") or len(sample_ids) != len(
            set(sample_ids)
        ):
            return False
    return True


def _evaluate_assay_result(result: dict[str, Any], maturity: Maturity | None) -> GateReport:
    conditions = result.get("conditions", [])
    exact_runs = [run for condition in conditions for run in condition.get("exact_runs", [])]
    method_runs = [run for condition in conditions for run in condition.get("method_runs", [])]
    runs = [*result.get("original_runs", []), *exact_runs, *method_runs]
    checkpoint_group = [run.get("checkpoint") for run in runs]
    prediction_group = [run.get("predictions") for run in runs]
    checks = {
        "recognized_maturity": maturity is not None,
        "process_completed": result.get("status") == "completed",
        "configuration_hash_present": isinstance(result.get("config_sha256"), str),
        "deletion_manifests_present_and_hashed": _assay_deletion_manifests_valid(result),
        "partition_lineage_complete": _assay_partition_lineage_valid(result),
        "all_numeric_outputs_finite": _all_finite(result),
        "checkpoints_present_and_hashed": _artifact_refs_valid(checkpoint_group, ".pt"),
        "raw_predictions_and_labels_present": _predictions_valid(prediction_group),
        "runtime_complete": all(
            result.get("runtime", {}).get(key) is not None
            for key in ("python", "platform", "torch", "device")
        ),
        "multiple_original_seeds": len(result.get("original_runs", [])) >= 2,
        "multiple_exact_seeds_per_condition": bool(conditions)
        and all(len(condition.get("exact_runs", [])) >= 2 for condition in conditions),
    }
    if method_runs:
        checks["split_complete_method_study_predictions"] = _split_complete_predictions_valid(
            [
                *[run.get("predictions") for run in exact_runs],
                *[run.get("predictions") for run in method_runs],
            ]
        )
    failures = tuple(name for name, passed in checks.items() if not passed)
    passed = not failures
    publication_eligible = passed and maturity in {Maturity.PUBLICATION, Maturity.REPRODUCED}
    return GateReport(passed, publication_eligible, checks, failures)


def evaluate_result(result: dict[str, Any]) -> GateReport:
    maturity_value = result.get("maturity")
    try:
        maturity = Maturity(str(maturity_value))
    except ValueError:
        maturity = None
    if result.get("schema_version") == 3 and "conditions" in result:
        return _evaluate_assay_result(result, maturity)
    deletion = result.get("deletion", {})
    forget = set(deletion.get("forget_ids", []))
    retain = set(deletion.get("retain_ids", []))
    checks = {
        "recognized_maturity": maturity is not None,
        "process_completed": result.get("status") == "completed",
        "configuration_hash_present": isinstance(result.get("config_sha256"), str),
        "forget_retain_disjoint": forget.isdisjoint(retain),
        "partition_reconstructs_train": len(forget | retain)
        == int(result.get("dataset", {}).get("train_count", -1)),
        "exact_exclusion_assertion": deletion.get("exact_retraining_exclusion_verified") is True,
        "all_numeric_outputs_finite": _all_finite(result),
        "checkpoints_present_and_hashed": _artifact_refs_valid(result.get("checkpoints"), ".pt"),
        "raw_predictions_and_labels_present": _predictions_valid(result.get("predictions")),
        "runtime_complete": all(
            result.get("runtime", {}).get(key) is not None
            for key in ("python", "platform", "torch", "device", "original_training_seconds")
        ),
        "multiple_exact_seeds": len(result.get("exact_retraining_metrics_by_seed", [])) >= 2,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    passed = not failures
    publication_eligible = passed and maturity in {Maturity.PUBLICATION, Maturity.REPRODUCED}
    return GateReport(passed, publication_eligible, checks, failures)


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
