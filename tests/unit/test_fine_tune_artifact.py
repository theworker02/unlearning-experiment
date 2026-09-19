import hashlib
import itertools
import json
from pathlib import Path

import yaml

from scripts.fine_tune_rules import select_candidate
from scripts.validate_fine_tune_artifact import validate_artifact


def test_fine_tune_artifact_validator_recomputes_selection_and_hashes(tmp_path: Path) -> None:
    protocol = yaml.safe_load(Path("study/preregistration/fine_tune_tuning_v1.yaml").read_text())
    protocol["scope"]["original_seeds"] = [1]
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = {"path": str(checkpoint), "sha256": hashlib.sha256(b"checkpoint").hexdigest()}
    grid = protocol["validation_only_tuning"]["grid"]
    keys = ("epochs", "learning_rate", "batch_size", "optimizer", "weight_decay")
    candidate_runs = []
    for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), start=1):
        candidate_runs.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "original_seed": 1,
                "hyperparameters": dict(zip(keys, values, strict=True)),
                "validation_accuracy": 0.8,
                "validation_accuracy_drop": 0.01,
                "forget_training_cross_entropy": float(index),
                "runtime_seconds": 1.0,
                "checkpoint": reference,
            }
        )
    result = {
        "experiment_id": protocol["experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": protocol_hash},
        "original_checkpoints": {"1": reference},
        "candidate_runs": candidate_runs,
        "selection": select_candidate(candidate_runs),
        "final_test": [{"original_seed": 1, "metrics": {}}],
        "overall_complete": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert validate_artifact(result_path, protocol_path)["status"] == "PASS"

    result["selection"]["selected"]["candidate_id"] = "candidate-01"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    tampered = validate_artifact(result_path, protocol_path)
    assert tampered["status"] == "FAIL"
    assert "selection_mismatch" in tampered["failures"]


def test_fine_tune_artifact_validator_fails_closed_when_missing(tmp_path: Path) -> None:
    result = validate_artifact(
        tmp_path / "missing.json", Path("study/preregistration/fine_tune_tuning_v1.yaml")
    )
    assert result["status"] == "MISSING"
