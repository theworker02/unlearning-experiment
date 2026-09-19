import hashlib
import itertools
import json
from pathlib import Path

import yaml

from scripts.scrub_sanity_rules import evaluate_mechanism
from scripts.scrub_tuning_rules import select_candidate
from scripts.validate_scrub_tuning_artifact import validate_artifact


def test_scrub_tuning_artifact_recomputes_mechanism_selection_and_hashes(
    tmp_path: Path,
) -> None:
    protocol = yaml.safe_load(Path("study/preregistration/scrub_tuning_v1.yaml").read_text())
    protocol["scope"]["original_seeds"] = [1]
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = {"path": str(checkpoint), "sha256": hashlib.sha256(b"checkpoint").hexdigest()}
    prerequisite_path = tmp_path / "sanity.json"
    prerequisite_path.write_text("{}", encoding="utf-8")
    prerequisite_reference = {
        "path": str(prerequisite_path),
        "sha256": hashlib.sha256(prerequisite_path.read_bytes()).hexdigest(),
        "test_metrics_ignored_for_tuning": True,
    }
    trace = [
        {
            "max_gradient_l2_mean": 1.0,
            "before_max": {"forget_teacher_student_kl": 0.0},
            "after_max": {"forget_teacher_student_kl": 1.0},
            "before_min": {"retain_objective": 2.0},
            "after_min": {"retain_objective": 1.0},
        }
    ]
    acceptance = evaluate_mechanism(trace)
    grid = protocol["validation_only_tuning"]["grid"]
    keys = tuple(grid)
    runs = []
    for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), start=1):
        runs.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "original_seed": 1,
                "hyperparameters": dict(zip(keys, values, strict=True)),
                "mechanism_trace": trace,
                "mechanism_acceptance": acceptance,
                "mechanism_valid": True,
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
        "prerequisite": prerequisite_reference,
        "original_checkpoints": {"1": reference},
        "candidate_runs": runs,
        "selection": select_candidate(runs),
        "final_test": [{"original_seed": 1, "metrics": {}}],
        "overall_complete": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert validate_artifact(result_path, protocol_path)["status"] == "PASS"

    result["candidate_runs"][0]["mechanism_valid"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")
    tampered = validate_artifact(result_path, protocol_path)
    assert tampered["status"] == "FAIL"
    assert "candidate-01_seed_1_mechanism_mismatch" in tampered["failures"]


def test_scrub_tuning_artifact_fails_closed_when_missing(tmp_path: Path) -> None:
    assert validate_artifact(
        tmp_path / "missing.json", Path("study/preregistration/scrub_tuning_v1.yaml")
    )["status"] == "MISSING"
