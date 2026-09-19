from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.neggrad_plus_rules import select_candidate
from scripts.validate_experiment_profiles import (
    validate_all,
    validate_neggrad_plus_artifacts,
    validate_scrub_sanity_artifact,
)


def test_publication_profiles_validate_and_fail_closed() -> None:
    result = validate_all()
    assert result["status"] == "PASS_WITH_MAIN_PROFILE_BLOCKED"
    assert result["pilot_job_count"] == {
        "original_training_jobs": 1,
        "exact_retraining_jobs": 15,
        "approximate_unlearning_jobs": 9,
        "total_jobs_excluding_sequential_development": 25,
    }
    assert set(result["main_unresolved_fields"]) == {
        "original_seed_count",
        "exact_retraining_seed_count",
        "candidate_seed_count",
    }
    assert result["main_count_decision"]["status"] == "NOT_APPLICABLE"
    assert result["pilot_method_blockers"] == ["scrub"]
    scrub = result["method_readiness"]["pilot.yaml"]["scrub"]
    assert scrub["status"] == "FAIL"
    assert "overall_acceptance_not_true" in scrub["failures"]
    assert result["method_readiness"]["pilot.yaml"]["fine_tune"]["status"] == "PASS"
    assert set(result["method_readiness"]["main.yaml"]) == {
        "fine_tune",
        "neggrad_plus",
        "scrub",
        "scrub_rewind",
        "ssd",
    }
    ssd = result["method_readiness"]["main.yaml"]["ssd"]
    assert ssd["status"] == "BLOCKED"
    assert ssd["sensitivity_artifact_validation"]["status"] == "MISSING"
    assert ssd["reproduction_artifact_validation"]["status"] == "MISSING"
    assert "passing_ssd_reproduction_artifacts" in ssd["failures"]
    assert "frozen_validation_only_tuning_budget" not in ssd["failures"]
    assert "reproduction_check_missing" not in ssd["failures"]
    for profile in result["method_readiness"].values():
        for method in profile.values():
            assert len(method["failures"]) == len(set(method["failures"]))


def test_scrub_sanity_artifact_requires_matching_protocol_trace_and_checkpoints(tmp_path) -> None:
    protocol = yaml.safe_load(
        Path("study/preregistration/scrub_sanity_reproduction_v1.yaml").read_text()
    )
    protocol["scope"]["original_seeds"] = [3101]
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-test-checkpoint")
    checkpoint_reference = {
        "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    trace = [
        {
            "max_gradient_l2_mean": 0.2,
            "before_max": {"forget_teacher_student_kl": 0.0},
            "after_max": {"forget_teacher_student_kl": 0.3},
            "before_min": {"retain_objective": 1.2},
            "after_min": {"retain_objective": 0.8},
        }
    ]
    acceptance = {
        "initial_student_teacher_logits_are_not_identical": True,
        "first_max_epoch_increases_forget_teacher_student_kl": True,
        "first_min_epoch_decreases_retain_objective_on_fixed_probe": True,
        "all_recorded_values_are_finite": True,
    }
    result = {
        "experiment_id": protocol["experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()},
        "runs": [
            {
                "original_seed": 3101,
                "mechanism_trace": trace,
                "acceptance": acceptance,
                "accepted": True,
                "original_checkpoint": checkpoint_reference,
                "unlearned_checkpoint": checkpoint_reference,
            }
        ],
        "overall_acceptance": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert validate_scrub_sanity_artifact(result_path, protocol_path)["status"] == "PASS"
    result["protocol"]["sha256"] = "stale"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    stale = validate_scrub_sanity_artifact(result_path, protocol_path)
    assert stale["status"] == "FAIL"
    assert "protocol_hash_mismatch" in stale["failures"]


def test_pilot_authorization_is_rejected_without_passing_scrub_artifact(tmp_path) -> None:
    source_root = Path("experiment_profiles")
    for name in ("pilot.yaml", "main.yaml", "extended.yaml"):
        profile = yaml.safe_load((source_root / name).read_text())
        if name == "pilot.yaml":
            profile["launch_authorized"] = True
            profile["method_requirements"]["scrub"]["required_artifact"] = str(
                tmp_path / "missing-result.json"
            )
        (tmp_path / name).write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="launch blocked by method readiness"):
        validate_all(tmp_path)


def test_main_count_cannot_be_inserted_without_frozen_artifact(tmp_path) -> None:
    source_root = Path("experiment_profiles")
    for name in ("pilot.yaml", "main.yaml", "extended.yaml"):
        profile = yaml.safe_load((source_root / name).read_text())
        if name == "main.yaml":
            profile["exact_retraining_seed_count"] = 15
        (tmp_path / name).write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid count decision"):
        validate_all(tmp_path)


def test_neggrad_artifact_validator_recomputes_selection_and_hashes(tmp_path) -> None:
    protocol = yaml.safe_load(
        Path("study/preregistration/neggrad_plus_reproduction_v1.yaml").read_text()
    )
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = {"path": str(checkpoint), "sha256": hashlib.sha256(b"checkpoint").hexdigest()}
    seeds = protocol["scope"]["original_seeds"]
    sanity_runs = [
        {
            "seed": seed,
            "accepted": True,
            "diagnostics": {
                "gradient_l2_mean": 1.0,
                "before": {"forget_cross_entropy": 1.0},
                "after": {"forget_cross_entropy": 2.0},
            },
        }
        for seed in seeds
    ]
    candidate_runs = []
    for index in range(1, 9):
        for seed in seeds:
            candidate_runs.append(
                {
                    "candidate_id": f"candidate-{index:02d}",
                    "original_seed": seed,
                    "hyperparameters": {"beta": 0.95 + index / 1000, "epochs": 5},
                    "validation_accuracy_drop": 0.01,
                    "target_class_validation_cross_entropy": float(index),
                    "checkpoint": reference,
                }
            )
    selection = select_candidate(candidate_runs)
    stage1 = {
        "experiment_id": protocol["artifact_policy"]["stage_1_experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": protocol_hash},
        "runs": sanity_runs,
        "overall_acceptance": True,
    }
    stage2 = {
        "experiment_id": protocol["artifact_policy"]["stage_2_experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": protocol_hash},
        "candidate_runs": candidate_runs,
        "original_checkpoints": {str(seed): reference for seed in seeds},
        "selection": selection,
        "final_test": [{"original_seed": seed, "metrics": {}} for seed in seeds],
    }
    stage1_path = tmp_path / "stage1.json"
    stage2_path = tmp_path / "stage2.json"
    stage1_path.write_text(json.dumps(stage1), encoding="utf-8")
    stage2_path.write_text(json.dumps(stage2), encoding="utf-8")
    assert validate_neggrad_plus_artifacts(
        stage1_path, stage2_path, protocol_path
    )["status"] == "PASS"
    stage2["selection"]["selected"]["candidate_id"] = "candidate-01"
    stage2_path.write_text(json.dumps(stage2), encoding="utf-8")
    tampered = validate_neggrad_plus_artifacts(stage1_path, stage2_path, protocol_path)
    assert tampered["status"] == "FAIL"
    assert "selection_mismatch" in tampered["failures"]
