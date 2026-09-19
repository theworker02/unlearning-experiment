from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.ssd_rules import select_candidate
from scripts.validate_ssd_artifacts import validate_artifacts


def test_ssd_artifact_validator_recomputes_selection_and_hashes(tmp_path: Path) -> None:
    protocol = yaml.safe_load(Path("study/preregistration/ssd_reproduction_v1.yaml").read_text())
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = {"path": str(checkpoint), "sha256": hashlib.sha256(b"checkpoint").hexdigest()}
    fisher = tmp_path / "full-fisher.pt"
    fisher.write_bytes(b"fisher")
    fisher_reference = {"path": str(fisher), "sha256": hashlib.sha256(b"fisher").hexdigest()}
    seeds = protocol["scope"]["original_seeds"]
    stage1_runs = [
        {
            "original_seed": seed,
            "accepted": True,
            "fisher_finite_nonnegative": True,
            "changed_parameter_l2": 1.0,
            "diagnostics": {
                "selected_parameter_count": 10,
                "selected_parameter_fraction": 0.1,
            },
        }
        for seed in seeds
    ]
    grid = protocol["stage_2_validation_only_tuning"]["grid"]
    hyperparameters = [
        {"alpha": alpha, "dampening_constant": dampening}
        for alpha in grid["alpha"]
        for dampening in grid["dampening_constant"]
    ]
    candidate_runs = []
    for index, candidate in enumerate(hyperparameters, start=1):
        for seed in seeds:
            candidate_runs.append(
                {
                    "candidate_id": f"candidate-{index:02d}",
                    "original_seed": seed,
                    "hyperparameters": candidate,
                    "validation_accuracy_drop": 0.01,
                    "forget_training_cross_entropy": float(index),
                    "selected_parameter_fraction": 0.1,
                    "checkpoint": reference,
                }
            )
    stage1 = {
        "experiment_id": protocol["artifact_policy"]["stage_1_experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": protocol_hash},
        "runs": stage1_runs,
        "overall_acceptance": True,
    }
    stage2 = {
        "experiment_id": protocol["artifact_policy"]["stage_2_experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": protocol_hash},
        "candidate_runs": candidate_runs,
        "original_checkpoints": {str(seed): reference for seed in seeds},
        "full_fisher_artifacts": {str(seed): fisher_reference for seed in seeds},
        "selection": select_candidate(candidate_runs),
        "final_test": [{"original_seed": seed, "metrics": {}} for seed in seeds],
    }
    stage1_path = tmp_path / "stage1.json"
    stage2_path = tmp_path / "stage2.json"
    stage1_path.write_text(json.dumps(stage1), encoding="utf-8")
    stage2_path.write_text(json.dumps(stage2), encoding="utf-8")

    assert validate_artifacts(stage1_path, stage2_path, protocol_path)["status"] == "PASS"

    stage2["full_fisher_artifacts"][str(seeds[0])]["sha256"] = "tampered"
    stage2_path.write_text(json.dumps(stage2), encoding="utf-8")
    tampered = validate_artifacts(stage1_path, stage2_path, protocol_path)
    assert tampered["status"] == "FAIL"
    assert f"original_seed_{seeds[0]}_full_fisher_invalid" in tampered["failures"]

    stage2["full_fisher_artifacts"][str(seeds[0])] = fisher_reference
    stage2["candidate_runs"][0]["hyperparameters"] = {
        "alpha": 999.0,
        "dampening_constant": 1.0,
    }
    stage2_path.write_text(json.dumps(stage2), encoding="utf-8")
    tampered = validate_artifacts(stage1_path, stage2_path, protocol_path)
    assert "candidate-01_hyperparameters_mismatch" in tampered["failures"]
