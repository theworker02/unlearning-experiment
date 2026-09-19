import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.ssd_fisher_sensitivity_rules import (
    conditions as build_conditions,
)
from scripts.ssd_fisher_sensitivity_rules import (
    fisher_distances,
    prediction_sensitivity,
    selected_mask,
    selected_set_metrics,
    state_l2,
)
from scripts.validate_ssd_fisher_sensitivity_artifact import validate_artifact


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_ssd_fisher_sensitivity_artifact_validator_fails_closed_when_missing(
    tmp_path: Path,
) -> None:
    result = validate_artifact(
        tmp_path / "missing.json",
        Path("study/preregistration/ssd_fisher_estimator_sensitivity_v1.yaml"),
    )
    assert result == {
        "eligible": False,
        "status": "MISSING",
        "failures": ["artifact_missing"],
    }


def test_ssd_fisher_sensitivity_artifact_validator_recomputes_all_metrics(
    tmp_path: Path,
) -> None:
    protocol = yaml.safe_load(
        Path("study/preregistration/ssd_fisher_estimator_sensitivity_v1.yaml").read_text()
    )
    protocol["scope"]["original_seeds"] = [1]
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    original = {"weight": torch.zeros(2)}
    updated = {"weight": torch.tensor([0.1, 0.0])}
    full = {"weight": torch.tensor([1.0, 1.0])}
    forget = {"weight": torch.tensor([20.0, 0.0])}
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 1])
    paths = {
        "original": tmp_path / "original.pt",
        "updated": tmp_path / "updated.pt",
        "full": tmp_path / "full.pt",
        "forget": tmp_path / "forget.pt",
        "predictions": tmp_path / "predictions.npz",
    }
    torch.save(original, paths["original"])
    torch.save(updated, paths["updated"])
    torch.save(full, paths["full"])
    torch.save(forget, paths["forget"])
    np.savez(paths["predictions"], probabilities=probabilities, labels=labels)
    mask = selected_mask(full, forget, alpha=10.0)
    comparison = {
        "full_fisher": fisher_distances(full, full),
        "forget_fisher": fisher_distances(forget, forget),
        "selection": selected_set_metrics(mask, mask),
        "model_update": {
            "parameter_change_l2": state_l2(updated, original),
            "pairwise_updated_parameter_l2": state_l2(updated, updated),
        },
        "validation_behavior": prediction_sensitivity(probabilities, probabilities, labels),
    }
    conditions = [
        {
            **condition,
            "full_fisher": _reference(paths["full"]),
            "forget_fisher": _reference(paths["forget"]),
            "updated_checkpoint": _reference(paths["updated"]),
            "validation_predictions": _reference(paths["predictions"]),
            "comparison_to_per_example_batch_1": comparison,
        }
        for condition in build_conditions(protocol)
    ]
    result = {
        "experiment_id": protocol["experiment_id"],
        "publication_evidence": False,
        "protocol": {"sha256": protocol_hash},
        "analytic_control": {
            "passed": True,
            "per_example": [[0.125, 0.125], [0.125, 0.125]],
            "minibatch_size_1": [[0.125, 0.125], [0.125, 0.125]],
            "minibatch_size_2": [[0.0625, 0.0625], [0.0625, 0.0625]],
        },
        "runs": [
            {
                "original_seed": 1,
                "original_checkpoint": _reference(paths["original"]),
                "conditions": conditions,
            }
        ],
        "overall_complete": True,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert validate_artifact(result_path, protocol_path)["status"] == "PASS"

    result["runs"][0]["conditions"][0]["comparison_to_per_example_batch_1"]["selection"][
        "selected_count"
    ] = 999
    result_path.write_text(json.dumps(result), encoding="utf-8")
    tampered = validate_artifact(result_path, protocol_path)
    assert tampered["status"] == "FAIL"
    assert "seed_1_per-example-batch-1_comparison_mismatch" in tampered["failures"]
