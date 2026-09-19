from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

if __package__:
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
else:
    from ssd_fisher_sensitivity_rules import (
        conditions as build_conditions,
    )
    from ssd_fisher_sensitivity_rules import (
        fisher_distances,
        prediction_sensitivity,
        selected_mask,
        selected_set_metrics,
        state_l2,
    )


def _hash_valid(reference: dict[str, object]) -> bool:
    path = Path(str(reference.get("path", "")))
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference.get(
        "sha256"
    )


def _load_tensors(reference: dict[str, object]) -> dict[str, torch.Tensor]:
    return torch.load(Path(str(reference["path"])), map_location="cpu", weights_only=True)


def _matches(reported: Any, recomputed: Any, *, tolerance: float = 1e-10) -> bool:
    if isinstance(recomputed, dict):
        return isinstance(reported, dict) and set(reported) == set(recomputed) and all(
            _matches(reported[key], value, tolerance=tolerance)
            for key, value in recomputed.items()
        )
    if isinstance(recomputed, float):
        return isinstance(reported, int | float) and math.isclose(
            float(reported), recomputed, rel_tol=tolerance, abs_tol=tolerance
        )
    return reported == recomputed


def validate_artifact(result_path: Path, protocol_path: Path) -> dict[str, object]:
    if not result_path.is_file():
        return {"eligible": False, "status": "MISSING", "failures": ["artifact_missing"]}
    failures: list[str] = []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, yaml.YAMLError) as error:
        return {"eligible": False, "status": "INVALID", "failures": [type(error).__name__]}
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if result.get("experiment_id") != protocol.get("experiment_id"):
        failures.append("experiment_id_mismatch")
    if result.get("publication_evidence") is not False:
        failures.append("publication_role_invalid")
    if result.get("protocol", {}).get("sha256") != protocol_hash:
        failures.append("protocol_hash_mismatch")
    analytic = result.get("analytic_control", {})
    expected_per_example = [[0.125, 0.125], [0.125, 0.125]]
    expected_minibatch_two = [[0.0625, 0.0625], [0.0625, 0.0625]]
    if (
        analytic.get("passed") is not True
        or analytic.get("per_example") != expected_per_example
        or analytic.get("minibatch_size_1") != expected_per_example
        or analytic.get("minibatch_size_2") != expected_minibatch_two
    ):
        failures.append("analytic_control_invalid")

    expected_seeds = [int(seed) for seed in protocol["scope"]["original_seeds"]]
    expected_conditions = build_conditions(protocol)
    expected_ids = [str(condition["condition_id"]) for condition in expected_conditions]
    runs = result.get("runs", [])
    if [run.get("original_seed") for run in runs] != expected_seeds:
        failures.append("seed_set_mismatch")
    alpha = float(protocol["scope"]["ssd_hyperparameters"]["alpha"])
    for run in runs:
        seed = run.get("original_seed")
        if not _hash_valid(run.get("original_checkpoint", {})):
            failures.append(f"seed_{seed}_original_checkpoint_invalid")
            continue
        original_state = _load_tensors(run["original_checkpoint"])
        conditions = run.get("conditions", [])
        if [condition.get("condition_id") for condition in conditions] != expected_ids:
            failures.append(f"seed_{seed}_condition_grid_mismatch")
            continue
        loaded = {}
        for condition, expected in zip(conditions, expected_conditions, strict=True):
            condition_id = str(condition.get("condition_id"))
            if any(condition.get(key) != expected[key] for key in expected):
                failures.append(f"seed_{seed}_{condition_id}_definition_mismatch")
            references = (
                condition.get("full_fisher", {}),
                condition.get("forget_fisher", {}),
                condition.get("updated_checkpoint", {}),
                condition.get("validation_predictions", {}),
            )
            if not all(_hash_valid(reference) for reference in references):
                failures.append(f"seed_{seed}_{condition_id}_artifact_hash_invalid")
                continue
            full_fisher = _load_tensors(condition["full_fisher"])
            forget_fisher = _load_tensors(condition["forget_fisher"])
            state = _load_tensors(condition["updated_checkpoint"])
            with np.load(Path(condition["validation_predictions"]["path"])) as predictions:
                probabilities = predictions["probabilities"]
                labels = predictions["labels"]
            if not all(
                torch.all(torch.isfinite(value)) and torch.all(value >= 0)
                for mapping in (full_fisher, forget_fisher)
                for value in mapping.values()
            ):
                failures.append(f"seed_{seed}_{condition_id}_fisher_invalid")
            loaded[condition_id] = {
                "full": full_fisher,
                "forget": forget_fisher,
                "state": state,
                "probabilities": probabilities,
                "labels": labels,
                "reported": condition.get("comparison_to_per_example_batch_1", {}),
            }
        if len(loaded) != len(expected_ids):
            continue
        reference = loaded["per-example-batch-1"]
        reference_mask = selected_mask(reference["full"], reference["forget"], alpha=alpha)
        for condition_id in expected_ids:
            condition = loaded[condition_id]
            mask = selected_mask(condition["full"], condition["forget"], alpha=alpha)
            recomputed = {
                "full_fisher": fisher_distances(condition["full"], reference["full"]),
                "forget_fisher": fisher_distances(condition["forget"], reference["forget"]),
                "selection": selected_set_metrics(mask, reference_mask),
                "model_update": {
                    "parameter_change_l2": state_l2(condition["state"], original_state),
                    "pairwise_updated_parameter_l2": state_l2(
                        condition["state"], reference["state"]
                    ),
                },
                "validation_behavior": prediction_sensitivity(
                    condition["probabilities"],
                    reference["probabilities"],
                    condition["labels"],
                ),
            }
            if not _matches(condition["reported"], recomputed):
                failures.append(f"seed_{seed}_{condition_id}_comparison_mismatch")
        for batch_size in (32, 128):
            candidate = loaded[f"per-example-batch-{batch_size}"]
            if any(
                not torch.equal(candidate[kind][name], reference[kind][name])
                for kind in ("full", "forget")
                for name in reference[kind]
            ):
                failures.append(f"seed_{seed}_per_example_batch_invariance_failed_{batch_size}")
    if result.get("overall_complete") is not True or not runs:
        failures.append("overall_complete_not_true")
    return {
        "eligible": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "artifact": str(result_path),
        "protocol_sha256": protocol_hash,
    }
