from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from pathlib import Path

import yaml

if __package__:
    from scripts.ssd_rules import select_candidate
else:
    from ssd_rules import select_candidate


def _hash_valid(reference: dict[str, object]) -> bool:
    path = Path(str(reference.get("path", "")))
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference.get(
        "sha256"
    )


def validate_artifacts(
    stage1_path: Path, stage2_path: Path, protocol_path: Path
) -> dict[str, object]:
    if not stage1_path.is_file() or not stage2_path.is_file():
        return {"eligible": False, "status": "MISSING", "failures": ["artifact_missing"]}
    failures: list[str] = []
    try:
        stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
        stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, yaml.YAMLError) as error:
        return {"eligible": False, "status": "INVALID", "failures": [type(error).__name__]}
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    expected_seeds = [int(seed) for seed in protocol["scope"]["original_seeds"]]
    expected_ids = {
        protocol["artifact_policy"]["stage_1_experiment_id"],
        protocol["artifact_policy"]["stage_2_experiment_id"],
    }
    if {stage1.get("experiment_id"), stage2.get("experiment_id")} != expected_ids:
        failures.append("experiment_id_mismatch")
    for stage in (stage1, stage2):
        if stage.get("publication_evidence") is not False:
            failures.append("publication_role_invalid")
        if stage.get("protocol", {}).get("sha256") != protocol_hash:
            failures.append("protocol_hash_mismatch")
    sanity_runs = stage1.get("runs", [])
    if [run.get("original_seed") for run in sanity_runs] != expected_seeds:
        failures.append("sanity_seed_set_mismatch")
    for run in sanity_runs:
        diagnostics = run.get("diagnostics", {})
        fraction = float(diagnostics.get("selected_parameter_fraction", float("nan")))
        changed_l2 = float(run.get("changed_parameter_l2", float("nan")))
        accepted = (
            int(diagnostics.get("selected_parameter_count", 0)) > 0
            and math.isfinite(fraction)
            and 0 < fraction < 1
            and math.isfinite(changed_l2)
            and changed_l2 > 0
            and run.get("fisher_finite_nonnegative") is True
        )
        if run.get("accepted") is not accepted:
            failures.append(f"seed_{run.get('original_seed')}_sanity_acceptance_mismatch")
    if stage1.get("overall_acceptance") is not True or not sanity_runs:
        failures.append("sanity_overall_acceptance_not_true")

    candidate_runs = stage2.get("candidate_runs", [])
    tuning = protocol["stage_2_validation_only_tuning"]
    candidate_count = int(tuning["candidate_count"])
    grid = tuning["grid"]
    expected_hyperparameters = {
        f"candidate-{index:02d}": {
            "alpha": float(alpha),
            "dampening_constant": float(dampening),
        }
        for index, (alpha, dampening) in enumerate(
            product(grid["alpha"], grid["dampening_constant"]), start=1
        )
    }
    expected_pairs = {
        (f"candidate-{index:02d}", seed)
        for index in range(1, candidate_count + 1)
        for seed in expected_seeds
    }
    observed_pairs = {
        (str(run.get("candidate_id")), int(run.get("original_seed", -1)))
        for run in candidate_runs
    }
    if len(candidate_runs) != len(expected_pairs) or observed_pairs != expected_pairs:
        failures.append("candidate_seed_grid_incomplete_or_duplicated")
    for run in candidate_runs:
        candidate_id = str(run.get("candidate_id"))
        if run.get("hyperparameters") != expected_hyperparameters.get(candidate_id):
            failures.append(f"{candidate_id}_hyperparameters_mismatch")
        scalar_metrics = (
            float(run.get("validation_accuracy_drop", float("nan"))),
            float(run.get("forget_training_cross_entropy", float("nan"))),
            float(run.get("selected_parameter_fraction", float("nan"))),
        )
        if not all(math.isfinite(value) for value in scalar_metrics):
            failures.append(f"{candidate_id}_seed_{run.get('original_seed')}_metrics_nonfinite")
        if not 0 <= scalar_metrics[2] <= 1:
            failures.append(
                f"{candidate_id}_seed_{run.get('original_seed')}_selected_fraction_invalid"
            )
        if not _hash_valid(run.get("checkpoint", {})):
            failures.append(
                f"{run.get('candidate_id')}_seed_{run.get('original_seed')}_checkpoint_invalid"
            )
    for seed in expected_seeds:
        key = str(seed)
        if not _hash_valid(stage2.get("original_checkpoints", {}).get(key, {})):
            failures.append(f"original_seed_{seed}_checkpoint_invalid")
        if not _hash_valid(stage2.get("full_fisher_artifacts", {}).get(key, {})):
            failures.append(f"original_seed_{seed}_full_fisher_invalid")
    if candidate_runs:
        try:
            recomputed = select_candidate(candidate_runs)
        except ValueError:
            failures.append("no_feasible_candidate")
        else:
            if stage2.get("selection") != recomputed:
                failures.append("selection_mismatch")
            final_seeds = [item.get("original_seed") for item in stage2.get("final_test", [])]
            if final_seeds != expected_seeds:
                failures.append("winner_only_test_evaluation_incomplete")
    return {
        "eligible": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "artifacts": [str(stage1_path), str(stage2_path)],
        "protocol_sha256": protocol_hash,
    }
