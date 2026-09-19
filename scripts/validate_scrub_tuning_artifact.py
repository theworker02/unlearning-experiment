from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import yaml

if __package__:
    from scripts.scrub_sanity_rules import evaluate_mechanism
    from scripts.scrub_tuning_rules import select_candidate
else:
    from scrub_sanity_rules import evaluate_mechanism
    from scrub_tuning_rules import select_candidate


def _hash_valid(reference: dict[str, object]) -> bool:
    path = Path(str(reference.get("path", "")))
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference.get(
        "sha256"
    )


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
    prerequisite = result.get("prerequisite", {})
    if (
        not _hash_valid(prerequisite)
        or prerequisite.get("test_metrics_ignored_for_tuning") is not True
    ):
        failures.append("prerequisite_artifact_invalid")
    seeds = [int(seed) for seed in protocol["scope"]["original_seeds"]]
    grid = protocol["validation_only_tuning"]["grid"]
    keys = tuple(grid)
    expected_hyperparameters = {
        f"candidate-{index:02d}": dict(zip(keys, values, strict=True))
        for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), start=1)
    }
    expected_pairs = {
        (candidate_id, seed) for candidate_id in expected_hyperparameters for seed in seeds
    }
    runs = result.get("candidate_runs", [])
    observed_pairs = {
        (str(run.get("candidate_id")), int(run.get("original_seed", -1))) for run in runs
    }
    if len(runs) != len(expected_pairs) or observed_pairs != expected_pairs:
        failures.append("candidate_seed_grid_incomplete_or_duplicated")
    for run in runs:
        candidate_id = str(run.get("candidate_id"))
        if run.get("hyperparameters") != expected_hyperparameters.get(candidate_id):
            failures.append(f"{candidate_id}_hyperparameters_mismatch")
        trace = run.get("mechanism_trace", [])
        try:
            acceptance = evaluate_mechanism(trace)
        except (IndexError, KeyError, TypeError):
            failures.append(f"{candidate_id}_seed_{run.get('original_seed')}_trace_invalid")
        else:
            if run.get("mechanism_acceptance") != acceptance or run.get(
                "mechanism_valid"
            ) is not all(acceptance.values()):
                failures.append(
                    f"{candidate_id}_seed_{run.get('original_seed')}_mechanism_mismatch"
                )
        numeric = (
            run.get("validation_accuracy"),
            run.get("validation_accuracy_drop"),
            run.get("forget_training_cross_entropy"),
            run.get("runtime_seconds"),
        )
        if not all(
            isinstance(value, int | float) and math.isfinite(float(value)) for value in numeric
        ):
            failures.append(f"{candidate_id}_seed_{run.get('original_seed')}_metrics_nonfinite")
        if not _hash_valid(run.get("checkpoint", {})):
            failures.append(f"{candidate_id}_seed_{run.get('original_seed')}_checkpoint_invalid")
    for seed in seeds:
        originals = result.get("original_checkpoints", {})
        if not _hash_valid(originals.get(str(seed), originals.get(seed, {}))):
            failures.append(f"original_seed_{seed}_checkpoint_invalid")
    if runs:
        try:
            recomputed = select_candidate(runs)
        except ValueError:
            failures.append("no_feasible_candidate")
        else:
            if result.get("selection") != recomputed:
                failures.append("selection_mismatch")
            if [item.get("original_seed") for item in result.get("final_test", [])] != seeds:
                failures.append("winner_only_test_evaluation_incomplete")
    if result.get("overall_complete") is not True:
        failures.append("overall_complete_not_true")
    return {
        "eligible": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "artifact": str(result_path),
        "protocol_sha256": protocol_hash,
    }
