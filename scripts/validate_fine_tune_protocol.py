from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

PROTOCOL = Path("study/preregistration/fine_tune_tuning_v1.yaml")


def validate(path: Path = PROTOCOL) -> dict[str, object]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if protocol.get("status") not in {"DRAFT_NOT_LOCKED", "LOCKED"}:
        errors.append("status must be DRAFT_NOT_LOCKED or LOCKED")
    if (
        protocol.get("status") == "DRAFT_NOT_LOCKED"
        and protocol.get("execution_authorized") is not False
    ):
        errors.append("draft execution must fail closed")
    seeds = protocol["scope"]["original_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        errors.append("exactly three distinct original seeds are required")
    tuning = protocol["validation_only_tuning"]
    selection = tuning["selection_data"]
    if selection.get("test_split") != "prohibited":
        errors.append("test split must be prohibited during tuning")
    if selection.get("exact_retraining_predictions_or_distances") != "prohibited":
        errors.append("exact-counterfactual outcomes must be prohibited during tuning")
    grid = tuning["grid"]
    keys = ("epochs", "learning_rate", "batch_size", "optimizer", "weight_decay")
    candidate_count = len(list(product(*(grid[key] for key in keys))))
    if candidate_count != tuning["candidate_count"]:
        errors.append("candidate_count does not match the declared Cartesian grid")
    compute = protocol["compute_control"]
    expected_runs = candidate_count * len(seeds)
    if compute["planned_candidate_runs"] != expected_runs:
        errors.append("planned candidate-run count is inconsistent")
    if compute["planned_total_model_updates"] != (
        compute["planned_original_trainings"] + expected_runs
    ):
        errors.append("planned total model-update count is inconsistent")
    if compute.get("launch_requires_measured_estimate") is not True:
        errors.append("measured compute estimate must precede launch")
    if tuning["selection_rule"].get("utility_constraint") != (
        "validation_accuracy_drop_from_original_at_most_0.05"
    ):
        errors.append("utility constraint must remain frozen at 0.05")
    if errors:
        raise SystemExit("fine-tuning protocol invalid:\n- " + "\n- ".join(errors))
    return {
        "status": "VALID_EXECUTION_DISABLED_PROTOCOL",
        "candidate_count": candidate_count,
        "candidate_runs": expected_runs,
        "planned_total_model_updates": compute["planned_total_model_updates"],
    }


if __name__ == "__main__":
    print(yaml.safe_dump(validate(), sort_keys=True).strip())
