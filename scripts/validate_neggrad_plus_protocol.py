from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

PROTOCOL = Path("study/preregistration/neggrad_plus_reproduction_v1.yaml")


def validate(path: Path = PROTOCOL) -> dict[str, object]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    status = protocol.get("status")
    if status not in {"DRAFT_NOT_LOCKED", "LOCKED"}:
        errors.append("status must be DRAFT_NOT_LOCKED or LOCKED")
    if (
        status == "DRAFT_NOT_LOCKED"
        and protocol.get("execution_authorized") is not False
    ):
        errors.append("draft execution must fail closed")
    if status == "LOCKED":
        amendment_path = Path(str(protocol.get("freeze_amendment", "")))
        if protocol.get("execution_authorized") is not True:
            errors.append("locked protocol must explicitly authorize execution")
        if not amendment_path.is_file():
            errors.append("locked protocol requires a freeze amendment")
        elif yaml.safe_load(amendment_path.read_text(encoding="utf-8")).get("status") != "LOCKED":
            errors.append("freeze amendment is not locked")
    seeds = protocol["scope"]["original_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != len(seeds):
        errors.append("exactly three distinct original seeds are required")
    tuning = protocol["stage_2_validation_only_tuning"]
    selection = tuning["selection_data"]
    if selection.get("test_split") != "prohibited":
        errors.append("test split must be prohibited during tuning")
    if selection.get("exact_retraining_predictions_or_distances") != "prohibited":
        errors.append("exact-counterfactual outcomes must be prohibited during tuning")
    grid = tuning["grid"]
    computed_candidates = len(
        list(
            product(
                grid["beta"],
                grid["epochs"],
                grid["optimizer"],
                grid["learning_rate"],
                grid["weight_decay"],
                grid["momentum"],
                grid["batch_size"],
            )
        )
    )
    if computed_candidates != tuning["candidate_count"]:
        errors.append("candidate_count does not match the declared Cartesian grid")
    compute = protocol["compute_control"]
    expected_candidate_runs = computed_candidates * len(seeds)
    if compute["planned_stage_2_candidate_runs"] != expected_candidate_runs:
        errors.append("planned candidate-run count is inconsistent")
    expected_updates = (
        compute["planned_original_trainings"]
        + compute["planned_stage_1_unlearning_runs"]
        + compute["planned_stage_2_candidate_runs"]
    )
    if compute["planned_total_model_updates"] != expected_updates:
        errors.append("planned total model-update count is inconsistent")
    if compute.get("launch_requires_measured_estimate") is not True:
        errors.append("measured compute estimate must precede launch")
    rule = tuning["selection_rule"]
    if rule.get("utility_constraint") != "validation_accuracy_drop_from_original_at_most_0.05":
        errors.append("utility constraint must remain frozen at 0.05")
    if errors:
        raise SystemExit("NegGrad+ protocol invalid:\n- " + "\n- ".join(errors))
    return {
        "status": f"VALID_{status}_PROTOCOL",
        "candidate_count": computed_candidates,
        "candidate_runs": expected_candidate_runs,
        "planned_total_model_updates": expected_updates,
    }


if __name__ == "__main__":
    result = validate()
    print(yaml.safe_dump(result, sort_keys=True).strip())
