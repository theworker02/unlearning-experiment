from __future__ import annotations

from pathlib import Path

import yaml

PROTOCOL = Path("study/preregistration/ssd_fisher_estimator_sensitivity_v1.yaml")


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
    estimators = protocol["estimators"]
    common = estimators["common_execution"]
    if common.get("model_mode") != "evaluation" or common.get("dataloader_shuffle") is not False:
        errors.append("estimator model mode and batch ordering must remain fixed")
    per_example_sizes = estimators["paper_aligned_per_example"]["loader_batch_sizes"]
    official_sizes = estimators["pinned_official_style_minibatch"]["batch_sizes"]
    if per_example_sizes != [1, 32, 128] or official_sizes != [1, 32, 128]:
        errors.append("both estimator batch-size grids must remain [1, 32, 128]")
    if protocol["data_access"].get("test_split") != "prohibited":
        errors.append("test split must be prohibited")
    if protocol["data_access"].get("exact_retraining_predictions_or_distances") != "prohibited":
        errors.append("exact-retraining outcomes must be prohibited")
    compute = protocol["compute_control"]
    expected_estimates = len(seeds) * (len(per_example_sizes) + len(official_sizes))
    if compute["planned_full_fisher_estimates"] != expected_estimates:
        errors.append("planned full-Fisher estimate count is inconsistent")
    if compute["planned_forget_fisher_estimates"] != expected_estimates:
        errors.append("planned forget-Fisher estimate count is inconsistent")
    if compute["planned_ssd_updates"] != expected_estimates:
        errors.append("planned SSD update count is inconsistent")
    if compute["planned_test_evaluations"] != 0:
        errors.append("test evaluations must remain zero")
    if compute.get("launch_requires_measured_estimate") is not True:
        errors.append("measured compute estimate must precede launch")
    if errors:
        raise SystemExit("SSD Fisher sensitivity protocol invalid:\n- " + "\n- ".join(errors))
    return {
        "status": "VALID_EXECUTION_DISABLED_PROTOCOL",
        "seed_count": len(seeds),
        "estimator_conditions_per_seed": len(per_example_sizes) + len(official_sizes),
        "planned_full_fisher_estimates": expected_estimates,
        "planned_ssd_updates": expected_estimates,
    }


if __name__ == "__main__":
    print(yaml.safe_dump(validate(), sort_keys=True).strip())
