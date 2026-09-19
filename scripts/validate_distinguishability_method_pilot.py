from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from exact_unlearning.methods import METHODS
from exact_unlearning.quality import evaluate_result

PLAN = Path("study/preregistration/phase3_distinguishability_method_pilot_v1.yaml")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(plan_path: Path = PLAN, *, require_locked: bool = False) -> dict[str, object]:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    status = plan.get("status")
    authorized = plan.get("execution", {}).get("authorized")
    if status not in {"DRAFT_NOT_LOCKED", "LOCKED"}:
        errors.append("method pilot status must be DRAFT_NOT_LOCKED or LOCKED")
    if status == "DRAFT_NOT_LOCKED" and authorized is not False:
        errors.append("draft plan must not authorize execution")
    if status == "LOCKED":
        amendment_path = Path(str(plan.get("freeze_amendment", "")))
        if authorized is not True:
            errors.append("locked plan must explicitly authorize execution")
        if not amendment_path.is_file():
            errors.append("locked plan requires a freeze amendment")
        else:
            amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
            if amendment.get("status") != "LOCKED":
                errors.append("freeze amendment is not locked")
            if amendment.get("source_draft_sha256") != plan.get("source_draft_sha256"):
                errors.append("freeze amendment does not bind the draft hash")
    if require_locked and status != "LOCKED":
        errors.append("execution requires a locked plan")
    source = plan["source_ensemble"]
    for key in ("result", "config"):
        path = Path(source[key])
        if not path.is_file() or sha256(path) != source[f"{key}_sha256"]:
            errors.append(f"source {key} hash mismatch")
    result = json.loads(Path(source["result"]).read_text(encoding="utf-8"))
    gate = evaluate_result(result)
    if not gate.passed:
        errors.append(f"source artifact gate failed: {gate.failures}")

    original_seeds = {int(item["seed"]) for item in result["original_runs"]}
    if original_seeds != set(source["original_seeds"]):
        errors.append("planned original seeds differ from the immutable ensemble")
    conditions = {item["condition_id"]: item for item in result["conditions"]}
    for planned in plan["conditions"]:
        condition = conditions.get(planned["condition_id"])
        if condition is None:
            errors.append(f"missing source condition {planned['condition_id']}")
            continue
        if not condition["metrics"]["js_divergence"]["assay_sensitive"]:
            errors.append(f"JS is not assay-sensitive for {planned['condition_id']}")
        exact_seeds = {int(item["seed"]) for item in condition["exact_runs"]}
        if not set(source["exact_seeds"]) <= exact_seeds:
            errors.append(f"exact seed references are incomplete for {planned['condition_id']}")
        manifest = Path(planned["deletion_manifest"])
        if not manifest.is_file() or sha256(manifest) != planned["deletion_manifest_sha256"]:
            errors.append(f"deletion manifest hash mismatch for {planned['condition_id']}")
        lineage = condition["partition_lineage"]
        if not lineage["exact_retraining_exclusion_verified"]:
            errors.append(f"exact exclusion failed for {planned['condition_id']}")

    for method in plan["methods"]:
        descriptor = METHODS.get(method["name"])
        if descriptor is None or not descriptor.implementation_provenance:
            errors.append(f"unregistered or provenance-free method: {method['name']}")
    semantics = plan["analysis"]["decision_semantics"]
    if semantics["equivalence_claim_permitted"] or semantics["confirmatory_pass_fail_permitted"]:
        errors.append("draft pilot improperly permits equivalence or confirmatory decisions")
    expected_runs = len(source["original_seeds"]) * len(plan["conditions"]) * len(plan["methods"])
    if plan["execution"]["planned_unlearning_runs"] != expected_runs:
        errors.append("planned unlearning-run count is inconsistent")
    expected_exact_evaluations = len(source["exact_seeds"]) * len(plan["conditions"])
    if plan["execution"]["new_exact_evaluation_artifacts"] != expected_exact_evaluations:
        errors.append("planned split-complete exact-evaluation count is inconsistent")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "PASS",
        "plan": plan_path.as_posix(),
        "source_experiment": result["experiment_id"],
        "conditions": len(plan["conditions"]),
        "methods": len(plan["methods"]),
        "planned_unlearning_runs": expected_runs,
        "new_trainings_avoided": len(source["original_seeds"])
        + len(source["exact_seeds"]) * len(plan["conditions"]),
        "execution_authorized": authorized,
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
