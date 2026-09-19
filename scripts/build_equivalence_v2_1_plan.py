from __future__ import annotations

import hashlib
import json
from pathlib import Path

if __package__:
    from scripts.build_equivalence_v2_plan import build_plan
else:
    from build_equivalence_v2_plan import build_plan

PROTOCOL = Path("study/preregistration/equivalence_margin_calibration_v2_1.yaml")
OUTPUT = Path("study/manifests/equivalence_margin_calibration_v2_1_plan.json")
CHECKSUM = OUTPUT.with_suffix(".sha256")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v2_1_plan() -> dict[str, object]:
    plan = build_plan()
    plan["plan_id"] = "equivalence-margin-calibration-v2.1-input-plan"
    plan["protocol"] = {"path": PROTOCOL.as_posix(), "sha256": _sha256(PROTOCOL)}
    plan["prediction_schema"]["arrays"]["true_labels"] = [
        plan["split_manifest"]["validation_sample_count"]
    ]
    plan["prediction_schema"]["label_dtype"] = "int64"
    plan["compute_and_storage"]["minimum_true_label_array_bytes"] = (
        len(plan["conditions"]) * plan["split_manifest"]["validation_sample_count"] * 8
    )
    for condition in plan["conditions"]:
        condition["combined_analysis_input_target"] = condition[
            "combined_analysis_input_target"
        ].replace("equivalence-margin-v2/", "equivalence-margin-v2-1/")
        for run in condition["calibration_runs"]:
            run["validation_prediction_target"] = run["validation_prediction_target"].replace(
                "equivalence-margin-v2/", "equivalence-margin-v2-1/"
            )
        for run in condition["holdout_runs"]:
            run["run_id"] = run["run_id"].replace("equivalence-v2-", "equivalence-v2-1-")
            run["checkpoint_target"] = run["checkpoint_target"].replace(
                "equivalence-margin-v2/", "equivalence-margin-v2-1/"
            )
            run["validation_prediction_target"] = run["validation_prediction_target"].replace(
                "equivalence-margin-v2/", "equivalence-margin-v2-1/"
            )
    plan["notes"].append(
        "V2.1 analysis inputs include frozen ground-truth labels for class-stratified selection."
    )
    return plan


def main() -> None:
    plan = build_v2_1_plan()
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"immutable plan differs: {OUTPUT}")
    OUTPUT.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    CHECKSUM.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps(plan["compute_and_storage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
