from __future__ import annotations

from pathlib import Path

import yaml


def validate(path: Path = Path("study/preregistration/scrub_sanity_reproduction_v1.yaml")) -> None:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    status = protocol.get("status")
    if status not in {"DRAFT_NOT_LOCKED", "LOCKED"}:
        errors.append("protocol status must be DRAFT_NOT_LOCKED or LOCKED")
    if (
        status == "DRAFT_NOT_LOCKED"
        and protocol.get("execution_authorized") is not False
    ):
        errors.append("draft protocol execution must fail closed")
    if status == "LOCKED":
        amendment_path = Path(str(protocol.get("freeze_amendment", "")))
        if protocol.get("execution_authorized") is not True:
            errors.append("locked protocol must explicitly authorize execution")
        if not amendment_path.is_file():
            errors.append("locked protocol requires a freeze amendment")
        else:
            amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
            if amendment.get("status") != "LOCKED":
                errors.append("freeze amendment is not locked")
    if protocol["scope"].get("architecture") != "small_cnn_bn":
        errors.append("SCRUB sanity protocol must use the validated BatchNorm architecture")
    tuning = protocol["tuning_policy"]
    if tuning.get("hyperparameter_search_permitted") is not False:
        errors.append("hyperparameter search is not permitted in the fixed-setting sanity check")
    if tuning.get("test_data_used_for_selection") is not False:
        errors.append("test data cannot be used for selection")
    if tuning.get("published_test_numbers_used_as_targets") is not False:
        errors.append("published test numbers cannot be optimization targets")
    if protocol["compute_control"].get("launch_requires_measured_estimate") is not True:
        errors.append("measured compute estimate must precede authorization")
    if len(protocol["acceptance_criteria"]) < 4:
        errors.append("mechanism acceptance criteria are incomplete")
    if errors:
        raise SystemExit("SCRUB sanity protocol invalid:\n- " + "\n- ".join(errors))
    print(f"SCRUB sanity protocol is internally valid ({status})")


if __name__ == "__main__":
    validate()
