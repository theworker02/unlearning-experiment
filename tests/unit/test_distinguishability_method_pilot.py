from __future__ import annotations

from copy import deepcopy

import yaml

from scripts.validate_distinguishability_method_pilot import PLAN, validate


def test_locked_method_pilot_plan_validates_with_explicit_authorization() -> None:
    result = validate()
    assert result["status"] == "PASS"
    assert result["planned_unlearning_runs"] == 30
    assert result["new_trainings_avoided"] == 25
    assert result["execution_authorized"] is True


def test_validator_rejects_equivalence_claim(tmp_path) -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    modified = deepcopy(plan)
    modified["analysis"]["decision_semantics"]["equivalence_claim_permitted"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(modified), encoding="utf-8")
    try:
        validate(path)
    except ValueError as error:
        assert "equivalence" in str(error)
    else:
        raise AssertionError("equivalence-permitting pilot plan was accepted")
