from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.run_distinguishability_method_pilot import main, validate_execution_plan


def test_execution_plan_reuses_source_ensembles_without_writing() -> None:
    result = validate_execution_plan()
    assert result["planned_unlearning_runs"] == 30
    assert result["expected_new_artifacts"] == 81
    assert result["source_original_artifacts_reused"] == 5
    assert result["source_exact_artifacts_reused"] == 20
    assert result["new_split_complete_exact_evaluations"] == 20
    # The immutable completed pilot may exist; planning itself must not write a new artifact.
    assert result["experiment_id"] == "phase3-fmnist-full-class-method-pilot-v1"


def test_runner_refuses_unlocked_unauthorized_plan(tmp_path: Path) -> None:
    source = Path("study/preregistration/phase3_distinguishability_method_pilot_v1.yaml")
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    plan["status"] = "DRAFT_NOT_LOCKED"
    plan["execution"]["authorized"] = False
    draft = tmp_path / "draft.yaml"
    draft.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="LOCKED and explicitly authorized"):
        main(draft)
    assert not (tmp_path / "raw").exists()
