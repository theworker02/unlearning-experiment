from __future__ import annotations

import hashlib
import json

from scripts.build_equivalence_v2_plan import CHECKSUM, OUTPUT, build_plan


def test_v2_plan_reuses_only_calibration_models_and_plans_disjoint_holdout() -> None:
    plan = build_plan()
    assert plan["execution_authorized"] is False
    assert plan["compute_and_storage"]["existing_exact_models_reused"] == 30
    assert plan["compute_and_storage"]["new_exact_training_jobs"] == 20
    assert plan["compute_and_storage"]["runtime_estimate"] is None
    for condition in plan["conditions"]:
        calibration = {item["seed"] for item in condition["calibration_runs"]}
        holdout = {item["seed"] for item in condition["holdout_runs"]}
        assert len(calibration) == 15
        assert len(holdout) == 10
        assert not calibration & holdout
        assert all(
            item["source"] == "reuse_exact_checkpoint" for item in condition["calibration_runs"]
        )
        assert all(
            item["source"] == "new_exact_training_required" for item in condition["holdout_runs"]
        )


def test_committed_v2_plan_and_checksum_are_reproducible() -> None:
    observed = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert observed == build_plan()
    digest, filename = CHECKSUM.read_text(encoding="utf-8").strip().split("  ")
    assert filename == OUTPUT.name
    assert digest == hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
