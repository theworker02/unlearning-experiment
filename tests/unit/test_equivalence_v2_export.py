from pathlib import Path

import pytest

from scripts.export_equivalence_v2_validation import export_reused, preflight

PLAN = Path("study/manifests/equivalence_margin_calibration_v2_plan.json")
PROTOCOL = Path("study/preregistration/equivalence_margin_calibration_v2.yaml")


def test_reusable_checkpoint_preflight_is_hash_bound_and_nonexecuting() -> None:
    result = preflight(PLAN, PROTOCOL)
    assert result["status"] == "PASS"
    assert result["checkpoint_count"] == 30
    assert result["condition_count"] == 2
    assert result["validation_sample_count"] == 1000
    assert result["test_predictions_accessed"] is False
    assert result["training_executed"] is False


def test_export_is_blocked_while_protocol_is_draft() -> None:
    with pytest.raises(ValueError, match="explicitly frozen"):
        export_reused(PLAN, PROTOCOL)
