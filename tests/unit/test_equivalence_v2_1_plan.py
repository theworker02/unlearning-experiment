import hashlib
import json

from scripts.build_equivalence_v2_1_plan import CHECKSUM, OUTPUT, build_v2_1_plan


def test_v2_1_plan_is_separate_and_includes_stratification_labels() -> None:
    plan = build_v2_1_plan()
    assert plan["execution_authorized"] is False
    assert plan["protocol"]["path"].endswith("equivalence_margin_calibration_v2_1.yaml")
    assert plan["prediction_schema"]["arrays"]["true_labels"] == [1000]
    assert plan["compute_and_storage"]["minimum_true_label_array_bytes"] == 16000
    assert all("v2-1" in item["combined_analysis_input_target"] for item in plan["conditions"])


def test_v2_1_plan_and_checksum_are_reproducible() -> None:
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == build_v2_1_plan()
    digest, filename = CHECKSUM.read_text(encoding="utf-8").strip().split("  ")
    assert filename == OUTPUT.name
    assert digest == hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
