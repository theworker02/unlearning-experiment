import json

from scripts.estimate_equivalence_v2_1_compute import LATEX, OUTPUT, REPORT, estimate


def test_compute_estimate_is_evidence_bound_and_incomplete_by_design() -> None:
    value = estimate()
    assert value["status"] == "EMPIRICAL_PLANNING_ESTIMATE_NOT_A_BENCHMARK"
    assert value["execution_authorized"] is False
    assert value["observed_exact_training_seconds"]["run_count"] == 30
    assert value["projected_serial_new_training_seconds"]["job_count"] == 20
    assert value["projected_serial_new_training_seconds"]["interval_95"][0] > 0
    assert "validation_prediction_export_time" in value["not_estimated"]
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == value
    assert REPORT.is_file()
    assert "not a benchmark" in REPORT.read_text(encoding="utf-8")
    assert "\\VTwoOneTrainingMedianSeconds" in LATEX.read_text(encoding="utf-8")
