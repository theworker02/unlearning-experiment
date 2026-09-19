from pathlib import Path

import yaml

from scripts.recommend_main_counts import (
    build_recommendation,
    recommend,
    validate_decision_artifact,
    validate_protocol,
)

PROTOCOL_PATH = Path("study/preregistration/main_count_decision_v1.yaml")


def _protocol():
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_repository_evidence_fails_closed_without_mutating_profile() -> None:
    before = Path("experiment_profiles/main.yaml").read_bytes()
    result = build_recommendation(PROTOCOL_PATH)
    assert result["status"] == "EQUIVALENCE_MARGIN_INADMISSIBLE"
    assert result["recommendation"] == {
        "original_seed_count": None,
        "exact_retraining_seed_count": 15,
        "candidate_seed_count": None,
    }
    assert result["diagnostics"]["largest_pilot_exact_precision_selection"] == 10
    assert result["profile_mutated"] is False
    assert Path("experiment_profiles/main.yaml").read_bytes() == before


def test_complete_synthetic_evidence_selects_frozen_counts() -> None:
    protocol = _protocol()
    exact = {
        "artifact_gate": {"passed": True},
        "conditions": [
            {
                "condition_id": item["condition_id"],
                "metrics": {item["metric"]: {"reference_precision": {"selected_seed_count": 10}}},
            }
            for item in protocol["inputs"]["exact_precision"]["conditions"]
        ],
    }
    original_precision = {
        "status": "PASS",
        "selected_seed_count": 8,
        "all_primary_conditions_pass": True,
        "validation_only": True,
        "protocol_id": "main-original-seed-precision-v1",
        "resampling_unit": "independently_trained_original_model_seed",
    }
    candidate_precision = {
        **original_precision,
        "protocol_id": "main-candidate-seed-precision-v1",
        "resampling_unit": "independently_initialized_candidate_unlearning_seed",
    }
    result = recommend(
        protocol,
        exact_precision=exact,
        equivalence={"decision": {"joint_primary_admissible": True}},
        original_precision=original_precision,
        candidate_precision=candidate_precision,
    )
    assert result["status"] == "READY_FOR_FREEZE"
    assert result["recommendation"] == {
        "original_seed_count": 8,
        "exact_retraining_seed_count": 15,
        "candidate_seed_count": 8,
    }


def test_protocol_is_nonexecuting_and_forbids_peeking() -> None:
    protocol = _protocol()
    validate_protocol(protocol)
    assert protocol["execution_authorized"] is False
    assert protocol["artifact"]["profile_update"] == "human_reviewed_separate_change_only"


def test_artifact_validator_recomputes_every_field_and_source_hash(tmp_path) -> None:
    result = build_recommendation(PROTOCOL_PATH)
    artifact_path = tmp_path / "decision.json"
    artifact_path.write_text(__import__("json").dumps(result), encoding="utf-8")
    valid = validate_decision_artifact(
        PROTOCOL_PATH,
        artifact_path,
        required_status="EQUIVALENCE_MARGIN_INADMISSIBLE",
    )
    assert valid["status"] == "PASS"

    result["recommendation"]["exact_retraining_seed_count"] = 20
    artifact_path.write_text(__import__("json").dumps(result), encoding="utf-8")
    tampered = validate_decision_artifact(
        PROTOCOL_PATH,
        artifact_path,
        required_status="EQUIVALENCE_MARGIN_INADMISSIBLE",
    )
    assert tampered["status"] == "FAIL"
    assert "count_decision_not_reproducible_from_current_evidence" in tampered["failures"]
