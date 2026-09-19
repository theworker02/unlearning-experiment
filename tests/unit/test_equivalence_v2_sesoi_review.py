from copy import deepcopy
from pathlib import Path

import yaml

from scripts.validate_equivalence_v2_sesoi_review import validate_review

REVIEW = Path("study/amendments/equivalence_margin_v2_sesoi_review.yaml")
STRATIFIED_REVIEW = Path("study/amendments/equivalence_margin_v2_1_sesoi_review.yaml")


def test_repository_review_is_valid_but_awaiting_human_decision() -> None:
    result = validate_review(yaml.safe_load(REVIEW.read_text(encoding="utf-8")))
    assert result == {
        "status": "AWAITING_HUMAN_DECISION",
        "eligible_for_freeze": False,
        "failures": [],
    }


def test_stratified_review_is_hash_bound_and_awaiting_human_decision() -> None:
    result = validate_review(yaml.safe_load(STRATIFIED_REVIEW.read_text(encoding="utf-8")))
    assert result == {
        "status": "AWAITING_HUMAN_DECISION",
        "eligible_for_freeze": False,
        "failures": [],
    }


def test_approval_requires_identity_rationale_timestamp_and_every_check() -> None:
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    review["decision"] = "APPROVE_AS_SPECIFIED"
    incomplete = validate_review(review)
    assert incomplete["status"] == "INVALID"
    assert "reviewer_missing" in incomplete["failures"]
    assert "approval_requires_every_scientific_check" in incomplete["failures"]

    approved = deepcopy(review)
    approved["reviewer"] = "Independent human reviewer"
    approved["reviewed_at_utc"] = "2026-09-02T00:00:00Z"
    approved["rationale"] = (
        "This synthetic fixture supplies a sufficiently detailed review rationale."
    )
    approved["required_checks"] = {key: True for key in approved["required_checks"]}
    result = validate_review(approved)
    assert result["status"] == "APPROVED_FOR_SEPARATE_FREEZE_AMENDMENT"
    assert result["eligible_for_freeze"] is True


def test_revision_is_recorded_but_never_authorizes_freeze() -> None:
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    review["decision"] = "REVISE_BEFORE_FREEZE"
    review["reviewer"] = "Independent human reviewer"
    review["reviewed_at_utc"] = "2026-09-02T00:00:00Z"
    review["rationale"] = (
        "Class-stratified selection is required before the protocol can be frozen."
    )
    result = validate_review(review)
    assert result["status"] == "REVIEWED_NOT_APPROVED"
    assert result["eligible_for_freeze"] is False
