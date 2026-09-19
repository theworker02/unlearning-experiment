from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

REVIEW = Path("study/amendments/equivalence_margin_v2_sesoi_review.yaml")
VALID_DECISIONS = {
    "APPROVE_AS_SPECIFIED",
    "REVISE_BEFORE_FREEZE",
    "REJECT_EQUIVALENCE_ENDPOINT",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_review(review: dict[str, object]) -> dict[str, object]:
    failures = []
    for key in ("reviewed_protocol", "reviewed_input_plan"):
        reference = review.get(key, {})
        path = Path(str(reference.get("path", "")))
        if not path.is_file() or _sha256(path) != reference.get("sha256"):
            failures.append(f"{key}_missing_or_hash_mismatch")
    report_path = Path(str(review.get("preflight_report", "")))
    if not report_path.is_file():
        failures.append("preflight_report_missing")
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            failures.append("preflight_report_invalid")
        else:
            if report.get("status") != "PASS":
                failures.append("preflight_report_not_passing")
            if report.get("protocol_sha256") != review["reviewed_protocol"]["sha256"]:
                failures.append("preflight_protocol_hash_mismatch")
            if report.get("plan_sha256") != review["reviewed_input_plan"]["sha256"]:
                failures.append("preflight_plan_hash_mismatch")
    decision = review.get("decision")
    if decision is None:
        status = "AWAITING_HUMAN_DECISION" if not failures else "INVALID"
        return {"status": status, "eligible_for_freeze": False, "failures": failures}
    if decision not in VALID_DECISIONS:
        failures.append("invalid_decision")
    if not isinstance(review.get("reviewer"), str) or not review.get("reviewer", "").strip():
        failures.append("reviewer_missing")
    timestamp = review.get("reviewed_at_utc")
    try:
        datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        failures.append("review_timestamp_invalid")
    if (
        not isinstance(review.get("rationale"), str)
        or len(review.get("rationale", "").strip()) < 40
    ):
        failures.append("review_rationale_too_short")
    checks = review.get("required_checks", {})
    if decision == "APPROVE_AS_SPECIFIED" and (
        not checks or not all(value is True for value in checks.values())
    ):
        failures.append("approval_requires_every_scientific_check")
    eligible = decision == "APPROVE_AS_SPECIFIED" and not failures
    return {
        "status": "APPROVED_FOR_SEPARATE_FREEZE_AMENDMENT"
        if eligible
        else "REVIEWED_NOT_APPROVED"
        if not failures
        else "INVALID",
        "eligible_for_freeze": eligible,
        "decision": decision,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = yaml.safe_load(args.review.read_text(encoding="utf-8"))
    print(json.dumps(validate_review(review), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
