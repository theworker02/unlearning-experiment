from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

COUNT_FIELDS = ("original_seed_count", "exact_retraining_seed_count", "candidate_seed_count")


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _source_reference(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": path.as_posix(), "present": False, "sha256": None}
    return {
        "path": path.as_posix(),
        "present": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def validate_protocol(protocol: dict[str, object]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("count decision protocol must use schema version 1")
    if protocol.get("execution_authorized") is not False:
        raise ValueError("count decision protocol must not authorize training")
    rules = protocol.get("rules", {})
    for field in COUNT_FIELDS:
        rule = rules.get(field, {})
        candidates = rule.get("candidates", [])
        minimum = rule.get("minimum")
        if not candidates or candidates != sorted(set(candidates)):
            raise ValueError(f"{field}: candidate grid must be nonempty, unique, and sorted")
        if not isinstance(minimum, int) or minimum < 2 or candidates[0] < minimum:
            raise ValueError(f"{field}: invalid minimum or candidate grid")
    if "forbidden" not in rules.get("no_peeking", "").lower():
        raise ValueError("protocol must explicitly forbid confirmatory-result peeking")


def _exact_precision_selection(artifact, conditions) -> int | None:
    if artifact is None or artifact.get("artifact_gate", {}).get("passed") is not True:
        return None
    indexed = {item.get("condition_id"): item for item in artifact.get("conditions", [])}
    selected = []
    for specification in conditions:
        condition = indexed.get(specification["condition_id"])
        metric = (
            None if condition is None else condition.get("metrics", {}).get(specification["metric"])
        )
        count = (
            None
            if metric is None
            else metric.get("reference_precision", {}).get("selected_seed_count")
        )
        if not isinstance(count, int):
            return None
        selected.append(count)
    return max(selected) if selected else None


def _validated_selected_count(
    artifact, allowed: list[int], minimum: int, *, protocol_id: str, resampling_unit: str
) -> int | None:
    if artifact is None or artifact.get("status") != "PASS":
        return None
    if (
        artifact.get("validation_only") is not True
        or artifact.get("protocol_id") != protocol_id
        or artifact.get("resampling_unit") != resampling_unit
    ):
        return None
    count = artifact.get("selected_seed_count")
    if not isinstance(count, int) or count not in allowed or count < minimum:
        raise ValueError("precision artifact selected a count outside the frozen grid")
    if artifact.get("all_primary_conditions_pass") is not True:
        return None
    return count


def recommend(protocol, *, exact_precision, equivalence, original_precision, candidate_precision):
    validate_protocol(protocol)
    rules = protocol["rules"]
    exact_observed = _exact_precision_selection(
        exact_precision, protocol["inputs"]["exact_precision"]["conditions"]
    )
    exact_minimum = rules["exact_retraining_seed_count"]["minimum"]
    exact_count = None if exact_observed is None else max(exact_minimum, exact_observed)
    original_count = _validated_selected_count(
        original_precision,
        rules["original_seed_count"]["candidates"],
        rules["original_seed_count"]["minimum"],
        protocol_id=protocol["inputs"]["original_seed_precision"]["protocol_id"],
        resampling_unit=protocol["inputs"]["original_seed_precision"]["resampling_unit"],
    )
    candidate_count = _validated_selected_count(
        candidate_precision,
        rules["candidate_seed_count"]["candidates"],
        rules["candidate_seed_count"]["minimum"],
        protocol_id=protocol["inputs"]["candidate_seed_precision"]["protocol_id"],
        resampling_unit=protocol["inputs"]["candidate_seed_precision"]["resampling_unit"],
    )
    margin_admissible = bool(
        equivalence and equivalence.get("decision", {}).get("joint_primary_admissible") is True
    )
    blockers = []
    if exact_count is None:
        blockers.append("exact_precision_evidence_missing_or_invalid")
    if original_count is None:
        blockers.append("original_seed_precision_evidence_missing_or_invalid")
    if candidate_count is None:
        blockers.append("candidate_seed_precision_evidence_missing_or_invalid")
    if not margin_admissible:
        blockers.append("joint_primary_equivalence_margin_inadmissible")
    if not blockers:
        status = "READY_FOR_FREEZE"
    elif "joint_primary_equivalence_margin_inadmissible" in blockers:
        status = "EQUIVALENCE_MARGIN_INADMISSIBLE"
    else:
        status = "INSUFFICIENT_EVIDENCE"
    return {
        "schema_version": 1,
        "decision_id": protocol["decision_id"],
        "status": status,
        "protocol_sha256": None,
        "recommendation": {
            "original_seed_count": original_count,
            "exact_retraining_seed_count": exact_count,
            "candidate_seed_count": candidate_count,
        },
        "diagnostics": {
            "largest_pilot_exact_precision_selection": exact_observed,
            "exact_minimum_applied": exact_minimum,
            "joint_primary_equivalence_margin_admissible": margin_admissible,
        },
        "blockers": blockers,
        "profile_mutated": False,
    }


def build_recommendation(protocol_path: Path) -> dict[str, object]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    inputs = protocol["inputs"]
    source_paths = {name: Path(specification["path"]) for name, specification in inputs.items()}
    for name, specification in inputs.items():
        if "protocol" in specification:
            source_paths[f"{name}_protocol"] = Path(specification["protocol"])
    original_precision = _load_json(source_paths["original_seed_precision"])
    candidate_precision = _load_json(source_paths["candidate_seed_precision"])
    for name, artifact in (
        ("original_seed_precision", original_precision),
        ("candidate_seed_precision", candidate_precision),
    ):
        if artifact is None:
            continue
        protocol_source = source_paths[f"{name}_protocol"]
        expected_hash = hashlib.sha256(protocol_source.read_bytes()).hexdigest()
        if artifact.get("protocol_sha256") != expected_hash:
            if name == "original_seed_precision":
                original_precision = None
            else:
                candidate_precision = None
    result = recommend(
        protocol,
        exact_precision=_load_json(source_paths["exact_precision"]),
        equivalence=_load_json(source_paths["equivalence_calibration"]),
        original_precision=original_precision,
        candidate_precision=candidate_precision,
    )
    result["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    result["source_artifacts"] = {
        name: _source_reference(path) for name, path in sorted(source_paths.items())
    }
    return result


def validate_decision_artifact(
    protocol_path: Path,
    artifact_path: Path,
    *,
    required_status: str = "READY_FOR_FREEZE",
) -> dict[str, object]:
    if not protocol_path.is_file() or not artifact_path.is_file():
        return {
            "eligible": False,
            "status": "MISSING",
            "failures": ["count_decision_protocol_or_artifact_missing"],
        }
    failures = []
    try:
        observed = json.loads(artifact_path.read_text(encoding="utf-8"))
        expected = build_recommendation(protocol_path)
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as error:
        return {
            "eligible": False,
            "status": "INVALID",
            "failures": [type(error).__name__],
        }
    if observed != expected:
        failures.append("count_decision_not_reproducible_from_current_evidence")
    if observed.get("status") != required_status:
        failures.append("count_decision_not_ready_for_freeze")
    return {
        "eligible": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "recomputed_status": expected.get("status"),
        "protocol_sha256": expected.get("protocol_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("study/preregistration/main_count_decision_v1.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(build_recommendation(args.protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
