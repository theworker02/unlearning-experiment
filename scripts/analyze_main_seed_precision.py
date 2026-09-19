from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

FORBIDDEN_TOKENS = ("test", "ranking", "winner", "confirmatory", "headline")


def validate_protocol(protocol: dict[str, object]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("execution_authorized") is not False:
        raise ValueError("invalid or execution-authorizing seed-precision protocol")
    counts = protocol.get("candidate_seed_counts", [])
    if counts != sorted(set(counts)) or not counts or counts[0] < 5:
        raise ValueError("candidate seed counts must be unique, sorted, and start at five or more")
    if protocol.get("minimum_available_pilot_seeds", 0) < counts[-1]:
        raise ValueError("pilot must contain every prespecified candidate count")
    if protocol.get("resampling_unit") not in {
        "independently_trained_original_model_seed",
        "independently_initialized_candidate_unlearning_seed",
    }:
        raise ValueError("invalid model-level resampling unit")
    if not protocol.get("groups"):
        raise ValueError("at least one frozen analysis group is required")


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any(token in str(key).lower() for token in FORBIDDEN_TOKENS)
            or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def scalar_seed_precision_curve(
    values: np.ndarray,
    *,
    candidate_counts: tuple[int, ...],
    repetitions: int,
    seed: int,
    confidence: float,
    maximum_relative_half_width: float,
) -> dict[str, object]:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or not np.isfinite(observations).all():
        raise ValueError("seed summaries must be a finite one-dimensional array")
    if len(observations) < max(candidate_counts) or np.any(observations < 0):
        raise ValueError("insufficient non-negative independent seed summaries")
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    curve = []
    selected = None
    for count in candidate_counts:
        estimates = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            sample = rng.choice(observations, size=count, replace=True)
            estimates[index] = np.mean(sample)
        low, center, high = np.quantile(estimates, [alpha, 0.5, 1 - alpha])
        relative_half_width = float((high - low) / (2 * center)) if center > 0 else float("inf")
        passes = relative_half_width <= maximum_relative_half_width
        curve.append(
            {
                "seed_count": count,
                "bootstrap_median_mean": float(center),
                "ci_low": float(low),
                "ci_high": float(high),
                "relative_interval_half_width": relative_half_width,
                "passes": passes,
            }
        )
        if selected is None and passes:
            selected = count
    return {"selected_seed_count": selected, "curve": curve}


def analyze(protocol: dict[str, object], evidence: dict[str, object]) -> dict[str, object]:
    validate_protocol(protocol)
    if _contains_forbidden_key(evidence):
        raise ValueError("evidence contains a forbidden result field")
    if evidence.get("split") != "validation" or evidence.get("publication_evidence") is not False:
        raise ValueError("only non-publication validation evidence is admissible")
    expected_groups = {item["group_id"] for item in protocol["groups"]}
    records = evidence.get("groups", [])
    if {item.get("group_id") for item in records} != expected_groups or len(records) != len(
        expected_groups
    ):
        raise ValueError("evidence group set does not match the frozen protocol")
    curves = []
    counts = tuple(protocol["candidate_seed_counts"])
    for offset, record in enumerate(sorted(records, key=lambda item: item["group_id"])):
        summaries = record.get("seed_summaries", [])
        seed_ids = [item.get("seed") for item in summaries]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError("seed identifiers must be unique within each group")
        precision = scalar_seed_precision_curve(
            np.asarray([item.get("value") for item in summaries], dtype=float),
            candidate_counts=counts,
            repetitions=protocol["bootstrap_repetitions"],
            seed=protocol["bootstrap_seed"] + offset,
            confidence=protocol["confidence_level"],
            maximum_relative_half_width=protocol["maximum_relative_interval_half_width"],
        )
        curves.append({"group_id": record["group_id"], **precision})
    selected_values = [item["selected_seed_count"] for item in curves]
    all_pass = all(value is not None for value in selected_values)
    selected = max(selected_values) if all_pass else None
    return {
        "schema_version": 1,
        "status": "PASS" if all_pass else "INSUFFICIENT_PRECISION",
        "protocol_id": protocol["protocol_id"],
        "validation_only": True,
        "resampling_unit": protocol["resampling_unit"],
        "selected_seed_count": selected,
        "all_primary_conditions_pass": all_pass,
        "groups": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("analysis_authorized") is not True
        or protocol.get("status") != "FROZEN_VALIDATION_ONLY"
    ):
        raise SystemExit("analysis blocked until the protocol is explicitly frozen")
    evidence = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(protocol, evidence)
    result["protocol_sha256"] = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
