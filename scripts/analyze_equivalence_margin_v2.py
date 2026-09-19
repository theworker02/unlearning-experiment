from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.equivalence_controls import (
    calibrate_excess_js_margin,
    leave_one_out_excess_js,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_ids(values: np.ndarray) -> str:
    canonical = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def analyze_manifest(protocol: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
    if manifest.get("publication_evidence") is not False or manifest.get("split") != "validation":
        raise ValueError("v2 calibration accepts validation-only non-publication evidence")
    if manifest.get("validation_split_sha256") != protocol["probe"]["expected_split_hash"]:
        raise ValueError("input manifest does not bind the frozen validation split")
    expected = {item["condition_id"] for item in protocol["scope"]["initial_conditions"]}
    records = manifest.get("conditions", [])
    if {item.get("condition_id") for item in records} != expected or len(records) != len(expected):
        raise ValueError("input conditions do not match the protocol")
    severities = protocol["positive_control"]["sensitivity_fractions"]
    primary_index = severities.index(protocol["positive_control"]["primary_fraction"])
    results = []
    for record in sorted(records, key=lambda item: item["condition_id"]):
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"{record['condition_id']}: input file missing or hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            probabilities = arrays["probabilities"]
            sample_ids = arrays["sample_ids"]
            model_seeds = arrays["model_seeds"]
            stratified = "within_ground_truth_class" in protocol["positive_control"]["selection"]
            true_labels = arrays["true_labels"] if stratified else None
        if _digest_ids(sample_ids) != protocol["probe"]["expected_split_hash"]:
            raise ValueError(f"{record['condition_id']}: sample IDs do not match the frozen split")
        if len(sample_ids) < protocol["probe"]["minimum_samples"]:
            raise ValueError(f"{record['condition_id']}: probe is too small")
        calibration_seeds = record.get("calibration_seeds", [])
        holdout_seeds = record.get("holdout_seeds", [])
        if len(calibration_seeds) != protocol["margin_estimation"]["calibration_seed_count"]:
            raise ValueError("calibration seed count does not match the protocol")
        if len(holdout_seeds) != protocol["margin_estimation"]["heldout_seed_count"]:
            raise ValueError("holdout seed count does not match the protocol")
        seed_to_index = {int(seed): index for index, seed in enumerate(model_seeds.tolist())}
        if len(seed_to_index) != len(model_seeds):
            raise ValueError("model seed identifiers must be unique")
        try:
            calibration_indices = np.asarray(
                [seed_to_index[int(seed)] for seed in calibration_seeds], dtype=int
            )
            holdout_indices = np.asarray(
                [seed_to_index[int(seed)] for seed in holdout_seeds], dtype=int
            )
        except KeyError as error:
            raise ValueError("seed manifest references an absent model") from error
        effects = np.column_stack(
            [
                leave_one_out_excess_js(
                    probabilities,
                    sample_ids,
                    fraction=fraction,
                    selection_seed=protocol["positive_control"]["selection_seed"],
                    strata=true_labels,
                )
                for fraction in severities
            ]
        )
        calibration = calibrate_excess_js_margin(
            effects,
            primary_index=primary_index,
            calibration_indices=calibration_indices,
            holdout_indices=holdout_indices,
            repetitions=protocol["margin_estimation"]["bootstrap_repetitions"],
            seed=protocol["margin_estimation"]["bootstrap_seed"],
            confidence=0.95,
            minimum_exceedance_probability=protocol["admissibility"][
                "minimum_heldout_primary_exceedance_probability"
            ],
        )
        results.append(
            {
                "condition_id": record["condition_id"],
                "input": {"path": path.as_posix(), "sha256": record["sha256"]},
                "severity_fractions": severities,
                "calibration": calibration,
            }
        )
    joint = all(item["calibration"]["admissible"] for item in results)
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "analysis_status": "VALIDATION_ONLY_CALIBRATION",
        "joint_primary_admissible": joint,
        "conditions": results,
        "publication_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("study/preregistration/equivalence_margin_calibration_v2.yaml"),
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("analysis_authorized") is not True
        or protocol.get("status") != "FROZEN_VALIDATION_ONLY"
    ):
        raise SystemExit("analysis blocked until the v2 protocol is explicitly frozen")
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    result = analyze_manifest(protocol, manifest)
    result["protocol_sha256"] = _sha256(args.protocol)
    result["input_manifest_sha256"] = _sha256(args.input_manifest)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"immutable analysis output differs: {args.output}")
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
