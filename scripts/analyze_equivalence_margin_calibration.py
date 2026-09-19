from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.phase3 import calibrate_distance_equivalence_margin
from exact_unlearning.quality import evaluate_result

REFERENCE_PATH = Path(
    "results/raw/phase3-fmnist-class-conditioned-assay-pilot-v3/result.json"
)
CALIBRATION_PATH = Path(
    "results/raw/phase3-fmnist-equivalence-margin-calibration-pilot-v1/result.json"
)
PLAN_PATH = Path("study/preregistration/equivalence_margin_calibration_v1.yaml")
OUTPUT_PATH = Path("results/processed/equivalence_margin_calibration_v1.json")
REPORT_PATH = Path("reports/equivalence_margin_calibration_v1.md")


def load_probabilities(reference: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(reference["path"]), allow_pickle=False) as arrays:
        return (
            arrays["test_probabilities"].astype(np.float64),
            arrays["test_ids"].astype(np.int64),
        )


def js_divergence(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> float:
    p, left_ids = left
    q, right_ids = right
    if not np.array_equal(left_ids, right_ids):
        raise ValueError("prediction artifacts use different test sample IDs")
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
    divergence += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return float(divergence.mean())


def reference_exact_pairs(condition: dict[str, object]) -> tuple[list[object], np.ndarray]:
    predictions = [load_probabilities(item["predictions"]) for item in condition["exact_runs"]]
    distances = [
        js_divergence(predictions[left], predictions[right])
        for left in range(len(predictions))
        for right in range(left + 1, len(predictions))
    ]
    return predictions, np.asarray(distances, dtype=np.float64)


def condition_class(condition_id: str) -> int:
    for target_class in (0, 5):
        if f"class{target_class}" in condition_id:
            return target_class
    raise ValueError(f"unsupported calibration condition: {condition_id}")


def main() -> None:
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    for name, result in (("reference", reference), ("calibration", calibration)):
        gate = evaluate_result(result)
        if not gate.passed:
            raise SystemExit(f"{name} artifact gate failed: {gate.failures}")

    references: dict[int, dict[str, object]] = {}
    for condition in reference["conditions"]:
        target_class = condition_class(condition["condition_id"])
        if float(condition["fraction"]) == 1.0:
            exact, pair_distances = reference_exact_pairs(condition)
            references[target_class] = {
                "condition_id": condition["condition_id"],
                "predictions": exact,
                "pair_distances": pair_distances,
            }
    if set(references) != {0, 5}:
        raise SystemExit("full-deletion reference ensembles are incomplete")

    records: list[dict[str, object]] = []
    minimum_detection = float(plan["decision"]["minimum_detection_probability"])
    for condition in calibration["conditions"]:
        target_class = condition_class(condition["condition_id"])
        reference_item = references[target_class]
        candidate_margin = float(np.quantile(reference_item["pair_distances"], 0.95))
        replica_statistics = []
        for run in condition["exact_runs"]:
            perturbed = load_probabilities(run["predictions"])
            distances = np.asarray(
                [js_divergence(perturbed, exact) for exact in reference_item["predictions"]]
            )
            replica_statistics.append(
                {
                    "seed": int(run["seed"]),
                    "median_distance_to_reference_ensemble": float(np.median(distances)),
                    "detected": bool(float(np.median(distances)) > candidate_margin),
                }
            )
        seed_statistics = np.asarray(
            [item["median_distance_to_reference_ensemble"] for item in replica_statistics]
        )
        calibration_check = calibrate_distance_equivalence_margin(
            reference_item["pair_distances"],
            seed_statistics,
            reference_quantile=0.95,
            minimum_detection_probability=minimum_detection,
        )
        residual_fraction = 1.0 - float(condition["fraction"])
        records.append(
            {
                "condition_id": condition["condition_id"],
                "reference_condition_id": reference_item["condition_id"],
                "target_class": target_class,
                "deleted_fraction": float(condition["fraction"]),
                "residual_forget_class_fraction": residual_fraction,
                "forget_count": int(condition["forget_count"]),
                "candidate_margin": candidate_margin,
                "replica_statistics": replica_statistics,
                "calibration": calibration_check,
                "primary": bool(np.isclose(residual_fraction, 0.01)),
            }
        )

    primary = [item for item in records if item["primary"]]
    joint_admissible = len(primary) == 2 and all(
        bool(item["calibration"]["admissible"]) for item in primary
    )
    payload = {
        "schema_version": 1,
        "analysis_status": "prospectively_frozen_pilot_calibration",
        "plan": PLAN_PATH.as_posix(),
        "reference_experiment": reference["experiment_id"],
        "calibration_experiment": calibration["experiment_id"],
        "endpoint": "js_divergence",
        "records": records,
        "decision": {
            "joint_primary_admissible": joint_admissible,
            "numerical_margin_status": "condition_specific_admissible"
            if joint_admissible
            else "not_admissible",
            "confirmatory_method_matrix_status": "still_blocked_pending_protocol_lock",
        },
        "limitations": [
            "The operational 1% contamination threshold is pilot-specific, not universal.",
            "Calibration covers two Fashion-MNIST classes and one reduced CNN recipe.",
            "A condition-specific q95 is not transferable to other datasets or architectures.",
            "Passing calibration does not itself demonstrate successful unlearning.",
        ],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Equivalence-margin calibration v1",
        "",
        "**Status:** prospectively frozen pilot calibration; not publication evidence.",
        "",
        "The primary controlled violation leaves 1% of the forbidden class in the training",
        "dataset. Five-percent and ten-percent residual conditions are diagnostics only. Each",
        "replica is summarized by its median JS divergence to all 15 condition-matched",
        "full-deletion exact models. Detection requires that statistic to exceed the exact/exact",
        "95th percentile, and at least 80% of 15 replicas must be detected in both classes.",
        "",
        "| Class | Residual class | Candidate q95 | Detected replicas | Detection "
        "| Primary | Admissible |",
        "| ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    ordered_records = sorted(
        records,
        key=lambda item: (item["target_class"], item["residual_forget_class_fraction"]),
    )
    for record in ordered_records:
        detected = sum(item["detected"] for item in record["replica_statistics"])
        lines.append(
            f"| {record['target_class']} | {record['residual_forget_class_fraction']:.0%} | "
            f"{record['candidate_margin']:.6f} | {detected}/15 | "
            f"{record['calibration']['detection_probability']:.1%} | "
            f"{'yes' if record['primary'] else 'no'} | "
            f"{'yes' if record['calibration']['admissible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            (
                "The 1% controlled violation passed in both classes. The condition-specific "
                "exact/exact q95 margins are admissible for the next Fashion-MNIST full-class "
                "pilot stage. The confirmatory method matrix remains blocked until the complete "
                "protocol is locked."
                if joint_admissible
                else "The 1% controlled violation did not pass in both classes. The numerical "
                "margin remains inadmissible, irrespective of the larger diagnostic conditions."
            ),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
            "## Evidence",
            "",
            f"- Frozen plan: `{PLAN_PATH.as_posix()}`",
            f"- Reference raw result: `{REFERENCE_PATH.as_posix()}`",
            f"- Calibration raw result: `{CALIBRATION_PATH.as_posix()}`",
            f"- Generated analysis: `{OUTPUT_PATH.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
