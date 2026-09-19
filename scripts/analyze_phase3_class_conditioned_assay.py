from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.metrics import linear_cka, squared_mmd
from exact_unlearning.phase3 import seed_count_precision_curve
from exact_unlearning.quality import evaluate_result

RESULT_PATH = Path(
    "results/raw/phase3-fmnist-class-conditioned-assay-pilot-v3/result.json"
)
OUTPUT_PATH = Path("results/processed/phase3_class_conditioned_assay_pilot.json")
REPORT_PATH = Path("reports/phase3_class_conditioned_assay_pilot.md")
DECISION_RULES_PATH = Path("study/preregistration/decision_rules.yaml")


def load_prediction(reference: dict[str, object]) -> dict[str, np.ndarray]:
    with np.load(str(reference["path"]), allow_pickle=False) as arrays:
        return {
            "probabilities": arrays["test_probabilities"].astype(np.float64),
            "labels": arrays["test_predicted_labels"].astype(np.int64),
            "features": arrays["test_features"].astype(np.float64),
            "ids": arrays["test_ids"].astype(np.int64),
        }


def distances(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    if not np.array_equal(left["ids"], right["ids"]):
        raise ValueError("prediction artifacts use different test sample IDs")
    p = np.clip(left["probabilities"], 1e-12, 1.0)
    q = np.clip(right["probabilities"], 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
    js += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return {
        "js_divergence": float(js.mean()),
        "prediction_disagreement": float(np.mean(left["labels"] != right["labels"])),
        "final_feature_cka_distance": float(
            1.0 - linear_cka(left["features"], right["features"])
        ),
        "final_feature_mmd": float(
            squared_mmd(left["features"][:256], right["features"][:256])
        ),
    }


def exact_pair_matrices(
    references: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    predictions = [load_prediction(item["predictions"]) for item in references]
    metric_names = tuple(distances(predictions[0], predictions[1]))
    matrices = {
        metric: np.zeros((len(predictions), len(predictions)), dtype=np.float64)
        for metric in metric_names
    }
    for left, right in combinations(range(len(predictions)), 2):
        values = distances(predictions[left], predictions[right])
        for metric, value in values.items():
            matrices[metric][left, right] = matrices[metric][right, left] = value
    return matrices


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    gate = evaluate_result(result)
    if not gate.passed:
        raise SystemExit(f"analysis blocked by artifact gate: {gate.failures}")

    rules = yaml.safe_load(DECISION_RULES_PATH.read_text(encoding="utf-8"))
    precision = rules["reference_precision"]
    candidate_counts = tuple(int(value) for value in precision["candidate_seed_counts"])
    condition_records: list[dict[str, object]] = []
    selected_counts: list[int | None] = []

    for condition_index, condition in enumerate(result["conditions"]):
        matrices = exact_pair_matrices(condition["exact_runs"])
        metric_records: dict[str, object] = {}
        for metric, assay in condition["metrics"].items():
            curve = seed_count_precision_curve(
                matrices[metric],
                candidate_counts=candidate_counts,
                repetitions=int(precision["repetitions"]),
                seed=int(precision["simulation_seed"]) + condition_index,
                confidence=float(precision["confidence_level"]),
                maximum_relative_half_width=float(
                    precision["maximum_relative_interval_half_width"]
                ),
            )
            selected_counts.append(curve["selected_seed_count"])
            metric_records[metric] = {
                "assay_sensitive": bool(assay["assay_sensitive"]),
                "separability_ratio": float(assay["separability_ratio"]),
                "minimum_detectable_ratio": float(assay["power_simulation"]["minimum_ratio"]),
                "exact_exact_q95": float(assay["exact_exact_q95"]),
                "reference_precision": curve,
            }
        condition_records.append(
            {
                "condition_id": condition["condition_id"],
                "fraction": float(condition["fraction"]),
                "forget_count": int(condition["forget_count"]),
                "manifest": condition["manifest"],
                "metrics": metric_records,
            }
        )

    sensitive = [
        {"condition_id": row["condition_id"], "metric": metric}
        for row in condition_records
        for metric, values in row["metrics"].items()
        if values["assay_sensitive"]
    ]
    every_metric_precision_ready = all(value is not None for value in selected_counts)
    payload = {
        "schema_version": 1,
        "analysis_status": "exploratory_calibration_not_confirmatory",
        "source_experiment": result["experiment_id"],
        "source_result": RESULT_PATH.as_posix(),
        "source_config_sha256": result["config_sha256"],
        "artifact_gate": {
            "passed": gate.passed,
            "publication_eligible": gate.publication_eligible,
            "failures": gate.failures,
        },
        "design": {
            "original_seed_count": len(result["original_runs"]),
            "exact_seed_count_per_condition": len(result["conditions"][0]["exact_runs"]),
            "condition_count": len(result["conditions"]),
            "training_count": len(result["original_runs"])
            + sum(len(item["exact_runs"]) for item in result["conditions"]),
        },
        "conditions": condition_records,
        "sensitive_condition_metrics": sensitive,
        "reference_precision_ready_for_all_metrics": every_metric_precision_ready,
        "decision": {
            "full_class_js_can_advance_to_equivalence_margin_calibration": len(sensitive) == 2
            and all(item["metric"] == "js_divergence" for item in sensitive),
            "broad_multimetric_confirmatory_protocol_can_freeze": False,
            "equivalence_margin_status": "not_calibrated",
            "confirmatory_method_matrix_status": "blocked",
        },
        "runtime": {
            "started_at_utc": result["started_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
            "original_training_seconds": float(sum(result["original_training_seconds"])),
            "exact_training_seconds": float(
                sum(sum(item["exact_training_seconds"]) for item in result["conditions"])
            ),
            "environment": result["runtime"],
        },
        "limitations": [
            "Fashion-MNIST is a pilot dataset and not sufficient publication evidence.",
            "Only one reduced CNN training recipe and two deleted classes were tested.",
            "Assay sensitivity does not establish an equivalence margin or successful unlearning.",
            "No approximate-unlearning method was evaluated in this calibration run.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    metric_order = (
        "js_divergence",
        "prediction_disagreement",
        "final_feature_cka_distance",
        "final_feature_mmd",
    )
    lines = [
        "# Phase 3 class-conditioned assay pilot",
        "",
        "**Status:** complete exploratory calibration pilot; not publication evidence.",
        "",
        "## Design and integrity",
        "",
        f"The immutable run `{result['experiment_id']}` trained five independent original models",
        "and fifteen independent exact-retraining models for each of three prospectively",
        "configured class-conditioned deletion sets. The run contains 50 trained models in",
        "total. The central schema-v3 artifact gate passed every integrity check; the gate",
        "correctly marks this PILOT run as ineligible for confirmatory publication claims.",
        "No approximate-unlearning method was evaluated, so method outcomes could not influence",
        "assay selection.",
        "",
        "## Assay result",
        "",
        "Full deletion of class 0 and class 5 produced an identifiable counterfactual signal",
        "under test-set Jensen--Shannon divergence. The 50% class-0 deletion and every tested",
        "prediction-disagreement or representation metric remained below its prospectively",
        "simulated minimum detectable separation.",
        "",
        "| Condition | Deleted | Metric | Ratio | Minimum detectable | Exact/exact q95 "
        "| Sensitive | k selected |",
        "| --- | ---: | --- | ---: | ---: | ---: | :---: | ---: |",
    ]
    for condition in condition_records:
        for metric in metric_order:
            values = condition["metrics"][metric]
            selected = values["reference_precision"]["selected_seed_count"]
            lines.append(
                f"| {condition['condition_id']} | {condition['forget_count']} "
                f"({condition['fraction']:.0%}) | {metric} | "
                f"{values['separability_ratio']:.3f} | "
                f"{values['minimum_detectable_ratio']:.2f} | "
                f"{values['exact_exact_q95']:.4f} | "
                f"{'yes' if values['assay_sensitive'] else 'no'} | "
                f"{selected if selected is not None else 'none'} |"
            )
    lines.extend(
        [
            "",
            "## Reference-ensemble precision",
            "",
            "Precision was recalculated from model-level exact/exact matrices. Each simulation",
            "draws independently trained model seeds without replacement; it does not resample",
            "dependent model pairs as independent observations. A selected k is the first of",
            "5, 10, or 15 whose 95% interval has at most 20% relative half-width.",
            "",
        ]
    )
    for condition in condition_records:
        js_precision = condition["metrics"]["js_divergence"]["reference_precision"]
        lines.append(f"### {condition['condition_id']} — JS divergence")
        lines.append("")
        lines.append("| k | Median pair mean | 95% interval | Relative half-width | Pass |")
        lines.append("| ---: | ---: | ---: | ---: | :---: |")
        for row in js_precision["curve"]:
            lines.append(
                f"| {row['seed_count']} | {row['median_pair_mean']:.6f} | "
                f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | "
                f"{row['relative_half_width']:.1%} | "
                f"{'yes' if row['meets_precision_rule'] else 'no'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "JS divergence advances to controlled equivalence-margin calibration for full-class",
            "deletion. The broad multimetric confirmatory protocol remains unfrozen. No",
            "approximate-method comparison may be interpreted confirmatorily until the numerical",
            "margin is calibrated prospectively and its detection requirement passes.",
            "",
            "## Failure history",
            "",
            "Two earlier attempts are retained rather than erased: v1 exhausted memory after",
            "eagerly loading large intermediate activations, and v2 encountered an immutable",
            "artifact-name collision. Both are recorded in `results/failures/failures.jsonl`;",
            "neither contributes observations to this analysis.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
            "## Evidence",
            "",
            f"- Raw result: `{RESULT_PATH.as_posix()}`",
            f"- Generated analysis: `{OUTPUT_PATH.as_posix()}`",
            "- Configuration: `configs/experiments/phase3_class_conditioned_assay_pilot.yaml`",
            "- Failure registry: `results/failures/failures.jsonl`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
