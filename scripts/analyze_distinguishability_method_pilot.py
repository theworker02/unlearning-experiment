from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.metrics import linear_cka, squared_mmd
from exact_unlearning.quality import evaluate_result

if __package__:
    from scripts.validate_distinguishability_method_pilot import PLAN, validate
else:
    from validate_distinguishability_method_pilot import PLAN, validate


RESULT = Path("results/raw/phase3-fmnist-full-class-method-pilot-v1/result.json")
OUTPUT = Path("results/processed/phase3_fmnist_full_class_method_pilot_v1.json")
REPORT = Path("reports/phase3_fmnist_full_class_method_pilot_v1.md")
SPLITS = ("test", "retain", "forget")
DISTANCE_METRICS = (
    "js_divergence",
    "prediction_disagreement",
    "final_feature_cka_distance",
    "final_feature_mmd",
)


def load_predictions(reference: dict[str, object]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(str(reference["path"]), allow_pickle=False) as arrays:
        return {
            split: {
                "ids": arrays[f"{split}_ids"].astype(np.int64),
                "true_labels": arrays[f"{split}_true_labels"].astype(np.int64),
                "probabilities": arrays[f"{split}_probabilities"].astype(np.float64),
                "predicted_labels": arrays[f"{split}_predicted_labels"].astype(np.int64),
                "features": arrays[f"{split}_features"].astype(np.float64),
            }
            for split in SPLITS
        }


def distances(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    if not np.array_equal(left["ids"], right["ids"]):
        raise ValueError("prediction artifacts use different split sample IDs")
    if not np.array_equal(left["true_labels"], right["true_labels"]):
        raise ValueError("prediction artifacts use different split labels")
    p = np.clip(left["probabilities"], 1e-12, 1.0)
    q = np.clip(right["probabilities"], 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
    js += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return {
        "js_divergence": float(js.mean()),
        "prediction_disagreement": float(
            np.mean(left["predicted_labels"] != right["predicted_labels"])
        ),
        "final_feature_cka_distance": float(
            1.0 - linear_cka(left["features"], right["features"])
        ),
        "final_feature_mmd": float(
            squared_mmd(left["features"][:256], right["features"][:256])
        ),
    }


def seed_bootstrap_mean(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("seed bootstrap requires at least two finite seed-level values")
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(repetitions)]
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "seed_count": len(array),
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "repetitions": repetitions,
        "resampling_unit": "original_model_seed",
    }


def exact_null(
    exact_predictions: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, list[float]]]:
    null = {
        split: {metric: [] for metric in DISTANCE_METRICS}
        for split in SPLITS
    }
    for left, right in combinations(range(len(exact_predictions)), 2):
        for split in SPLITS:
            measured = distances(
                exact_predictions[left][split], exact_predictions[right][split]
            )
            for metric, value in measured.items():
                null[split][metric].append(value)
    return null


def smoothed_percentile(value: float, reference: list[float]) -> float:
    array = np.asarray(reference, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("empirical percentile requires a finite reference")
    return float((1 + np.sum(array <= value)) / (len(array) + 1))


def validate_method_grid(result: dict[str, object], plan: dict[str, object]) -> None:
    expected = {
        (condition["condition_id"], method["name"], int(seed))
        for condition in plan["conditions"]
        for method in plan["methods"]
        for seed in plan["source_ensemble"]["original_seeds"]
    }
    observed_rows = [
        (condition["condition_id"], run["method"], int(run["original_seed"]))
        for condition in result["conditions"]
        for run in condition["method_runs"]
    ]
    observed = set(observed_rows)
    if len(observed_rows) != len(observed):
        raise ValueError("method-run grid contains duplicate condition/method/original-seed rows")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"method-run grid mismatch; missing={missing}, extra={extra}")
    expected_exact = set(plan["source_ensemble"]["exact_seeds"])
    for condition in result["conditions"]:
        observed_exact = {int(run["seed"]) for run in condition["exact_runs"]}
        if observed_exact != expected_exact:
            raise ValueError(
                f"exact-reference seed grid mismatch for {condition['condition_id']}"
            )


def validate_only(plan_path: Path = PLAN) -> dict[str, object]:
    summary = validate(plan_path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    return {
        **summary,
        "analysis_status": "ready_but_no_result_required",
        "splits": list(SPLITS),
        "distance_metrics": list(DISTANCE_METRICS),
        "statistical_unit": "original_model_seed",
        "equivalence_claim_permitted": plan["analysis"]["decision_semantics"][
            "equivalence_claim_permitted"
        ],
        "confirmatory_pass_fail_permitted": plan["analysis"]["decision_semantics"][
            "confirmatory_pass_fail_permitted"
        ],
    }


def analyze(result_path: Path = RESULT, plan_path: Path = PLAN) -> dict[str, object]:
    if not result_path.is_file():
        raise FileNotFoundError(f"method pilot result is unavailable: {result_path}")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = evaluate_result(result)
    if not gate.passed:
        raise ValueError(f"method pilot artifact gate failed: {gate.failures}")
    if result["source_result_sha256"] != plan["source_ensemble"]["result_sha256"]:
        raise ValueError("method pilot source ensemble differs from the analysis plan")
    validate_method_grid(result, plan)

    condition_outputs = []
    bootstrap_seed = 81173
    for condition_index, condition in enumerate(result["conditions"]):
        exact = [load_predictions(run["predictions"]) for run in condition["exact_runs"]]
        null = exact_null(exact)
        methods = {}
        for method_index, method in enumerate(plan["methods"]):
            runs = sorted(
                [run for run in condition["method_runs"] if run["method"] == method["name"]],
                key=lambda run: int(run["original_seed"]),
            )
            per_seed = []
            for run in runs:
                candidate = load_predictions(run["predictions"])
                split_summaries = {}
                for split in SPLITS:
                    measured = [distances(candidate[split], oracle[split]) for oracle in exact]
                    split_summaries[split] = {
                        metric: {
                            "median_distance_to_exact_ensemble": float(
                                np.median([item[metric] for item in measured])
                            ),
                            "exact_null_percentile": smoothed_percentile(
                                float(np.median([item[metric] for item in measured])),
                                null[split][metric],
                            ),
                        }
                        for metric in DISTANCE_METRICS
                    }
                per_seed.append(
                    {
                        "original_seed": int(run["original_seed"]),
                        "unlearning_seed": int(run["seed"]),
                        "duration_seconds": float(run["duration_seconds"]),
                        "distances": split_summaries,
                        "conventional_metrics": run["metrics"],
                    }
                )
            primary_values = np.asarray(
                [
                    item["distances"]["test"]["js_divergence"][
                        "median_distance_to_exact_ensemble"
                    ]
                    for item in per_seed
                ]
            )
            methods[method["name"]] = {
                "role": method["role"],
                "per_seed": per_seed,
                "primary_test_js": seed_bootstrap_mean(
                    primary_values,
                    repetitions=5000,
                    seed=bootstrap_seed + condition_index * 100 + method_index,
                ),
                "interpretation": "continuous_exploratory_discrepancy_not_equivalence",
            }
        condition_outputs.append(
            {
                "condition_id": condition["condition_id"],
                "exact_seed_count": len(exact),
                "exact_null": {
                    split: {
                        metric: {
                            "pair_count": len(values),
                            "median": float(np.median(values)),
                            "q95": float(np.quantile(values, 0.95)),
                        }
                        for metric, values in metrics.items()
                    }
                    for split, metrics in null.items()
                },
                "methods": methods,
            }
        )

    payload = {
        "schema_version": 1,
        "analysis_status": "exploratory_continuous_not_confirmatory",
        "source_experiment": result["experiment_id"],
        "source_result": result_path.as_posix(),
        "statistical_unit": "original_model_seed",
        "bootstrap": {"repetitions": 5000, "confidence": 0.95, "base_seed": bootstrap_seed},
        "conditions": condition_outputs,
        "decision_semantics": plan["analysis"]["decision_semantics"],
        "limitations": [
            "Five original-model seeds support exploratory uncertainty intervals only.",
            "Exact/exact model pairs define a null distribution but are not treated as "
            "independent.",
            "No equivalence margin passed calibration, so no equivalence status is computed.",
            "Metric contradictions remain continuous exploratory comparisons without "
            "PASS/FAIL labels.",
        ],
    }
    return payload


def write_analysis(payload: dict[str, object]) -> None:
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Full-class distinguishability method pilot",
        "",
        "**Status:** exploratory continuous analysis; not equivalence or confirmatory evidence.",
        "",
        "| Condition | Method | Mean seed-level median test JS | 95% seed bootstrap interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for condition in payload["conditions"]:
        for method, values in condition["methods"].items():
            primary = values["primary_test_js"]
            lines.append(
                f"| {condition['condition_id']} | {method} | {primary['estimate']:.6f} | "
                f"[{primary['ci_low']:.6f}, {primary['ci_high']:.6f}] |"
            )
    lines.extend(
        [
            "",
            "No row is labeled equivalent, successful, failed, or confirmatory. Conventional",
            "metrics, exact-null percentiles, and multidimensional distances remain available in",
            f"`{OUTPUT.as_posix()}` for evidence-bound exploratory analysis.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the full-class method pilot")
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_only:
        print(json.dumps(validate_only(arguments.plan), indent=2, sort_keys=True))
        return
    payload = analyze(arguments.result, arguments.plan)
    write_analysis(payload)
    print(OUTPUT)
    print(REPORT)


if __name__ == "__main__":
    main()
