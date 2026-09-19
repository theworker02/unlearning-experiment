# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.quality import evaluate_result


def distances(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    p = np.clip(left["probabilities"], 1e-12, 1.0)
    q = np.clip(right["probabilities"], 1e-12, 1.0)
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
    js += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return {
        "js_divergence": float(js.mean()),
        "prediction_disagreement": float(np.mean(left["labels"] != right["labels"])),
    }


def load_test(reference: dict[str, object]) -> dict[str, np.ndarray]:
    with np.load(str(reference["path"]), allow_pickle=False) as arrays:
        return {
            "probabilities": arrays["test_probabilities"].astype(np.float64),
            "labels": arrays["test_predicted_labels"].astype(np.int64),
            "ids": arrays["test_ids"].astype(np.int64),
        }


def bootstrap_replica_mean(
    pair_matrix: np.ndarray, repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    size = pair_matrix.shape[0]
    estimates = []
    for _ in range(repetitions):
        sampled = rng.integers(0, size, size=size)
        values = [
            pair_matrix[left, right] for left, right in combinations(sampled, 2) if left != right
        ]
        if values:
            estimates.append(float(np.mean(values)))
    return np.asarray(estimates)


def main() -> None:
    plan = yaml.safe_load(Path("research/exact_seed_convergence_plan.yaml").read_text())
    result_path = Path("results/raw") / plan["condition"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = evaluate_result(result)
    if not gate.passed:
        raise SystemExit(f"convergence analysis blocked by artifact gate: {gate.failures}")
    references = result["predictions"]["oracles"]
    seeds = [int(reference["seed"]) for reference in references]
    if seeds != plan["ordered_oracle_seeds"]:
        raise SystemExit("oracle seed order differs from the frozen convergence plan")
    oracle_predictions = [load_test(reference) for reference in references]
    ids_differ = any(
        not np.array_equal(oracle_predictions[0]["ids"], item["ids"]) for item in oracle_predictions
    )
    if ids_differ:
        raise SystemExit("oracle prediction sample IDs differ")

    rng = np.random.default_rng(int(plan["bootstrap"]["seed"]))
    repetitions = int(plan["bootstrap"]["repetitions"])
    records = []
    for population_size in plan["prefix_population_sizes"]:
        subset = oracle_predictions[:population_size]
        for metric in plan["primary_descriptive_metrics"]:
            matrix = np.full((population_size, population_size), np.nan)
            values = []
            for left, right in combinations(range(population_size), 2):
                value = distances(subset[left], subset[right])[metric]
                matrix[left, right] = matrix[right, left] = value
                values.append(value)
            bootstrap = bootstrap_replica_mean(matrix, repetitions, rng)
            records.append(
                {
                    "population_size": population_size,
                    "metric": metric,
                    "pair_count": len(values),
                    "mean": float(np.mean(values)),
                    "standard_deviation_across_pairs": float(np.std(values, ddof=1)),
                    "empirical_95th_percentile": float(np.quantile(values, 0.95)),
                    "replica_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "replica_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                }
            )

    stability = {}
    threshold = float(plan["stability_diagnostic"]["threshold"])
    for metric in plan["primary_descriptive_metrics"]:
        by_k = {row["population_size"]: row for row in records if row["metric"] == metric}
        relative_change = abs(by_k[15]["mean"] - by_k[10]["mean"]) / by_k[15]["mean"]
        stability[metric] = {
            "relative_change_k10_to_k15": relative_change,
            "threshold": threshold,
            "passes_pilot_diagnostic": relative_change <= threshold,
        }

    methods = {}
    for method, reference in result["predictions"]["methods"].items():
        candidate = load_test(reference)
        methods[method] = {
            metric: [distances(candidate, oracle)[metric] for oracle in oracle_predictions]
            for metric in plan["primary_descriptive_metrics"]
        }
    method_summaries = {}
    for method, metric_values in methods.items():
        method_summaries[method] = {}
        for metric, values in metric_values.items():
            exact_row = next(
                row for row in records if row["population_size"] == 15 and row["metric"] == metric
            )
            method_summaries[method][metric] = {
                "approximate_exact_mean": float(np.mean(values)),
                "approximate_exact_standard_deviation": float(np.std(values, ddof=1)),
                "exact_exact_mean": exact_row["mean"],
                "exact_exact_empirical_95th_percentile": exact_row["empirical_95th_percentile"],
                "mean_below_exact_exact_95th_percentile": float(np.mean(values))
                <= exact_row["empirical_95th_percentile"],
            }
    payload = {
        "schema_version": 1,
        "analysis_status": "pilot_descriptive_not_confirmatory",
        "source_experiment": result["experiment_id"],
        "source_result": result_path.as_posix(),
        "ordered_oracle_seeds": seeds,
        "bootstrap": plan["bootstrap"],
        "records": records,
        "stability": stability,
        "approximate_exact_distances": methods,
        "approximate_exact_summaries": method_summaries,
        "limitations": [
            "One dataset, architecture, deletion set, and original training seed.",
            "Bootstrap resamples exact replicas; intervals are pilot precision diagnostics.",
            "The 10% stability threshold is a design diagnostic, not an equivalence margin.",
        ],
    }
    output = Path("results/processed/exact_seed_convergence_fashion_pilot.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Exact-retraining seed convergence pilot",
        "",
        f"Source experiment: `{result['experiment_id']}`. This is PILOT evidence, not a",
        "confirmatory seed-count justification.",
        "",
        "| k | Metric | Pairs | Mean | Pair SD | 95% replica-bootstrap interval |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['population_size']} | {row['metric']} | {row['pair_count']} | "
            f"{row['mean']:.6f} | {row['standard_deviation_across_pairs']:.6f} | "
            f"[{row['replica_bootstrap_ci_low']:.6f}, "
            f"{row['replica_bootstrap_ci_high']:.6f}] |"
        )
    lines.extend(["", "## Frozen stability diagnostic", ""])
    for metric, status in stability.items():
        lines.append(
            f"- {metric}: relative k=10→15 change {status['relative_change_k10_to_k15']:.3%}; "
            f"diagnostic {'PASS' if status['passes_pilot_diagnostic'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "## Approximate/exact descriptive comparison at k=15",
            "",
            "| Method | Metric | Approx/exact mean | Exact/exact mean | Exact/exact 95th percentile | Below percentile |",
            "| --- | --- | ---: | ---: | ---: | :---: |",
        ]
    )
    for method, metric_summaries in method_summaries.items():
        for metric, summary in metric_summaries.items():
            lines.append(
                f"| {method} | {metric} | {summary['approximate_exact_mean']:.6f} | "
                f"{summary['exact_exact_mean']:.6f} | "
                f"{summary['exact_exact_empirical_95th_percentile']:.6f} | "
                f"{'yes' if summary['mean_below_exact_exact_95th_percentile'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "All comparisons are descriptive for one pilot condition. Being below an empirical",
            "pairwise percentile is not an equivalence test and does not establish successful",
            "unlearning, privacy, or cross-condition generalization.",
        ]
    )
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in payload["limitations"]]])
    Path("reports/exact_seed_convergence_pilot.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
