from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from exact_unlearning.quality import evaluate_result  # noqa: E402


def main() -> None:
    analysis_path = Path("results/processed/exact_seed_convergence_fashion_pilot.json")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    result_path = Path(analysis["source_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not evaluate_result(result).passed:
        raise SystemExit("F002 source experiment does not pass artifact validation")
    metrics = ["js_divergence", "prediction_disagreement"]
    labels = ["Mean Jensen–Shannon divergence", "Mean prediction disagreement"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        rows = [row for row in analysis["records"] if row["metric"] == metric]
        x = [row["population_size"] for row in rows]
        y = [row["mean"] for row in rows]
        low = [value - row["replica_bootstrap_ci_low"] for value, row in zip(y, rows, strict=True)]
        high = [
            row["replica_bootstrap_ci_high"] - value
            for value, row in zip(y, rows, strict=True)
        ]
        axis.errorbar(x, y, yerr=[low, high], marker="o", color="#174ea6", capsize=4)
        axis.axvline(10, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_xlabel("Exact-retraining replicas, k")
        axis.set_ylabel(label)
        axis.set_xticks(x)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Exact-retraining estimate convergence — Fashion-MNIST pilot")
    figure.text(
        0.5,
        0.01,
        "Points: exact/exact pair means; bars: 95% replica-bootstrap intervals. Pilot only.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    output = Path("visualizations/empirical/F002_seed_convergence.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    csv_path = Path("results/processed/exact_seed_convergence_fashion_pilot.csv")
    source_artifacts = [result_path, analysis_path, csv_path]
    sidecar = {
        "schema_version": 1,
        "figure_id": "F002",
        "title": "Exact-retraining seed convergence in the Fashion-MNIST schema-v2 pilot",
        "type": "empirical",
        "contains_empirical_data": True,
        "research_maturity": "PILOT",
        "publication_figure": False,
        "benchmark_version": "0.5.0",
        "source_experiments": [analysis["source_experiment"]],
        "source_data": "results/processed/exact_seed_convergence_fashion_pilot.csv",
        "analysis_artifact": analysis_path.as_posix(),
        "source_artifacts": [
            {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in source_artifacts
        ],
        "generation_script": "visualizations/scripts/build_f002_seed_convergence.py",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": result.get("git_commit"),
        "output": output.as_posix(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "uncertainty": "95% replica-bootstrap interval; 2000 resamples; pilot diagnostic",
        "limitations": "One dataset, architecture, deletion set, and original-model seed.",
    }
    metadata = Path("visualizations/metadata/F002.yaml")
    metadata.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
