# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT_ID = "fashion-cnn-random-01pct-schema2-rocm-pilot-v1"


def main() -> None:
    result_path = Path("results/raw") / EXPERIMENT_ID / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    convergence = json.loads(
        Path("results/processed/exact_seed_convergence_fashion_pilot.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = result["runtime"]
    oracle_total = sum(runtime["oracle_training_seconds"])
    method_total = sum(method["duration_seconds"] for method in result["methods"].values())
    lines = [
        "# Schema-v2 Fashion-MNIST counterfactual-fidelity pilot",
        "",
        "> **PILOT evidence — not publication evidence and not a confirmatory test.**",
        "",
        f"Experiment: `{EXPERIMENT_ID}`. The run used the frozen 1% random deletion manifest",
        "and 15 exact-retraining seeds specified before execution.",
        "",
        "## Infrastructure outcome",
        "",
        "- Artifact quality gate: PASS.",
        "- Frozen deletion manifest binding and hash validation: PASS.",
        "- Forgotten/retained disjointness and exact-retraining exclusion: PASS.",
        "- Immutable checkpoints and prediction artifacts with IDs, labels, logits, and probabilities: PASS.",
        "- Finite metric and probability validation: PASS.",
        "- Failed runs in this condition: none.",
        "",
        "## Observed runtime",
        "",
        f"- Device: `{runtime['device']}` through ROCm `{runtime['rocm_version']}`; CUDA runtime: none.",
        f"- Original training: {runtime['original_training_seconds']:.3f} seconds.",
        f"- Fifteen exact retrainings: {oracle_total:.3f} seconds total.",
        f"- Three approximate methods: {method_total:.3f} seconds total.",
        "",
        "These wall-clock values describe this WSL/RX 7800 XT pilot only and are not normalized",
        "hardware-independent compute measurements.",
        "",
        "## Exact-retraining variability and convergence",
        "",
    ]
    for metric, status in convergence["stability"].items():
        lines.append(
            f"- {metric}: k=10→15 mean changed by "
            f"{status['relative_change_k10_to_k15']:.3%}; frozen 10% pilot diagnostic "
            f"{'PASS' if status['passes_pilot_diagnostic'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "The complete k=3/5/8/10/15 estimates and replica-bootstrap intervals are in",
            "`reports/exact_seed_convergence_pilot.md`.",
            "",
            "## Approximate/exact comparison",
            "",
            "| Method | Test accuracy | Forget accuracy | JS mean to exact | Disagreement mean to exact |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    summaries = convergence["approximate_exact_summaries"]
    for method, values in result["methods"].items():
        lines.append(
            f"| {method} | {values['test_metrics']['accuracy']:.4f} | "
            f"{values['forget_metrics']['accuracy']:.4f} | "
            f"{summaries[method]['js_divergence']['approximate_exact_mean']:.6f} | "
            f"{summaries[method]['prediction_disagreement']['approximate_exact_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "For all three methods, the mean approximate/exact JS divergence and prediction",
            "disagreement were below the exact/exact empirical 95th percentile in this condition.",
            "This is not an equivalence test. Forget accuracy remained high, so this condition does",
            "not demonstrate successful forgetting under a criterion requiring reduced forget-set",
            "accuracy. No contradiction classification was preregistered for this pilot.",
            "",
            "## Are the metrics informative?",
            "",
            "The exact/exact distances changed materially from k=3 to larger populations and then",
            "met the frozen k=10→15 stability diagnostic. This supports using multiple exact seeds",
            "and shows why a single oracle would be unstable. JS divergence and prediction",
            "disagreement provided nonzero, distinguishable scales, but their scientific utility",
            "across deletion regimes remains unestablished.",
            "",
            "## What must change before Tier-2",
            "",
            "- Repeat across original-model seeds; this pilot varies exact-retraining seeds only.",
            "- Add a second deletion fraction and high-loss condition under schema v2.",
            "- Freeze conventional-pass and counterfactual-fail criteria before contradiction analysis.",
            "- Validate stronger privacy attacks and baseline tuning parity.",
            "- Extend to CIFAR-10 and at least one independently licensed tabular dataset.",
            "- Treat seed convergence as condition-specific until replicated.",
            "",
            "## Scientific assessment",
            "",
            "The pilot validates the schema-v2 machinery and makes exact-retraining variance",
            "measurable. It does not answer the paper's central cross-condition research questions.",
        ]
    )
    output = Path("reports/schema2_fashion_pilot.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
