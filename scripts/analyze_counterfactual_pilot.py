# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import yaml

from exact_unlearning.config import load_config
from exact_unlearning.data import load_dataset
from exact_unlearning.metrics import loss_membership_auc, predict, predictive_metrics
from exact_unlearning.models import build_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float | int | list[float]]:
    ordered = sorted(values)
    percentile_index = round(0.95 * (len(ordered) - 1))
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "empirical_95th_percentile": ordered[percentile_index],
        "n": len(values),
        "values": values,
    }


def load_oracle_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    config = result["config"]
    data = load_dataset(config["dataset"])
    deletion = result["deletion"]
    forget_positions = torch.as_tensor(deletion["forget_positions"], dtype=torch.long)
    retain_positions = torch.as_tensor(deletion["retain_positions"], dtype=torch.long)
    records = []
    for checkpoint in result["checkpoints"]["oracles"]:
        checkpoint_path = Path(checkpoint["path"])
        if sha256(checkpoint_path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint checksum mismatch: {checkpoint_path}")
        model = build_model(
            config["model"]["architecture"],
            tuple(data.x_train.shape[1:]),
            int(config["model"]["hidden_dim"]),
            int(data.y_train.max()) + 1,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        test_prediction = predict(model, data.x_test)
        forget_prediction = predict(model, data.x_train[forget_positions])
        retain_prediction = predict(model, data.x_train[retain_positions])
        bins = int(config["evaluation"]["ece_bins"])
        records.append(
            {
                "seed": checkpoint["seed"],
                "test": predictive_metrics(test_prediction, data.y_test.numpy(), bins),
                "forget": predictive_metrics(
                    forget_prediction, data.y_train[forget_positions].numpy(), bins
                ),
                "retain": predictive_metrics(
                    retain_prediction, data.y_train[retain_positions].numpy(), bins
                ),
                "loss_membership_auc_forget_vs_test": loss_membership_auc(
                    forget_prediction.probabilities,
                    data.y_train[forget_positions].numpy(),
                    test_prediction.probabilities,
                    data.y_test.numpy(),
                ),
            }
        )
    return records


def envelope_flag(method: dict[str, Any], split: str, metric: str) -> bool:
    return bool(method["within_empirical_retraining_envelope_by_split"][split][metric])


def contradiction_matrix(result: dict[str, Any], method: dict[str, Any]) -> dict[str, bool]:
    privacy_reference = result["exact_retraining_privacy_reference"]
    privacy_difference = method["privacy_fidelity_absolute_auc_difference"]
    privacy_pass = (
        privacy_difference["mean"]
        <= privacy_reference["pairwise_absolute_auc_difference"]["empirical_95th_percentile"]
    )
    forget_pass = envelope_flag(method, "forget", "absolute_accuracy_difference")
    retain_pass = envelope_flag(method, "retain", "absolute_accuracy_difference")
    calibration_pass = all(
        envelope_flag(method, "test", metric)
        for metric in (
            "absolute_nll_difference",
            "absolute_brier_difference",
            "absolute_ece_difference",
        )
    )
    prediction_pass = all(
        envelope_flag(method, "test", metric)
        for metric in ("top1_disagreement", "js_divergence", "total_variation")
    )
    representation_pass = envelope_flag(method, "test", "linear_cka_distance")
    counterfactual_pass = prediction_pass and calibration_pass and representation_pass
    return {
        "forget": forget_pass,
        "retain": retain_pass,
        "privacy": privacy_pass,
        "calibration": calibration_pass,
        "prediction": prediction_pass,
        "representation": representation_pass,
        "counterfactual": counterfactual_pass,
        "evaluation_contradiction": forget_pass
        and retain_pass
        and privacy_pass
        and not counterfactual_pass,
    }


def analyze_condition(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "completed":
        raise RuntimeError(f"pilot artifact is not completed: {path}")
    canonical_config = json.dumps(result["config"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical_config).hexdigest() != result["config_sha256"]:
        raise RuntimeError(f"config hash mismatch: {path}")
    oracle_metrics = load_oracle_metrics(result)
    started = datetime.fromisoformat(result["started_at_utc"])
    completed = datetime.fromisoformat(result["completed_at_utc"])
    methods = {}
    for name, method in result["methods"].items():
        methods[name] = {
            "test_accuracy": method["test_metrics"]["accuracy"],
            "forget_accuracy": method["forget_metrics"]["accuracy"],
            "retain_accuracy": method["retain_metrics"]["accuracy"],
            "membership_auc": method["loss_membership_auc_forget_vs_test"],
            "runtime_seconds": method["duration_seconds"],
            "speedup": method["speedup_vs_mean_retraining"],
            "test_js_mean": method["distance_to_retrained_distribution_by_split"]["test"][
                "js_divergence"
            ]["mean"],
            "test_js_normalized": method["distance_normalized_by_retraining_dispersion_by_split"][
                "test"
            ]["js_divergence"],
            "test_disagreement_mean": method["distance_to_retrained_distribution_by_split"]["test"][
                "top1_disagreement"
            ]["mean"],
            "test_cka_distance_mean": method["distance_to_retrained_distribution_by_split"]["test"][
                "linear_cka_distance"
            ]["mean"],
            "matrix": contradiction_matrix(result, method),
        }
    return {
        "experiment_id": result["experiment_id"],
        "raw_artifact": str(path),
        "raw_sha256": sha256(path),
        "strategy": result["config"]["deletion"]["strategy"],
        "fraction": result["config"]["deletion"]["fraction"],
        "train_count": result["dataset"]["train_count"],
        "forgotten_count": result["deletion"]["count"],
        "wall_seconds": (completed - started).total_seconds(),
        "oracle_training_seconds": summarize(result["runtime"]["oracle_training_seconds"]),
        "oracle_test_accuracy": summarize(
            [record["test"]["accuracy"] for record in oracle_metrics]
        ),
        "oracle_forget_accuracy": summarize(
            [record["forget"]["accuracy"] for record in oracle_metrics]
        ),
        "oracle_membership_auc": summarize(
            [record["loss_membership_auc_forget_vs_test"] for record in oracle_metrics]
        ),
        "exact_exact": {
            metric: result["exact_retraining_dispersion_by_split"]["test"][metric]
            for metric in ("js_divergence", "top1_disagreement", "linear_cka_distance")
        },
        "methods": methods,
    }


def mark(value: bool) -> str:
    return "within" if value else "outside"


def build_report(analysis: dict[str, Any]) -> str:
    conditions = analysis["conditions"]
    contradiction_count = sum(
        method["matrix"]["evaluation_contradiction"]
        for condition in conditions
        for method in condition["methods"].values()
    )
    lines = [
        "# Counterfactual Fidelity Pilot",
        "",
        "> **Pilot evidence only — not publication evidence or a confirmatory result.**",
        "",
        "This report is generated from immutable Fashion-MNIST artifacts listed in the processed "
        "summary. The pilot uses a deterministic 10,000-example training pool (9,000 train, 1,000 "
        "validation), 2,500 test examples, a small CNN, one original seed, five independent exact "
        "retraining seeds, and three unlearning methods. SCRUB is the paper-derived core max/min "
        "objective without privacy rewinding; its Fashion-MNIST settings are a declared adaptation.",
        "",
        "## Exact-retraining variability",
        "",
        "| Condition | Oracle accuracy range | JS exact/exact mean (p95) | Top-1 disagreement mean (p95) | CKA distance mean (p95) | Mean oracle train s | Total wall s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        accuracy = condition["oracle_test_accuracy"]
        js = condition["exact_exact"]["js_divergence"]
        disagreement = condition["exact_exact"]["top1_disagreement"]
        cka = condition["exact_exact"]["linear_cka_distance"]
        lines.append(
            f"| {condition['strategy']} {100 * condition['fraction']:.1f}% | "
            f"{accuracy['minimum']:.4f}–{accuracy['maximum']:.4f} | "
            f"{js['mean']:.4f} ({js['empirical_95th_percentile']:.4f}) | "
            f"{disagreement['mean']:.4f} ({disagreement['empirical_95th_percentile']:.4f}) | "
            f"{cka['mean']:.4f} ({cka['empirical_95th_percentile']:.4f}) | "
            f"{condition['oracle_training_seconds']['mean']:.2f} | {condition['wall_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Approximate-to-exact diagnostics",
            "",
            "| Condition | Method | Test acc | Forget acc | Retain acc | MIA AUC | JS / exact mean | Disagreement | CKA distance | Counterfactual envelope | Speedup |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for condition in conditions:
        label = f"{condition['strategy']} {100 * condition['fraction']:.1f}%"
        for name, method in condition["methods"].items():
            lines.append(
                f"| {label} | {name} | {method['test_accuracy']:.4f} | "
                f"{method['forget_accuracy']:.4f} | {method['retain_accuracy']:.4f} | "
                f"{method['membership_auc']:.4f} | {method['test_js_normalized']:.3f} | "
                f"{method['test_disagreement_mean']:.4f} | {method['test_cka_distance_mean']:.4f} | "
                f"{mark(method['matrix']['counterfactual'])} | {method['speedup']:.2f}× |"
            )
    lines.extend(
        [
            "",
            "## Descriptive contradiction matrix",
            "",
            "‘Within’ means the candidate mean did not exceed the empirical exact/exact 95th "
            "percentile for the declared diagnostic. With only ten exact/exact pairs, this is not "
            "an equivalence test and must not be read as statistical acceptance of equivalence.",
            "",
            "| Condition | Method | Forget | Retain | Privacy | Calibration | Prediction | Representation | Overall | Contradiction |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for condition in conditions:
        label = f"{condition['strategy']} {100 * condition['fraction']:.1f}%"
        for name, method in condition["methods"].items():
            matrix = method["matrix"]
            lines.append(
                f"| {label} | {name} | {mark(matrix['forget'])} | {mark(matrix['retain'])} | "
                f"{mark(matrix['privacy'])} | {mark(matrix['calibration'])} | "
                f"{mark(matrix['prediction'])} | {mark(matrix['representation'])} | "
                f"{mark(matrix['counterfactual'])} | "
                f"{'observed' if matrix['evaluation_contradiction'] else 'not observed'} |"
            )
    lines.extend(
        [
            "",
            "## Answers to pilot questions",
            "",
            "1. **Can exact-retraining variability be measured reliably?** The pipeline measures "
            "it reproducibly and the five replicas produce nonzero exact/exact distributions. Five "
            "seeds (ten pairs) are too few to estimate a reliable 95th-percentile tail; Tier-2 needs "
            "prospective precision/power calculations and preferably more independent replicas.",
            "2. **Are approximate/exact differences larger than exact/exact differences?** All methods "
            "remained inside the descriptive prediction and representation envelopes. All three "
            "high-loss methods fell outside the calibration envelope. These exploratory flags are "
            "not formal evidence of difference or equivalence.",
            f"3. **Do conventional and counterfactual metrics disagree?** The post-run exploratory "
            f"matrix found {contradiction_count} surface-pass/counterfactual-fail cells, all from "
            "high-loss deletion and calibration. The composite matrix rule was not frozen before the "
            "pilot and cannot support a confirmatory claim. Additionally, "
            "the high-loss condition makes the loss-based MIA AUC far from 0.5 even for exact retraining, "
            "demonstrating that a naïve near-chance privacy rule is confounded by deletion composition.",
            "4. **Is variance manageable?** Runtime variance is small, but behavioral exact/exact "
            "variance is material enough that single-oracle comparisons would be unsafe. The full "
            "study should model seed and condition effects rather than normalize by one noisy mean.",
            "5. **Which metrics appear informative?** JS divergence, top-1 disagreement, CKA distance, "
            "calibration gaps, and exact-referenced MIA are all operational. The MIA result shows why "
            "raw attack AUC is not interpretable without the exact reference. No scalar aggregate is justified.",
            "6. **What compute would the full study require?** This pilot's per-condition wall times "
            "are reported above. Linear extrapolation is inappropriate for CIFAR/ResNet; Tier-2 must "
            "benchmark one full condition on target hardware, then budget the frozen manifest.",
            "7. **Are assumptions clearly wrong?** Treating exact retraining as one deterministic point, "
            "treating MIA≈0.5 as a universal privacy criterion, and treating an empirical p95 from ten "
            "dependent pairs as a formal equivalence boundary are all untenable.",
            "8. **Is the question still worthwhile?** Yes as a falsifiable measurement study: the null "
            "pilot did not manufacture a contradiction, while it exposed reference variance and a "
            "privacy-metric composition confound that the publication design must control.",
            "",
            "## Before Tier-2",
            "",
            "Freeze equivalence margins and a hierarchical/paired analysis before final runs; benchmark "
            "CIFAR-10/ResNet-18 hardware; add LiRA-style and calibrated loss attacks; add local-neighbor "
            "and corruption probes; execute repeated candidate seeds; validate SCRUB hyperparameters "
            "without selecting on final test outcomes; and implement sequential stage-matched oracles. "
            "The current repository is not ready for confirmatory claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    manifest = yaml.safe_load(
        Path("research/counterfactual_fidelity_pilot_manifest.yaml").read_text(encoding="utf-8")
    )
    conditions = []
    for planned in manifest["conditions"]:
        config = load_config(planned["config"])
        path = Path(config.raw["output_root"]) / config.experiment_id / "result.json"
        conditions.append(analyze_condition(path))
    analysis = {
        "schema_version": 1,
        "stage": "counterfactual_fidelity_pilot",
        "publication_evidence": False,
        "manifest": "research/counterfactual_fidelity_pilot_manifest.yaml",
        "conditions": conditions,
    }
    processed = Path("results/processed/counterfactual_fidelity_pilot.json")
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = Path("reports/counterfactual_fidelity_pilot.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(analysis), encoding="utf-8")
    print(processed)
    print(report)


if __name__ == "__main__":
    main()
