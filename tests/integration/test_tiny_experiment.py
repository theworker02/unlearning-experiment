from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.config import load_config
from exact_unlearning.experiment import run_experiment


def test_tiny_experiment_writes_complete_immutable_record(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/experiments/digits_linear_tiny.yaml").read_text())
    raw["experiment_id"] = "integration-tiny"
    raw["output_root"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result_path = run_experiment(load_config(config_path))
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert len(result["runtime"]["oracle_training_seconds"]) == 2
    assert set(result["methods"]) == {"fine_tune", "naive_gradient_ascent"}
    assert "js_divergence" in result["exact_retraining_dispersion_on_test"]
    assert result["deletion"]["exact_retraining_exclusion_verified"]
    assert not set(result["deletion"]["forget_ids"]).intersection(result["deletion"]["retain_ids"])
    assert result["checkpoints"]["original"]["sha256"]
    assert result["maturity"] == "VALIDATION"
    assert result["predictions"]["original"]["sha256"]
    for method in result["methods"].values():
        local = method["local_behavior_on_test"]
        assert local["feature_space"] == "normalized_input_flattened"
        assert local["oracle_count"] == 2
        assert len(local["distance_bins"]) == 4
        assert (
            sum(item["count"] for item in local["distance_bins"]) == result["dataset"]["test_count"]
        )
        conditioned = method["class_conditioned_disagreement_to_exact_on_test"]
        assert set(conditioned) == {str(label) for label in range(10)}
        assert all(summary["n"] == 2 for summary in conditioned.values())
        curves = method["reliability_curves"]
        assert len(curves["test"]) == raw["evaluation"]["ece_bins"]
        assert len(curves["retain"]) == raw["evaluation"]["ece_bins"]
        assert len(curves["forget"]) == raw["evaluation"]["ece_bins"]
        assert set(method["membership_inference_audit"]) == {
            "loss",
            "confidence",
            "entropy",
            "margin",
        }
        assert set(method["privacy_fidelity_to_exact"]) == {
            "loss",
            "confidence",
            "entropy",
            "margin",
        }
    with np.load(result["predictions"]["original"]["path"], allow_pickle=False) as arrays:
        assert arrays["artifact_schema_version"].item() == 2
        assert "test_logits" in arrays.files
        assert "test_features" in arrays.files
        assert arrays["test_features"].shape[0] == result["dataset"]["test_count"]
    assert result["quality_gate"]["passed"]


def test_zero_deletion_runs_end_to_end(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/experiments/digits_linear_tiny.yaml").read_text())
    raw["experiment_id"] = "integration-zero"
    raw["output_root"] = str(tmp_path)
    raw["deletion"]["fraction"] = 0.0
    raw["unlearning"]["methods"] = ["fine_tune"]
    config_path = tmp_path / "zero.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = yaml.safe_load(run_experiment(load_config(config_path)).read_text(encoding="utf-8"))
    assert result["deletion"]["count"] == 0
    assert result["methods"]["fine_tune"]["forget_metrics"] is None
    assert result["methods"]["fine_tune"]["loss_membership_auc_forget_vs_test"] is None
    assert result["methods"]["fine_tune"]["membership_inference_audit"] is None
    assert result["methods"]["fine_tune"]["privacy_fidelity_to_exact"] is None
    assert result["methods"]["fine_tune"]["local_behavior_on_test"] is None


def test_complete_deterministic_runs_reproduce_metrics(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/experiments/digits_linear_tiny.yaml").read_text())
    raw["output_root"] = str(tmp_path)
    results = []
    for run_id in ("repeat-a", "repeat-b"):
        raw["experiment_id"] = run_id
        path = tmp_path / f"{run_id}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        results.append(
            yaml.safe_load(run_experiment(load_config(path)).read_text(encoding="utf-8"))
        )
    assert results[0]["deletion"]["forget_ids"] == results[1]["deletion"]["forget_ids"]
    assert (
        results[0]["exact_retraining_dispersion_on_test"]
        == results[1]["exact_retraining_dispersion_on_test"]
    )
    for method in results[0]["methods"]:
        assert (
            results[0]["methods"][method]["test_metrics"]
            == results[1]["methods"][method]["test_metrics"]
        )
