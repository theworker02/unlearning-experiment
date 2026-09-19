from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import os
import platform
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import sklearn
import torch

from .config import ExperimentConfig
from .contracts import UnlearningContext
from .data import DatasetSplit, load_dataset
from .deletion import select_deletion, select_deletion_from_manifest
from .devices import resolve_device
from .methods import run_unlearning_method
from .metrics import (
    class_conditioned_disagreement,
    local_behavior_profile,
    loss_membership_auc,
    membership_audit,
    model_discrepancy,
    parameter_distance,
    predict,
    predictive_metrics,
    reliability_curve,
)
from .models import Classifier, build_model
from .quality import evaluate_result
from .registry import (
    reserve_experiment,
    write_checkpoint,
    write_prediction_artifact,
    write_raw_result,
)
from .seeding import seed_everything
from .training import train_model


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _index_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _prediction_arrays(
    model: Classifier,
    data: DatasetSplit,
    retain_positions: np.ndarray,
    forget_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"artifact_schema_version": np.array(2, dtype=np.int64)}
    splits = {
        "test": (data.test_ids, data.x_test, data.y_test),
        "validation": (data.validation_ids, data.x_validation, data.y_validation),
        "retain": (
            data.train_ids[retain_positions],
            data.x_train[retain_positions],
            data.y_train[retain_positions],
        ),
    }
    if len(forget_positions):
        splits["forget"] = (
            data.train_ids[forget_positions],
            data.x_train[forget_positions],
            data.y_train[forget_positions],
        )
    for split, (ids, x, y) in splits.items():
        prediction = predict(model, x)
        arrays[f"{split}_ids"] = np.asarray(ids, dtype=np.int64)
        arrays[f"{split}_true_labels"] = y.cpu().numpy().astype(np.int64, copy=False)
        arrays[f"{split}_probabilities"] = prediction.probabilities.astype(np.float32)
        arrays[f"{split}_features"] = prediction.features.astype(np.float32)
        if prediction.logits is not None:
            arrays[f"{split}_logits"] = prediction.logits.astype(np.float32)
        arrays[f"{split}_predicted_labels"] = prediction.labels.astype(np.int64, copy=False)
    return arrays


def _new_model(
    config: dict[str, Any], seed: int, input_shape: tuple[int, ...], classes: int
) -> Classifier:
    seed_everything(seed, bool(config["training"]["deterministic"]))
    return build_model(
        str(config["model"]["architecture"]),
        input_shape,
        int(config["model"]["hidden_dim"]),
        classes,
    )


def _train_from_scratch(
    config: dict[str, Any],
    data: DatasetSplit,
    positions: np.ndarray,
    seed: int,
    *,
    forbidden_positions: np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[Classifier, float]:
    if forbidden_positions is not None and np.intersect1d(positions, forbidden_positions).size:
        raise AssertionError("exact retraining positions include a forgotten example")
    model = _new_model(config, seed, tuple(data.x_train.shape[1:]), int(data.y_train.max()) + 1)
    training = config["training"]
    result = train_model(
        model,
        data.x_train[positions],
        data.y_train[positions],
        seed=seed,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        deterministic=bool(training["deterministic"]),
        device=device,
        optimizer_name=str(training.get("optimizer", "adamw")),
        momentum=float(training.get("momentum", 0.9)),
    )
    return result.model, result.duration_seconds


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(mean(values)),
        "standard_deviation": float(stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "median": float(np.median(array)),
        "empirical_95th_percentile": float(np.quantile(array, 0.95)),
        "n": len(values),
        "values": [float(value) for value in values],
    }


def _compare_to_oracles(
    candidate: Classifier,
    oracles: list[Classifier],
    x: torch.Tensor,
    y: torch.Tensor,
    bins: int,
) -> dict[str, dict[str, float]]:
    candidate_prediction = predict(candidate, x)
    observations: dict[str, list[float]] = {}
    for oracle in oracles:
        comparison = model_discrepancy(
            candidate_prediction, predict(oracle, x), y.cpu().numpy(), bins
        )
        comparison.update(parameter_distance(candidate, oracle))
        for key, value in comparison.items():
            observations.setdefault(key, []).append(value)
    return {key: _summary(values) for key, values in observations.items()}


def _oracle_dispersion(
    oracles: list[Classifier], x: torch.Tensor, y: torch.Tensor, bins: int
) -> dict[str, dict[str, float]]:
    observations: dict[str, list[float]] = {}
    for left, right in itertools.combinations(oracles, 2):
        comparison = model_discrepancy(predict(left, x), predict(right, x), y.cpu().numpy(), bins)
        comparison.update(parameter_distance(left, right))
        for key, value in comparison.items():
            observations.setdefault(key, []).append(value)
    return {key: _summary(values) for key, values in observations.items()}


def _local_profiles_to_oracles(
    candidate: Classifier,
    oracles: list[Classifier],
    forgotten_x: torch.Tensor,
    probe_x: torch.Tensor,
    *,
    distance_bins: int = 4,
) -> dict[str, Any]:
    """Compare candidate/oracle behavior by input distance from forgotten samples."""
    forgotten_features = forgotten_x.detach().cpu().numpy().reshape(len(forgotten_x), -1)
    probe_features = probe_x.detach().cpu().numpy().reshape(len(probe_x), -1)
    candidate_probabilities = predict(candidate, probe_x).probabilities
    profiles = [
        local_behavior_profile(
            forgotten_features,
            probe_features,
            candidate_probabilities,
            predict(oracle, probe_x).probabilities,
            distance_bins=min(distance_bins, len(probe_x)),
        )
        for oracle in oracles
    ]
    bins = []
    for index in range(len(profiles[0])):
        first = profiles[0][index]
        bins.append(
            {
                "distance_bin": first["distance_bin"],
                "count": first["count"],
                "minimum_distance": first["minimum_distance"],
                "maximum_distance": first["maximum_distance"],
                "mean_distance": first["mean_distance"],
                "mean_js_divergence_to_exact": _summary(
                    [float(profile[index]["mean_js_divergence"]) for profile in profiles]
                ),
                "top1_change_rate_to_exact": _summary(
                    [float(profile[index]["top1_change_rate"]) for profile in profiles]
                ),
            }
        )
    return {
        "feature_space": "normalized_input_flattened",
        "probe_split": "test",
        "oracle_count": len(oracles),
        "distance_bins": bins,
    }


def _class_conditioned_to_oracles(
    candidate: Classifier,
    oracles: list[Classifier],
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, dict[str, float]]:
    candidate_labels = predict(candidate, x).labels
    truth = y.detach().cpu().numpy()
    observations: dict[int, list[float]] = {}
    for oracle in oracles:
        rates = class_conditioned_disagreement(candidate_labels, predict(oracle, x).labels, truth)
        for label, rate in rates.items():
            observations.setdefault(label, []).append(rate)
    return {str(label): _summary(values) for label, values in sorted(observations.items())}


def _normalized_distance(
    candidate: dict[str, dict[str, float]], dispersion: dict[str, dict[str, float]]
) -> dict[str, float | None]:
    normalized: dict[str, float | None] = {}
    for metric, summary in candidate.items():
        baseline = dispersion.get(metric, {}).get("mean", 0.0)
        normalized[metric] = float(summary["mean"] / baseline) if baseline > 1e-12 else None
    return normalized


def _envelope_status(
    candidate: dict[str, dict[str, float]], dispersion: dict[str, dict[str, float]]
) -> dict[str, bool | None]:
    status: dict[str, bool | None] = {}
    for metric, summary in candidate.items():
        threshold = dispersion.get(metric, {}).get("empirical_95th_percentile")
        status[metric] = summary["mean"] <= threshold if threshold is not None else None
    return status


def run_experiment(config: ExperimentConfig) -> Path:
    started_at = dt.datetime.now(dt.UTC).isoformat()
    raw = config.raw
    reserve_experiment(raw["output_root"], config.experiment_id)
    dataset_config = raw["dataset"]
    data = load_dataset(dataset_config)
    device = resolve_device(str(raw["training"].get("device", "auto")))
    all_positions = np.arange(len(data.train_ids), dtype=np.int64)
    original, original_seconds = _train_from_scratch(
        raw, data, all_positions, int(raw["original_seed"]), device=device
    )
    deletion_config = raw["deletion"]
    if deletion_config.get("manifest"):
        deletion = select_deletion_from_manifest(
            deletion_config["manifest"],
            data.train_ids,
            expected_dataset=data.metadata.name,
            expected_dataset_sha256=data.dataset_sha256,
            expected_strategy=str(deletion_config["strategy"]),
            expected_fraction=float(deletion_config["fraction"]),
            expected_seed=int(deletion_config["seed"]),
        )
    else:
        deletion = select_deletion(
            str(deletion_config["strategy"]),
            float(deletion_config["fraction"]),
            int(deletion_config["seed"]),
            data.train_ids,
            data.x_train,
            data.y_train,
            original,
            int(deletion_config["target_class"])
            if deletion_config.get("target_class") is not None
            else None,
        )
    oracles: list[Classifier] = []
    oracle_durations: list[float] = []
    for seed in raw["oracle_seeds"]:
        model, duration = _train_from_scratch(
            raw,
            data,
            deletion.retain_positions,
            int(seed),
            forbidden_positions=deletion.forget_positions,
            device=device,
        )
        oracles.append(model)
        oracle_durations.append(duration)

    oracle_privacy_scores: list[float] = []
    oracle_privacy_audits: list[dict[str, dict[str, float]]] = []
    if len(deletion.forget_positions):
        for oracle in oracles:
            oracle_forget = predict(oracle, data.x_train[deletion.forget_positions])
            oracle_test = predict(oracle, data.x_test)
            oracle_privacy_scores.append(
                loss_membership_auc(
                    oracle_forget.probabilities,
                    data.y_train[deletion.forget_positions].numpy(),
                    oracle_test.probabilities,
                    data.y_test.numpy(),
                )
            )
            oracle_privacy_audits.append(
                membership_audit(
                    oracle_forget.probabilities,
                    data.y_train[deletion.forget_positions].numpy(),
                    oracle_test.probabilities,
                    data.y_test.numpy(),
                )
            )
    oracle_privacy_pairwise = [
        abs(left - right)
        for index, left in enumerate(oracle_privacy_scores)
        for right in oracle_privacy_scores[index + 1 :]
    ]
    bins = int(raw["evaluation"]["ece_bins"])
    oracle_metrics_by_seed = []
    for seed, oracle in zip(raw["oracle_seeds"], oracles, strict=True):
        oracle_metrics_by_seed.append(
            {
                "seed": int(seed),
                "test": predictive_metrics(predict(oracle, data.x_test), data.y_test.numpy(), bins),
                "test_reliability_curve": reliability_curve(
                    predict(oracle, data.x_test).probabilities, data.y_test.numpy(), bins
                ),
                "retain": predictive_metrics(
                    predict(oracle, data.x_train[deletion.retain_positions]),
                    data.y_train[deletion.retain_positions].numpy(),
                    bins,
                ),
                "forget": predictive_metrics(
                    predict(oracle, data.x_train[deletion.forget_positions]),
                    data.y_train[deletion.forget_positions].numpy(),
                    bins,
                )
                if len(deletion.forget_positions)
                else None,
            }
        )

    checkpoint_references: dict[str, Any] = {
        "original": write_checkpoint(
            raw["output_root"], config.experiment_id, "original", original.state_dict()
        ),
        "oracles": [],
        "methods": {},
    }
    prediction_references: dict[str, Any] = {
        "original": write_prediction_artifact(
            raw["output_root"],
            config.experiment_id,
            "original",
            _prediction_arrays(
                original, data, deletion.retain_positions, deletion.forget_positions
            ),
        ),
        "oracles": [],
        "methods": {},
    }
    for seed, oracle in zip(raw["oracle_seeds"], oracles, strict=True):
        checkpoint_references["oracles"].append(
            {
                "seed": int(seed),
                **write_checkpoint(
                    raw["output_root"],
                    config.experiment_id,
                    f"oracle-{seed}",
                    oracle.state_dict(),
                ),
            }
        )
        prediction_references["oracles"].append(
            {
                "seed": int(seed),
                **write_prediction_artifact(
                    raw["output_root"],
                    config.experiment_id,
                    f"oracle-{seed}",
                    _prediction_arrays(
                        oracle, data, deletion.retain_positions, deletion.forget_positions
                    ),
                ),
            }
        )

    unlearning_config = raw["unlearning"]
    methods: dict[str, Any] = {}
    for method_index, method in enumerate(unlearning_config["methods"]):
        unlearning_seed = int(raw["original_seed"]) + 10_000 + method_index
        method_config = {**unlearning_config, "batch_size": raw["training"]["batch_size"]}
        descriptor, outcome = run_unlearning_method(
            method,
            UnlearningContext(
                original=original,
                x_retain=data.x_train[deletion.retain_positions],
                y_retain=data.y_train[deletion.retain_positions],
                x_forget=data.x_train[deletion.forget_positions],
                y_forget=data.y_train[deletion.forget_positions],
                seed=unlearning_seed,
                config=method_config,
                device=device,
            ),
        )
        test_prediction = predict(outcome.model, data.x_test)
        retain_prediction = predict(outcome.model, data.x_train[deletion.retain_positions])
        bins = int(raw["evaluation"]["ece_bins"])
        comparisons_by_split = {
            "test": _compare_to_oracles(outcome.model, oracles, data.x_test, data.y_test, bins),
            "retain": _compare_to_oracles(
                outcome.model,
                oracles,
                data.x_train[deletion.retain_positions],
                data.y_train[deletion.retain_positions],
                bins,
            ),
        }
        if len(deletion.forget_positions):
            comparisons_by_split["forget"] = _compare_to_oracles(
                outcome.model,
                oracles,
                data.x_train[deletion.forget_positions],
                data.y_train[deletion.forget_positions],
                bins,
            )
        comparisons = comparisons_by_split["test"]
        forget_metrics = None
        membership_auc = None
        method_privacy_audit = None
        if len(deletion.forget_positions):
            forget_prediction = predict(outcome.model, data.x_train[deletion.forget_positions])
            forget_metrics = predictive_metrics(
                forget_prediction,
                data.y_train[deletion.forget_positions].numpy(),
                int(raw["evaluation"]["ece_bins"]),
            )
            membership_auc = loss_membership_auc(
                forget_prediction.probabilities,
                data.y_train[deletion.forget_positions].numpy(),
                test_prediction.probabilities,
                data.y_test.numpy(),
            )
            method_privacy_audit = membership_audit(
                forget_prediction.probabilities,
                data.y_train[deletion.forget_positions].numpy(),
                test_prediction.probabilities,
                data.y_test.numpy(),
            )
        checkpoint_references["methods"][method] = write_checkpoint(
            raw["output_root"], config.experiment_id, f"method-{method}", outcome.model.state_dict()
        )
        prediction_references["methods"][method] = write_prediction_artifact(
            raw["output_root"],
            config.experiment_id,
            f"method-{method}",
            _prediction_arrays(
                outcome.model, data, deletion.retain_positions, deletion.forget_positions
            ),
        )
        methods[method] = {
            "simple_engineering_baseline": descriptor.simple_baseline,
            "literature_reference": descriptor.literature_reference,
            "implementation_provenance": descriptor.implementation_provenance,
            "implementation_notes": descriptor.implementation_notes,
            "seed": unlearning_seed,
            "duration_seconds": outcome.duration_seconds,
            "speedup_vs_mean_retraining": float(mean(oracle_durations) / outcome.duration_seconds)
            if outcome.duration_seconds > 0
            else None,
            "test_metrics": predictive_metrics(
                test_prediction, data.y_test.numpy(), int(raw["evaluation"]["ece_bins"])
            ),
            "reliability_curves": {
                "test": reliability_curve(test_prediction.probabilities, data.y_test.numpy(), bins),
                "retain": reliability_curve(
                    retain_prediction.probabilities,
                    data.y_train[deletion.retain_positions].numpy(),
                    bins,
                ),
                "forget": reliability_curve(
                    forget_prediction.probabilities,
                    data.y_train[deletion.forget_positions].numpy(),
                    bins,
                )
                if len(deletion.forget_positions)
                else None,
            },
            "forget_metrics": forget_metrics,
            "retain_metrics": predictive_metrics(
                retain_prediction,
                data.y_train[deletion.retain_positions].numpy(),
                int(raw["evaluation"]["ece_bins"]),
            ),
            "loss_membership_auc_forget_vs_test": membership_auc,
            "privacy_fidelity_absolute_auc_difference": _summary(
                [abs(membership_auc - value) for value in oracle_privacy_scores]
            )
            if membership_auc is not None
            else None,
            "membership_inference_audit": method_privacy_audit,
            "privacy_fidelity_to_exact": {
                signal: {
                    output: _summary(
                        [
                            abs(method_privacy_audit[signal][output] - audit[signal][output])
                            for audit in oracle_privacy_audits
                        ]
                    )
                    for output in ("roc_auc", "attack_advantage", "tpr_at_fpr")
                }
                for signal in ("loss", "confidence", "entropy", "margin")
            }
            if method_privacy_audit is not None
            else None,
            "distance_to_retrained_distribution": comparisons,
            "distance_to_retrained_distribution_by_split": comparisons_by_split,
            "local_behavior_on_test": _local_profiles_to_oracles(
                outcome.model,
                oracles,
                data.x_train[deletion.forget_positions],
                data.x_test,
            )
            if len(deletion.forget_positions)
            else None,
            "class_conditioned_disagreement_to_exact_on_test": _class_conditioned_to_oracles(
                outcome.model, oracles, data.x_test, data.y_test
            ),
        }

    bins = int(raw["evaluation"]["ece_bins"])
    dispersion_by_split = {
        "test": _oracle_dispersion(oracles, data.x_test, data.y_test, bins),
        "retain": _oracle_dispersion(
            oracles,
            data.x_train[deletion.retain_positions],
            data.y_train[deletion.retain_positions],
            bins,
        ),
    }
    if len(deletion.forget_positions):
        dispersion_by_split["forget"] = _oracle_dispersion(
            oracles,
            data.x_train[deletion.forget_positions],
            data.y_train[deletion.forget_positions],
            bins,
        )
    dispersion = dispersion_by_split["test"]
    for method in methods.values():
        method["distance_normalized_by_retraining_dispersion"] = _normalized_distance(
            method["distance_to_retrained_distribution"], dispersion
        )
        method["distance_normalized_by_retraining_dispersion_by_split"] = {
            split: _normalized_distance(values, dispersion_by_split[split])
            for split, values in method["distance_to_retrained_distribution_by_split"].items()
        }
        method["within_empirical_retraining_envelope_by_split"] = {
            split: _envelope_status(values, dispersion_by_split[split])
            for split, values in method["distance_to_retrained_distribution_by_split"].items()
        }
    result = {
        "schema_version": 2,
        "experiment_id": config.experiment_id,
        "maturity": config.maturity.value,
        "status": "completed",
        "research_stage": (
            "Tier-0/Tier-1 infrastructure validation — not publication evidence."
            if config.maturity.value == "VALIDATION"
            else "Counterfactual Fidelity Pilot — not publication evidence."
        ),
        "started_at_utc": started_at,
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config": raw,
        "config_sha256": config.sha256,
        "git_commit": _git_commit(),
        "dataset": {
            "name": data.metadata.name,
            "version": data.metadata.version,
            "source": data.metadata.source,
            "license": data.metadata.license,
            "sklearn_version": sklearn.__version__,
            "sha256_after_scaling": data.dataset_sha256,
            "train_count": len(data.train_ids),
            "validation_count": len(data.validation_ids),
            "test_count": len(data.test_ids),
            "train_ids": data.train_ids.tolist(),
            "validation_ids": data.validation_ids.tolist(),
            "test_ids": data.test_ids.tolist(),
        },
        "deletion": {
            "count": len(deletion.forget_positions),
            "retained_count": len(deletion.retain_positions),
            "forget_ids": deletion.forget_ids.tolist(),
            "forget_positions": deletion.forget_positions.tolist(),
            "retain_ids": data.train_ids[deletion.retain_positions].tolist(),
            "retain_positions": deletion.retain_positions.tolist(),
            "sha256": deletion.sha256,
            "manifest": (
                {
                    "path": str(deletion_config["manifest"]),
                    "sha256": hashlib.sha256(
                        Path(deletion_config["manifest"]).read_bytes()
                    ).hexdigest(),
                }
                if deletion_config.get("manifest")
                else None
            ),
            "forget_ids_sha256": _index_digest(deletion.forget_ids),
            "retain_ids_sha256": _index_digest(data.train_ids[deletion.retain_positions]),
            "train_ids_sha256": _index_digest(data.train_ids),
            "exact_retraining_exclusion_verified": not np.intersect1d(
                deletion.forget_positions, deletion.retain_positions
            ).size,
            "retained_manifest_matches_complement": np.array_equal(
                deletion.retain_positions,
                np.setdiff1d(all_positions, deletion.forget_positions, assume_unique=True),
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "rocm_version": torch.version.hip,
            "original_training_seconds": original_seconds,
            "oracle_training_seconds": oracle_durations,
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "git_dirty": _git_dirty(),
            "determinism_environment": {
                key: os.environ.get(key)
                for key in (
                    "CUBLAS_WORKSPACE_CONFIG",
                    "HIPBLAS_WORKSPACE_CONFIG",
                    "HSA_ENABLE_DXG_DETECTION",
                    "PYTHONHASHSEED",
                )
            },
        },
        "exact_retraining_dispersion_on_test": dispersion,
        "exact_retraining_dispersion_by_split": dispersion_by_split,
        "exact_retraining_privacy_reference": {
            "loss_membership_auc_forget_vs_test": _summary(oracle_privacy_scores)
            if oracle_privacy_scores
            else None,
            "pairwise_absolute_auc_difference": _summary(oracle_privacy_pairwise)
            if oracle_privacy_pairwise
            else None,
            "audits_by_seed": [
                {"seed": int(seed), "signals": audit}
                for seed, audit in zip(raw["oracle_seeds"], oracle_privacy_audits, strict=True)
            ]
            if oracle_privacy_audits
            else [],
        },
        "exact_retraining_metrics_by_seed": oracle_metrics_by_seed,
        "methods": methods,
        "checkpoints": checkpoint_references,
        "predictions": prediction_references,
        "interpretation": "No conclusion is generated automatically from a single run.",
    }
    result["quality_gate"] = evaluate_result(result).as_dict()
    return write_raw_result(raw["output_root"], config.experiment_id, result)
