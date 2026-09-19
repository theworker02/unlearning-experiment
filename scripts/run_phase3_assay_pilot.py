from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import partition_lineage, select_deletion_from_manifest
from exact_unlearning.devices import resolve_device
from exact_unlearning.experiment import _train_from_scratch
from exact_unlearning.metrics import linear_cka, predict, squared_mmd
from exact_unlearning.phase3 import minimum_detectable_separation, separability_ratio
from exact_unlearning.quality import evaluate_result
from exact_unlearning.registry import (
    reserve_experiment,
    write_checkpoint,
    write_prediction_artifact,
    write_raw_result,
)

CONFIG = Path("configs/experiments/phase3_multi_original_assay_pilot.yaml")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_arrays(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    ids: np.ndarray,
    *,
    partition_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    prediction = predict(model, x)
    arrays = {
        "artifact_schema_version": np.array(2, dtype=np.int64),
        "test_ids": ids.astype(np.int64),
        "test_true_labels": y.cpu().numpy().astype(np.int64),
        "test_probabilities": prediction.probabilities.astype(np.float32),
        "test_predicted_labels": prediction.labels.astype(np.int64),
        "test_logits": prediction.logits.astype(np.float32),
        "test_features": prediction.features.astype(np.float32),
    }
    if partition_sha256 is not None:
        arrays["partition_sha256"] = np.array(partition_sha256)
    return arrays


def distance(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    p = np.clip(left["test_probabilities"], 1e-12, 1.0)
    q = np.clip(right["test_probabilities"], 1e-12, 1.0)
    midpoint = (p + q) / 2
    js = np.mean(
        0.5 * np.sum(p * np.log(p / midpoint), axis=1)
        + 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    )
    return {
        "js_divergence": float(js),
        "prediction_disagreement": float(
            np.mean(left["test_predicted_labels"] != right["test_predicted_labels"])
        ),
        "final_feature_cka_distance": float(
            1 - linear_cka(left["test_features"], right["test_features"])
        ),
        "final_feature_mmd": squared_mmd(left["test_features"][:256], right["test_features"][:256]),
    }


def load_reference(reference: dict[str, str]) -> dict[str, np.ndarray]:
    required = ("test_probabilities", "test_predicted_labels", "test_features")
    with np.load(reference["path"], allow_pickle=False) as arrays:
        return {name: arrays[name] for name in required}


def validate_config(config_path: Path) -> dict[str, int]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw.get("maturity") != "PILOT":
        raise ValueError("assay calibration runner only accepts PILOT configurations")
    original_seeds = list(raw.get("original_seeds", []))
    exact_seeds = list(raw.get("exact_seeds", []))
    conditions = list(raw.get("conditions", []))
    if len(original_seeds) < 2 or len(set(original_seeds)) != len(original_seeds):
        raise ValueError("at least two unique original seeds are required")
    if len(exact_seeds) < 2 or len(set(exact_seeds)) != len(exact_seeds):
        raise ValueError("at least two unique exact seeds are required")
    if not conditions:
        raise ValueError("at least one assay condition is required")
    condition_ids = [str(condition.get("condition_id", "")) for condition in conditions]
    if any(
        not value or not value.replace("-", "").replace("_", "").isalnum()
        for value in condition_ids
    ):
        raise ValueError("every condition requires a safe condition_id")
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition_id values must be unique")
    data = load_dataset(raw["dataset"])
    for condition in conditions:
        select_deletion_from_manifest(
            condition["manifest"],
            data.train_ids,
            expected_dataset=data.metadata.name,
            expected_dataset_sha256=data.dataset_sha256,
            expected_strategy=str(condition["strategy"]),
            expected_fraction=float(condition["fraction"]),
            expected_seed=int(condition["seed"]),
            expected_target_class=condition.get("target_class"),
        )
    return {
        "original_seed_count": len(original_seeds),
        "exact_seed_count": len(exact_seeds),
        "condition_count": len(conditions),
        "planned_training_count": len(original_seeds) + len(exact_seeds) * len(conditions),
    }


def main(config_path: Path = CONFIG) -> None:
    validate_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = raw["experiment_id"]
    reserve_experiment(raw["output_root"], experiment_id)
    started = dt.datetime.now(dt.UTC).isoformat()
    data = load_dataset(raw["dataset"])
    device = resolve_device(str(raw["training"]["device"]))
    all_positions = np.arange(len(data.train_ids), dtype=np.int64)
    original_refs = []
    original_durations = []
    for seed in raw["original_seeds"]:
        model, duration = _train_from_scratch(raw, data, all_positions, int(seed), device=device)
        checkpoint = write_checkpoint(
            raw["output_root"], experiment_id, f"original-{seed}", model.state_dict()
        )
        predictions = write_prediction_artifact(
            raw["output_root"],
            experiment_id,
            f"original-{seed}",
            prediction_arrays(model, data.x_test, data.y_test, data.test_ids),
        )
        original_refs.append({"seed": seed, "checkpoint": checkpoint, "predictions": predictions})
        original_durations.append(duration)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    originals = [load_reference(item["predictions"]) for item in original_refs]
    conditions = []
    for condition in raw["conditions"]:
        manifest_path = Path(condition["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        deletion = select_deletion_from_manifest(
            manifest_path,
            data.train_ids,
            expected_dataset=data.metadata.name,
            expected_dataset_sha256=data.dataset_sha256,
            expected_strategy=str(condition["strategy"]),
            expected_fraction=float(condition["fraction"]),
            expected_seed=int(condition["seed"]),
            expected_target_class=condition.get("target_class"),
        )
        lineage = partition_lineage(deletion, data.train_ids)
        exact_refs = []
        durations = []
        for seed in raw["exact_seeds"]:
            model, duration = _train_from_scratch(
                raw,
                data,
                deletion.retain_positions,
                int(seed),
                forbidden_positions=deletion.forget_positions,
                device=device,
            )
            prefix = f"exact-{condition['condition_id']}-{seed}"
            checkpoint = write_checkpoint(
                raw["output_root"], experiment_id, prefix, model.state_dict()
            )
            predictions = write_prediction_artifact(
                raw["output_root"],
                experiment_id,
                prefix,
                prediction_arrays(
                    model,
                    data.x_test,
                    data.y_test,
                    data.test_ids,
                    partition_sha256=str(lineage["partition_sha256"]),
                ),
            )
            exact_refs.append(
                {
                    "seed": seed,
                    "partition_sha256": lineage["partition_sha256"],
                    "checkpoint": checkpoint,
                    "predictions": predictions,
                }
            )
            durations.append(duration)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        exact = [load_reference(item["predictions"]) for item in exact_refs]
        original_exact_records = [distance(left, right) for left in originals for right in exact]
        exact_pair_records = []
        for left, right in combinations(range(len(exact)), 2):
            exact_pair_records.append((left, right, distance(exact[left], exact[right])))
        metrics = {}
        for metric in (
            "js_divergence",
            "prediction_disagreement",
            "final_feature_cka_distance",
            "final_feature_mmd",
        ):
            original_exact = np.asarray([record[metric] for record in original_exact_records])
            pair_matrix = np.zeros((len(exact), len(exact)), dtype=float)
            exact_pairs = []
            for left, right, record in exact_pair_records:
                value = record[metric]
                pair_matrix[left, right] = pair_matrix[right, left] = value
                exact_pairs.append(value)
            seed_null = np.asarray(
                [np.mean(np.delete(pair_matrix[index], index)) for index in range(len(exact))]
            )
            spec = raw["statistics"]["candidate_separation_ratios"]
            ratios = np.arange(spec["start"], spec["stop"] + spec["step"] / 2, spec["step"])
            power = minimum_detectable_separation(
                seed_null,
                alpha=raw["statistics"]["alpha"],
                target_power=raw["statistics"]["target_power"],
                repetitions=raw["statistics"]["simulation_repetitions"],
                seed=raw["statistics"]["simulation_seed"],
                candidate_ratios=ratios,
            )
            ratio = separability_ratio(original_exact, np.asarray(exact_pairs))
            metrics[metric] = {
                "original_seed_count": len(originals),
                "exact_seed_count": len(exact),
                "original_exact_median": float(np.median(original_exact)),
                "exact_exact_q95": float(np.quantile(exact_pairs, 0.95)),
                "separability_ratio": ratio,
                "power_simulation": power,
                "assay_sensitive": power["minimum_ratio"] is not None
                and ratio >= power["minimum_ratio"],
            }
        conditions.append(
            {
                "condition_id": condition["condition_id"],
                "fraction": condition["fraction"],
                "manifest": {
                    "path": manifest_path.as_posix(),
                    "sha256": sha256(manifest_path),
                    "manifest_hash": manifest["manifest_hash"],
                },
                "forget_count": len(deletion.forget_positions),
                "partition_lineage": lineage,
                "metrics_partition_sha256": lineage["partition_sha256"],
                "exact_runs": exact_refs,
                "exact_training_seconds": durations,
                "metrics": metrics,
            }
        )
    result = {
        "schema_version": 3,
        "experiment_id": experiment_id,
        "maturity": "PILOT",
        "status": "completed",
        "interpretation": "Assay calibration only; approximate methods were not evaluated.",
        "started_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config": raw,
        "config_sha256": sha256(config_path),
        "dataset": {
            "name": data.metadata.name,
            "sha256": data.dataset_sha256,
            "train_count": len(data.train_ids),
            "test_count": len(data.test_ids),
        },
        "original_runs": original_refs,
        "original_training_seconds": original_durations,
        "conditions": conditions,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "rocm": torch.version.hip,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "hsa_dxg": os.environ.get("HSA_ENABLE_DXG_DETECTION"),
        },
    }
    result["quality_gate"] = evaluate_result(result).as_dict()
    destination = write_raw_result(raw["output_root"], experiment_id, result)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a preregistered Phase-3 assay pilot")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_only:
        print(json.dumps(validate_config(arguments.config), indent=2, sort_keys=True))
    else:
        main(arguments.config)
