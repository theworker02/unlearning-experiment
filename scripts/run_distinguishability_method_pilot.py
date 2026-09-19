from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import yaml

from exact_unlearning.contracts import UnlearningContext
from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import select_deletion_from_manifest
from exact_unlearning.devices import resolve_device
from exact_unlearning.experiment import _prediction_arrays
from exact_unlearning.methods import run_unlearning_method
from exact_unlearning.metrics import loss_membership_auc, predict, predictive_metrics
from exact_unlearning.models import build_model
from exact_unlearning.quality import evaluate_result
from exact_unlearning.registry import (
    reserve_experiment,
    write_checkpoint,
    write_failure_record,
    write_prediction_artifact,
    write_raw_result,
)

if __package__:
    from scripts.validate_distinguishability_method_pilot import PLAN, sha256, validate
else:
    from validate_distinguishability_method_pilot import PLAN, sha256, validate


def load_checkpoint_model(
    reference: dict[str, object],
    source_config: dict[str, object],
    data: object,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = Path(str(reference["checkpoint"]["path"]))
    if sha256(checkpoint) != reference["checkpoint"]["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    model = build_model(
        str(source_config["model"]["architecture"]),
        tuple(data.x_train.shape[1:]),
        int(source_config["model"]["hidden_dim"]),
        int(data.y_train.max()) + 1,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def validate_execution_plan(plan_path: Path = PLAN) -> dict[str, object]:
    summary = validate(plan_path, require_locked=True)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    source = json.loads(Path(plan["source_ensemble"]["result"]).read_text(encoding="utf-8"))
    exact_evaluations = len(plan["source_ensemble"]["exact_seeds"]) * len(plan["conditions"])
    expected_artifacts = summary["planned_unlearning_runs"] * 2 + exact_evaluations + 1
    return {
        **summary,
        "experiment_id": plan["experiment_id"],
        "expected_new_artifacts": expected_artifacts,
        "source_original_artifacts_reused": len(source["original_runs"]),
        "source_exact_artifacts_reused": len(plan["source_ensemble"]["exact_seeds"])
        * len(plan["conditions"]),
        "new_split_complete_exact_evaluations": exact_evaluations,
    }


def _main(plan_path: Path = PLAN) -> None:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if plan["status"] != "LOCKED" or plan["execution"]["authorized"] is not True:
        raise SystemExit("execution blocked: plan must be LOCKED and explicitly authorized")
    validate_execution_plan(plan_path)

    source_path = Path(plan["source_ensemble"]["result"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_config = source["config"]
    data = load_dataset(source_config["dataset"])
    device = resolve_device(str(source_config["training"]["device"]))
    output_root = plan["execution"]["output_root"]
    experiment_id = plan["experiment_id"]
    reserve_experiment(output_root, experiment_id)
    started = dt.datetime.now(dt.UTC).isoformat()

    originals_by_seed = {int(item["seed"]): item for item in source["original_runs"]}
    source_conditions = {item["condition_id"]: item for item in source["conditions"]}
    source_condition_configs = {
        item["condition_id"]: item for item in source_config["conditions"]
    }
    selected_exact_seeds = set(plan["source_ensemble"]["exact_seeds"])
    conditions = []
    for planned_condition in plan["conditions"]:
        condition_id = planned_condition["condition_id"]
        source_condition = source_conditions[condition_id]
        condition_config = source_condition_configs[condition_id]
        deletion = select_deletion_from_manifest(
            planned_condition["deletion_manifest"],
            data.train_ids,
            expected_dataset=data.metadata.name,
            expected_dataset_sha256=data.dataset_sha256,
            expected_strategy=str(condition_config["strategy"]),
            expected_fraction=float(condition_config["fraction"]),
            expected_seed=int(condition_config["seed"]),
            expected_target_class=int(planned_condition["target_class"]),
        )
        lineage = source_condition["partition_lineage"]
        source_exact_runs = [
            item
            for item in source_condition["exact_runs"]
            if int(item["seed"]) in selected_exact_seeds
        ]
        exact_runs = []
        for source_exact in source_exact_runs:
            exact_model = load_checkpoint_model(source_exact, source_config, data, device)
            exact_arrays = _prediction_arrays(
                exact_model, data, deletion.retain_positions, deletion.forget_positions
            )
            exact_arrays["partition_sha256"] = np.array(lineage["partition_sha256"])
            exact_predictions = write_prediction_artifact(
                output_root,
                experiment_id,
                f"reference-exact-{condition_id}-{source_exact['seed']}",
                exact_arrays,
            )
            test = predict(exact_model, data.x_test)
            retain = predict(exact_model, data.x_train[deletion.retain_positions])
            forget = predict(exact_model, data.x_train[deletion.forget_positions])
            bins = 15
            exact_runs.append(
                {
                    "seed": int(source_exact["seed"]),
                    "partition_sha256": lineage["partition_sha256"],
                    "checkpoint": source_exact["checkpoint"],
                    "predictions": exact_predictions,
                    "metrics": {
                        "test": predictive_metrics(test, data.y_test.numpy(), bins),
                        "retain": predictive_metrics(
                            retain,
                            data.y_train[deletion.retain_positions].numpy(),
                            bins,
                        ),
                        "forget": predictive_metrics(
                            forget,
                            data.y_train[deletion.forget_positions].numpy(),
                            bins,
                        ),
                        "loss_membership_auc": loss_membership_auc(
                            forget.probabilities,
                            data.y_train[deletion.forget_positions].numpy(),
                            test.probabilities,
                            data.y_test.numpy(),
                        ),
                    },
                }
            )
            del exact_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        method_runs = []
        for original_seed in plan["source_ensemble"]["original_seeds"]:
            original_reference = originals_by_seed[int(original_seed)]
            for method_index, method_spec in enumerate(plan["methods"]):
                model = load_checkpoint_model(
                    original_reference, source_config, data, device
                )
                unlearning_seed = int(original_seed) + 10_000 + method_index
                method_config = {
                    **plan["unlearning"],
                    "batch_size": int(source_config["training"]["batch_size"]),
                }
                descriptor, outcome = run_unlearning_method(
                    method_spec["name"],
                    UnlearningContext(
                        original=model,
                        x_retain=data.x_train[deletion.retain_positions],
                        y_retain=data.y_train[deletion.retain_positions],
                        x_forget=data.x_train[deletion.forget_positions],
                        y_forget=data.y_train[deletion.forget_positions],
                        seed=unlearning_seed,
                        config=method_config,
                        device=device,
                    ),
                )
                name = (
                    f"method-{condition_id}-original{original_seed}-{method_spec['name']}"
                )
                checkpoint = write_checkpoint(
                    output_root, experiment_id, name, outcome.model.state_dict()
                )
                arrays = _prediction_arrays(
                    outcome.model, data, deletion.retain_positions, deletion.forget_positions
                )
                arrays["partition_sha256"] = np.array(lineage["partition_sha256"])
                predictions = write_prediction_artifact(
                    output_root, experiment_id, name, arrays
                )
                test = predict(outcome.model, data.x_test)
                retain = predict(
                    outcome.model, data.x_train[deletion.retain_positions]
                )
                forget = predict(
                    outcome.model, data.x_train[deletion.forget_positions]
                )
                bins = 15
                method_runs.append(
                    {
                        "method": method_spec["name"],
                        "role": method_spec["role"],
                        "original_seed": int(original_seed),
                        "seed": unlearning_seed,
                        "partition_sha256": lineage["partition_sha256"],
                        "duration_seconds": outcome.duration_seconds,
                        "simple_engineering_baseline": descriptor.simple_baseline,
                        "literature_reference": descriptor.literature_reference,
                        "implementation_provenance": descriptor.implementation_provenance,
                        "implementation_notes": descriptor.implementation_notes,
                        "checkpoint": checkpoint,
                        "predictions": predictions,
                        "metrics": {
                            "test": predictive_metrics(test, data.y_test.numpy(), bins),
                            "retain": predictive_metrics(
                                retain,
                                data.y_train[deletion.retain_positions].numpy(),
                                bins,
                            ),
                            "forget": predictive_metrics(
                                forget,
                                data.y_train[deletion.forget_positions].numpy(),
                                bins,
                            ),
                            "loss_membership_auc": loss_membership_auc(
                                forget.probabilities,
                                data.y_train[deletion.forget_positions].numpy(),
                                test.probabilities,
                                data.y_test.numpy(),
                            ),
                        },
                    }
                )
                del model, outcome
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        conditions.append(
            {
                "condition_id": condition_id,
                "fraction": source_condition["fraction"],
                "manifest": source_condition["manifest"],
                "forget_count": source_condition["forget_count"],
                "partition_lineage": lineage,
                "metrics_partition_sha256": lineage["partition_sha256"],
                "exact_runs": exact_runs,
                "metrics": source_condition["metrics"],
                "method_runs": method_runs,
            }
        )

    result = {
        "schema_version": 3,
        "experiment_id": experiment_id,
        "maturity": "PILOT",
        "status": "completed",
        "interpretation": (
            "Exploratory continuous distinguishability pilot; equivalence and confirmatory "
            "pass/fail claims are prohibited."
        ),
        "started_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config": plan,
        "config_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_experiment": source["experiment_id"],
        "source_result_sha256": sha256(source_path),
        "dataset": source["dataset"],
        "original_runs": [
            originals_by_seed[int(seed)]
            for seed in plan["source_ensemble"]["original_seeds"]
        ],
        "conditions": conditions,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "rocm": torch.version.hip,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "hsa_dxg": os.environ.get("HSA_ENABLE_DXG_DETECTION"),
            "aggregate_unlearning_seconds": float(
                sum(
                    run["duration_seconds"]
                    for condition in conditions
                    for run in condition["method_runs"]
                )
            ),
            "mean_unlearning_seconds": float(
                mean(
                    run["duration_seconds"]
                    for condition in conditions
                    for run in condition["method_runs"]
                )
            ),
        },
    }
    result["quality_gate"] = evaluate_result(result).as_dict()
    if not result["quality_gate"]["passed"]:
        raise RuntimeError(f"completed pilot failed artifact gate: {result['quality_gate']}")
    print(write_raw_result(output_root, experiment_id, result))


def main(plan_path: Path = PLAN) -> None:
    """Run once and append a failure record if any post-reservation operation fails."""
    try:
        _main(plan_path)
    except Exception as error:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        output_root = str(plan["execution"]["output_root"])
        experiment_id = str(plan["experiment_id"])
        failure_id = dt.datetime.now(dt.UTC).strftime("failure-%Y%m%dT%H%M%S%fZ")
        write_failure_record(
            output_root,
            experiment_id,
            {
                "failure_id": failure_id,
                "experiment_id": experiment_id,
                "failure_category": type(error).__name__,
                "exception": str(error),
                "plan": str(plan_path),
                "plan_sha256": sha256(plan_path),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "hip": torch.version.hip,
                    "hsa_dxg": os.environ.get("HSA_ENABLE_DXG_DETECTION"),
                },
                "invalidated_batch": True,
                "remediation": (
                    "Preserve the partial append-only artifact directory. Revise the protocol "
                    "under a new experiment ID and revalidate before retrying."
                ),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the locked distinguishability method pilot")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_only:
        print(json.dumps(validate_execution_plan(arguments.plan), indent=2, sort_keys=True))
    else:
        main(arguments.plan)
