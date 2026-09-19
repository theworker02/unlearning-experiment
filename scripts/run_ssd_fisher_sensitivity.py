from __future__ import annotations

import datetime as dt
import hashlib
import platform
from pathlib import Path

import torch
import yaml
from torch import nn

from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import partition_lineage, select_deletion
from exact_unlearning.devices import resolve_device
from exact_unlearning.experiment import _git_commit
from exact_unlearning.metrics import predict
from exact_unlearning.models import build_model
from exact_unlearning.registry import (
    reserve_experiment,
    write_checkpoint,
    write_failure_record,
    write_prediction_artifact,
    write_raw_result,
)
from exact_unlearning.seeding import seed_everything
from exact_unlearning.ssd import (
    apply_selective_synaptic_dampening,
    estimate_diagonal_fisher,
    estimate_minibatch_gradient_square_fisher,
)
from exact_unlearning.training import train_model

if __package__:
    from scripts.ssd_fisher_sensitivity_rules import (
        conditions,
        fisher_distances,
        prediction_sensitivity,
        selected_mask,
        selected_set_metrics,
        state_l2,
    )
    from scripts.validate_ssd_fisher_sensitivity_protocol import PROTOCOL, validate
else:
    from ssd_fisher_sensitivity_rules import (
        conditions,
        fisher_distances,
        prediction_sensitivity,
        selected_mask,
        selected_set_metrics,
        state_l2,
    )
    from validate_ssd_fisher_sensitivity_protocol import PROTOCOL, validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conditions(protocol: dict[str, object]) -> list[dict[str, object]]:
    return conditions(protocol)


def _analytic_control() -> dict[str, object]:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 0])
    per_example = estimate_diagonal_fisher(model, x, y, batch_size=2)["weight"]
    minibatch_one = estimate_minibatch_gradient_square_fisher(
        model, x, y, batch_size=1
    )["weight"]
    minibatch_two = estimate_minibatch_gradient_square_fisher(
        model, x, y, batch_size=2
    )["weight"]
    expected_per_example = torch.full((2, 2), 0.125)
    expected_minibatch_two = torch.full((2, 2), 0.0625)
    passed = (
        torch.equal(per_example, expected_per_example)
        and torch.equal(minibatch_one, expected_per_example)
        and torch.equal(minibatch_two, expected_minibatch_two)
    )
    return {
        "passed": passed,
        "per_example": per_example.tolist(),
        "minibatch_size_1": minibatch_one.tolist(),
        "minibatch_size_2": minibatch_two.tolist(),
        "expected_per_example": expected_per_example.tolist(),
        "expected_minibatch_size_2": expected_minibatch_two.tolist(),
    }


def validate_execution_plan(path: Path = PROTOCOL) -> dict[str, object]:
    return {**validate(path), "protocol_sha256": _sha256(path)}


def main(path: Path = PROTOCOL) -> None:
    plan = validate_execution_plan(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["status"] != "LOCKED" or protocol["execution_authorized"] is not True:
        raise SystemExit("execution blocked: SSD Fisher protocol must be LOCKED and authorized")
    output_root = protocol["artifact_policy"]["output_root"]
    experiment_id = protocol["experiment_id"]
    reserve_experiment(output_root, experiment_id)
    started = dt.datetime.now(dt.UTC).isoformat()
    try:
        analytic = _analytic_control()
        if not analytic["passed"]:
            raise RuntimeError("SSD Fisher synthetic analytic control failed")
        data = load_dataset(protocol["scope"]["dataset_config"])
        training = protocol["original_training"]
        device = resolve_device(str(training["device"]))
        seed_everything(0, bool(training["deterministic"]))
        selector_model = build_model(
            protocol["scope"]["architecture"],
            tuple(data.x_train.shape[1:]),
            int(protocol["scope"]["hidden_dim"]),
            int(data.y_train.max()) + 1,
        )
        deletion = select_deletion(
            "class",
            1.0,
            0,
            data.train_ids,
            data.x_train,
            data.y_train,
            selector_model,
            target_class=int(protocol["scope"]["deletion"]["target_class"]),
        )
        lineage = partition_lineage(deletion, data.train_ids)
        x_forget = data.x_train[deletion.forget_positions]
        y_forget = data.y_train[deletion.forget_positions]
        alpha = float(protocol["scope"]["ssd_hyperparameters"]["alpha"])
        dampening = float(protocol["scope"]["ssd_hyperparameters"]["dampening_constant"])
        runs = []
        for raw_seed in protocol["scope"]["original_seeds"]:
            seed = int(raw_seed)
            seed_everything(seed, bool(training["deterministic"]))
            model = build_model(
                protocol["scope"]["architecture"],
                tuple(data.x_train.shape[1:]),
                int(protocol["scope"]["hidden_dim"]),
                int(data.y_train.max()) + 1,
            )
            trained = train_model(
                model,
                data.x_train,
                data.y_train,
                seed=seed,
                epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                deterministic=bool(training["deterministic"]),
                device=device,
                optimizer_name=str(training["optimizer"]),
            )
            original_reference = write_checkpoint(
                output_root, experiment_id, f"original-{seed}", trained.model.state_dict()
            )
            condition_objects = []
            for condition in _conditions(protocol):
                estimator = (
                    estimate_diagonal_fisher
                    if condition["estimator"] == "paper_aligned_per_example"
                    else estimate_minibatch_gradient_square_fisher
                )
                batch_size = int(condition["batch_size"])
                full_fisher = estimator(
                    trained.model,
                    data.x_train,
                    data.y_train,
                    batch_size=batch_size,
                    device=device,
                )
                forget_fisher = estimator(
                    trained.model,
                    x_forget,
                    y_forget,
                    batch_size=batch_size,
                    device=device,
                )
                updated, diagnostics = apply_selective_synaptic_dampening(
                    trained.model,
                    full_fisher,
                    forget_fisher,
                    alpha=alpha,
                    dampening_constant=dampening,
                )
                condition_id = str(condition["condition_id"])
                full_reference = write_checkpoint(
                    output_root, experiment_id, f"{condition_id}-full-{seed}", full_fisher
                )
                forget_reference = write_checkpoint(
                    output_root, experiment_id, f"{condition_id}-forget-{seed}", forget_fisher
                )
                checkpoint = write_checkpoint(
                    output_root,
                    experiment_id,
                    f"{condition_id}-updated-{seed}",
                    updated.state_dict(),
                )
                prediction = predict(updated, data.x_validation)
                prediction_reference = write_prediction_artifact(
                    output_root,
                    experiment_id,
                    f"{condition_id}-validation-{seed}",
                    {
                        "probabilities": prediction.probabilities,
                        "labels": data.y_validation.numpy(),
                    },
                )
                condition_objects.append(
                    {
                        **condition,
                        "full_fisher": full_reference,
                        "forget_fisher": forget_reference,
                        "updated_checkpoint": checkpoint,
                        "validation_predictions": prediction_reference,
                        "diagnostics": diagnostics,
                        "_full": full_fisher,
                        "_forget": forget_fisher,
                        "_state": updated.state_dict(),
                        "_probabilities": prediction.probabilities,
                    }
                )
            reference = condition_objects[0]
            reference_mask = selected_mask(
                reference["_full"], reference["_forget"], alpha=alpha
            )
            for condition in condition_objects:
                mask = selected_mask(condition["_full"], condition["_forget"], alpha=alpha)
                condition["comparison_to_per_example_batch_1"] = {
                    "full_fisher": fisher_distances(condition["_full"], reference["_full"]),
                    "forget_fisher": fisher_distances(
                        condition["_forget"], reference["_forget"]
                    ),
                    "selection": selected_set_metrics(mask, reference_mask),
                    "model_update": {
                        "parameter_change_l2": state_l2(
                            condition["_state"], trained.model.state_dict()
                        ),
                        "pairwise_updated_parameter_l2": state_l2(
                            condition["_state"], reference["_state"]
                        ),
                    },
                    "validation_behavior": prediction_sensitivity(
                        condition["_probabilities"],
                        reference["_probabilities"],
                        data.y_validation.numpy(),
                    ),
                }
                for private_key in ("_full", "_forget", "_state", "_probabilities"):
                    del condition[private_key]
            runs.append(
                {
                    "original_seed": seed,
                    "original_checkpoint": original_reference,
                    "conditions": condition_objects,
                }
            )
        write_raw_result(
            output_root,
            experiment_id,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "publication_evidence": False,
                "protocol": {"path": str(path), "sha256": plan["protocol_sha256"]},
                "git_commit": _git_commit(),
                "started_at": started,
                "finished_at": dt.datetime.now(dt.UTC).isoformat(),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "device": str(device),
                },
                "dataset": {
                    "name": data.metadata.name,
                    "version": data.metadata.version,
                    "sha256": data.dataset_sha256,
                },
                "partition_lineage": lineage,
                "analytic_control": analytic,
                "runs": runs,
                "overall_complete": True,
            },
        )
    except Exception as error:
        failure_id = dt.datetime.now(dt.UTC).strftime("failure-%Y%m%dT%H%M%S%fZ")
        write_failure_record(
            output_root,
            experiment_id,
            {
                "failure_id": failure_id,
                "experiment_id": experiment_id,
                "failure_category": type(error).__name__,
                "exception": str(error),
                "protocol_sha256": plan["protocol_sha256"],
                "invalidated_batch": True,
                "remediation": "review failure and create a new protocol version",
            },
        )
        raise


if __name__ == "__main__":
    main()
