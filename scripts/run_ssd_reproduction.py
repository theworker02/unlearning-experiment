from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import math
import platform
from pathlib import Path

import torch
import yaml

from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import partition_lineage, select_deletion
from exact_unlearning.devices import resolve_device
from exact_unlearning.experiment import _git_commit
from exact_unlearning.metrics import predict, predictive_metrics
from exact_unlearning.models import build_model
from exact_unlearning.registry import (
    reserve_experiment,
    write_checkpoint,
    write_failure_record,
    write_raw_result,
)
from exact_unlearning.seeding import seed_everything
from exact_unlearning.ssd import (
    estimate_diagonal_fisher,
    selective_synaptic_dampening,
)
from exact_unlearning.training import per_example_loss, train_model

if __package__:
    from scripts.ssd_rules import select_candidate
    from scripts.validate_ssd_protocol import PROTOCOL, validate
else:
    from ssd_rules import select_candidate
    from validate_ssd_protocol import PROTOCOL, validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidates(protocol: dict[str, object]) -> list[dict[str, object]]:
    grid = protocol["stage_2_validation_only_tuning"]["grid"]
    return [
        {
            "candidate_id": f"candidate-{index:02d}",
            "hyperparameters": {
                "alpha": float(alpha),
                "dampening_constant": float(dampening),
            },
        }
        for index, (alpha, dampening) in enumerate(
            itertools.product(grid["alpha"], grid["dampening_constant"]), start=1
        )
    ]


def validate_execution_plan(path: Path = PROTOCOL) -> dict[str, object]:
    summary = validate(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        **summary,
        "protocol_sha256": _sha256(path),
        "experiment_family": protocol["experiment_family"],
    }


def _parameter_change_l2(original: torch.nn.Module, updated: torch.nn.Module) -> float:
    squared = sum(
        float((after.detach() - before.detach()).square().sum())
        for before, after in zip(original.parameters(), updated.parameters(), strict=True)
    )
    return math.sqrt(squared)


def _fisher_finite_nonnegative(fisher: dict[str, torch.Tensor]) -> bool:
    return all(
        torch.all(torch.isfinite(value)) and torch.all(value >= 0)
        for value in fisher.values()
    )


def main(path: Path = PROTOCOL) -> None:
    plan = validate_execution_plan(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["status"] != "LOCKED" or protocol["execution_authorized"] is not True:
        raise SystemExit("execution blocked: SSD protocol must be LOCKED and authorized")
    output_root = protocol["artifact_policy"]["output_root"]
    stage1_id = protocol["artifact_policy"]["stage_1_experiment_id"]
    stage2_id = protocol["artifact_policy"]["stage_2_experiment_id"]
    reserve_experiment(output_root, stage1_id)
    reserve_experiment(output_root, stage2_id)
    started = dt.datetime.now(dt.UTC).isoformat()
    try:
        data = load_dataset(protocol["scope"]["dataset_config"])
        training = protocol["original_training"]
        fisher_config = protocol["fisher_estimation"]
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
        originals: dict[int, torch.nn.Module] = {}
        full_fishers: dict[int, dict[str, torch.Tensor]] = {}
        original_accuracy: dict[int, float] = {}
        original_references: dict[int, dict[str, str]] = {}
        fisher_references: dict[int, dict[str, str]] = {}
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
            originals[seed] = trained.model
            original_prediction = predict(trained.model, data.x_validation)
            original_accuracy[seed] = predictive_metrics(
                original_prediction, data.y_validation.numpy(), 15
            )["accuracy"]
            original_references[seed] = write_checkpoint(
                output_root, stage1_id, f"original-{seed}", trained.model.state_dict()
            )
            full_fisher = estimate_diagonal_fisher(
                trained.model,
                data.x_train,
                data.y_train,
                batch_size=int(fisher_config["batch_size"]),
                device=device,
            )
            full_fishers[seed] = full_fisher
            fisher_references[seed] = write_checkpoint(
                output_root, stage2_id, f"full-fisher-{seed}", full_fisher
            )

        fixed = protocol["stage_1_fixed_sanity"]["hyperparameters"]
        stage1_runs = []
        for seed, original in originals.items():
            outcome = selective_synaptic_dampening(
                original,
                x_forget,
                y_forget,
                full_fisher=full_fishers[seed],
                alpha=float(fixed["alpha"]),
                dampening_constant=float(fixed["dampening_constant"]),
                batch_size=int(fisher_config["batch_size"]),
                device=device,
            )
            diagnostics = outcome.diagnostics or {}
            fraction = float(diagnostics.get("selected_parameter_fraction", float("nan")))
            changed_l2 = _parameter_change_l2(original, outcome.model)
            accepted = (
                _fisher_finite_nonnegative(full_fishers[seed])
                and int(diagnostics.get("selected_parameter_count", 0)) > 0
                and math.isfinite(fraction)
                and 0 < fraction < 1
                and math.isfinite(changed_l2)
                and changed_l2 > 0
            )
            stage1_runs.append(
                {
                    "original_seed": seed,
                    "accepted": accepted,
                    "fisher_finite_nonnegative": _fisher_finite_nonnegative(
                        full_fishers[seed]
                    ),
                    "changed_parameter_l2": changed_l2,
                    "diagnostics": diagnostics,
                }
            )
        write_raw_result(
            output_root,
            stage1_id,
            {
                "schema_version": 1,
                "experiment_id": stage1_id,
                "publication_evidence": False,
                "protocol": {"path": str(path), "sha256": plan["protocol_sha256"]},
                "partition_lineage": lineage,
                "runs": stage1_runs,
                "overall_acceptance": all(run["accepted"] for run in stage1_runs),
            },
        )
        if not all(run["accepted"] for run in stage1_runs):
            raise RuntimeError("SSD fixed-setting sanity stage failed")

        candidate_runs = []
        models: dict[tuple[str, int], torch.nn.Module] = {}
        for candidate in _candidates(protocol):
            candidate_id = str(candidate["candidate_id"])
            hyperparameters = candidate["hyperparameters"]
            for seed, original in originals.items():
                outcome = selective_synaptic_dampening(
                    original,
                    x_forget,
                    y_forget,
                    full_fisher=full_fishers[seed],
                    alpha=float(hyperparameters["alpha"]),
                    dampening_constant=float(hyperparameters["dampening_constant"]),
                    batch_size=int(fisher_config["batch_size"]),
                    device=device,
                )
                validation = predict(outcome.model, data.x_validation)
                accuracy = predictive_metrics(validation, data.y_validation.numpy(), 15)[
                    "accuracy"
                ]
                forget_loss = float(
                    per_example_loss(outcome.model, x_forget, y_forget).mean()
                )
                checkpoint = write_checkpoint(
                    output_root,
                    stage2_id,
                    f"{candidate_id}-seed-{seed}",
                    outcome.model.state_dict(),
                )
                diagnostics = outcome.diagnostics or {}
                candidate_runs.append(
                    {
                        "candidate_id": candidate_id,
                        "original_seed": seed,
                        "hyperparameters": hyperparameters,
                        "validation_accuracy": accuracy,
                        "validation_accuracy_drop": original_accuracy[seed] - accuracy,
                        "forget_training_cross_entropy": forget_loss,
                        "selected_parameter_fraction": diagnostics[
                            "selected_parameter_fraction"
                        ],
                        "diagnostics": diagnostics,
                        "checkpoint": checkpoint,
                    }
                )
                models[(candidate_id, seed)] = outcome.model
        selection = select_candidate(candidate_runs)
        selected_id = selection["selected"]["candidate_id"]
        final_test = []
        for raw_seed in protocol["scope"]["original_seeds"]:
            seed = int(raw_seed)
            test_prediction = predict(models[(selected_id, seed)], data.x_test)
            final_test.append(
                {
                    "original_seed": seed,
                    "metrics": predictive_metrics(test_prediction, data.y_test.numpy(), 15),
                }
            )
        write_raw_result(
            output_root,
            stage2_id,
            {
                "schema_version": 1,
                "experiment_id": stage2_id,
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
                "original_checkpoints": original_references,
                "full_fisher_artifacts": fisher_references,
                "candidate_runs": candidate_runs,
                "selection": selection,
                "final_test": final_test,
            },
        )
    except Exception as error:
        failure_id = dt.datetime.now(dt.UTC).strftime("failure-%Y%m%dT%H%M%S%fZ")
        for experiment_id in (stage1_id, stage2_id):
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
