from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import platform
from pathlib import Path

import torch
import yaml

from exact_unlearning.contracts import UnlearningContext
from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import partition_lineage, select_deletion
from exact_unlearning.devices import resolve_device
from exact_unlearning.experiment import _git_commit
from exact_unlearning.methods import run_unlearning_method
from exact_unlearning.metrics import predict, predictive_metrics
from exact_unlearning.models import build_model
from exact_unlearning.registry import (
    reserve_experiment,
    write_checkpoint,
    write_failure_record,
    write_raw_result,
)
from exact_unlearning.seeding import seed_everything
from exact_unlearning.training import per_example_loss, train_model

if __package__:
    from scripts.neggrad_plus_rules import select_candidate
    from scripts.validate_neggrad_plus_protocol import PROTOCOL, validate
else:
    from neggrad_plus_rules import select_candidate
    from validate_neggrad_plus_protocol import PROTOCOL, validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidates(protocol: dict[str, object]) -> list[dict[str, object]]:
    grid = protocol["stage_2_validation_only_tuning"]["grid"]
    keys = (
        "beta",
        "epochs",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "momentum",
        "batch_size",
    )
    return [
        {"candidate_id": f"candidate-{index:02d}", **dict(zip(keys, values, strict=True))}
        for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), start=1)
    ]


def validate_execution_plan(path: Path = PROTOCOL) -> dict[str, object]:
    summary = validate(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        **summary,
        "protocol_sha256": _sha256(path),
        "experiment_family": protocol["experiment_family"],
    }


def main(path: Path = PROTOCOL) -> None:
    plan = validate_execution_plan(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["status"] != "LOCKED" or protocol["execution_authorized"] is not True:
        raise SystemExit("execution blocked: NegGrad+ protocol must be LOCKED and authorized")
    output_root = protocol["artifact_policy"]["output_root"]
    stage1_id = protocol["artifact_policy"]["stage_1_experiment_id"]
    stage2_id = protocol["artifact_policy"]["stage_2_experiment_id"]
    reserve_experiment(output_root, stage1_id)
    reserve_experiment(output_root, stage2_id)
    started = dt.datetime.now(dt.UTC).isoformat()
    stage1_completed = False
    try:
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
            "class", 1.0, 0, data.train_ids, data.x_train, data.y_train, selector_model,
            target_class=int(protocol["scope"]["deletion"]["target_class"]),
        )
        lineage = partition_lineage(deletion, data.train_ids)
        originals = {}
        original_validation_accuracy = {}
        original_references = {}
        for seed in protocol["scope"]["original_seeds"]:
            seed_everything(int(seed), bool(training["deterministic"]))
            model = build_model(
                protocol["scope"]["architecture"], tuple(data.x_train.shape[1:]),
                int(protocol["scope"]["hidden_dim"]), int(data.y_train.max()) + 1,
            )
            trained = train_model(
                model, data.x_train, data.y_train, seed=int(seed),
                epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                deterministic=bool(training["deterministic"]), device=device,
                optimizer_name=str(training["optimizer"]),
            )
            originals[int(seed)] = trained.model
            prediction = predict(trained.model, data.x_validation)
            original_validation_accuracy[int(seed)] = predictive_metrics(
                prediction, data.y_validation.numpy(), 15
            )["accuracy"]
            original_references[int(seed)] = write_checkpoint(
                output_root, stage1_id, f"original-{seed}", trained.model.state_dict()
            )
        fixed = protocol["stage_1_fixed_sanity"]["hyperparameters"]
        stage1_runs = []
        for seed, original in originals.items():
            _, outcome = run_unlearning_method(
                "neggrad_plus",
                UnlearningContext(
                    original, data.x_train[deletion.retain_positions],
                    data.y_train[deletion.retain_positions],
                    data.x_train[deletion.forget_positions],
                    data.y_train[deletion.forget_positions], seed + 10_000,
                    {**fixed, "batch_size": fixed["batch_size"]}, device,
                ),
            )
            diagnostics = outcome.diagnostics or {}
            accepted = (
                diagnostics.get("gradient_l2_mean", 0.0) > 0
                and diagnostics["after"]["forget_cross_entropy"]
                > diagnostics["before"]["forget_cross_entropy"]
            )
            stage1_runs.append({"seed": seed, "accepted": accepted, "diagnostics": diagnostics})
        write_raw_result(
            output_root, stage1_id,
            {"schema_version": 1, "experiment_id": stage1_id, "publication_evidence": False,
             "protocol": {"path": str(path), "sha256": plan["protocol_sha256"]},
             "partition_lineage": lineage, "runs": stage1_runs,
             "overall_acceptance": all(run["accepted"] for run in stage1_runs)},
        )
        stage1_completed = True
        if not all(run["accepted"] for run in stage1_runs):
            raise RuntimeError("NegGrad+ fixed-setting sanity stage failed")
        target_mask = data.y_validation == int(protocol["scope"]["deletion"]["target_class"])
        candidate_runs = []
        models = {}
        for candidate_index, candidate in enumerate(_candidates(protocol), start=1):
            for seed, original in originals.items():
                _, outcome = run_unlearning_method(
                    "neggrad_plus",
                    UnlearningContext(
                        original, data.x_train[deletion.retain_positions],
                        data.y_train[deletion.retain_positions],
                        data.x_train[deletion.forget_positions],
                        data.y_train[deletion.forget_positions], seed + 10_000 + candidate_index,
                        candidate, device,
                    ),
                )
                validation = predict(outcome.model, data.x_validation)
                accuracy = predictive_metrics(validation, data.y_validation.numpy(), 15)["accuracy"]
                target_loss = float(
                    per_example_loss(
                        outcome.model,
                        data.x_validation[target_mask],
                        data.y_validation[target_mask],
                    ).mean()
                )
                checkpoint = write_checkpoint(
                    output_root, stage2_id, f"{candidate['candidate_id']}-seed-{seed}",
                    outcome.model.state_dict(),
                )
                run = {
                    "candidate_id": candidate["candidate_id"], "original_seed": seed,
                    "hyperparameters": candidate, "validation_accuracy": accuracy,
                    "validation_accuracy_drop": original_validation_accuracy[seed] - accuracy,
                    "target_class_validation_cross_entropy": target_loss,
                    "diagnostics": outcome.diagnostics, "checkpoint": checkpoint,
                }
                candidate_runs.append(run)
                models[(candidate["candidate_id"], seed)] = outcome.model
        selection = select_candidate(candidate_runs)
        selected_id = selection["selected"]["candidate_id"]
        final_test = []
        for seed in protocol["scope"]["original_seeds"]:
            prediction = predict(models[(selected_id, int(seed))], data.x_test)
            final_test.append(
                {"original_seed": int(seed), "metrics": predictive_metrics(
                    prediction, data.y_test.numpy(), 15
                )}
            )
        write_raw_result(
            output_root, stage2_id,
            {"schema_version": 1, "experiment_id": stage2_id, "publication_evidence": False,
             "protocol": {"path": str(path), "sha256": plan["protocol_sha256"]},
             "git_commit": _git_commit(), "started_at": started,
             "finished_at": dt.datetime.now(dt.UTC).isoformat(),
             "environment": {"python": platform.python_version(), "torch": torch.__version__,
                             "device": str(device)},
             "dataset": {"name": data.metadata.name, "version": data.metadata.version,
                         "sha256": data.dataset_sha256},
             "partition_lineage": lineage, "original_checkpoints": original_references,
             "candidate_runs": candidate_runs, "selection": selection, "final_test": final_test},
        )
    except Exception as error:
        failure_id = dt.datetime.now(dt.UTC).strftime("failure-%Y%m%dT%H%M%S%fZ")
        affected_experiment_ids = (stage2_id,) if stage1_completed else (stage1_id, stage2_id)
        for experiment_id in affected_experiment_ids:
            write_failure_record(
                output_root, experiment_id,
                {"failure_id": failure_id, "experiment_id": experiment_id,
                 "failure_category": type(error).__name__, "exception": str(error),
                 "protocol_sha256": plan["protocol_sha256"],
                 "invalidated_batch": not stage1_completed,
                 "completed_sanity_stage_preserved": stage1_completed,
                 "remediation": "review failure and create a new protocol version"},
            )
        raise


if __name__ == "__main__":
    main()
