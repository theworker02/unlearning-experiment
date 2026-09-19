from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import platform
from pathlib import Path

import torch
import yaml

from exact_unlearning.contracts import UnlearningContext
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
from exact_unlearning.scrub import scrub
from exact_unlearning.seeding import seed_everything
from exact_unlearning.training import train_model

if __package__:
    from scripts.scrub_sanity_rules import evaluate_mechanism
    from scripts.validate_scrub_sanity_protocol import validate
else:
    from scrub_sanity_rules import evaluate_mechanism
    from validate_scrub_sanity_protocol import validate

PROTOCOL = Path("study/preregistration/scrub_sanity_reproduction_v1.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_execution_plan(path: Path = PROTOCOL) -> dict[str, object]:
    validate(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "experiment_id": protocol["experiment_id"],
        "status": protocol["status"],
        "execution_authorized": protocol["execution_authorized"],
        "planned_original_trainings": len(protocol["scope"]["original_seeds"]),
        "planned_unlearning_runs": len(protocol["scope"]["original_seeds"]),
        "protocol_sha256": _sha256(path),
    }


def main(path: Path = PROTOCOL) -> None:
    summary = validate_execution_plan(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["status"] != "LOCKED" or protocol["execution_authorized"] is not True:
        raise SystemExit("execution blocked: SCRUB sanity protocol must be LOCKED and authorized")

    output_root = protocol["artifacts"]["immutable_output_root"]
    experiment_id = protocol["experiment_id"]
    reserve_experiment(output_root, experiment_id)
    started = dt.datetime.now(dt.UTC).isoformat()
    try:
        data = load_dataset(protocol["scope"]["dataset_config"])
        training = protocol["original_training"]
        device = resolve_device(str(training["device"]))
        deletion = select_deletion(
            "class",
            1.0,
            0,
            data.train_ids,
            data.x_train,
            data.y_train,
            build_model(
                protocol["scope"]["architecture"],
                tuple(data.x_train.shape[1:]),
                int(protocol["scope"]["hidden_dim"]),
                int(data.y_train.max()) + 1,
            ),
            target_class=int(protocol["scope"]["deletion"]["target_class"]),
        )
        lineage = partition_lineage(deletion, data.train_ids)
        runs = []
        for seed in protocol["scope"]["original_seeds"]:
            seed_everything(int(seed), bool(training["deterministic"]))
            original = build_model(
                protocol["scope"]["architecture"],
                tuple(data.x_train.shape[1:]),
                int(protocol["scope"]["hidden_dim"]),
                int(data.y_train.max()) + 1,
            )
            trained = train_model(
                original,
                data.x_train,
                data.y_train,
                seed=int(seed),
                epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                deterministic=bool(training["deterministic"]),
                device=device,
                optimizer_name=str(training["optimizer"]),
            )
            unlearning_seed = int(seed) + 10_000
            outcome = scrub(
                UnlearningContext(
                    original=trained.model,
                    x_retain=data.x_train[deletion.retain_positions],
                    y_retain=data.y_train[deletion.retain_positions],
                    x_forget=data.x_train[deletion.forget_positions],
                    y_forget=data.y_train[deletion.forget_positions],
                    seed=unlearning_seed,
                    config={
                        **protocol["fixed_hyperparameters"],
                        "batch_size": int(training["batch_size"]),
                    },
                    device=device,
                )
            )
            trace = outcome.diagnostics["mechanism_trace"] if outcome.diagnostics else []
            acceptance = evaluate_mechanism(trace)
            test_prediction = predict(outcome.model, data.x_test)
            runs.append(
                {
                    "original_seed": int(seed),
                    "unlearning_seed": unlearning_seed,
                    "training_duration_seconds": trained.duration_seconds,
                    "unlearning_duration_seconds": outcome.duration_seconds,
                    "mechanism_trace": trace,
                    "acceptance": acceptance,
                    "accepted": all(acceptance.values()),
                    "test_metrics": predictive_metrics(
                        test_prediction, data.y_test.numpy(), 15
                    ),
                    "original_checkpoint": write_checkpoint(
                        output_root, experiment_id, f"original-{seed}", trained.model.state_dict()
                    ),
                    "unlearned_checkpoint": write_checkpoint(
                        output_root, experiment_id, f"scrub-{seed}", outcome.model.state_dict()
                    ),
                }
            )
        write_raw_result(
            output_root,
            experiment_id,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "study_class": protocol["study_class"],
                "publication_evidence": False,
                "protocol": {"path": str(path), "sha256": summary["protocol_sha256"]},
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
                "runs": runs,
                "overall_acceptance": all(run["accepted"] for run in runs),
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
                "protocol_sha256": summary["protocol_sha256"],
                "environment": {"python": platform.python_version(), "torch": torch.__version__},
                "invalidated_batch": True,
                "remediation": "review failure and create a new protocol version",
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a locked SCRUB mechanism-sanity protocol")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate_only:
        print(validate_execution_plan(arguments.protocol))
    else:
        main(arguments.protocol)
