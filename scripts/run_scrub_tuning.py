from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
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
from exact_unlearning.training import per_example_loss

if __package__:
    from scripts.scrub_sanity_rules import evaluate_mechanism
    from scripts.scrub_tuning_rules import select_candidate
    from scripts.validate_experiment_profiles import validate_scrub_sanity_artifact
    from scripts.validate_scrub_tuning_protocol import PROTOCOL, validate
else:
    from scrub_sanity_rules import evaluate_mechanism
    from scrub_tuning_rules import select_candidate
    from validate_experiment_profiles import validate_scrub_sanity_artifact
    from validate_scrub_tuning_protocol import PROTOCOL, validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidates(protocol: dict[str, object]) -> list[dict[str, object]]:
    grid = protocol["validation_only_tuning"]["grid"]
    keys = tuple(grid)
    return [
        {
            "candidate_id": f"candidate-{index:02d}",
            "hyperparameters": dict(zip(keys, values, strict=True)),
        }
        for index, values in enumerate(itertools.product(*(grid[key] for key in keys)), start=1)
    ]


def validate_execution_plan(path: Path = PROTOCOL) -> dict[str, object]:
    return {**validate(path), "protocol_sha256": _sha256(path)}


def main(path: Path = PROTOCOL) -> None:
    plan = validate_execution_plan(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol["status"] != "LOCKED" or protocol["execution_authorized"] is not True:
        raise SystemExit("execution blocked: SCRUB tuning protocol must be LOCKED and authorized")
    prerequisite_path = Path(protocol["prerequisite"]["required_artifact"])
    prerequisite_protocol = Path(protocol["prerequisite"]["protocol"])
    prerequisite_validation = validate_scrub_sanity_artifact(
        prerequisite_path, prerequisite_protocol
    )
    if not prerequisite_validation["eligible"]:
        raise SystemExit("execution blocked: passing SCRUB sanity artifact is required")
    prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    output_root = protocol["artifact_policy"]["output_root"]
    experiment_id = protocol["experiment_id"]
    reserve_experiment(output_root, experiment_id)
    started = dt.datetime.now(dt.UTC).isoformat()
    try:
        data = load_dataset(protocol["scope"]["dataset_config"])
        device = resolve_device("auto")
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
        x_retain = data.x_train[deletion.retain_positions]
        y_retain = data.y_train[deletion.retain_positions]
        x_forget = data.x_train[deletion.forget_positions]
        y_forget = data.y_train[deletion.forget_positions]
        originals = {}
        original_accuracy = {}
        original_references = {}
        prerequisite_runs = {int(run["original_seed"]): run for run in prerequisite["runs"]}
        for raw_seed in protocol["scope"]["original_seeds"]:
            seed = int(raw_seed)
            reference = prerequisite_runs[seed]["original_checkpoint"]
            model = build_model(
                protocol["scope"]["architecture"],
                tuple(data.x_train.shape[1:]),
                int(protocol["scope"]["hidden_dim"]),
                int(data.y_train.max()) + 1,
            )
            state = torch.load(
                Path(reference["path"]), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
            model.to(device)
            originals[seed] = model
            original_references[seed] = reference
            original_prediction = predict(model, data.x_validation)
            original_accuracy[seed] = predictive_metrics(
                original_prediction, data.y_validation.numpy(), 15
            )["accuracy"]
        candidate_runs = []
        models = {}
        for candidate_index, candidate in enumerate(_candidates(protocol), start=1):
            candidate_id = str(candidate["candidate_id"])
            hyperparameters = candidate["hyperparameters"]
            for seed, original in originals.items():
                outcome = scrub(
                    UnlearningContext(
                        original,
                        x_retain,
                        y_retain,
                        x_forget,
                        y_forget,
                        seed + 20_000 + candidate_index,
                        hyperparameters,
                        device,
                    )
                )
                trace = outcome.diagnostics["mechanism_trace"] if outcome.diagnostics else []
                mechanism_acceptance = evaluate_mechanism(trace)
                validation = predict(outcome.model, data.x_validation)
                accuracy = predictive_metrics(validation, data.y_validation.numpy(), 15)[
                    "accuracy"
                ]
                forget_loss = float(
                    per_example_loss(outcome.model, x_forget, y_forget).mean()
                )
                checkpoint = write_checkpoint(
                    output_root,
                    experiment_id,
                    f"{candidate_id}-seed-{seed}",
                    outcome.model.state_dict(),
                )
                candidate_runs.append(
                    {
                        "candidate_id": candidate_id,
                        "original_seed": seed,
                        "hyperparameters": hyperparameters,
                        "mechanism_trace": trace,
                        "mechanism_acceptance": mechanism_acceptance,
                        "mechanism_valid": all(mechanism_acceptance.values()),
                        "validation_accuracy": accuracy,
                        "validation_accuracy_drop": original_accuracy[seed] - accuracy,
                        "forget_training_cross_entropy": forget_loss,
                        "runtime_seconds": outcome.duration_seconds,
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
            experiment_id,
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "publication_evidence": False,
                "protocol": {"path": str(path), "sha256": plan["protocol_sha256"]},
                "prerequisite": {
                    "path": str(prerequisite_path),
                    "sha256": _sha256(prerequisite_path),
                    "protocol_sha256": prerequisite_validation["protocol_sha256"],
                    "test_metrics_ignored_for_tuning": True,
                },
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
                "candidate_runs": candidate_runs,
                "selection": selection,
                "final_test": final_test,
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
