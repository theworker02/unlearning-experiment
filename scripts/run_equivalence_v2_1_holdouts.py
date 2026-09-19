from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from exact_unlearning.data import load_dataset
from exact_unlearning.deletion import partition_lineage, select_deletion_from_manifest
from exact_unlearning.devices import resolve_device
from exact_unlearning.metrics import predict
from exact_unlearning.models import build_model
from exact_unlearning.registry import reserve_experiment, write_checkpoint, write_raw_result
from exact_unlearning.training import train_model

if __package__:
    from scripts.export_equivalence_v2_validation import _sha256, preflight
else:
    from export_equivalence_v2_validation import _sha256, preflight


PROTOCOL = Path("study/preregistration/equivalence_margin_calibration_v2_1.yaml")
PLAN = Path("study/manifests/equivalence_margin_calibration_v2_1_frozen_plan_v2.json")


def _write_npz_once(path: Path, **arrays: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation prediction: {path}")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _validate_freeze(
    protocol_path: Path, plan_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_VALIDATION_ONLY"
        or protocol.get("execution_authorized") is not True
    ):
        raise SystemExit("holdout training blocked: protocol is not frozen and authorized")
    amendment_path = Path(str(protocol["freeze_amendment"]))
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("status") != "LOCKED":
        raise SystemExit("holdout training blocked: freeze amendment is not locked")
    if amendment["source_protocol"]["sha256"] != _sha256(protocol_path):
        raise SystemExit("holdout training blocked: amendment/protocol hash mismatch")
    if amendment["frozen_input_plan"]["sha256"] != _sha256(plan_path):
        raise SystemExit("holdout training blocked: amendment/plan hash mismatch")
    preflight(plan_path, protocol_path)
    return protocol, plan


def main(protocol_path: Path = PROTOCOL, plan_path: Path = PLAN) -> None:
    protocol, plan = _validate_freeze(protocol_path, plan_path)
    config = yaml.safe_load(Path(plan["training_config"]["path"]).read_text(encoding="utf-8"))
    data = load_dataset(config["dataset"])
    device = resolve_device(str(config["training"]["device"]))
    classes = int(data.y_train.max()) + 1
    condition_configs = {item["condition_id"]: item for item in config["conditions"]}
    root = Path("results/raw/equivalence-margin-v2-1")
    for condition in plan["conditions"]:
        condition_id = str(condition["condition_id"])
        source_condition = condition_configs[condition_id]
        deletion = select_deletion_from_manifest(
            condition["deletion_manifest"]["path"],
            data.train_ids,
            expected_dataset=data.metadata.name,
            expected_dataset_sha256=data.dataset_sha256,
            expected_strategy=str(source_condition["strategy"]),
            expected_fraction=float(source_condition["fraction"]),
            expected_seed=int(source_condition["seed"]),
            expected_target_class=int(source_condition["target_class"]),
        )
        reserve_experiment(root, condition_id)
        runs = []
        try:
            for item in condition["holdout_runs"]:
                seed = int(item["seed"])
                model = build_model(
                    config["model"]["architecture"],
                    tuple(data.x_train.shape[1:]),
                    int(config["model"]["hidden_dim"]),
                    classes,
                )
                trained = train_model(
                    model,
                    data.x_train[deletion.retain_positions],
                    data.y_train[deletion.retain_positions],
                    seed=seed,
                    epochs=int(config["training"]["epochs"]),
                    batch_size=int(config["training"]["batch_size"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    weight_decay=float(config["training"]["weight_decay"]),
                    deterministic=bool(config["training"]["deterministic"]),
                    device=device,
                    optimizer_name=str(config["training"].get("optimizer", "adamw")),
                    momentum=float(config["training"].get("momentum", 0.9)),
                )
                checkpoint = write_checkpoint(
                    root, condition_id, f"exact-{seed}", trained.model.state_dict()
                )
                expected_checkpoint = item["checkpoint_target"]
                if checkpoint["path"] != expected_checkpoint:
                    raise AssertionError("checkpoint target differs from frozen plan")
                output = _write_npz_once(
                    Path(item["validation_prediction_target"]),
                    probabilities=predict(trained.model, data.x_validation).probabilities.astype(
                        np.float32
                    ),
                    sample_ids=data.validation_ids.astype(np.int64),
                    model_seed=np.asarray(seed, dtype=np.int64),
                    true_labels=data.y_validation.numpy().astype(np.int64),
                )
                runs.append(
                    {
                        "seed": seed,
                        "duration_seconds": trained.duration_seconds,
                        "checkpoint": checkpoint,
                        "validation_prediction": output,
                    }
                )
            write_raw_result(
                root,
                condition_id,
                {
                    "schema_version": 1,
                    "experiment_id": f"equivalence-v2-1-{condition_id}-holdouts",
                    "status": "COMPLETED_VALIDATION_ONLY",
                    "publication_evidence": False,
                    "protocol": {
                        "path": protocol_path.as_posix(),
                        "sha256": _sha256(protocol_path),
                    },
                    "plan": {"path": plan_path.as_posix(), "sha256": _sha256(plan_path)},
                    "dataset": {"name": data.metadata.name, "sha256": data.dataset_sha256},
                    "partition_lineage": partition_lineage(deletion, data.train_ids),
                    "device": str(device),
                    "runs": runs,
                },
            )
        except Exception:
            # The reserved directory is intentionally retained; a new protocol/version is required.
            raise
    print("completed validation-only v2.1 heldout exact controls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--plan", type=Path, default=PLAN)
    arguments = parser.parse_args()
    main(arguments.protocol, arguments.plan)
