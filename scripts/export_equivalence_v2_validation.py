from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exact_unlearning.data import load_dataset  # noqa: E402
from exact_unlearning.metrics import predict  # noqa: E402
from exact_unlearning.models import build_model  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_schema(state: dict[str, torch.Tensor]) -> str:
    description = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(state.items())
    ]
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: checkpoint is not a nonempty state dictionary")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise ValueError(f"{path}: checkpoint contains a non-tensor state entry")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"{path}: checkpoint contains a nonfinite tensor")
    return state


def preflight(plan_path: Path, protocol_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if plan.get("execution_authorized") is not False:
        raise ValueError("v2 input plan must remain non-execution-authorizing")
    if plan.get("protocol", {}).get("sha256") != _sha256(protocol_path):
        raise ValueError("input plan does not match the current v2 protocol")
    config_path = Path(plan["training_config"]["path"])
    if _sha256(config_path) != plan["training_config"]["sha256"]:
        raise ValueError("training configuration hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_model = build_model(
        config["model"]["architecture"],
        (1, 28, 28),
        int(config["model"]["hidden_dim"]),
        10,
    )
    schemas = set()
    checkpoints = []
    for condition in plan["conditions"]:
        planned_seeds = [item["seed"] for item in condition["calibration_runs"]]
        if planned_seeds != protocol["margin_estimation"]["calibration_seeds"]:
            raise ValueError("plan calibration seeds differ from the v2 protocol")
        for run in condition["calibration_runs"]:
            checkpoint = Path(run["checkpoint"]["path"])
            if not checkpoint.is_file() or _sha256(checkpoint) != run["checkpoint"]["sha256"]:
                raise ValueError(f"checkpoint missing or hash mismatch: {checkpoint}")
            state = _load_state(checkpoint)
            expected_model.load_state_dict(state, strict=True)
            schemas.add(_state_schema(state))
            checkpoints.append(checkpoint.as_posix())
    if len(schemas) != 1:
        raise ValueError("reusable checkpoints do not share one state schema")
    split_path = Path(plan["split_manifest"]["path"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if _sha256(split_path) != plan["split_manifest"]["sha256"]:
        raise ValueError("split manifest file hash mismatch")
    if split["split_hashes"]["validation"] != protocol["probe"]["expected_split_hash"]:
        raise ValueError("validation split differs from the v2 protocol")
    return {
        "schema_version": 1,
        "status": "PASS",
        "checkpoint_count": len(checkpoints),
        "condition_count": len(plan["conditions"]),
        "validation_sample_count": len(split["splits"]["validation"]),
        "state_schema_sha256": next(iter(schemas)),
        "plan_sha256": _sha256(plan_path),
        "protocol_sha256": _sha256(protocol_path),
        "test_predictions_accessed": False,
        "training_executed": False,
    }


def _write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[name], value) for name, value in arrays.items()
            ):
                raise ValueError(f"immutable validation artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_json_once(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"immutable preflight report differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def export_reused(plan_path: Path, protocol_path: Path) -> dict[str, object]:
    report = preflight(plan_path, protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("analysis_authorized") is not True
        or protocol.get("status") != "FROZEN_VALIDATION_ONLY"
    ):
        raise ValueError("validation export blocked until the v2 protocol is explicitly frozen")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(plan["training_config"]["path"]).read_text(encoding="utf-8"))
    data = load_dataset(config["dataset"])
    expected_ids = np.asarray(
        json.loads(Path(plan["split_manifest"]["path"]).read_text(encoding="utf-8"))["splits"][
            "validation"
        ],
        dtype=np.int64,
    )
    if not np.array_equal(data.validation_ids, expected_ids):
        raise ValueError("loaded validation IDs differ from the frozen split manifest")
    exported = []
    for condition in plan["conditions"]:
        probabilities = []
        model_seeds = []
        stratified = "within_ground_truth_class" in protocol["positive_control"]["selection"]
        for run in condition["calibration_runs"]:
            model = build_model(
                config["model"]["architecture"],
                tuple(data.x_train.shape[1:]),
                int(config["model"]["hidden_dim"]),
                len(torch.unique(data.y_train)),
            )
            model.load_state_dict(_load_state(Path(run["checkpoint"]["path"])), strict=True)
            values = predict(model, data.x_validation).probabilities.astype(np.float32)
            target = Path(run["validation_prediction_target"])
            _write_npz_once(
                target,
                probabilities=values,
                sample_ids=data.validation_ids.astype(np.int64),
                model_seed=np.asarray(run["seed"], dtype=np.int64),
                **(
                    {"true_labels": data.y_validation.numpy().astype(np.int64)}
                    if stratified
                    else {}
                ),
            )
            probabilities.append(values)
            model_seeds.append(run["seed"])
            exported.append({"path": target.as_posix(), "sha256": _sha256(target)})
        ensemble_target = Path(condition["combined_analysis_input_target"])
        _write_npz_once(
            ensemble_target,
            probabilities=np.asarray(probabilities, dtype=np.float32),
            sample_ids=data.validation_ids.astype(np.int64),
            model_seeds=np.asarray(model_seeds, dtype=np.int64),
            **({"true_labels": data.y_validation.numpy().astype(np.int64)} if stratified else {}),
        )
        exported.append({"path": ensemble_target.as_posix(), "sha256": _sha256(ensemble_target)})
    return {**report, "exported": exported, "export_count": len(exported)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "export-reused"))
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("study/manifests/equivalence_margin_calibration_v2_plan.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("study/preregistration/equivalence_margin_calibration_v2.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        preflight(args.plan, args.protocol)
        if args.mode == "preflight"
        else export_reused(args.plan, args.protocol)
    )
    if args.output is not None:
        _write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
