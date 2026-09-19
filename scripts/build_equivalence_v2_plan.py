from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROTOCOL = Path("study/preregistration/equivalence_margin_calibration_v2.yaml")
CONFIG = Path("configs/experiments/phase3_class_conditioned_assay_pilot.yaml")
SPLIT_MANIFEST = Path("datasets/fashion_mnist/split_manifest.json")
SOURCE_ROOT = Path("results/raw/phase3-fmnist-class-conditioned-assay-pilot-v3")
OUTPUT = Path("study/manifests/equivalence_margin_calibration_v2_plan.json")
CHECKSUM = OUTPUT.with_suffix(".sha256")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan() -> dict[str, object]:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    calibration_seeds = protocol["margin_estimation"]["calibration_seeds"]
    holdout_seeds = protocol["margin_estimation"]["heldout_seeds"]
    if len(calibration_seeds) != protocol["margin_estimation"]["calibration_seed_count"]:
        raise ValueError("calibration seed manifest does not match the protocol count")
    if len(holdout_seeds) != protocol["margin_estimation"]["heldout_seed_count"]:
        raise ValueError("holdout seed manifest does not match the protocol count")
    if set(calibration_seeds) & set(holdout_seeds):
        raise ValueError("calibration and holdout seed manifests overlap")
    validation_ids = split["splits"]["validation"]
    if len(validation_ids) < protocol["probe"]["minimum_samples"]:
        raise ValueError("frozen validation split is smaller than the protocol requires")
    if split["split_hashes"]["validation"] != protocol["probe"]["expected_split_hash"]:
        raise ValueError("validation split hash differs from the protocol")
    conditions = []
    measured_checkpoint_bytes = []
    for condition in protocol["scope"]["initial_conditions"]:
        condition_id = condition["condition_id"]
        target_class = 0 if "class0" in condition_id else 5
        deletion_manifest = Path(
            f"deletion_manifests/phase3/fashion_mnist/r1-class{target_class}-f1.00-seed3201.json"
        )
        calibration = []
        for seed in calibration_seeds:
            checkpoint = SOURCE_ROOT / "checkpoints" / f"exact-{condition_id}-{seed}.pt"
            if not checkpoint.is_file():
                raise ValueError(f"reusable checkpoint missing: {checkpoint}")
            measured_checkpoint_bytes.append(checkpoint.stat().st_size)
            calibration.append(
                {
                    "seed": seed,
                    "source": "reuse_exact_checkpoint",
                    "checkpoint": {"path": checkpoint.as_posix(), "sha256": sha256(checkpoint)},
                    "validation_prediction_target": (
                        f"results/validation/equivalence-margin-v2/{condition_id}/exact-{seed}.npz"
                    ),
                }
            )
        holdout = [
            {
                "seed": seed,
                "source": "new_exact_training_required",
                "run_id": f"equivalence-v2-{condition_id}-exact-{seed}",
                "checkpoint_target": (
                    f"results/raw/equivalence-margin-v2/{condition_id}/checkpoints/exact-{seed}.pt"
                ),
                "validation_prediction_target": (
                    f"results/validation/equivalence-margin-v2/{condition_id}/exact-{seed}.npz"
                ),
            }
            for seed in holdout_seeds
        ]
        conditions.append(
            {
                **condition,
                "deletion_manifest": {
                    "path": deletion_manifest.as_posix(),
                    "sha256": sha256(deletion_manifest),
                },
                "calibration_runs": calibration,
                "holdout_runs": holdout,
                "combined_analysis_input_target": (
                    f"results/validation/equivalence-margin-v2/{condition_id}/ensemble.npz"
                ),
            }
        )
    checkpoint_bytes = measured_checkpoint_bytes[0]
    if any(value != checkpoint_bytes for value in measured_checkpoint_bytes):
        raise ValueError("source checkpoint sizes are inconsistent")
    condition_count = len(conditions)
    model_count = len(calibration_seeds) + len(holdout_seeds)
    probability_bytes = condition_count * model_count * len(validation_ids) * 10 * 4
    return {
        "schema_version": 1,
        "plan_id": "equivalence-margin-calibration-v2-input-plan",
        "status": "DRAFT_INPUT_PLAN_NOT_EXECUTION_AUTHORIZED",
        "execution_authorized": False,
        "protocol": {"path": PROTOCOL.as_posix(), "sha256": sha256(PROTOCOL)},
        "training_config": {"path": CONFIG.as_posix(), "sha256": sha256(CONFIG)},
        "split_manifest": {
            "path": SPLIT_MANIFEST.as_posix(),
            "sha256": sha256(SPLIT_MANIFEST),
            "validation_split_sha256": split["split_hashes"]["validation"],
            "validation_sample_count": len(validation_ids),
        },
        "prediction_schema": {
            "format": "npz_allow_pickle_false",
            "arrays": {
                "probabilities": [model_count, len(validation_ids), 10],
                "sample_ids": [len(validation_ids)],
                "model_seeds": [model_count],
            },
            "probability_dtype": "float32",
        },
        "conditions": conditions,
        "compute_and_storage": {
            "existing_exact_models_reused": condition_count * len(calibration_seeds),
            "new_exact_training_jobs": condition_count * len(holdout_seeds),
            "validation_prediction_exports": condition_count * model_count,
            "measured_bytes_per_existing_checkpoint": checkpoint_bytes,
            "projected_new_checkpoint_bytes": checkpoint_bytes
            * condition_count
            * len(holdout_seeds),
            "minimum_probability_array_bytes": probability_bytes,
            "runtime_estimate": None,
            "runtime_blocker": "target_device_benchmark_not_yet_measured",
        },
        "notes": [
            (
                "Existing checkpoints require fresh validation predictions; "
                "test predictions are forbidden."
            ),
            "Projected bytes exclude NPZ metadata, sample IDs, logits, and environment records.",
            "This plan does not freeze the one-percent SESOI or authorize any training.",
        ],
    }


def main() -> None:
    plan = build_plan()
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"immutable plan differs: {OUTPUT}")
    OUTPUT.write_text(payload, encoding="utf-8")
    disk_digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    CHECKSUM.write_text(f"{disk_digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps(plan["compute_and_storage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
