from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PLAN = Path("study/manifests/equivalence_margin_calibration_v2_1_frozen_plan_v2.json")
OUTPUT = Path("study/manifests/equivalence_margin_calibration_v2_1_inputs.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[name], value) for name, value in arrays.items()
            ):
                raise ValueError(f"immutable assembled input differs: {path}")
        return
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    conditions = []
    for condition in plan["conditions"]:
        runs = [*condition["calibration_runs"], *condition["holdout_runs"]]
        probabilities: list[np.ndarray] = []
        seeds: list[int] = []
        sample_ids: np.ndarray | None = None
        true_labels: np.ndarray | None = None
        for run in runs:
            path = Path(run["validation_prediction_target"])
            if not path.is_file():
                raise ValueError(f"missing validation prediction: {path}")
            with np.load(path, allow_pickle=False) as arrays:
                current_ids = arrays["sample_ids"].astype(np.int64)
                current_labels = arrays["true_labels"].astype(np.int64)
                if sample_ids is None:
                    sample_ids, true_labels = current_ids, current_labels
                elif not np.array_equal(sample_ids, current_ids) or not np.array_equal(
                    true_labels, current_labels
                ):
                    raise ValueError("validation inputs do not share frozen IDs and labels")
                probabilities.append(arrays["probabilities"].astype(np.float32))
                seeds.append(int(run["seed"]))
        target = Path("results/validation/equivalence-margin-v2-1") / str(
            condition["condition_id"]
        ) / "ensemble-with-holdouts.npz"
        _write_npz_once(
            target,
            probabilities=np.asarray(probabilities, dtype=np.float32),
            sample_ids=sample_ids,
            model_seeds=np.asarray(seeds, dtype=np.int64),
            true_labels=true_labels,
        )
        conditions.append(
            {
                "condition_id": condition["condition_id"],
                "path": target.as_posix(),
                "sha256": _sha256(target),
                "calibration_seeds": [item["seed"] for item in condition["calibration_runs"]],
                "holdout_seeds": [item["seed"] for item in condition["holdout_runs"]],
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol_id": "equivalence-margin-calibration-v2.1",
        "publication_evidence": False,
        "split": "validation",
        "validation_split_sha256": plan["split_manifest"]["validation_split_sha256"],
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "conditions": conditions,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != payload:
        raise ValueError(f"immutable input manifest differs: {OUTPUT}")
    OUTPUT.write_text(payload, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
