from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from exact_unlearning.quality import evaluate_result

FIELDS = [
    "schema_version",
    "model_id",
    "state",
    "role",
    "method",
    "source_experiment_id",
    "dataset",
    "dataset_sha256",
    "architecture",
    "seed",
    "deletion_scenario",
    "checkpoint_path",
    "checkpoint_sha256",
    "maturity",
    "validation_status",
    "publication_eligible",
    "reproduction_command",
]


def collect(group: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    found = []
    if isinstance(group, dict):
        if "path" in group and "sha256" in group:
            return [(prefix.strip("."), group)]
        for key, value in group.items():
            found.extend(collect(value, f"{prefix}.{key}"))
    elif isinstance(group, list):
        for index, value in enumerate(group):
            found.extend(collect(value, f"{prefix}.{index}"))
    return found


def main() -> None:
    rows = []
    for path in sorted(Path("results/raw").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        gate = evaluate_result(result)
        config_paths = [
            candidate
            for candidate in Path("configs/experiments").glob("*.yaml")
            if f"experiment_id: {result['experiment_id']}" in candidate.read_text(encoding="utf-8")
        ]
        config_path = config_paths[0].as_posix() if len(config_paths) == 1 else ""
        for role, reference in collect(result.get("checkpoints", {})):
            seed = reference.get("seed", "")
            state = (
                "exact_retrained"
                if role.startswith("oracles")
                else "unlearned"
                if role.startswith("methods")
                else "original"
            )
            method = role.split(".", 1)[1] if role.startswith("methods") else ""
            rows.append(
                {
                    "schema_version": 1,
                    "model_id": f"{result['experiment_id']}:{role}",
                    "state": state,
                    "role": role,
                    "method": method,
                    "source_experiment_id": result["experiment_id"],
                    "dataset": result["dataset"]["name"],
                    "dataset_sha256": result["dataset"].get(
                        "sha256", result["dataset"].get("sha256_after_scaling")
                    ),
                    "architecture": result["config"]["model"]["architecture"],
                    "seed": seed,
                    "deletion_scenario": result["config"]["deletion"]["strategy"],
                    "checkpoint_path": reference["path"],
                    "checkpoint_sha256": reference["sha256"],
                    "maturity": result.get("maturity", "LEGACY_UNDECLARED"),
                    "validation_status": "PASS" if gate.passed else "FAIL_ARTIFACT_GATE",
                    "publication_eligible": gate.publication_eligible,
                    "reproduction_command": (
                        f"unlearn-bench run --config {config_path}"
                        if config_path
                        else "unavailable"
                    ),
                }
            )
    output = Path("model_zoo/model_manifest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} model records")


if __name__ == "__main__":
    main()
