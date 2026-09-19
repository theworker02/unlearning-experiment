from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from exact_unlearning.quality import evaluate_result

FIELDS = [
    "experiment_id",
    "phase",
    "maturity",
    "status",
    "dataset",
    "architecture",
    "method",
    "deletion_regime",
    "deletion_fraction",
    "seed",
    "commit",
    "start_time",
    "finish_time",
    "runtime_seconds",
    "validation_status",
    "publication_eligibility",
    "failure_reason",
    "superseding_run",
    "raw_artifact",
]


def _legacy_maturity(result: dict[str, Any]) -> str:
    if "maturity" in result:
        return str(result["maturity"])
    return "PILOT" if "fashion" in result["experiment_id"] else "VALIDATION"


def _completed_rows(path: Path) -> list[dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    report = evaluate_result(result)
    maturity = _legacy_maturity(result)
    common = {
        "experiment_id": result["experiment_id"],
        "phase": result.get("research_stage", "legacy development run"),
        "maturity": maturity,
        "status": result.get("status"),
        "dataset": result.get("dataset", {}).get("name"),
        "architecture": result.get("config", {}).get("model", {}).get("architecture"),
        "deletion_regime": result.get("config", {}).get("deletion", {}).get("strategy"),
        "deletion_fraction": result.get("config", {}).get("deletion", {}).get("fraction"),
        "commit": result.get("git_commit"),
        "start_time": result.get("started_at_utc"),
        "finish_time": result.get("completed_at_utc"),
        "validation_status": "PASS" if report.passed else "FAIL: " + ",".join(report.failures),
        "publication_eligibility": report.publication_eligible,
        "failure_reason": "",
        "superseding_run": "",
        "raw_artifact": path.as_posix(),
    }
    rows = []
    for method, values in result.get("methods", {}).items():
        rows.append(
            {
                **common,
                "method": method,
                "seed": values.get("seed"),
                "runtime_seconds": values.get("duration_seconds"),
            }
        )
    if not rows:
        rows.append({**common, "method": "", "seed": "", "runtime_seconds": ""})
    return rows


def _failure_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path("results/failures").glob("**/*.json")):
        failure = json.loads(path.read_text(encoding="utf-8"))
        config = failure.get("configuration", {})
        rows.append(
            {
                "experiment_id": failure.get("experiment_id"),
                "phase": "attempted experiment",
                "maturity": config.get("maturity"),
                "status": "failed",
                "dataset": config.get("dataset", {}).get("name"),
                "architecture": config.get("model", {}).get("architecture"),
                "method": "batch",
                "deletion_regime": config.get("deletion", {}).get("strategy"),
                "deletion_fraction": config.get("deletion", {}).get("fraction"),
                "seed": config.get("original_seed"),
                "commit": failure.get("git_commit"),
                "start_time": failure.get("timestamp_utc"),
                "finish_time": failure.get("timestamp_utc"),
                "runtime_seconds": "",
                "validation_status": "FAIL",
                "publication_eligibility": False,
                "failure_reason": failure.get("failure_category"),
                "superseding_run": failure.get("superseding_run", ""),
                "raw_artifact": path.as_posix(),
            }
        )
    return rows


def main() -> None:
    rows = []
    for path in sorted(Path("results/raw").glob("*/result.json")):
        rows.extend(_completed_rows(path))
    rows.extend(_failure_rows())
    output = Path("research/experiment_ledger.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output}: {len(rows)} method/run records")


if __name__ == "__main__":
    main()
