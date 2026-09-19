from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from exact_unlearning.quality import evaluate_result, load_result


def main() -> None:
    records = []
    for path in sorted(Path("results/raw").glob("*/result.json")):
        result = load_result(path)
        report = evaluate_result(result)
        records.append(
            {
                "experiment_id": result["experiment_id"],
                "declared_or_inferred_maturity": result.get(
                    "maturity", "PILOT" if "fashion" in result["experiment_id"] else "VALIDATION"
                ),
                "artifact": path.as_posix(),
                **report.as_dict(),
            }
        )
    counts = Counter(failure for record in records for failure in record["failures"])
    payload = {
        "schema_version": 1,
        "records": records,
        "failure_counts": dict(sorted(counts.items())),
        "publication_eligible_count": sum(record["publication_eligible"] for record in records),
        "interpretation": "Legacy failures are not repaired by mutating immutable raw evidence.",
    }
    output = Path("results/processed/phase3_quality_evaluation.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase III quality evaluation",
        "",
        "This evaluates existing immutable runs under the new fail-closed rules. A failed gate",
        "does not mean the process never ran; it means the artifact cannot support publication.",
        "",
        "| Experiment | Maturity | Gate | Publication eligible | Failures |",
        "| --- | --- | --- | :---: | --- |",
    ]
    for record in records:
        failures = ", ".join(record["failures"]) or "none"
        lines.append(
            f"| {record['experiment_id']} | {record['declared_or_inferred_maturity']} | "
            f"{'PASS' if record['passed'] else 'FAIL'} | "
            f"{'yes' if record['publication_eligible'] else 'no'} | {failures} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "No existing run is promoted to publication evidence. Legacy pilot artifacts predate",
            "mandatory raw-prediction capture and explicit maturity metadata. New schema-v2 runs",
            "can pass artifact postconditions but still remain PILOT or VALIDATION by design.",
        ]
    )
    Path("reports/phase3_quality_evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
