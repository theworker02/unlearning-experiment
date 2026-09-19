from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from exact_unlearning.quality import evaluate_result


def main() -> None:
    rows: list[str] = []
    inputs = sorted(Path("results/raw").glob("*/result.json"))
    accepted: list[Path] = []
    for path in inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not evaluate_result(result).passed:
            continue
        accepted.append(path)
        for method, values in result["methods"].items():
            test = values["test_metrics"]
            js = values["distance_to_retrained_distribution"]["js_divergence"]
            rows.append(
                f"{result['experiment_id']} & {method} & {test['accuracy']:.3f} & "
                f"{js['mean']:.4f} $\\pm$ {js['standard_deviation']:.4f} \\\\"
            )
    if not rows:
        raise SystemExit("no raw results found; refusing to create an empty evidence table")
    output = Path("results/tables/smoke_summary.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\\begin{tabular}{llrr}\nExperiment & Method & Accuracy & JS divergence \\\\\n"
        "\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": "scripts/build_tables.py",
        "inputs": [
            {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in accepted
        ],
        "output": output.as_posix(),
        "scope": "development evidence only",
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
