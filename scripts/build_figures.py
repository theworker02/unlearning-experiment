from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from exact_unlearning.quality import evaluate_result


def main() -> None:
    labels: list[str] = []
    values: list[float] = []
    errors: list[float] = []
    accepted: list[Path] = []
    for path in sorted(Path("results/raw").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if not evaluate_result(result).passed:
            continue
        accepted.append(path)
        for method, metrics in result["methods"].items():
            js = metrics["distance_to_retrained_distribution"]["js_divergence"]
            labels.append(f"{result['experiment_id']}\n{method}")
            values.append(js["mean"])
            errors.append(js["standard_deviation"])
    if not values:
        raise SystemExit("no raw results found; refusing to create an empty figure")
    figure, axis = plt.subplots(figsize=(max(6, len(values) * 1.8), 4))
    axis.bar(labels, values, yerr=errors, capsize=4, color="#3768a6")
    axis.set_ylabel("Mean Jensen–Shannon divergence to retrained models")
    axis.set_title("Development runs — not publication evidence")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    output = Path("results/figures/smoke_js_divergence.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    provenance = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": "scripts/build_figures.py",
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
