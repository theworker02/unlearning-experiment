from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from exact_unlearning.quality import evaluate_result


def registry_count(filename: str, key: str, status: str | None = None) -> int:
    entries = yaml.safe_load(Path("benchmark", filename).read_text(encoding="utf-8"))[key]
    if status is None:
        return len(entries)
    return sum(entry.get("status") == status for entry in entries)


def main() -> None:
    results: list[dict[str, Any]] = []
    for path in sorted(Path("results/raw").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        gate = evaluate_result(result)
        results.append(
            {
                "experiment_id": result["experiment_id"],
                "artifact": path.as_posix(),
                "gate_passed": gate.passed,
                "publication_eligible": gate.publication_eligible,
                "failures": list(gate.failures),
            }
        )
    with Path("model_zoo/model_manifest.csv").open(encoding="utf-8", newline="") as handle:
        models = list(csv.DictReader(handle))
    manifest_count = len(list(Path("deletion_manifests").glob("**/*.json")))
    questions = yaml.safe_load(
        Path("research/research_questions.yaml").read_text(encoding="utf-8")
    )["questions"]
    presentation_path = Path("site/presentation/presentation-data.json")
    presentation_slides = (
        len(json.loads(presentation_path.read_text(encoding="utf-8"))["slides"])
        if presentation_path.is_file()
        else 0
    )
    payload = {
        "schema_version": 1,
        "benchmark_version": Path("benchmark/VERSION").read_text(encoding="utf-8").strip(),
        "assigned_maturity": "0.5.0",
        "maturity_label": "Research Infrastructure Candidate",
        "counts": {
            "dataset_registry_entries": registry_count("datasets.yaml", "datasets"),
            "available_datasets": registry_count("datasets.yaml", "datasets", "available"),
            "architecture_entries": registry_count("architectures.yaml", "architectures"),
            "method_entries": registry_count("methods.yaml", "methods"),
            "deletion_manifests": manifest_count,
            "raw_experiment_records": len(results),
            "gate_passing_records": sum(item["gate_passed"] for item in results),
            "publication_eligible_records": sum(item["publication_eligible"] for item in results),
            "indexed_model_checkpoints": len(models),
            "verified_claims": 0,
            "empirical_publication_figures": 0,
            "empirical_pilot_figures": sum(
                1
                for path in Path("visualizations/metadata").glob("*.yaml")
                if yaml.safe_load(path.read_text(encoding="utf-8")).get("type") == "empirical"
            ),
            "research_questions": len(questions),
            "inconclusive_research_questions": sum(
                question["status"] == "INCONCLUSIVE" for question in questions
            ),
            "presentation_slides": presentation_slides,
        },
        "capabilities": {
            "immutable_result_registry": True,
            "dataset_reconstruction": True,
            "exact_retraining_population_runner": True,
            "literature_backed_method_pilot": True,
            "sequential_runner": "stage_matched_orchestration_and_integration_tests",
            "confirmatory_results": False,
            "paper_data_release": False,
            "external_reproduction": False,
        },
        "raw_records": results,
    }
    output = Path("release/metadata/repository_inventory.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Repository inventory",
        "",
        "Generated from repository artifacts. It does not infer completed science from planned",
        "registry entries.",
        "",
        f"Assigned maturity: **v{payload['assigned_maturity']} — {payload['maturity_label']}**",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        *[f"| {key.replace('_', ' ')} | {value} |" for key, value in payload["counts"].items()],
        "",
        "No raw record currently passes the Phase III artifact gate. v1.0, v2.0, and v2.3",
        "therefore remain blocked regardless of implemented presentation infrastructure.",
    ]
    Path("reports/repository_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
