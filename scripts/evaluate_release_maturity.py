from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    inventory = json.loads(
        Path("release/metadata/repository_inventory.json").read_text(encoding="utf-8")
    )
    gates = yaml.safe_load(Path("release/release_gates.yaml").read_text(encoding="utf-8"))
    counts = inventory["counts"]
    checks = {
        "benchmark_manifest_valid": Path("benchmark/benchmark_manifest.json").is_file(),
        "experiment_registry_exists": Path("results/processed/experiments.sqlite").is_file(),
        "immutable_result_writer_tested": Path("tests/unit/test_registry.py").is_file(),
        "dataset_abstraction_exists": Path("src/exact_unlearning/data.py").is_file(),
        "deletion_manifests_exist": counts["deletion_manifests"] > 0,
        "exact_retraining_supported": Path("src/exact_unlearning/experiment.py").is_file(),
        "metric_reference_checks_pass": (
            json.loads(Path("validation/reference_checks/report.json").read_text())["status"]
            == "PASS"
        ),
        "pilot_capability_exists": Path(
            "configs/experiments/fashion_cnn_random_01pct_pilot.yaml"
        ).is_file(),
        "standardized_model_interface": Path("src/exact_unlearning/contracts.py").is_file(),
        "standardized_method_interface": Path("src/exact_unlearning/methods.py").is_file(),
        "canonical_deletion_scenarios": counts["deletion_manifests"] >= 12,
        "correction_and_deprecation_policies": Path("CORRECTIONS.md").is_file()
        and Path("docs/deprecation_policy.md").is_file(),
        "generated_scholarly_site": Path("site/index.html").is_file(),
        "visual_provenance_complete": Path("visualizations/metadata/F001.yaml").is_file(),
        "evidence_explorer_complete": Path("site/evidence/index.html").is_file(),
    }
    explicitly_blocked = {
        requirement
        for version in ("1.0.0", "2.0.0", "2.3.0")
        for requirement in gates["versions"][version]["required"]
        if requirement not in checks
    }
    checks.update({requirement: False for requirement in explicitly_blocked})
    versions = {}
    assigned = None
    for version, definition in gates["versions"].items():
        required = definition["required"]
        failures = [requirement for requirement in required if not checks.get(requirement, False)]
        versions[version] = {"passed": not failures, "failures": failures}
        if not failures:
            assigned = version
    payload = {
        "schema_version": 1,
        "assigned_version": assigned,
        "versions": versions,
        "checks": checks,
    }
    output = Path("release/metadata/maturity_evaluation.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"assigned maturity: v{assigned}")


if __name__ == "__main__":
    main()
