# ruff: noqa: E501
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/fashion-cnn-random-01pct-schema2-rocm-pilot-v1/result.json"
CONVERGENCE = ROOT / "results/processed/exact_seed_convergence_fashion_pilot.json"
SEPARABILITY = ROOT / "results/processed/phase3/PILOT-FMNIST-IID-DELETE-001-v3-r2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def latex_method(name: str) -> str:
    return {
        "fine_tune": "Fine-tuning",
        "naive_gradient_ascent": "Naive gradient ascent",
        "scrub": "SCRUB",
    }.get(name, name.replace("_", " ").title())


def write_tables(raw: dict, paper_results: dict) -> dict[str, str]:
    destination = ROOT / "paper/tables"
    destination.mkdir(parents=True, exist_ok=True)
    exact = raw["exact_retraining_metrics_by_seed"]
    summary_lines = [
        "\\begin{tabular}{lrrr}\n",
        "\\toprule\n",
        "Split and metric & Minimum & Mean & Maximum \\\\\n",
        "\\midrule\n",
    ]
    for split in ("forget", "retain", "test"):
        for metric in ("accuracy", "nll", "brier", "ece"):
            values = [float(record[split][metric]) for record in exact]
            label = (
                f"{split.title()} {metric.upper() if metric in {'nll', 'ece'} else metric.title()}"
            )
            summary_lines.append(
                f"{label} & {min(values):.4f} & {sum(values) / len(values):.4f} & {max(values):.4f} \\\\\n"
            )
    summary_lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    (destination / "exact_metric_ranges.tex").write_text("".join(summary_lines), encoding="utf-8")

    runtime = paper_results["runtime_reference_seconds"]
    runtime_table = (
        "\\begin{tabular}{lr}\n\\toprule\nOperation & Seconds \\\\\n\\midrule\n"
        f"Original training & {runtime['original_training']:.2f} \\\\\n"
        f"15 exact retrainings, aggregate & {runtime['exact_total']:.2f} \\\\\n"
        f"Approximate methods, aggregate & {runtime['approximate_total']:.2f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (destination / "runtime.tex").write_text(runtime_table, encoding="utf-8")

    method_lines = [
        "\\begin{tabular}{lrrrr}\n",
        "\\toprule\n",
        "Method & Seconds & Forget acc. & Retain acc. & Test acc. \\\\\n",
        "\\midrule\n",
    ]
    for name in sorted(raw["methods"]):
        record = raw["methods"][name]
        method_lines.append(
            f"{latex_method(name)} & {record['duration_seconds']:.2f} & "
            f"{record['forget_metrics']['accuracy']:.3f} & "
            f"{record['retain_metrics']['accuracy']:.3f} & "
            f"{record['test_metrics']['accuracy']:.3f} \\\\\n"
        )
    method_lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    (destination / "method_descriptives.tex").write_text("".join(method_lines), encoding="utf-8")

    integrity = raw["quality_gate"]["checks"]
    gate_lines = ["\\begin{tabular}{ll}\n\\toprule\nGate & P001 status \\\\\n\\midrule\n"]
    for name, passed in sorted(integrity.items()):
        gate_lines.append(
            f"{name.replace('_', ' ').title()} & {'PASS' if passed else 'FAIL'} \\\\\n"
        )
    gate_lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    (destination / "integrity_gates.tex").write_text("".join(gate_lines), encoding="utf-8")
    return {path.name: sha256(path) for path in sorted(destination.glob("*.tex"))}


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    convergence = json.loads(CONVERGENCE.read_text(encoding="utf-8"))
    separability = json.loads(SEPARABILITY.read_text(encoding="utf-8"))
    stability = convergence["stability"]
    sep = separability["separability"]["metrics"]
    method_seconds = sum(float(item["duration_seconds"]) for item in raw["methods"].values())
    paper_results = {
        "schema_version": 1,
        "release": "R001",
        "experiment": "P001",
        "source_artifacts": {
            "raw_result": {"path": RAW.relative_to(ROOT).as_posix(), "sha256": sha256(RAW)},
            "convergence": {
                "path": CONVERGENCE.relative_to(ROOT).as_posix(),
                "sha256": sha256(CONVERGENCE),
            },
            "separability": {
                "path": SEPARABILITY.relative_to(ROOT).as_posix(),
                "sha256": sha256(SEPARABILITY),
            },
        },
        "dataset": {
            "name": "Fashion-MNIST",
            "dataset_sha256": raw["dataset"]["sha256_after_scaling"],
            "train_count": raw["dataset"]["train_count"],
            "validation_count": raw["dataset"]["validation_count"],
            "test_count": raw["dataset"]["test_count"],
        },
        "deletion": {
            "fraction": raw["config"]["deletion"]["fraction"],
            "count": raw["deletion"]["count"],
            "manifest_sha256": raw["deletion"]["manifest"]["sha256"],
        },
        "exact_reference": {
            "count": len(raw["config"]["oracle_seeds"]),
            "seeds": raw["config"]["oracle_seeds"],
        },
        "convergence": {
            "threshold": stability["js_divergence"]["threshold"],
            "js_change": stability["js_divergence"]["relative_change_k10_to_k15"],
            "disagreement_change": stability["prediction_disagreement"][
                "relative_change_k10_to_k15"
            ],
            "js_pass": stability["js_divergence"]["passes_pilot_diagnostic"],
            "disagreement_pass": stability["prediction_disagreement"]["passes_pilot_diagnostic"],
        },
        "separability": {
            "js_ratio": sep["js_divergence"]["separability_ratio"],
            "disagreement_ratio": sep["prediction_disagreement"]["separability_ratio"],
            "classification": separability["decision"]["overall_class"],
        },
        "runtime_reference_seconds": {
            "original_training": raw["runtime"]["original_training_seconds"],
            "exact_total": sum(raw["runtime"]["oracle_training_seconds"]),
            "approximate_total": method_seconds,
        },
        "methods": sorted(raw["methods"]),
        "runtime_environment": raw["runtime"],
        "claim_boundary": {
            "supported": [
                "P001 artifact and exclusion gates passed",
                "The frozen k=10-to-15 convergence diagnostic passed for two output metrics",
                "The tested IID deletion was retrospectively assay-insensitive under those metrics",
            ],
            "unsupported": [
                "successful unlearning",
                "certified removal",
                "privacy equivalence",
                "cross-dataset or cross-architecture generalization",
            ],
        },
    }
    publication = ROOT / "results/publication"
    publication.mkdir(parents=True, exist_ok=True)
    result_path = publication / "paper-results.json"
    result_path.write_text(
        json.dumps(paper_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table_hashes = write_tables(raw, paper_results)

    reproduction = {
        "study": "counterfactual-fidelity-machine-unlearning",
        "release": "R001",
        "experiment": "P001",
        "schema": 2,
        "maturity": "retrospective_pilot",
        "dataset": paper_results["dataset"],
        "deletion": paper_results["deletion"],
        "exact_reference": paper_results["exact_reference"],
        "hardware_reference": {
            "gpu": "AMD Radeon RX 7800 XT",
            "backend": "ROCm",
            "rocm": raw["runtime"]["rocm_version"],
            "torch": raw["runtime"]["torch"],
            "platform": raw["runtime"]["platform"],
            "python": raw["runtime"]["python"],
        },
        "validation": {"convergence_threshold": paper_results["convergence"]["threshold"]},
        "expected": {
            "convergence": paper_results["convergence"],
            "runtime_reference_seconds": paper_results["runtime_reference_seconds"],
            "runtime_values_are_acceptance_criteria": False,
        },
        "figures": ["F002"],
        "identifiers": {"osf_doi": None, "zenodo_doi": None, "preprint_identifier": None},
    }
    (ROOT / "reproduction.yaml").write_text(
        yaml.safe_dump(reproduction, sort_keys=False), encoding="utf-8"
    )

    provenance = {
        "schema_version": 1,
        "release": "R001",
        "experiment": "P001",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "manuscript_commit": commit(),
        "commit_status": "resolved" if commit() else "unavailable_not_a_git_worktree",
        "results_sha256": sha256(result_path),
        "references_sha256": sha256(ROOT / "paper/references.bib"),
        "experiment_registry_sha256": sha256(ROOT / "study/registry/experiments.yaml"),
        "claim_graph_sha256": sha256(ROOT / "research/claims.yaml"),
        "figures": {"F002": sha256(ROOT / "figures/F002/F002.svg")},
        "tables": table_hashes,
        "schema_coverage": [2],
    }
    (ROOT / "paper/provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result_path.relative_to(ROOT))
    print("reproduction.yaml")
    print("paper/provenance.json")


if __name__ == "__main__":
    main()
