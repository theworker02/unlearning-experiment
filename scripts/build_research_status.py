from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml


def main() -> None:
    quality = json.loads(
        Path("results/processed/phase3_quality_evaluation.json").read_text(encoding="utf-8")
    )
    connection = sqlite3.connect("results/processed/experiments.sqlite")
    try:
        by_dataset = connection.execute(
            "SELECT dataset, COUNT(*) FROM experiments GROUP BY dataset ORDER BY dataset"
        ).fetchall()
        by_method = connection.execute(
            "SELECT method, COUNT(*) FROM experiments GROUP BY method ORDER BY method"
        ).fetchall()
    finally:
        connection.close()
    failures = sum(not record["passed"] for record in quality["records"])
    passing = len(quality["records"]) - failures
    with Path("research/claims.yaml").open(encoding="utf-8") as handle:
        claims = yaml.safe_load(handle)
    verified_claims = sum(
        claim.get("verification_status") == "verified" for claim in claims["claims"]
    )
    lines = [
        "# Research status",
        "",
        "Generated laboratory bookkeeping for Benchmark 0.5. Counts are registry rows, not",
        "independent scientific conclusions.",
        "",
        f"- Raw experiment records: {len(quality['records'])}",
        f"- Records failing current artifact gates: {failures}",
        f"- Records passing current artifact gates: {passing}",
        f"- Publication-eligible raw records: {quality['publication_eligible_count']}",
        "",
        "## Registry rows by dataset",
        "",
        *[f"- {name}: {count}" for name, count in by_dataset],
        "",
        "## Registry rows by method",
        "",
        *[f"- {name or 'unrecorded'}: {count}" for name, count in by_method],
        "",
        "## Pending critical work",
        "",
        "- Revise the decision protocol after the frozen equivalence-margin calibration failed.",
        "- Keep JS as a bounded distinguishability endpoint; do not claim equivalence.",
        "- Select and audit tabular datasets; implement and validate CIFAR-10.",
        "- Freeze and publicly register C001 before inspecting confirmatory method results.",
    ]
    Path("reports/research_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readiness = {
        "Literature": "PARTIAL",
        "Novelty": "PARTIAL",
        "Datasets": "PARTIAL",
        "Model training": "PARTIAL",
        "Baselines": "PARTIAL",
        "Exact retraining": "PARTIAL",
        "Statistics": "PARTIAL",
        "Reproducibility": "PARTIAL",
        "Figures": "PARTIAL",
        "Tables": "PARTIAL",
        "Claims": "PARTIAL",
        "Manuscript": "PARTIAL",
        "Artifact release": "FAIL",
    }
    content = [
        "# Publication readiness",
        "",
        "A critical FAIL blocks publication. This scorecard records present evidence, not intent.",
        "",
        "| Area | Status |",
        "| --- | --- |",
        *[f"| {area} | {status} |" for area, status in readiness.items()],
        "",
        "Publication is blocked: no run currently passes the publication evidence gate, the",
        "confirmatory protocol is not frozen, and the prospectively tested exact/exact-q95",
        "equivalence margins were inadmissible. Evidence-linked pilot claims validate, but they",
        "are not publication-grade confirmatory evidence.",
    ]
    Path("PUBLICATION_READINESS.md").write_text("\n".join(content) + "\n", encoding="utf-8")

    evidence = [
        "# Evidence index",
        "",
        "No primary conclusion is registered. Insufficient publication evidence.",
        "",
        "Three Fashion-MNIST pilot records pass the current artifact gate. The class-conditioned",
        "assay found full-class JS distinguishability in two tested classes; the subsequent",
        "prospectively frozen margin calibration failed its primary rule. Seven historical or",
        "development records remain gate-failing. No pilot is confirmatory or",
        "publication-eligible;",
        "see `reports/phase3_class_conditioned_assay_pilot.md`,",
        "`reports/equivalence_margin_calibration_v1.md`, and",
        "`reports/phase3_quality_evaluation.md`.",
        "",
        f"Registered verified claims: {verified_claims}.",
    ]
    Path("EVIDENCE.md").write_text("\n".join(evidence) + "\n", encoding="utf-8")
    print("reports/research_status.md, PUBLICATION_READINESS.md, EVIDENCE.md")


if __name__ == "__main__":
    main()
