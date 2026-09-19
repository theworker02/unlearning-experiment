from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    source = yaml.safe_load(Path("research/research_questions.yaml").read_text(encoding="utf-8"))
    questions = source["questions"]
    inventory = json.loads(
        Path("release/metadata/repository_inventory.json").read_text(encoding="utf-8")
    )
    if inventory["counts"]["publication_eligible_records"] != 0:
        raise SystemExit("synthesis template must be replaced when eligible evidence exists")

    lines = [
        "# Research questions",
        "",
        "Benchmark protocol: **0.5.0**. One gate-passing pilot informs design, but no",
        "question has a confirmatory or publication-eligible empirical answer.",
        "",
        "| ID | Research question | Eligible primary evidence | Status | Uncertainty | Figure |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for question in questions:
        evidence = ", ".join(question["experiments"]) or "none"
        lines.append(
            f"| {question['id']} | {question['question']} | {evidence} | "
            f"{question['status']} | "
            f"{question['uncertainty']} | none |"
        )
    lines.extend(["", "## Detailed register", ""])
    for question in questions:
        lines.extend(
            [
                f"### {question['id']} — {question['question']}",
                "",
                f"- Hypothesis: {question['hypothesis']}",
                f"- Status: **{question['status']}**",
                f"- Observation: {question['reason']}",
                "- Statistical inference: none; the required eligible sample is absent.",
                "- Interpretation: the hypothesis is neither supported nor rejected.",
                f"- Required evidence: {question['required_evidence']}",
                f"- Planned analysis: {question['planned_analysis']}",
                f"- Uncertainty: {question['uncertainty']}",
                f"- Limitation: {'; '.join(question['limitations'])}",
                "",
            ]
        )
    Path("research/RESEARCH_QUESTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conclusions = {
        "schema_version": 1,
        "benchmark_version": "0.5.0",
        "conclusions": [
            {
                "id": f"CONC-{index:03d}",
                "question": question["id"],
                "statement": "Insufficient eligible evidence to support or reject the hypothesis.",
                "status": "inconclusive",
                "observation": question["reason"],
                "statistical_inference": None,
                "interpretation": "No empirical conclusion is permitted.",
                "evidence": question["experiments"],
                "effect_size": None,
                "uncertainty": question["uncertainty"],
                "limitations": question["limitations"],
            }
            for index, question in enumerate(questions, start=1)
        ],
    }
    Path("research/conclusions.yaml").write_text(
        yaml.safe_dump(conclusions, sort_keys=False), encoding="utf-8"
    )

    results = [
        "# Results",
        "",
        "Benchmark protocol: **0.5.0**.",
        "",
        "> **Insufficient evidence for the central research questions.** One schema-v2 pilot",
        "> passes artifact validation but remains non-confirmatory and publication-ineligible.",
    ]
    sections = [
        "Primary findings",
        "Exact-retraining variability",
        "Metric disagreement",
        "Sequential deletion",
        "Deletion-strategy sensitivity",
        "Cross-dataset findings",
        "Cross-architecture findings",
        "Privacy findings",
        "Calibration findings",
        "Representation findings",
        "Compute–fidelity tradeoffs",
        "Negative and inconclusive findings",
    ]
    for section in sections:
        results.extend(
            [
                "",
                f"## {section}",
                "",
                "No confirmatory analysis, publication figure, or evidence-backed answer exists.",
            ]
        )
    results.extend(
        [
            "",
            "## What can be stated",
            "",
            "A new Fashion-MNIST pilot validates schema-v2 capture and measures exact-seed",
            "convergence for one condition. Its frozen k=10→15 diagnostics passed, but the",
            "condition does not establish method effectiveness, equivalence, contradiction, or",
            "generalization. Six historical records remain ineligible provenance.",
        ]
    )
    Path("RESULTS.md").write_text("\n".join(results) + "\n", encoding="utf-8")

    learned = """# What we learned

The project has not yet learned a defensible general answer to its machine-unlearning questions.

It established that the earlier development records are insufficient under the present evidence
standard: they lack explicit maturity and required immutable prediction artifacts, and most lack
the current per-seed exact-reference record. Consequently, no method comparison, contradiction,
sequential trend, or compute–fidelity conclusion is defensible.

One new schema-v2 Fashion-MNIST pilot passed every artifact gate and made exact-retraining variance
measurable across 15 replicas. The frozen pilot diagnostic found that exact/exact mean JS divergence
and prediction disagreement changed by less than 10% from k=10 to k=15. Approximate models fell
below exact/exact 95th-percentile distances for those two dimensions, while forget-set accuracy
remained high. These observations describe one condition; they are neither equivalence tests nor
evidence of successful forgetting.

The scientifically justified next step is replication across original seeds, deletion fractions,
and a high-loss condition before freezing the larger benchmark matrix. This conclusion concerns
research readiness, not general unlearning performance.
"""
    Path("WHAT_WE_LEARNED.md").write_text(learned, encoding="utf-8")
    print("research/RESEARCH_QUESTIONS.md, research/conclusions.yaml, RESULTS.md")


if __name__ == "__main__":
    main()
