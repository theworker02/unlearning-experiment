# ruff: noqa: E501
from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

import yaml

NAV = [
    ("Home", "/index.html"),
    ("Benchmark", "/benchmark/index.html"),
    ("Datasets", "/datasets/index.html"),
    ("Models", "/models/index.html"),
    ("Experiments", "/experiments/index.html"),
    ("Figures", "/figures/index.html"),
    ("Evidence", "/evidence/index.html"),
    ("Reproduce", "/reproduce/index.html"),
    ("Presentation", "/presentation/index.html"),
    ("Results", "/results/index.html"),
    ("Limitations", "/limitations/index.html"),
    ("Citation", "/citation/index.html"),
]


def page(title: str, body: str) -> str:
    navigation = "".join(f'<a href="{url}">{label}</a>' for label, url in NAV)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Counterfactual Fidelity Benchmark</title>
<link rel="stylesheet" href="/assets/style.css"></head><body>
<header><div><strong>Counterfactual Fidelity Benchmark</strong><span>Protocol 0.5.0</span></div>
<nav>{navigation}</nav></header><main><h1>{html.escape(title)}</h1>{body}</main>
<footer>Generated from repository artifacts. Benchmark 0.5.0 — pre-publication.</footer></body></html>"""


def write(route: str, title: str, body: str) -> None:
    output = Path("site") / route / "index.html" if route else Path("site/index.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page(title, body), encoding="utf-8")


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def main() -> None:
    inventory = json.loads(
        Path("release/metadata/repository_inventory.json").read_text(encoding="utf-8")
    )
    counts = inventory["counts"]
    datasets = yaml.safe_load(Path("benchmark/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    methods = yaml.safe_load(Path("benchmark/methods.yaml").read_text(encoding="utf-8"))["methods"]
    with Path("model_zoo/model_manifest.csv").open(encoding="utf-8", newline="") as handle:
        models = list(csv.DictReader(handle))
    with Path("research/experiment_ledger.csv").open(encoding="utf-8", newline="") as handle:
        experiments = list(csv.DictReader(handle))
    claims = yaml.safe_load(Path("research/claims.yaml").read_text(encoding="utf-8"))["claims"]
    questions = yaml.safe_load(
        Path("research/research_questions.yaml").read_text(encoding="utf-8")
    )["questions"]

    style = """body{margin:0;color:#17212b;background:#fff;font:16px/1.55 Arial,sans-serif}header{padding:18px 5vw;border-bottom:1px solid #cbd5e1;background:#f8fafc}header div{display:flex;gap:16px;align-items:baseline}header span{color:#64748b}nav{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px}a{color:#174ea6}main{max-width:1100px;margin:45px auto;padding:0 24px}h1{font-size:2rem}h2{margin-top:2.2rem}.notice{border-left:5px solid #a16207;background:#fffbeb;padding:14px 18px}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.fact{border:1px solid #cbd5e1;padding:14px}.fact b{display:block;font-size:1.5rem}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #cbd5e1;padding:9px;text-align:left;vertical-align:top}th{background:#f1f5f9}code{background:#f1f5f9;padding:2px 5px}footer{margin-top:60px;padding:25px 5vw;border-top:1px solid #cbd5e1;color:#475569}@media print{nav{display:none}}"""
    assets = Path("site/assets")
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "style.css").write_text(style, encoding="utf-8")

    facts = "".join(
        f"<div class='fact'><b>{value}</b>{html.escape(key.replace('_', ' '))}</div>"
        for key, value in counts.items()
    )
    write(
        "",
        "Evaluating unlearning against exact-retraining distributions",
        "<p class='notice'><strong>Status:</strong> Research Infrastructure Candidate v0.5.0. "
        "No experiment is publication-eligible and no empirical conclusion is registered.</p>"
        "<h2>Research question</h2><p>When do conventional forgetting criteria indicate success "
        "while an approximately unlearned model remains distinguishable from independently "
        "trained models that never encountered the forgotten data?</p>"
        f"<h2>Artifact inventory</h2><div class='facts'>{facts}</div>"
        "<h2>What is released</h2><p>A pilot benchmark specification, dataset and deletion "
        "manifests, immutable experiment machinery, quality gates, and development records. "
        "Confirmatory results, a paper-data release, and a publication manuscript do not yet exist.</p>",
    )
    write(
        "benchmark",
        "Benchmark specification",
        "<p>Benchmark 0.5 defines valid-run provenance, exact-retraining populations, extension "
        "boundaries, and multidimensional fidelity. It is not a frozen publication protocol.</p>"
        + table(
            ["Method", "Category", "Status"],
            [[m["id"], m["category"], m["status"]] for m in methods],
        ),
    )
    write(
        "datasets",
        "Dataset catalog",
        "<p>Counts and status come from the benchmark registry. Digits is CI-only.</p>"
        + table(
            ["Dataset", "Research role", "Status"],
            [[d["id"], d["role"], d["status"]] for d in datasets],
        ),
    )
    write(
        "models",
        "Model catalog",
        "<p>Only existing indexed checkpoints are shown. All currently indexed models are legacy "
        "development artifacts and are not publication-eligible.</p>"
        + table(
            ["Model ID", "Dataset", "Architecture", "Role", "Checkpoint hash", "Eligible"],
            [
                [
                    m["model_id"],
                    m["dataset"],
                    m["architecture"],
                    m["state"],
                    m["checkpoint_sha256"][:12],
                    m["publication_eligible"],
                ]
                for m in models
            ],
        ),
    )
    write(
        "experiments",
        "Experiment catalog",
        table(
            ["Experiment", "Dataset", "Architecture", "Method", "Deletion", "Validation"],
            [
                [
                    e["experiment_id"],
                    e["dataset"],
                    e["architecture"],
                    e["method"],
                    e["deletion_regime"],
                    e["validation_status"],
                ]
                for e in experiments
            ],
        ),
    )
    figure_target = Path("site/figures/F001_framework.svg")
    figure_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("visualizations/conceptual/F001_framework.svg", figure_target)
    convergence_target = Path("site/figures/F002_seed_convergence.svg")
    shutil.copyfile("visualizations/empirical/F002_seed_convergence.svg", convergence_target)
    write(
        "figures",
        "Figures",
        "<p class='notice'>F002 is empirical PILOT evidence. No publication figure exists because "
        "there is no confirmatory or publication-eligible result.</p>"
        "<h2>F001 — Counterfactual evaluation framework</h2>"
        "<img src='/figures/F001_framework.svg' alt='Conceptual counterfactual evaluation framework' style='max-width:100%'>"
        "<p>Conceptual; contains no empirical data.</p>"
        "<h2>F002 — Exact-retraining seed convergence</h2>"
        "<a href='/figures/f002/'><img src='/figures/F002_seed_convergence.svg' "
        "alt='Pilot exact-retraining seed convergence' style='max-width:100%'></a>"
        "<p>Empirical PILOT figure; not publication evidence.</p>",
    )
    write(
        "figures/f002",
        "F002 — Exact-retraining seed convergence",
        "<img src='/figures/F002_seed_convergence.svg' alt='Pilot exact-retraining seed "
        "convergence' style='max-width:100%'>"
        "<h2>Question</h2><p>Do exact/exact distance estimates stabilize as independently "
        "retrained replicas are added in this pilot condition?</p>"
        "<h2>Axes and unit</h2><p>x is the ordered prefix population size k. y is the mean "
        "exact/exact test-set distance. The unit is a pair of independently trained models; "
        "uncertainty uses replica bootstrap resampling.</p>"
        "<h2>Allowed interpretation</h2><p>The frozen k=10→15 diagnostic passed for JS "
        "divergence and prediction disagreement in this condition.</p>"
        "<h2>Disallowed interpretation</h2><p>This does not prove k=15 is adequate elsewhere, "
        "establish equivalence, or demonstrate successful unlearning.</p>"
        "<h2>Data and reproduction</h2><p><code>python "
        "scripts/analyze_exact_seed_convergence.py</code><br><code>python "
        "visualizations/scripts/build_f002_seed_convergence.py</code></p>",
    )
    claim_rows = [[c.get("claim_id"), c.get("text"), c.get("verification_status")] for c in claims]
    evidence_body = (
        table(["Claim", "Statement", "Status"], claim_rows)
        if claim_rows
        else "<p class='notice'>No claims are registered. Insufficient publication evidence.</p>"
    )
    write("evidence", "Evidence explorer", evidence_body)
    write(
        "evidence/strongest",
        "Strongest evidence",
        "<p class='notice'>No confirmatory finding qualifies. One pilot passed the artifact gate. "
        "Its frozen k=10→15 stability diagnostic passed for JS divergence and prediction "
        "disagreement. This one-condition observation does not establish method effectiveness "
        "or a generally adequate seed count.</p><p><a href='/figures/f002/'>Inspect F002, its "
        "interpretation limits, data, and reproduction commands.</a></p>",
    )
    uncertain_rows = [
        [question["id"], question["question"], question["status"], question["uncertainty"]]
        for question in questions
    ]
    write(
        "evidence/uncertain",
        "Where the evidence is weak",
        "<p>Every central question is currently inconclusive.</p>"
        + table(["Question", "Research question", "Status", "Uncertainty"], uncertain_rows),
    )
    for question in questions:
        write(
            f"results/{question['id'].lower()}",
            f"{question['id']} — {question['question']}",
            f"<p class='notice'><strong>{question['status']}.</strong> {question['reason']}</p>"
            f"<h2>Required evidence</h2><p>{question['required_evidence']}</p>"
            f"<h2>Planned analysis</h2><p>{question['planned_analysis']}</p>"
            f"<h2>Uncertainty</h2><p>{question['uncertainty']}</p>",
        )
    write(
        "reproduce",
        "Reproduction",
        '<pre><code>python -m pip install -e ".[dev]"\nmake verify\nmake corpus\n'
        "unlearn-bench validate\nunlearn-bench status</code></pre>"
        "<p>The flagship reproduction target is intentionally blocked until a confirmatory "
        "flagship configuration and expected range are frozen.</p>",
    )
    write(
        "limitations",
        "Limitations",
        "<ul><li>No raw run passes the current artifact gate.</li><li>Only Digits and "
        "Fashion-MNIST are locally available; neither establishes primary-benchmark "
        "generalization.</li><li>Sequential, robust privacy, CIFAR, and tabular confirmatory "
        "corpora are absent.</li><li>The statistical plan is not frozen.</li></ul>",
    )
    write(
        "citation",
        "Citation",
        "<p>The repository contains CITATION.cff, but the benchmark and paper are not archival "
        "releases and have no DOI. Do not cite pilot outputs as publication findings.</p>",
    )
    write("paper", "Paper", "<p>No evidence-complete manuscript is available.</p>")
    result_links = "".join(
        f"<li><a href='/results/{question['id'].lower()}/'>{question['id']}</a>: "
        f"{question['status']}</li>"
        for question in questions
    )
    write(
        "results",
        "Results",
        "<p class='notice'>No verified benchmark result release is available.</p>"
        f"<h2>Research-question views</h2><ul>{result_links}</ul>",
    )
    write(
        "methodology",
        "Methodology",
        "<p>See the repository methodology and Benchmark 0.5 specification.</p>",
    )
    print("site/index.html")


if __name__ == "__main__":
    main()
