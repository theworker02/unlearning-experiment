# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path("output/pdf/counterfactual_fidelity_working_preprint.pdf")
TMP = Path("tmp/pdfs")


def load_evidence() -> dict[str, object]:
    raw = json.loads(
        Path("results/raw/fashion-cnn-random-01pct-schema2-rocm-pilot-v1/result.json").read_text()
    )
    convergence = json.loads(
        Path("results/processed/exact_seed_convergence_fashion_pilot.json").read_text()
    )
    separability = json.loads(
        Path("results/processed/phase3/PILOT-FMNIST-IID-DELETE-001-v3-r2.json").read_text()
    )
    return {"raw": raw, "convergence": convergence, "separability": separability}


def build_plot() -> Drawing:
    rows = list(
        csv.DictReader(
            Path("results/processed/exact_seed_convergence_fashion_pilot.csv").open(
                encoding="utf-8"
            )
        )
    )
    drawing = Drawing(490, 190)
    drawing.add(
        String(245, 176, "P001 exact-retraining convergence", textAnchor="middle", fontSize=11)
    )
    for offset, metric, title in (
        (0, "js_divergence", "Jensen-Shannon divergence"),
        (250, "prediction_disagreement", "Prediction disagreement"),
    ):
        selected = [row for row in rows if row["metric"] == metric]
        chart = LinePlot()
        chart.x = 38 + offset
        chart.y = 30
        chart.width = 190
        chart.height = 115
        chart.data = [[(int(row["population_size"]), float(row["mean"])) for row in selected]]
        chart.lines[0].strokeColor = colors.HexColor("#174ea6")
        chart.lines[0].strokeWidth = 1.5
        chart.lines[0].symbol = None
        chart.xValueAxis.valueMin = 3
        chart.xValueAxis.valueMax = 15
        chart.xValueAxis.valueSteps = [3, 5, 8, 10, 15]
        chart.yValueAxis.valueMin = 0
        chart.yValueAxis.valueMax = max(float(row["mean"]) for row in selected) * 1.25
        drawing.add(chart)
        drawing.add(String(133 + offset, 153, title, textAnchor="middle", fontSize=8.5))
        drawing.add(
            String(133 + offset, 12, "Exact replicas, k", textAnchor="middle", fontSize=7.5)
        )
    return drawing


def page(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.72 * inch, 0.45 * inch, "Working preprint - pilot evidence only")
    canvas.drawRightString(7.78 * inch, 0.45 * inch, str(document.page))
    canvas.restoreState()


def main() -> None:
    evidence = load_evidence()
    raw = evidence["raw"]
    convergence = evidence["convergence"]
    separability = evidence["separability"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PaperTitle", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=10
        )
    )
    styles.add(
        ParagraphStyle(
            name="Status",
            parent=styles["BodyText"],
            borderColor=colors.HexColor("#9b1c1c"),
            borderWidth=0.8,
            borderPadding=8,
            backColor=colors.HexColor("#fff5f5"),
            textColor=colors.HexColor("#701a1a"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AbstractBody",
            parent=styles["BodyText"],
            leftIndent=24,
            rightIndent=24,
            leading=13,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10.5,
            textColor=colors.HexColor("#444444"),
            spaceAfter=10,
        )
    )
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Measuring Counterfactual Fidelity in Machine Unlearning",
        author="Magnexis",
    )
    story = [
        Paragraph("Measuring Counterfactual Fidelity in Machine Unlearning", styles["PaperTitle"]),
        Paragraph(
            "Anonymous working preprint - August 2026",
            ParagraphStyle(
                name="Byline", parent=styles["BodyText"], alignment=TA_CENTER, spaceAfter=14
            ),
        ),
        Paragraph(
            "<b>Working preprint.</b> This manuscript reports infrastructure and exploratory pilot observations only. No confirmatory or successful-unlearning claim has been established.",
            styles["Status"],
        ),
        Paragraph("Abstract", styles["Heading1"]),
        Paragraph(
            "Machine-unlearning evaluations often compare an approximate intervention with one retrained model, although stochastic training produces a population of counterfactual models. We present an evidence-traceable framework that first tests whether a deletion request is identifiable relative to exact-retraining variability, then assesses functional, privacy, representation, utility, and computational fidelity. A Fashion-MNIST pilot validates immutable artifacts and multi-seed reference estimation. Retrospective analysis finds that its one-percent IID deletion is not separable under two output metrics. The result illustrates why proximity to retraining in an insensitive assay is not evidence of removal. Confirmatory method comparisons remain pending.",
            styles["AbstractBody"],
        ),
    ]
    sections = [
        (
            "1. Introduction",
            "Approximate machine unlearning aims to remove selected training influence without complete retraining. The relevant counterfactual is not one parameter vector but the distribution produced by independently training on the retained data. This study asks when ordinary success criteria agree with that counterfactual target. Before judging an approximate method, it asks whether deletion causes a detectable original-to-exact change. If not, the condition is assay-insensitive and supports neither success nor failure.",
        ),
        (
            "2. Related work",
            "Certified removal formalizes closeness to retraining; partitioned and gradient-based systems address deletion cost; deep-learning methods introduce empirical objectives. Sekhari et al. study machine unlearning and generalization. Thudi et al. argue that auditable claims must be made at the algorithmic level. Membership inference is retained as a bounded privacy audit rather than proof of deletion. Fashion-MNIST is treated as a formally cited dataset.",
        ),
        (
            "3. Problem formulation",
            "For randomized training T, dataset D, deletion set F, and unlearning operator U, the reference population is {T_s(D minus F)} over independent seeds s. Counterfactual fidelity compares U(T(D),F) with this population across multiple observable and latent dimensions. Exact-to-exact distances define natural training variation. Original-to-exact distance establishes assay sensitivity.",
        ),
        (
            "4. Evaluation framework",
            "The framework separates functional output behavior, probability divergence, calibration, privacy attacks, intermediate representations, relearning behavior, retained utility, and cost. Pairwise exact distances are not independent observations because trained models recur across pairs. Inference therefore resamples at the trained-model or deletion-manifest level. Equivalence requires intervals inside prospectively frozen bounds; absence of a statistically detected difference is not equivalence.",
        ),
        (
            "5. Experimental design",
            f"P001 uses a deterministic 9,000-example Fashion-MNIST training split, a small CNN, one frozen IID deletion request containing {raw['deletion']['count']} examples, one original seed, and {len(raw['config']['oracle_seeds'])} exact-retraining seeds. AdamW training uses four epochs, batch size 128, learning rate 0.001, and weight decay 0.0001. Phase 3 adds independent original seeds and increasing deletion fractions before approximate methods are tested.",
        ),
    ]
    for heading, body in sections:
        story.extend(
            [
                Paragraph(heading, styles["Heading1"]),
                Paragraph(body, styles["BodyText"]),
                Spacer(1, 8),
            ]
        )
    plot = build_plot()
    story.extend(
        [
            plot,
            Paragraph(
                "Figure 1. Exact-to-exact convergence estimates generated from P001 prediction artifacts. The archived source table also contains 95% replica-bootstrap intervals. This is a pilot seed-count diagnostic, not an equivalence test.",
                styles["Caption"],
            ),
        ]
    )
    stability = convergence["stability"]
    metrics = separability["separability"]["metrics"]
    result_table = Table(
        [
            ["Pilot quantity", "Observed value", "Interpretation"],
            [
                "k=10 to 15 JS change",
                f"{100 * stability['js_divergence']['relative_change_k10_to_k15']:.2f}%",
                "Passed frozen 10% diagnostic",
            ],
            [
                "k=10 to 15 disagreement change",
                f"{100 * stability['prediction_disagreement']['relative_change_k10_to_k15']:.2f}%",
                "Passed frozen 10% diagnostic",
            ],
            [
                "JS separability ratio",
                f"{metrics['js_divergence']['separability_ratio']:.3f}",
                "Below exploratory threshold",
            ],
            [
                "Disagreement separability ratio",
                f"{metrics['prediction_disagreement']['separability_ratio']:.3f}",
                "Below exploratory threshold",
            ],
        ],
        colWidths=[2.15 * inch, 1.35 * inch, 3.0 * inch],
        repeatRows=1,
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aaaaaa")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
            ]
        )
    )
    story.extend(
        [
            Paragraph("6. Pilot results", styles["Heading1"]),
            Paragraph(
                "P001 completed its artifact gate. Its frozen convergence changes are reported below. Retrospective separability analysis classified both evaluated output metrics as insensitive, yielding U0. Approximate-method performance is therefore not interpreted as successful or failed forgetting.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            result_table,
            Spacer(1, 10),
        ]
    )
    story.extend(
        [
            Paragraph("7. Discussion", styles["Heading1"]),
            Paragraph(
                "The pilot exposes an identifiability problem rather than a method ranking. Ordinary IID deletions may leave the learned function nearly unchanged compared with training variation. In that regime, doing nothing and successful removal can both appear close to retraining. Stronger class-conditioned, high-influence, and controlled-memorization conditions are required before a strong empirical unlearning claim can be evaluated.",
                styles["BodyText"],
            ),
            Paragraph("8. Limitations", styles["Heading1"]),
            Paragraph(
                "P001 covers one reduced dataset, one architecture, one deletion manifest, and one original seed. Its separability analysis is retrospective. The membership audit is incomplete, representation metrics are not confirmatory, and empirical equivalence is not a legal deletion certificate. No result generalizes to CIFAR, tabular data, larger architectures, sequential deletion, or other methods.",
                styles["BodyText"],
            ),
            Paragraph("9. Conclusion", styles["Heading1"]),
            Paragraph(
                "The evidence supports the operation of the immutable pilot pipeline and motivates reference separability as a precondition for empirical unlearning claims. Whether approximate methods attain functional, privacy, representation, and utility equivalence remains open for a prospectively registered confirmatory study.",
                styles["BodyText"],
            ),
            PageBreak(),
            Paragraph("References", styles["Heading1"]),
        ]
    )
    references = [
        "[1] A. Sekhari, J. Acharya, G. Kamath, and A. T. Suresh. Remember What You Want to Forget: Algorithms for Machine Unlearning. NeurIPS 34, 2021.",
        "[2] A. Thudi, H. Jia, I. Shumailov, and N. Papernot. On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning. USENIX Security, pp. 4007-4022, 2022.",
        "[3] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747v2, 2017.",
        "[4] C. Guo, T. Goldstein, A. Hannun, and L. van der Maaten. Certified Data Removal from Machine Learning Models. ICML, 2020.",
    ]
    story.extend(Paragraph(item, styles["BodyText"]) for item in references)
    story.extend(
        [
            Spacer(1, 14),
            Paragraph("Artifact statement", styles["Heading1"]),
            Paragraph(
                "All reported numbers are generated from immutable P001 raw predictions and derived JSON artifacts. Research Release R001 and OSF records have not been published; no DOI exists. The human author remains responsible for interpretation, originality, citation accuracy, and venue compliance.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story, onFirstPage=page, onLaterPages=page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
