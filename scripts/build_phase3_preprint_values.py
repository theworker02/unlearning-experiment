from __future__ import annotations

import json
from pathlib import Path

ASSAY = Path("results/processed/phase3_class_conditioned_assay_pilot.json")
MARGIN = Path("results/processed/equivalence_margin_calibration_v1.json")
COMMANDS = Path("paper/generated/phase3_values.tex")
SECTION = Path("paper/generated/phase3_calibration.tex")


def latex_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    assay = json.loads(ASSAY.read_text(encoding="utf-8"))
    margin = json.loads(MARGIN.read_text(encoding="utf-8"))
    full_class = {
        condition["condition_id"]: condition for condition in assay["conditions"]
    }
    class0 = full_class["r1-class0-f1p00"]["metrics"]["js_divergence"]
    class5 = full_class["r1-class5-f1p00"]["metrics"]["js_divergence"]
    primary = {
        record["target_class"]: record for record in margin["records"] if record["primary"]
    }
    commands = {
        "PhaseThreeClassZeroJSRatio": f"{class0['separability_ratio']:.3f}",
        "PhaseThreeClassFiveJSRatio": f"{class5['separability_ratio']:.3f}",
        "PhaseThreeClassZeroMinimum": f"{class0['minimum_detectable_ratio']:.2f}",
        "PhaseThreeClassFiveMinimum": f"{class5['minimum_detectable_ratio']:.2f}",
        "PhaseThreeClassZeroMargin": f"{primary[0]['candidate_margin']:.6f}",
        "PhaseThreeClassFiveMargin": f"{primary[5]['candidate_margin']:.6f}",
        "PhaseThreeClassZeroDetection": (
            f"{100 * primary[0]['calibration']['detection_probability']:.1f}\\%"
        ),
        "PhaseThreeClassFiveDetection": (
            f"{100 * primary[5]['calibration']['detection_probability']:.1f}\\%"
        ),
    }
    COMMANDS.write_text(
        "".join(latex_command(name, value) for name, value in commands.items()),
        encoding="utf-8",
    )

    lines = [
        r"\subsection{Prospective class-conditioned assay calibration}",
        r"A separate prospective pilot used five original seeds and 15 exact-retraining seeds",
        r"per condition, without evaluating approximate methods. Full deletion of Fashion-MNIST",
        r"class 0 and class 5 produced JS-divergence separability ratios of",
        r"\PhaseThreeClassZeroJSRatio{} and \PhaseThreeClassFiveJSRatio{}, exceeding their",
        r"simulated minimum detectable ratios of \PhaseThreeClassZeroMinimum{} and",
        r"\PhaseThreeClassFiveMinimum{}. Half-class deletion and every tested disagreement,",
        r"CKA-distance, and MMD endpoint remained assay-insensitive. A model-level precision",
        r"simulation selected ten exact seeds for JS under the frozen 20\% interval rule in both",
        r"full-class conditions.",
        "",
        r"\subsection{Prospective equivalence-margin calibration}",
        r"A second frozen pilot tested whether the condition-specific exact/exact 95th percentile",
        r"could detect a controlled violation in which 1\% of the forbidden class remained in",
        r"training. The candidate margins were \PhaseThreeClassZeroMargin{} for class 0 and",
        r"\PhaseThreeClassFiveMargin{} for class 5. Detection probabilities were only",
        r"\PhaseThreeClassZeroDetection{} and \PhaseThreeClassFiveDetection{}, respectively,",
        r"below the prospectively required 80\% in both cases. The candidate margins are therefore",
        r"inadmissible. Larger contamination levels were designated diagnostics and cannot rescue",
        r"the failed primary rule. No approximate-method equivalence claim is made.",
    ]
    SECTION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(COMMANDS)
    print(SECTION)


if __name__ == "__main__":
    main()
