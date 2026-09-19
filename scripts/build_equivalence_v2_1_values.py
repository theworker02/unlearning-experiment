from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("results/processed/equivalence_margin_calibration_v2_1.json")
OUTPUT = Path("paper/generated/equivalence_v2_1_values.tex")


def main() -> None:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    if result.get("joint_primary_admissible") is not True:
        raise SystemExit(
            "v2.1 calibration did not jointly pass; no positive-margin manuscript values"
        )
    values = {item["condition_id"]: item["calibration"] for item in result["conditions"]}
    required = {"r1-class0-f1p00", "r1-class5-f1p00"}
    if set(values) != required:
        raise SystemExit("v2.1 conditions differ from the manuscript calibration scope")
    payload = "\n".join(
        [
            "% Generated from immutable v2.1 validation-only calibration output.",
            (
                "\\newcommand{\\VTwoOneClassZeroMargin}"
                f"{{{values['r1-class0-f1p00']['candidate_margin']:.7f}}}"
            ),
            (
                "\\newcommand{\\VTwoOneClassFiveMargin}"
                f"{{{values['r1-class5-f1p00']['candidate_margin']:.7f}}}"
            ),
            (
                "\\newcommand{\\VTwoOneHoldoutRate}"
                f"{{{100 * values['r1-class0-f1p00']['heldout_primary_exceedance_probability']:.0f}\\%}}"
            ),
            "",
        ]
    )
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"generated manuscript values differ: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
