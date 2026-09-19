from __future__ import annotations

import json
from pathlib import Path

import yaml

ALLOWED_STATUSES = {
    "SUPPORTED",
    "PARTIALLY SUPPORTED",
    "NOT SUPPORTED",
    "INCONCLUSIVE",
    "EXPLORATORY ONLY",
}


def main() -> None:
    errors = []
    inventory = json.loads(
        Path("release/metadata/repository_inventory.json").read_text(encoding="utf-8")
    )
    questions = yaml.safe_load(
        Path("research/research_questions.yaml").read_text(encoding="utf-8")
    )["questions"]
    conclusions = yaml.safe_load(Path("research/conclusions.yaml").read_text(encoding="utf-8"))[
        "conclusions"
    ]
    presentation = json.loads(
        Path("site/presentation/presentation-data.json").read_text(encoding="utf-8")
    )
    if len(questions) != len(conclusions):
        errors.append("research-question and conclusion counts differ")
    for question, conclusion in zip(questions, conclusions, strict=True):
        if question["status"] not in ALLOWED_STATUSES:
            errors.append(f"{question['id']}: invalid status")
        if conclusion["question"] != question["id"]:
            errors.append(f"{question['id']}: conclusion mapping differs")
    if inventory["counts"]["publication_eligible_records"] == 0:
        if any(question["status"] != "INCONCLUSIVE" for question in questions):
            errors.append("question status overstates an empty eligible evidence set")
        if presentation["contains_empirical_results"] is not False:
            errors.append("presentation claims empirical results despite empty eligible evidence")
    required_slide_fields = {
        "title",
        "body",
        "notes",
        "method",
        "statistics",
        "evidence",
        "limitations",
        "reviewer",
    }
    for index, slide in enumerate(presentation["slides"], start=1):
        missing = required_slide_fields - slide.keys()
        if missing:
            errors.append(f"slide {index}: missing {sorted(missing)}")
    for route in ["strongest", "uncertain"]:
        if not Path("site/evidence", route, "index.html").is_file():
            errors.append(f"missing evidence route: {route}")
    if errors:
        raise SystemExit("synthesis validation failed:\n- " + "\n- ".join(errors))
    print(
        f"validated synthesis: {len(questions)} inconclusive questions, "
        f"{len(presentation['slides'])} slides"
    )


if __name__ == "__main__":
    main()
