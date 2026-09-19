from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_all_research_questions_are_inconclusive_without_eligible_evidence() -> None:
    inventory = json.loads(
        Path("release/metadata/repository_inventory.json").read_text(encoding="utf-8")
    )
    questions = yaml.safe_load(
        Path("research/research_questions.yaml").read_text(encoding="utf-8")
    )["questions"]
    assert inventory["counts"]["publication_eligible_records"] == 0
    assert questions
    assert all(question["status"] == "INCONCLUSIVE" for question in questions)
    assert any(question["experiments"] for question in questions)


def test_presentation_is_nonempirical_and_has_technical_notes() -> None:
    presentation = json.loads(
        Path("site/presentation/presentation-data.json").read_text(encoding="utf-8")
    )
    assert presentation["contains_empirical_results"] is False
    assert 8 <= len(presentation["slides"]) < 18
    assert all(slide["notes"] and slide["limitations"] for slide in presentation["slides"])
