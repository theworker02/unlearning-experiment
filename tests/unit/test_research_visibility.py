from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_current_release_maturity_is_not_overstated() -> None:
    maturity = json.loads(
        Path("release/metadata/maturity_evaluation.json").read_text(encoding="utf-8")
    )
    assert maturity["assigned_version"] == "0.5.0"
    assert maturity["versions"]["0.5.0"]["passed"]
    assert not maturity["versions"]["1.0.0"]["passed"]
    assert not maturity["versions"]["2.3.0"]["passed"]


def test_site_reports_no_verified_results() -> None:
    homepage = Path("site/index.html").read_text(encoding="utf-8")
    evidence = Path("site/evidence/index.html").read_text(encoding="utf-8")
    assert "Research Infrastructure Candidate v0.5.0" in homepage
    assert "No experiment is publication-eligible" in homepage
    assert "No claims are registered" in evidence


def test_generated_site_routes_exist() -> None:
    routes = [
        "benchmark",
        "datasets",
        "models",
        "experiments",
        "figures",
        "evidence",
        "reproduce",
        "methodology",
        "limitations",
        "citation",
        "paper",
        "results",
    ]
    assert all(Path("site", route, "index.html").is_file() for route in routes)


def test_conceptual_figure_cannot_be_mistaken_for_empirical_evidence() -> None:
    metadata = yaml.safe_load(Path("visualizations/metadata/F001.yaml").read_text(encoding="utf-8"))
    assert metadata["type"] == "conceptual"
    assert metadata["contains_empirical_data"] is False
    assert metadata["source_experiments"] == []
