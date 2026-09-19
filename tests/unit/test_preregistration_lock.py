from __future__ import annotations

from pathlib import Path

import yaml


def test_multiple_testing_family_is_machine_readable_and_nonempty() -> None:
    rules = yaml.safe_load(Path("study/preregistration/decision_rules.yaml").read_text())
    testing = rules["multiple_testing"]
    family = testing["family_definition"]
    assert testing["method"] == "holm"
    assert family["primary_family"]
    assert set(family["secondary_families"]) == {"utility", "privacy", "representation"}
    assert family["gatekeeping"]
    assert testing["active"] is False
    assert testing["status"] == "SUSPENDED_PENDING_REVISED_CONFIRMATORY_FAMILY"


def test_confirmatory_protocol_remains_unfrozen() -> None:
    phase = yaml.safe_load(Path("study/preregistration/phase3.yaml").read_text())
    rules = yaml.safe_load(Path("study/preregistration/decision_rules.yaml").read_text())
    assert phase["status"] == "DRAFT_NOT_LOCKED"
    assert rules["status"] == "DRAFT_NOT_LOCKED"
    assert rules["equivalence"]["bounds"] is None
    assert rules["separability"]["threshold"] is None
