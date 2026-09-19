from __future__ import annotations

from pathlib import Path

import yaml

from exact_unlearning.methods import METHODS


def test_every_registered_method_has_a_license_audit_entry() -> None:
    audit = yaml.safe_load(Path("research/method_license_audit.yaml").read_text())
    assert set(METHODS) <= set(audit["methods"])
    assert audit["policy"][
        "official_repository_code_must_not_be_redistributed_under_project_mit_license"
    ]


def test_scrub_release_remains_blocked_on_human_provenance_review() -> None:
    audit = yaml.safe_load(Path("research/method_license_audit.yaml").read_text())
    scrub = audit["methods"]["scrub"]
    assert scrub["third_party_source_included"] is False
    assert scrub["official_license_class"] == "noncommercial_academic_software_license"
    assert "human_clean_room_provenance_review" in scrub["unresolved"]
    assert scrub["release_status"].startswith("blocked")


def test_scrub_reproduction_audit_separates_conformance_from_reproduction() -> None:
    record = yaml.safe_load(Path("research/method_reproduction/scrub.yaml").read_text())
    assert record["implementation"]["status"] == (
        "objective_conformance_verified_local_small_cnn_applicability_failed"
    )
    assert record["official_implementation"]["source_comparison_performed"] is False
    assert record["official_implementation"]["behavioral_comparison_performed"] is False
    assert record["documented_deviations"]
    assert record["applicability_finding"]["runtime_policy"].startswith(
        "future SCRUB runs fail closed"
    )
    assert "human_clean_room_provenance_review" in record["release_blockers"]


def test_ssd_audit_verifies_upstream_license_without_claiming_local_eligibility() -> None:
    audit = yaml.safe_load(Path("research/method_license_audit.yaml").read_text())
    ssd = audit["methods"]["ssd"]
    assert ssd["official_license_class"] == "MIT"
    assert ssd["implementation"] == "local_clean_room_equation_4_core_registry_disabled"
    assert ssd["third_party_source_included"] is False
    assert any("without_licenses" in blocker for blocker in ssd["unresolved"])
    assert ssd["release_status"].startswith("blocked")


def test_ssd_reproduction_record_separates_original_method_from_adaptive_extension() -> None:
    record = yaml.safe_load(Path("research/method_reproduction/ssd.yaml").read_text())
    assert record["official_implementation"]["license_class"] == "MIT"
    assert record["official_implementation"]["version_pin"] == (
        "cdfdc0e35c1908e032a6e150d882b0fa17833f85"
    )
    assert record["official_implementation"]["source_comparison_performed"] is True
    assert record["official_implementation"]["behavioral_comparison_performed"] is False
    assert record["official_implementation"]["static_comparison"]["deliberate_local_deviations"]
    assert record["local_implementation"]["status"].startswith("equation_4_core")
    assert record["local_implementation"]["runtime_registry_status"] == "intentionally_disabled"
    assert record["scope_boundary"]["adaptive_extension_status"].startswith("separate")
    assert record["release_blockers"]


def test_ssd_upstream_audit_does_not_overstate_transitive_license_resolution() -> None:
    audit = yaml.safe_load(Path("research/method_license_audit.yaml").read_text())["methods"]["ssd"]
    inventory = audit["upstream_inventory"]
    assert inventory["git_submodules"] == 0
    assert inventory["bundled_model_checkpoints"] == 0
    assert inventory["dependency_manifest_or_lockfile"] == "absent"
    findings = audit["upstream_attribution_findings"]
    assert findings["avalanche"]["license_verified"] is True
    assert findings["bad_teaching_unlearning"]["license_verified"] is True
    assert findings["fast_machine_unlearning"]["license_verified"] is True
    assert findings["pytorch_cifar100"]["license_verified"] is True
    assert findings["fast_machine_unlearning"]["license_class"].startswith("no_license_file")
    assert findings["pytorch_cifar100"]["license_class"].startswith("no_license_file")


def test_method_readiness_registration_flags_match_runtime_registry() -> None:
    readiness = yaml.safe_load(Path("research/method_readiness.yaml").read_text())["methods"]
    declared_registered = {name for name, record in readiness.items() if record["registered"]}
    assert declared_registered == set(METHODS)
