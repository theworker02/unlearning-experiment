import json
import re
from pathlib import Path

import yaml


def test_citation_metadata_has_no_fake_doi() -> None:
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert "doi" not in citation
    assert citation["authors"]


def test_registry_does_not_call_p001_preregistered() -> None:
    registry = yaml.safe_load(Path("study/registry/experiments.yaml").read_text())
    assert registry["experiments"]["P001"]["preregistered"] is False
    assert registry["experiments"]["C001"]["experiment_started"] is False


def test_zenodo_metadata_is_explicitly_unpublished() -> None:
    metadata = yaml.safe_load(Path("release/zenodo/R001-metadata.yaml").read_text())
    assert metadata["status"] == "DRAFT_NOT_DEPOSITED"
    assert metadata["doi"] is None


def test_p001_archived_indices_are_disjoint() -> None:
    import numpy as np

    forget = np.load("experiments/P001/manifests/deletion_indices.npy", allow_pickle=False)
    retain = np.load("experiments/P001/manifests/retained_indices.npy", allow_pickle=False)
    assert not np.intersect1d(forget, retain).size


def test_required_verified_citations_are_audited() -> None:
    audit = json.loads(Path("research/citation_audit.json").read_text())
    by_key = {entry["key"]: entry for entry in audit["references"]}
    for key in ("sekhari2021remember", "thudi2022auditable", "xiao2017fashion"):
        assert by_key[key]["verification_status"].startswith("verified_primary_source")


def test_every_manuscript_citation_is_audited_and_bibliographed() -> None:
    audit = json.loads(Path("research/citation_audit.json").read_text(encoding="utf-8"))
    audited_keys = {entry["key"] for entry in audit["references"]}
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("paper").glob("*.tex"))
    )
    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", manuscript)
        for key in group.split(",")
        if key.strip()
    }
    bibliography = Path("paper/references.bib").read_text(encoding="utf-8")
    bibliography_keys = set(re.findall(r"@[A-Za-z]+\{\s*([^,\s]+)", bibliography))

    assert cited_keys <= audited_keys
    assert cited_keys <= bibliography_keys
    assert bibliography_keys <= audited_keys


def test_prospective_design_claims_are_not_classified_as_experiments() -> None:
    claims = yaml.safe_load(Path("research/claims.yaml").read_text(encoding="utf-8"))["claims"]
    prospective = [claim for claim in claims if claim["claim_id"] in {"D002-C01", "D002-C02"}]
    assert {claim["classification"] for claim in prospective} == {
        "PROSPECTIVE_DESIGN",
        "PROCESS_STATE",
    }
    assert all(not claim["experiment_ids"] for claim in prospective)
    assert all(claim["evidence"].get("sha256") for claim in prospective)


def test_compute_projection_is_labeled_as_planning_not_benchmark() -> None:
    claims = yaml.safe_load(Path("research/claims.yaml").read_text(encoding="utf-8"))["claims"]
    claim = next(item for item in claims if item["claim_id"] == "D002-C03")
    assert claim["classification"] == "EMPIRICAL_PLANNING_ESTIMATE"
    assert "not a fresh benchmark" in claim["uncertainty"]


def test_hardware_report_distinguishes_historical_validation_from_current_drift() -> None:
    drift = json.loads(
        Path("reports/hardware_environment_drift_2026-09-02.json").read_text(encoding="utf-8")
    )
    assert drift["status"] == "BLOCKED_ENVIRONMENT_DRIFT"
    assert drift["torch_import"]["available"] is False
    assert drift["training_executed"] is False
    report = Path("reports/hardware_validation.md").read_text(encoding="utf-8")
    assert "historical" in report
    assert "current benchmark is therefore blocked" in report
