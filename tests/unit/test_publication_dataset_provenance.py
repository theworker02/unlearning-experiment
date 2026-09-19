from __future__ import annotations

from pathlib import Path

import yaml


def _dataset(name: str) -> dict[str, object]:
    return yaml.safe_load(Path(f"research/datasets/{name}.yaml").read_text())


def test_every_planned_publication_dataset_has_a_canonical_source() -> None:
    for name in ("cifar10", "cifar100", "uci_dry_bean"):
        record = _dataset(name)
        assert record["schema_version"] == 1
        assert str(record["canonical_source"]).startswith("https://")
    for name in ("cifar10", "cifar100"):
        record = _dataset(name)
        assert record["acquisition"]["status"] == "not_acquired"


def test_cifar_redistribution_fails_closed_while_license_is_unresolved() -> None:
    for name in ("cifar10", "cifar100"):
        license_record = _dataset(name)["license"]
        assert license_record["status"] == "not_stated_on_canonical_dataset_page"
        assert license_record["release_action"].startswith("block_redistribution")


def test_dry_bean_doi_and_license_are_explicit() -> None:
    record = _dataset("uci_dry_bean")
    assert record["doi"] == "10.24432/C50S4B"
    assert record["license"]["identifier"] == "CC-BY-4.0"
    assert record["reported_structure"] == {
        "samples": 13611,
        "features": 16,
        "classes": 7,
        "missing_values": False,
    }
    assert record["acquisition"]["status"] == "acquired_2026-09-02"
    assert len(record["acquisition"]["source_archive_sha256"]) == 64
