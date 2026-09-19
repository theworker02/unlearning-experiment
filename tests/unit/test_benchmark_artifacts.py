from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def test_benchmark_registry_identifiers_are_unique() -> None:
    registries = {
        "datasets.yaml": "datasets",
        "architectures.yaml": "architectures",
        "methods.yaml": "methods",
        "deletion_regimes.yaml": "regimes",
        "metrics.yaml": "metrics",
    }
    for filename, key in registries.items():
        document = yaml.safe_load(Path("benchmark", filename).read_text(encoding="utf-8"))
        identifiers = [entry["id"] for entry in document[key]]
        assert len(identifiers) == len(set(identifiers))


def test_benchmark_manifest_matches_registered_files() -> None:
    path = Path("benchmark/benchmark_manifest.json")
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        source = Path(item["path"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]


def test_deletion_manifests_hash_exact_sample_ids() -> None:
    for path in Path("deletion_manifests").glob("**/*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        encoded = json.dumps(manifest["sample_ids"], separators=(",", ":"))
        assert hashlib.sha256(encoded.encode()).hexdigest() == manifest["sample_ids_sha256"]
