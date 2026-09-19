from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    inventory_path = Path("release/metadata/repository_inventory.json")
    maturity_path = Path("release/metadata/maturity_evaluation.json")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    release_files = [
        Path("benchmark/benchmark_manifest.json"),
        inventory_path,
        maturity_path,
        Path("research/experiment_ledger.csv"),
        Path("model_zoo/model_manifest.csv"),
        Path("EVIDENCE.md"),
        Path("PUBLICATION_READINESS.md"),
    ]
    payload = {
        "schema_version": 1,
        "release_kind": "research_infrastructure_candidate",
        "benchmark_version": inventory["benchmark_version"],
        "assigned_maturity": maturity["assigned_version"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "counts": inventory["counts"],
        "empirical_results_included": False,
        "doi": None,
        "files": [file_record(path) for path in release_files],
    }
    output = Path("release/manifests/release_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    checksums = Path("release/checksums/SHA256SUMS")
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text(f"{checksum}  release/manifests/release_manifest.json\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
