from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    version = Path("benchmark/VERSION").read_text(encoding="utf-8").strip()
    files = [
        Path("benchmark/SPECIFICATION.md"),
        Path("benchmark/datasets.yaml"),
        Path("benchmark/architectures.yaml"),
        Path("benchmark/methods.yaml"),
        Path("benchmark/deletion_regimes.yaml"),
        Path("benchmark/metrics.yaml"),
        Path("benchmark/statistical_protocol.yaml"),
    ]
    for path in files[1:]:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise SystemExit(f"unsupported schema in {path}")
    manifest = {
        "schema_version": 1,
        "benchmark_version": version,
        "status": "pilot",
        "files": [{"path": path.as_posix(), "sha256": sha256(path)} for path in files],
    }
    output = Path("benchmark/benchmark_manifest.json")
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
