from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    path = Path("release/manifests/release_manifest.json")
    release = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for item in release["files"]:
        source = Path(item["path"])
        if not source.is_file():
            errors.append(f"missing {source}")
            continue
        if source.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch {source}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"hash mismatch {source}")
    expected = Path("release/checksums/SHA256SUMS").read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append("release manifest checksum mismatch")
    if release["empirical_results_included"]:
        errors.append("v0.5 infrastructure release must not claim empirical result inclusion")
    if errors:
        raise SystemExit("release validation failed:\n- " + "\n- ".join(errors))
    print(f"validated release manifest: v{release['assigned_maturity']}")


if __name__ == "__main__":
    main()
