from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def verify_checksum_file(path: Path) -> list[str]:
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        artifact = Path(name) if "/" in name else path.parent / name
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.is_file() else None
        if actual != expected:
            failures.append(name)
    return failures


def main() -> None:
    checks = {
        "artifact hashes": verify_checksum_file(Path("provenance/artifacts.sha256")),
        "manifest hashes": verify_checksum_file(Path("provenance/manifests.sha256")),
        "figure hashes": verify_checksum_file(Path("provenance/figures.sha256")),
    }
    forget = np.load("experiments/P001/manifests/deletion_indices.npy", allow_pickle=False)
    retain = np.load("experiments/P001/manifests/retained_indices.npy", allow_pickle=False)
    if np.intersect1d(forget, retain).size:
        checks["partition disjoint"] = ["overlap"]
    else:
        checks["partition disjoint"] = []
    for label, failures in checks.items():
        print(f"{label}: {'PASS' if not failures else 'FAIL'}")
    if any(checks.values()):
        raise SystemExit("P001 archive verification failed")
    print("SCIENTIFIC ARTIFACT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
