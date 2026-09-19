from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("release/R001"))
    args = parser.parse_args()
    root = args.release.resolve()
    required = [
        "preprint/manuscript-v1.0.pdf",
        "preprint/supplement.pdf",
        "registration/registration-response.md",
        "registration/C001-prospective-protocol.yaml",
        "reproduction/reproduction.yaml",
        "reproduction/P001/manifests/deletion_manifest.json",
        "artifacts/P001-raw/result.json",
        "artifacts/metrics/exact-seed-convergence.json",
        "figures/F002/F002.svg",
        "code/source.tar.gz",
        "SHA256SUMS",
        "release-manifest.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"R001 missing required files: {missing}")
    checked = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file() or sha256(target) != expected:
            raise SystemExit(f"R001 checksum failure: {relative}")
        checked += 1
    if checked == 0:
        raise SystemExit("R001 checksum manifest is empty")
    print(f"R001 RELEASE VERIFICATION: PASS ({checked} payload files)")
    print("Publication identifiers: intentionally unassigned; external publication not performed")


if __name__ == "__main__":
    main()
