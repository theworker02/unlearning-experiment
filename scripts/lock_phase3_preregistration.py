from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path("study/preregistration")
FILES = (
    "phase3.yaml",
    "hypotheses.md",
    "metrics.yaml",
    "decision_rules.yaml",
    "manifests.sha256",
    "seeds.txt",
)


def canonical_digest() -> tuple[str, dict[str, str]]:
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), hashes


def validate_ready() -> None:
    phase = yaml.safe_load((ROOT / "phase3.yaml").read_text(encoding="utf-8"))
    rules = yaml.safe_load((ROOT / "decision_rules.yaml").read_text(encoding="utf-8"))
    blockers = []
    if phase.get("status") != "FROZEN_BEFORE_CONFIRMATORY_EXECUTION":
        blockers.append("phase3.status")
    if rules.get("status") != "FROZEN_BEFORE_CONFIRMATORY_EXECUTION":
        blockers.append("decision_rules.status")
    if rules["separability"].get("threshold") is None:
        blockers.append("separability.threshold")
    if rules["equivalence"].get("bounds") is None:
        blockers.append("equivalence.bounds")
    family = rules["multiple_testing"].get("family_definition")
    if not isinstance(family, dict) or not family.get("primary_family"):
        blockers.append("multiple_testing.family_definition")
    if blockers:
        raise SystemExit(f"preregistration is not lock-ready: {', '.join(blockers)}")


def main() -> None:
    validate_ready()
    digest, hashes = canonical_digest()
    destination = ROOT / "phase3_preregistration.sha256"
    payload = json.dumps({"sha256": digest, "files": hashes}, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise SystemExit("preregistration is already locked; use a numbered amendment") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(destination)


if __name__ == "__main__":
    main()
