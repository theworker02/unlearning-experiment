from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _load_claims() -> list[dict[str, Any]]:
    document = yaml.safe_load(Path("research/claims.yaml").read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("claims"), list):
        raise SystemExit("research/claims.yaml must contain a claims list")
    return document["claims"]


def main() -> None:
    errors: list[str] = []
    claims = _load_claims()
    seen: set[str] = set()
    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        if not claim_id or claim_id in seen:
            errors.append(f"missing or duplicate claim id: {claim_id!r}")
        seen.add(claim_id)
        evidence = claim.get("evidence", {})
        artifacts = evidence.get("result_files", [])
        analyses = evidence.get("statistical_analysis", [])
        if not artifacts or not analyses or claim.get("verification_status") != "verified":
            errors.append(f"{claim_id}: incomplete or unverified evidence mapping")
        for item in [*artifacts, *analyses]:
            path = Path(item)
            if not path.is_file():
                errors.append(f"{claim_id}: missing {path}")
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        for item, expected in evidence.get("sha256", {}).items():
            path = Path(item)
            if not path.is_file():
                errors.append(f"{claim_id}: hash-bound evidence missing {path}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"{claim_id}: evidence hash mismatch {path}")
    if errors:
        raise SystemExit("claim validation failed:\n- " + "\n- ".join(errors))
    print(f"validated {len(claims)} evidence-linked claims")


if __name__ == "__main__":
    main()
