from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml


def main() -> None:
    errors: list[str] = []
    audit = json.loads(Path("research/citation_audit.json").read_text(encoding="utf-8"))
    pending = [
        item.get("key", "unknown")
        for item in audit["references"]
        if not str(item.get("verification_status", "")).startswith("verified_")
    ]
    if pending:
        errors.append(f"unverified citations: {pending}")

    claims = yaml.safe_load(Path("research/claims.yaml").read_text(encoding="utf-8"))["claims"]
    if not claims:
        errors.append("no verified evidence-linked claims")
    if any(claim.get("verification_status") != "verified" for claim in claims):
        errors.append("claim graph contains unverified claims")
    for claim in claims:
        for item, expected in claim.get("evidence", {}).get("sha256", {}).items():
            path = Path(item)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(
                    f"{claim.get('claim_id')}: hash-bound claim evidence mismatch: {path}"
                )

    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("paper").glob("*.tex"))
    )
    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", manuscript)
        for key in group.split(",")
        if key.strip()
    }
    audited_keys = {item.get("key") for item in audit["references"]}
    bibliography = Path("paper/references.bib").read_text(encoding="utf-8")
    bibliography_keys = set(re.findall(r"@[A-Za-z]+\{\s*([^,\s]+)", bibliography))
    if missing_audit := sorted(cited_keys - audited_keys):
        errors.append(f"manuscript citations missing from citation audit: {missing_audit}")
    if missing_bibliography := sorted(cited_keys - bibliography_keys):
        errors.append(f"manuscript citations missing from bibliography: {missing_bibliography}")
    if unaudited_bibliography := sorted(bibliography_keys - audited_keys):
        errors.append(f"bibliography entries missing from citation audit: {unaudited_bibliography}")
    placeholders = re.findall(r"(?i)\b(?:TODO|TBD|FIXME|XX+)\b", manuscript)
    if placeholders:
        errors.append(f"manuscript placeholders remain: {sorted(set(placeholders))}")
    risky = re.findall(
        r"(?i)\b(?:proves|guarantees|universally|definitive|unprecedented|state-of-the-art)\b",
        manuscript,
    )
    if risky:
        errors.append(f"language audit requires human review: {sorted(set(risky))}")
    pattern = r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}"
    for reference in re.findall(pattern, manuscript):
        candidate = Path("paper") / reference
        if not candidate.exists() and not candidate.with_suffix(".tex").exists():
            errors.append(f"missing manuscript dependency: {candidate}")

    if errors:
        raise SystemExit("paper build blocked:\n- " + "\n- ".join(errors))
    print("paper evidence, citation, placeholder, and dependency gates passed")


if __name__ == "__main__":
    main()
