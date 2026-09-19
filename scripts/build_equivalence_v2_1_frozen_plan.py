from __future__ import annotations

import hashlib
import json
from pathlib import Path

if __package__:
    from scripts.build_equivalence_v2_1_plan import build_v2_1_plan
else:
    from build_equivalence_v2_1_plan import build_v2_1_plan


OUTPUT = Path("study/manifests/equivalence_margin_calibration_v2_1_frozen_plan.json")
CHECKSUM = OUTPUT.with_suffix(".sha256")


def main() -> None:
    plan = build_v2_1_plan()
    plan["plan_id"] = "equivalence-margin-calibration-v2.1-frozen-input-plan"
    plan["status"] = "FROZEN_VALIDATION_ONLY"
    plan["execution_authorized"] = True
    plan["notes"].append(
        "Frozen after the recorded class-balanced 1% SESOI decision; validation-only scope."
    )
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"immutable frozen plan differs: {OUTPUT}")
    OUTPUT.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    CHECKSUM.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps(plan["compute_and_storage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
