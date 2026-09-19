"""Generate non-publication SCRUB mechanism-sanity outputs from immutable raw evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAW = Path("results/raw/scrub-fmnist-bn-sanity-v2/result.json")
PROCESSED = Path("results/processed/scrub_fmnist_bn_sanity_v2.json")
REPORT = Path("reports/scrub_fmnist_bn_sanity_v2.md")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    runs = raw["runs"]
    criteria = sorted(runs[0]["acceptance"])
    failed = [run["original_seed"] for run in runs if not run["accepted"]]
    summary = {
        "schema_version": 1,
        "analysis_kind": "development_method_mechanism_sanity",
        "publication_evidence": False,
        "raw_result": {"path": RAW.as_posix(), "sha256": sha256(RAW)},
        "run_count": len(runs),
        "accepted_run_count": sum(bool(run["accepted"]) for run in runs),
        "failed_run_count": len(failed),
        "failed_original_seeds": failed,
        "criterion_pass_counts": {
            criterion: sum(bool(run["acceptance"][criterion]) for run in runs)
            for criterion in criteria
        },
        "overall_acceptance": bool(raw["overall_acceptance"]),
        "decision": "SCRUB_EXCLUDED_FROM_COMPARATIVE_PILOT",
        "reason": (
            "The fixed mechanism acceptance criteria require every seed to pass; one seed did "
            "not decrease the retain objective after its first min epoch."
        ),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if PROCESSED.exists() and PROCESSED.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"immutable processed analysis differs: {PROCESSED}")
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED.write_text(encoded, encoding="utf-8")
    REPORT.write_text(
        "# SCRUB BatchNorm mechanism sanity (v2)\n\n"
        "Status: completed development-method check; not publication evidence.\n\n"
        f"Raw evidence: `{RAW.as_posix()}` (SHA-256 `{summary['raw_result']['sha256']}`).\n\n"
        f"- Runs: {summary['run_count']}\n"
        f"- Fully accepted: {summary['accepted_run_count']}\n"
        f"- Failed seeds: {summary['failed_original_seeds']}\n"
        f"- Overall acceptance: `{summary['overall_acceptance']}`\n\n"
        "## Decision\n\n"
        "SCRUB is excluded from the comparative pilot. The BatchNorm architecture produced a "
        "non-identical initial teacher/student signal in every run, but the fixed retain-objective "
        "recovery criterion failed for one of three seeds. This is a method-validity outcome, not "
        "an equivalence, privacy, or comparative-performance result.\n",
        encoding="utf-8",
    )
    print(PROCESSED)
    print(REPORT)


if __name__ == "__main__":
    main()
