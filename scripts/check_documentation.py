"""Check key documentation links and release metadata without heavy ML dependencies."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "REPRODUCE.md",
    "REPRODUCIBILITY.md",
    "VERSIONING.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CITATION.md",
    "codemeta.json",
    "LICENSE",
    "LICENSE_TRANSITION_NOTICE.md",
    "benchmark/VERSION",
    "pyproject.toml",
)

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

def main() -> int:
    failures: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            failures.append(f"required file missing: {name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    software = str(tomllib.loads(read("pyproject.toml"))["project"]["version"])
    benchmark = read("benchmark/VERSION").strip()
    module_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', read("src/exact_unlearning/__init__.py"), re.M)
    cff_match = re.search(r"^version:\s*['\"]?([\d.]+)", read("CITATION.cff"), re.M)
    metadata = json.loads(read("codemeta.json"))
    for source, version in (
        ("module", module_match.group(1) if module_match else None),
        ("CITATION.cff", cff_match.group(1) if cff_match else None),
        ("codemeta.json", str(metadata.get("version"))),
    ):
        if version != software:
            failures.append(f"{source} version {version!r} != software {software!r}")
    readme = read("README.md")
    if f"Software source version:** \`{software}\`" not in readme:
        failures.append("README does not state software version")
    if f"benchmark protocol:** \`{benchmark}\`" not in readme.lower():
        failures.append("README does not state benchmark protocol version")
    if "proprietary" not in read("LICENSE").lower():
        failures.append("LICENSE does not identify the proprietary rights status")
    if "license: mit" in read("CITATION.cff").lower():
        failures.append("CITATION.cff incorrectly identifies an MIT license")
    for name in ("README.md", "REPRODUCE.md", "REPRODUCIBILITY.md", "VERSIONING.md", "CITATION.md"):
        body = read(name)
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', body):
            target = target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            normalized = (ROOT / target).resolve()
            if not normalized.is_relative_to(ROOT.resolve()) or not normalized.exists():
                failures.append(f"broken relative link in {name}: {target}")

    if failures:
        for message in failures:
            print(f"FAIL: {message}", file=sys.stderr)
        return 1
    print(f"Documentation checks PASS — software {software}; benchmark {benchmark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
