# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "release/R001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: str, destination: Path) -> None:
    path = ROOT / source
    if not path.exists():
        raise SystemExit(f"required release input missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, destination)
    else:
        shutil.copy2(path, destination)


def make_source_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    included = [
        "src",
        "scripts",
        "configs",
        "tests",
        "paper",
        "research",
        "study",
        "figures/F002",
        "experiments/P001",
        "pyproject.toml",
        "Makefile",
        "README.md",
        "LICENSE",
    ]
    excluded_suffixes = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pyc"}

    def clean(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in Path(info.name).parts or Path(info.name).suffix in excluded_suffixes:
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for item in included:
            source = ROOT / item
            if source.exists():
                archive.add(source, arcname=f"unlearning-experiment/{item}", filter=clean)


def main() -> None:
    if DESTINATION.exists():
        raise SystemExit("release/R001 already exists; immutable releases are never overwritten")
    with tempfile.TemporaryDirectory(prefix="r001-build-", dir=ROOT / "release") as temporary:
        stage = Path(temporary) / "R001"
        copy("paper/manuscript.pdf", stage / "preprint/manuscript-v1.0.pdf")
        copy("paper/manuscript.tex", stage / "preprint/manuscript-v1.0.tex")
        copy("paper/manuscript.md", stage / "preprint/manuscript-v1.0.md")
        copy("paper/macros.tex", stage / "preprint/macros.tex")
        copy("paper/references.bib", stage / "preprint/references.bib")
        copy("paper/metadata.yaml", stage / "preprint/metadata.yaml")
        copy("paper/provenance.json", stage / "preprint/provenance.json")
        copy("paper/supplementary/supplement.pdf", stage / "preprint/supplement.pdf")
        copy("paper/tables", stage / "tables")
        copy(
            "study/registrations/R001-registration-response.md",
            stage / "registration/registration-response.md",
        )
        copy(
            "study/registrations/R001-registration-metadata.json",
            stage / "registration/registration-metadata.json",
        )
        copy(
            "study/registrations/C001-prospective-protocol.yaml",
            stage / "registration/C001-prospective-protocol.yaml",
        )
        copy("reproduction.yaml", stage / "reproduction/reproduction.yaml")
        copy("experiments/P001", stage / "reproduction/P001")
        copy("environment", stage / "reproduction/environment")
        copy("REPRODUCIBILITY.md", stage / "reproduction/README.md")
        copy(
            "results/publication/paper-results.json",
            stage / "reproduction/expected-results/paper-results.json",
        )
        copy(
            "results/raw/fashion-cnn-random-01pct-schema2-rocm-pilot-v1",
            stage / "artifacts/P001-raw",
        )
        copy(
            "results/processed/exact_seed_convergence_fashion_pilot.json",
            stage / "artifacts/metrics/exact-seed-convergence.json",
        )
        copy(
            "results/processed/phase3/PILOT-FMNIST-IID-DELETE-001-v3-r2.json",
            stage / "artifacts/metrics/reference-separability.json",
        )
        copy("provenance", stage / "artifacts/provenance")
        copy("figures/F002", stage / "figures/F002")
        for source in (
            "CITATION.cff",
            "codemeta.json",
            "LICENSE",
            "CLAIMS.md",
            "METHODS.md",
            "PROTOCOL.md",
            "HISTORY.md",
            "KNOWN_VARIABILITY.md",
            "CORRECTIONS.md",
        ):
            copy(source, stage / Path(source).name)
        (stage / "AUTHORS.md").write_text(
            "# Authors\n\nMatthew Looney (Magnexis).\n",
            encoding="utf-8",
        )
        (stage / "CONTRIBUTIONS.md").write_text(
            "# Contributions\n\nHuman review is required before assigning a formal contribution taxonomy.\n",
            encoding="utf-8",
        )
        (stage / "DISCLOSURES.md").write_text(
            "# Disclosures\n\nAutomated systems assisted with code, analysis tooling, organization, and drafting. "
            "The human author retains responsibility for all scientific and publication decisions.\n",
            encoding="utf-8",
        )
        (stage / "PUBLICATION_STATUS.md").write_text(
            "# Publication status\n\nThis is a locally frozen release candidate. No OSF, Zenodo, "
            "GitHub release, DOI, or preprint submission has been created.\n",
            encoding="utf-8",
        )
        (stage / "data/README.md").parent.mkdir(parents=True, exist_ok=True)
        (stage / "data/README.md").write_text(
            "# Data\n\nFashion-MNIST is not redistributed here. Deterministic acquisition, split IDs, "
            "derived deletion manifests, hashes, and provenance are included.\n",
            encoding="utf-8",
        )
        make_source_archive(stage / "code/source.tar.gz")
        verification = """#!/usr/bin/env bash
set -euo pipefail
python scripts/verify_r001_release.py --release release/R001
"""
        reproduce = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python scripts/run_experiment.py --config configs/experiments/fashion_cnn_random_01pct_schema2_rocm_pilot.yaml
"""
        (stage / "reproduction/verify.sh").write_text(verification, encoding="utf-8", newline="\n")
        (stage / "reproduction/reproduce.sh").write_text(reproduce, encoding="utf-8", newline="\n")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        sums = "".join(f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in files)
        (stage / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
        manifest = {
            "schema_version": 1,
            "release": "R001",
            "experiment": "P001",
            "immutable": True,
            "file_count_excluding_manifest": len(files),
            "identifiers": {"osf": None, "zenodo": None, "preprint": None},
            "sha256sums_sha256": sha256(stage / "SHA256SUMS"),
        }
        (stage / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.move(stage, DESTINATION)
    print(DESTINATION.relative_to(ROOT))


if __name__ == "__main__":
    main()
