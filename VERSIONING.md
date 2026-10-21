# Versioning and releases

This repository separates the **software/tooling version** from the **scientific benchmark protocol version**. Both use SemVer-like `MAJOR.MINOR.PATCH` identifiers, but they answer different questions.

| Identifier | Source of truth | Current value | Meaning |
| --- | --- | --- | --- |
| Software package | `pyproject.toml` and `src/exact_unlearning/__init__.py` | `0.5.1` | Installable code and documentation |
| Machine-readable metadata | `CITATION.cff`, `codemeta.json` | `0.5.1` | Cite the source snapshot; must match package |
| Scientific benchmark | `benchmark/VERSION` | `0.5.0` | Measurement/protocol contract; changes only after methodological review |

## SemVer policy for software

- **PATCH** (`0.5.0 → 0.5.1`): backwards-compatible documentation, metadata, correctness or test fixes that do not redefine scientific results.
- **MINOR** (`0.5.x → 0.6.0`): backwards-compatible features, datasets, methods, metrics or APIs. Before 1.0, a breaking experimental interface change may require a minor bump with migration notes.
- **MAJOR**: stable-interface incompatible releases after 1.0, or an intentionally new major software generation.

Breaking evidence formats, identifier semantics, comparison methods, preregistered hypotheses, or acceptance rules **also** require a separately reviewed benchmark protocol change. A software bump by itself never promotes pilots to publication evidence.

## Release checklist

1. Confirm legal rights and scientific scope; evaluate license changes separately and update `LICENSE_TRANSITION_NOTICE.md` if applicable.
2. Review experimental integrity, provenance manifests, claims and quality gates. Do not re-label legacy data.
3. Update `pyproject.toml`, `src/exact_unlearning/__init__.py`, `CITATION.cff`, `codemeta.json`, and `CHANGELOG.md` for the same software version.
4. Run `python scripts/check_documentation.py`, `python -m ruff check .`, `python -m pytest`, and required scientific validation on an appropriate environment. Record exact versions/hashes and known failures. A documentation-only CI pass is not an experiment pass.
5. Commit the versioned source. If issuing a public release, create an **annotated tag** (example `git tag -a v0.5.1 -m "Software 0.5.1"`) on the **verified commit** and push it. Then publish release notes tied to that exact tag.
6. Archive software/data artifacts *only* when actual payloads and digests exist and license/permissions permit; record their separate persistent identifiers and version-specific provenance.
7. Update the README badges and citation metadata if a public release or verified DOI changes status.

**Current status:** `0.5.1` identifies source on the repository branch; this task does not create a Git tag, a GitHub Release, an archival software DOI, or a new scientific data package. The [pilot preprint DOI](https://doi.org/10.5281/zenodo.23540870) is a separate research publication identifier.

## GitHub About text and topics

The GitHub repository's About **description** and **topics** are maintained in GitHub settings, not through files committed into the repository. Proposed strings are in [`.github/REPOSITORY_METADATA.md`](.github/REPOSITORY_METADATA.md). Do not claim those GitHub settings have changed unless verified from GitHub.

## Corrections

For software changes, update `CHANGELOG.md`. For changed scientific assumptions, results, metric definitions, or evidence lineage, also update `SCIENTIFIC_CHANGELOG.md`, relevant protocol version, and [CORRECTIONS.md](CORRECTIONS.md) as needed. Never modify an immutable raw result just to make newer conclusions appear supported.
