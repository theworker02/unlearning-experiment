# Contributing

Do not add manually typed research results. Add or update tests with behavioral changes,
preserve failed-run metadata, and document algorithm sources and deviations. New datasets
require provenance, license, version, checksum, deterministic preprocessing, and stable
example IDs. New methods require a verified source and must not be labeled faithful until
their implementation has been checked against the publication and, where available,
licensed reference code.

## Contribution classes

### Software contribution

Include focused tests, preserve schemas or supply a migration, and do not change scientific
semantics incidentally.

### Scientific-method contribution

State the hypothesis or methodological change, rationale, affected protocol versions, validation
plan, and whether existing experiments require rerunning. Record the decision and deviation.

### New benchmark method

Add the method registry entry, citation audit record, implementation provenance and license,
configuration schema, scientific sanity/reference check, and at least one immutable pilot run.
Simple baselines must be labeled as such.

### New dataset

Provide the source and license audit, deterministic loader, stable IDs, checksums, dataset card,
split manifest, leakage/duplicate validation, and reconstruction command before experiments.

### Reproduction report

Use `reviews/reproduction/REPORT_TEMPLATE.md`. Attach configuration hashes and raw machine-readable
outputs. A discrepancy is valuable evidence and should not be rewritten as agreement.

### Scientific correction

Follow `CORRECTIONS.md`. Never overwrite released evidence or reuse a changed model ID.
