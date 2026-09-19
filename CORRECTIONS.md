# Scientific correction policy

Corrections are append-only and versioned. A typo with no semantic effect may receive a documented
patch release. Metadata errors require corrected metadata plus retained superseded files. Analysis
errors require a new result version, affected-claim list, regenerated outputs, and an explanation
of whether conclusions changed. A result-changing implementation bug requires affected experiments
to be invalidated in the ledger and rerun under new experiment IDs. A methodological flaw requires
benchmark deprecation or a new protocol version; historical artifacts remain inspectable.

Never replace a released model, dataset, result, figure, or claim in place. Correction records must
identify discovery date, reporter, scope, root cause, affected versions, remediation, checksums, and
whether retraction or withdrawal is warranted.

## 2026-08-31 — Phase 3 pilot separability analysis revision

The first exploratory `PILOT-FMNIST-IID-DELETE-001-v3.json` artifact used dependent
exact/exact pair distances as the resampling input to its power simulation. It remains
preserved. `PILOT-FMNIST-IID-DELETE-001-v3-r2.json` supersedes it and uses one mean
distance summary per independently trained exact-retraining seed. Neither artifact is
confirmatory evidence and no manuscript claim depends on the superseded artifact.
