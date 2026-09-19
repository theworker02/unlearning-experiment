# Scientific changelog

## Benchmark 0.5.0 — infrastructure candidate

- Defined counterfactual evaluation against exact-retraining populations.
- Added explicit experiment maturity and fail-closed confirmatory preconditions.
- Required immutable predictions, logits, labels, sample IDs, checkpoint hashes, and environment
  metadata for new schema-v2 runs.
- Added independently checked metrics and property/fault tests.
- Added canonical pilot random-deletion manifests and schema-versioned dataset packages.
- Re-evaluated six historical raw records. None passes the new artifact gate; no historical record
  was modified.

Methodological implication: historical pilot summaries cannot be promoted to publication evidence.
New pilot runs must use the schema-v2 output contract before seed-convergence analysis.
