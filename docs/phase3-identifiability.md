# Phase 3: identifiability and strong-unlearning validation

The completed schema-v2 pilot is frozen as `PILOT-FMNIST-IID-DELETE-001`. Phase 3
does not reinterpret that run as publication evidence and does not infer failed
forgetting from its high absolute forget-set accuracy.

The first gate is reference separability: an experiment can evaluate unlearning only
when removing the deletion set produces a detectable original-to-exact change relative
to seed-level exact-to-exact variability. Assay-insensitive conditions remain valid
negative scientific results but receive outcome `U0`, not a pass or failure.

Schema v3 is an append-only analysis envelope referencing immutable schema-v2 training
records. Its decision classes are U0 assay-insensitive, U1 forgetting failure, U2
functional-only equivalence, U3 strong empirical equivalence, and U4 inconclusive.

The preregistration is intentionally marked `DRAFT_NOT_LOCKED`. No confirmatory run may
start until its power-derived threshold, equivalence bounds, multiplicity family,
manifests, and hashes are frozen. Changes after locking require a numbered amendment.

