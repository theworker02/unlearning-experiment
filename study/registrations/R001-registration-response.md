# R001 retrospective registration response

## Title

Auditable Machine-Unlearning Evaluation — P001 Fashion-MNIST Pilot Study Record

## Registration status

**Retrospective.** P001 was complete before this response was prepared. This document is
not a preregistration and must not be represented as one. Public OSF identifier and DOI:
not assigned.

## Research questions

1. Can an approximate post-deletion model be evaluated against a distribution of models
   independently trained without the deletion set?
2. Is that reference ensemble stable under the P001 seed-count diagnostic?
3. Is the P001 deletion request itself distinguishable from retraining variability under
   the selected output metrics?

## Completed design

- Dataset: reduced Fashion-MNIST pilot split (9,000 train, 1,000 validation, 2,500 test).
- Deletion: one immutable random IID 1% manifest containing 90 training examples.
- Original reference: one model, seed fixed by the archived configuration.
- Exact reference: 15 independent seeds, 2001–2015, trained on the retained complement.
- Approximate methods: fine-tuning, naïve gradient ascent, and SCRUB.
- Pilot metrics: Jensen–Shannon divergence and prediction disagreement.
- Frozen seed diagnostic: relative change from k=10 to k=15 must be at most 10%.

## Exclusion and failure rules

A run is invalid rather than a statistical observation when its dataset, split, deletion,
configuration, checkpoint, prediction, or metric integrity gate fails; when deleted and
retained indices overlap; when an exact retrain includes a deleted index; or when required
metrics are non-finite without a documented cause. Failures remain in the failure registry.

## Analysis

Exact/exact pairwise distances form a descriptive empirical reference distribution. The
frozen convergence diagnostic compares the mean distance estimated with the first ten exact
seeds against all fifteen. Subsequent retrospective separability analysis uses model seed as
the independent unit and is explicitly exploratory.

## Results and deviations

Both frozen convergence diagnostics passed: 4.48% for Jensen–Shannon divergence and 3.14%
for prediction disagreement. Retrospective separability analysis classified both probes as
assay-insensitive. This later analysis was not preregistered and does not support a method-
success conclusion. Consult `research/deviations.md`, `CORRECTIONS.md`, and NR001 for the
complete amendment and negative-result record.

## Relationship to C001

P001 motivates a prospective study using multiple original seeds, multiple deletion
manifests, stronger deletion regimes, expanded metrics, and frozen equivalence and
separability decisions. C001 must be publicly registered before any confirmatory training.
