# Study description

## Research question

When do conventional machine-unlearning metrics indicate success while an approximately
unlearned model remains distinguishable from models trained without the deleted data?

## Definitions and experimental units

Exact retraining trains independently on the retained-data complement. Approximate
unlearning modifies a previously trained model. A stochastic trained model—not a
pairwise model comparison—is the independent experimental unit. Exact/exact pairs are
dependent because models recur across pairs.

Counterfactual fidelity is multidimensional: functional predictions, probability
distributions, calibration, privacy audits, representations, relearning behavior, and
retained utility are not collapsed into one score.

## Data, models, and deletion mechanisms

Digits is infrastructure-only. P001 uses a documented subset of Fashion-MNIST and a
small CNN. Phase 3 covers IID, class-conditioned, hard-example, and isolated synthetic
memorization regimes. Each deletion request is identified by exact sample IDs and hashes.

## Reference, methods, and metrics

The reference is a distribution of independent retain-only retrainings. Simple baselines
remain explicitly labeled. Metrics include prediction disagreement, JS divergence, NLL,
Brier, ECE, signed forget/retain/test deltas, membership audits, CKA, MMD, and cost.

## Statistical design and quality gates

Reference separability is evaluated before approximate-method fidelity. Model/seed-level
and hierarchical bootstraps preserve the experimental unit. Equivalence requires a
confidence interval inside prospectively frozen bounds; failure to reject difference is
not equivalence. Outcomes are U0 assay-insensitive, U1 failure, U2 functional-only, U3
strong empirical equivalence, or U4 inconclusive.

## Provenance and evidence status

Raw runs are immutable. Derived artifacts preserve source hashes. P001 is an exploratory
pilot completed before any external registration. Its retrospective registration package
must never be described as preregistration. No confirmatory study or verified headline
claim currently exists.

## Reproduction, archives, and citation

See `PROTOCOL.md`, `REPRODUCIBILITY.md`, `CLAIMS.md`, and `CITATION.md`. Zenodo R001 and
OSF R001/C001 identifiers remain pending; no DOI has been assigned.

