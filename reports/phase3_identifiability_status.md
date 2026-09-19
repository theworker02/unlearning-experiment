# Phase 3 identifiability status

## Frozen pilot interpretation

`PILOT-FMNIST-IID-DELETE-001` is complete and hash-pinned. Its high absolute
forget-set accuracy is not evidence of residual memorization: IID Fashion-MNIST images
remain classifiable by a model that never trained on those particular images.

The retrospective schema-v3 analysis classified the deletion assay as
`U0_ASSAY_INSENSITIVE`. Original-to-exact separability was 0.442 for JS divergence and
0.577 for prediction disagreement. The exploratory 80%-power simulations required
ratios of 1.75 and 1.50 respectively. These simulations use seed-level exact-model
summaries, but the pilot contains only one original model and therefore cannot justify
a confirmatory threshold.

## Implemented foundations

- Append-only schema-v3 envelope validation and writing.
- Seed/model-level bootstrap utilities and bootstrap equivalence decisions.
- Explicit U0–U4 outcome logic.
- Signed reference deltas for accuracy, NLL, Brier, ECE, confidence, and margin.
- Loss, confidence, entropy, and margin MIA audits with AUC, advantage, and low-FPR TPR.
- Layer activation capture, linear CKA support, and Gaussian-kernel MMD.
- Deterministic hard-example ranking and isolated memorization-canary generation.
- Draft fail-closed preregistration and separate exploratory/confirmatory profiles.

## Still required before confirmatory training

- Run a multi-original-seed assay pilot across increasing deletion influence.
- Validate metric-specific k=5/10/15 convergence, including representation and privacy metrics.
- Freeze the power-derived separability threshold and equivalence margins.
- Freeze multiplicity families, R2 rankings, R3 canary specification, and all hashes.
- Validate the memorization precondition and relearning protocol.
- Lock the preregistration; amendments thereafter must be append-only.

No Phase 3 method result, strong-unlearning conclusion, or publication claim is made.

