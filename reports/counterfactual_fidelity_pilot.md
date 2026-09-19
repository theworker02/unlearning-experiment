# Counterfactual Fidelity Pilot

> **Pilot evidence only — not publication evidence or a confirmatory result.**

This report is generated from immutable Fashion-MNIST artifacts listed in the processed summary. The pilot uses a deterministic 10,000-example training pool (9,000 train, 1,000 validation), 2,500 test examples, a small CNN, one original seed, five independent exact retraining seeds, and three unlearning methods. SCRUB is the paper-derived core max/min objective without privacy rewinding; its Fashion-MNIST settings are a declared adaptation.

## Exact-retraining variability

| Condition | Oracle accuracy range | JS exact/exact mean (p95) | Top-1 disagreement mean (p95) | CKA distance mean (p95) | Mean oracle train s | Total wall s |
|---|---:|---:|---:|---:|---:|---:|
| random 1.0% | 0.8304–0.8492 | 0.0188 (0.0284) | 0.0942 (0.1199) | 0.0206 (0.0280) | 2.43 | 60.5 |
| random 5.0% | 0.8292–0.8456 | 0.0180 (0.0296) | 0.0936 (0.1280) | 0.0204 (0.0265) | 3.86 | 97.0 |
| high_loss 5.0% | 0.8228–0.8476 | 0.0243 (0.0381) | 0.0958 (0.1274) | 0.0196 (0.0279) | 3.94 | 98.8 |

## Approximate-to-exact diagnostics

| Condition | Method | Test acc | Forget acc | Retain acc | MIA AUC | JS / exact mean | Disagreement | CKA distance | Counterfactual envelope | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| random 1.0% | fine_tune | 0.8472 | 0.8444 | 0.8677 | 0.4993 | 0.606 | 0.0713 | 0.0234 | within | 4.04× |
| random 1.0% | naive_gradient_ascent | 0.8508 | 0.8222 | 0.8661 | 0.4925 | 0.639 | 0.0734 | 0.0225 | within | 2.44× |
| random 1.0% | scrub | 0.8488 | 0.8222 | 0.8692 | 0.4951 | 0.700 | 0.0804 | 0.0230 | within | 1.42× |
| random 5.0% | fine_tune | 0.8404 | 0.8911 | 0.8616 | 0.5221 | 0.781 | 0.0862 | 0.0224 | within | 2.86× |
| random 5.0% | naive_gradient_ascent | 0.8528 | 0.8933 | 0.8646 | 0.5225 | 0.652 | 0.0744 | 0.0215 | within | 1.60× |
| random 5.0% | scrub | 0.8436 | 0.8889 | 0.8654 | 0.5215 | 0.656 | 0.0758 | 0.0206 | within | 1.08× |
| high_loss 5.0% | fine_tune | 0.8396 | 0.0000 | 0.9096 | 0.0365 | 0.726 | 0.0869 | 0.0234 | outside | 2.31× |
| high_loss 5.0% | naive_gradient_ascent | 0.8332 | 0.0000 | 0.9039 | 0.0340 | 0.882 | 0.0878 | 0.0236 | outside | 1.41× |
| high_loss 5.0% | scrub | 0.8404 | 0.0000 | 0.9088 | 0.0347 | 0.758 | 0.0819 | 0.0221 | outside | 0.97× |

## Descriptive contradiction matrix

‘Within’ means the candidate mean did not exceed the empirical exact/exact 95th percentile for the declared diagnostic. With only ten exact/exact pairs, this is not an equivalence test and must not be read as statistical acceptance of equivalence.

| Condition | Method | Forget | Retain | Privacy | Calibration | Prediction | Representation | Overall | Contradiction |
|---|---|---|---|---|---|---|---|---|---|
| random 1.0% | fine_tune | within | within | within | within | within | within | within | not observed |
| random 1.0% | naive_gradient_ascent | within | within | within | within | within | within | within | not observed |
| random 1.0% | scrub | within | within | within | within | within | within | within | not observed |
| random 5.0% | fine_tune | within | within | outside | within | within | within | within | not observed |
| random 5.0% | naive_gradient_ascent | within | within | outside | within | within | within | within | not observed |
| random 5.0% | scrub | within | within | within | within | within | within | within | not observed |
| high_loss 5.0% | fine_tune | within | within | within | outside | within | within | outside | observed |
| high_loss 5.0% | naive_gradient_ascent | within | within | within | outside | within | within | outside | observed |
| high_loss 5.0% | scrub | within | within | within | outside | within | within | outside | observed |

## Answers to pilot questions

1. **Can exact-retraining variability be measured reliably?** The pipeline measures it reproducibly and the five replicas produce nonzero exact/exact distributions. Five seeds (ten pairs) are too few to estimate a reliable 95th-percentile tail; Tier-2 needs prospective precision/power calculations and preferably more independent replicas.
2. **Are approximate/exact differences larger than exact/exact differences?** All methods remained inside the descriptive prediction and representation envelopes. All three high-loss methods fell outside the calibration envelope. These exploratory flags are not formal evidence of difference or equivalence.
3. **Do conventional and counterfactual metrics disagree?** The post-run exploratory matrix found 3 surface-pass/counterfactual-fail cells, all from high-loss deletion and calibration. The composite matrix rule was not frozen before the pilot and cannot support a confirmatory claim. Additionally, the high-loss condition makes the loss-based MIA AUC far from 0.5 even for exact retraining, demonstrating that a naïve near-chance privacy rule is confounded by deletion composition.
4. **Is variance manageable?** Runtime variance is small, but behavioral exact/exact variance is material enough that single-oracle comparisons would be unsafe. The full study should model seed and condition effects rather than normalize by one noisy mean.
5. **Which metrics appear informative?** JS divergence, top-1 disagreement, CKA distance, calibration gaps, and exact-referenced MIA are all operational. The MIA result shows why raw attack AUC is not interpretable without the exact reference. No scalar aggregate is justified.
6. **What compute would the full study require?** This pilot's per-condition wall times are reported above. Linear extrapolation is inappropriate for CIFAR/ResNet; Tier-2 must benchmark one full condition on target hardware, then budget the frozen manifest.
7. **Are assumptions clearly wrong?** Treating exact retraining as one deterministic point, treating MIA≈0.5 as a universal privacy criterion, and treating an empirical p95 from ten dependent pairs as a formal equivalence boundary are all untenable.
8. **Is the question still worthwhile?** Yes as a falsifiable measurement study: the null pilot did not manufacture a contradiction, while it exposed reference variance and a privacy-metric composition confound that the publication design must control.

## Before Tier-2

Freeze equivalence margins and a hierarchical/paired analysis before final runs; benchmark CIFAR-10/ResNet-18 hardware; add LiRA-style and calibrated loss attacks; add local-neighbor and corruption probes; execute repeated candidate seeds; validate SCRUB hyperparameters without selecting on final test outcomes; and implement sequential stage-matched oracles. The current repository is not ready for confirmatory claims.
