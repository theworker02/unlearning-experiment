# Phase 3 class-conditioned assay pilot

**Status:** complete exploratory calibration pilot; not publication evidence.

## Design and integrity

The immutable run `phase3-fmnist-class-conditioned-assay-pilot-v3` trained five independent original models
and fifteen independent exact-retraining models for each of three prospectively
configured class-conditioned deletion sets. The run contains 50 trained models in
total. The central schema-v3 artifact gate passed every integrity check; the gate
correctly marks this PILOT run as ineligible for confirmatory publication claims.
No approximate-unlearning method was evaluated, so method outcomes could not influence
assay selection.

## Assay result

Full deletion of class 0 and class 5 produced an identifiable counterfactual signal
under test-set Jensen--Shannon divergence. The 50% class-0 deletion and every tested
prediction-disagreement or representation metric remained below its prospectively
simulated minimum detectable separation.

| Condition | Deleted | Metric | Ratio | Minimum detectable | Exact/exact q95 | Sensitive | k selected |
| --- | ---: | --- | ---: | ---: | ---: | :---: | ---: |
| r1-class0-f0p50 | 450 (50%) | js_divergence | 0.509 | 1.45 | 0.0366 | no | 10 |
| r1-class0-f0p50 | 450 (50%) | prediction_disagreement | 0.597 | 1.40 | 0.1795 | no | 10 |
| r1-class0-f0p50 | 450 (50%) | final_feature_cka_distance | 0.661 | 1.55 | 0.0349 | no | 10 |
| r1-class0-f0p50 | 450 (50%) | final_feature_mmd | 0.848 | 1.15 | 0.7325 | no | 5 |
| r1-class0-f1p00 | 900 (100%) | js_divergence | 1.958 | 1.45 | 0.0318 | yes | 10 |
| r1-class0-f1p00 | 900 (100%) | prediction_disagreement | 1.180 | 1.40 | 0.1481 | no | 10 |
| r1-class0-f1p00 | 900 (100%) | final_feature_cka_distance | 0.871 | 1.40 | 0.0267 | no | 10 |
| r1-class0-f1p00 | 900 (100%) | final_feature_mmd | 0.793 | 1.15 | 0.8050 | no | 5 |
| r1-class5-f1p00 | 900 (100%) | js_divergence | 2.275 | 1.60 | 0.0348 | yes | 10 |
| r1-class5-f1p00 | 900 (100%) | prediction_disagreement | 1.194 | 1.35 | 0.1628 | no | 10 |
| r1-class5-f1p00 | 900 (100%) | final_feature_cka_distance | 0.774 | 1.50 | 0.0388 | no | 10 |
| r1-class5-f1p00 | 900 (100%) | final_feature_mmd | 0.782 | 1.20 | 0.8137 | no | 5 |

## Reference-ensemble precision

Precision was recalculated from model-level exact/exact matrices. Each simulation
draws independently trained model seeds without replacement; it does not resample
dependent model pairs as independent observations. A selected k is the first of
5, 10, or 15 whose 95% interval has at most 20% relative half-width.

### r1-class0-f0p50 — JS divergence

| k | Median pair mean | 95% interval | Relative half-width | Pass |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0.022082 | [0.014990, 0.028911] | 31.5% | no |
| 10 | 0.022501 | [0.018673, 0.025068] | 14.2% | yes |
| 15 | 0.022219 | [0.022219, 0.022219] | 0.0% | yes |

### r1-class0-f1p00 — JS divergence

| k | Median pair mean | 95% interval | Relative half-width | Pass |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0.017504 | [0.012752, 0.023902] | 31.9% | no |
| 10 | 0.017712 | [0.015190, 0.019986] | 13.5% | yes |
| 15 | 0.017700 | [0.017700, 0.017700] | 0.0% | yes |

### r1-class5-f1p00 — JS divergence

| k | Median pair mean | 95% interval | Relative half-width | Pass |
| ---: | ---: | ---: | ---: | :---: |
| 5 | 0.020768 | [0.014414, 0.029109] | 35.4% | no |
| 10 | 0.020932 | [0.017273, 0.023897] | 15.8% | yes |
| 15 | 0.020888 | [0.020888, 0.020888] | 0.0% | yes |

## Decision

JS divergence advances to controlled equivalence-margin calibration for full-class
deletion. The broad multimetric confirmatory protocol remains unfrozen. No
approximate-method comparison may be interpreted confirmatorily until the numerical
margin is calibrated prospectively and its detection requirement passes.

## Failure history

Two earlier attempts are retained rather than erased: v1 exhausted memory after
eagerly loading large intermediate activations, and v2 encountered an immutable
artifact-name collision. Both are recorded in `results/failures/failures.jsonl`;
neither contributes observations to this analysis.

## Limitations

- Fashion-MNIST is a pilot dataset and not sufficient publication evidence.
- Only one reduced CNN training recipe and two deleted classes were tested.
- Assay sensitivity does not establish an equivalence margin or successful unlearning.
- No approximate-unlearning method was evaluated in this calibration run.

## Evidence

- Raw result: `results/raw/phase3-fmnist-class-conditioned-assay-pilot-v3/result.json`
- Generated analysis: `results/processed/phase3_class_conditioned_assay_pilot.json`
- Configuration: `configs/experiments/phase3_class_conditioned_assay_pilot.yaml`
- Failure registry: `results/failures/failures.jsonl`
