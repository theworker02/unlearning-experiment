# Equivalence-margin calibration v1

**Status:** prospectively frozen pilot calibration; not publication evidence.

The primary controlled violation leaves 1% of the forbidden class in the training
dataset. Five-percent and ten-percent residual conditions are diagnostics only. Each
replica is summarized by its median JS divergence to all 15 condition-matched
full-deletion exact models. Detection requires that statistic to exceed the exact/exact
95th percentile, and at least 80% of 15 replicas must be detected in both classes.

| Class | Residual class | Candidate q95 | Detected replicas | Detection | Primary | Admissible |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 0 | 1% | 0.031825 | 1/15 | 6.7% | yes | no |
| 0 | 5% | 0.031825 | 0/15 | 0.0% | no | no |
| 0 | 10% | 0.031825 | 5/15 | 33.3% | no | no |
| 5 | 1% | 0.034819 | 0/15 | 0.0% | yes | no |
| 5 | 5% | 0.034819 | 11/15 | 73.3% | no | no |
| 5 | 10% | 0.034819 | 15/15 | 100.0% | no | yes |

## Frozen decision

The 1% controlled violation did not pass in both classes. The numerical margin remains inadmissible, irrespective of the larger diagnostic conditions.

## Limitations

- The operational 1% contamination threshold is pilot-specific, not universal.
- Calibration covers two Fashion-MNIST classes and one reduced CNN recipe.
- A condition-specific q95 is not transferable to other datasets or architectures.
- Passing calibration does not itself demonstrate successful unlearning.

## Evidence

- Frozen plan: `study/preregistration/equivalence_margin_calibration_v1.yaml`
- Reference raw result: `results/raw/phase3-fmnist-class-conditioned-assay-pilot-v3/result.json`
- Calibration raw result: `results/raw/phase3-fmnist-equivalence-margin-calibration-pilot-v1/result.json`
- Generated analysis: `results/processed/equivalence_margin_calibration_v1.json`
