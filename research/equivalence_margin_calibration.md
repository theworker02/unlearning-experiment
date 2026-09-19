# Prospective equivalence-margin calibration

Status: **V1 EXECUTED — CANDIDATE NUMERICAL MARGINS NOT ADMISSIBLE**

For every non-negative counterfactual distance, construct the condition-matched exact/exact
reference distribution without using approximate-method outputs. Its prespecified 95th
percentile is the candidate upper equivalence margin; the lower bound is zero.

Before generating controlled perturbations, define in natural units the smallest change that
would matter scientifically for that endpoint. Generate independent perturbed-model replicas
under frozen seeds. The candidate margin is admissible only when at least 80% of the resulting
distances exceed it. A failed calibration marks the metric or proposed margin insensitive; it
does not justify widening the margin after approximate results are known.

The executable implementation is
`exact_unlearning.phase3.calibrate_distance_equivalence_margin`. Any replacement numerical
margin remains blocked until a new perturbation mechanism, seed policy, and metric-specific
meaningful change are frozen and executed prospectively.

## Calibration v1 result

The frozen v1 pilot used full-class Fashion-MNIST deletion as the reference and left 1% of
the forbidden class in the training dataset as the primary controlled violation. Fifteen
independently trained perturbation replicas were compared with fifteen condition-matched exact
full-deletion replicas. The primary detection probabilities were 1/15 for class 0 and 0/15
for class 5, below the frozen 80% requirement. The exact/exact q95 candidates are therefore
not admissible equivalence margins. Larger 5% and 10% contamination conditions were diagnostic
only and cannot rescue the failed primary rule.

The immutable raw evidence is
`results/raw/phase3-fmnist-equivalence-margin-calibration-pilot-v1/result.json`; the generated
decision record is `results/processed/equivalence_margin_calibration_v1.json`. No approximate
method output was used in this calibration.
