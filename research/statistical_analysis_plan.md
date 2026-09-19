# Statistical analysis plan

Version: 0.2-draft  
Status: **DRAFT — NOT FROZEN — CONFIRMATORY EXECUTION BLOCKED**  
Freeze commit: unavailable (repository is not yet a Git worktree)

## Current primary descriptive outcome

For the next exploratory method pilot, the endpoint is the mean across original-model seeds
of each seed's median test-set Jensen–Shannon divergence to ten condition-matched exact
replicas. The original-model seed is the statistical unit. This endpoint is descriptive and
does not define a confirmatory success threshold.

## Hypotheses and decision boundary

The null model is behavioral consistency with models trained without the forgotten samples.
No equivalence hypothesis is currently admissible. Failure to reject a difference will not
be interpreted as equivalence, and an exact-null percentile is a descriptive location rather
than a calibrated PASS/FAIL rule.

## Completed equivalence-margin calibration

The candidate upper margin was the condition-matched exact/exact 95th percentile. Calibration
v1 was executed prospectively for full deletion of Fashion-MNIST classes 0 and 5. The frozen
one-percent residual-class perturbation was detected in only 1/15 class-0 replicas and 0/15
class-5 replicas, below the required 80% in both conditions. The candidate margins are
therefore inadmissible. Larger diagnostic perturbations cannot rescue the failed primary rule.
A future equivalence claim requires a new prospectively justified and independently calibrated
margin.

The class-stratified v2.1 replacement calibration is frozen and completed as validation-only
evidence. It defines an interpretable smallest effect of interest as a deterministic top-1/top-2
probability swap on exactly 1% of the fixed 1,000-example validation probe, allocated as one
sample per ground-truth class. Fifteen exact-model seeds estimated a conservative one-sided
lower bound for the positive-control effect; ten disjoint seeds then had to show at least 80%
exceedance and strictly increasing effects at 0.5%, 1%, and 2% swap severities. Neither test
predictions nor approximate-method outputs were accessed. Both predeclared full-class
conditions passed this calibration, producing condition-specific JS margins of 0.0040597
(class 0) and 0.0031345 (class 5). V1 remains preserved as a failed calibration; the v2.1
margin is not transferable beyond this Fashion-MNIST/small-CNN assay.

## Main-study replication-count decision

The machine-readable procedure in `study/preregistration/main_count_decision_v1.yaml` freezes
candidate grids, minimums, precision requirements, forbidden inputs, and the equivalence gate
before confirmatory method outputs may be inspected. Current evidence supports 15 as the
exact-retraining floor: the pilot precision selection is 10, raised to C001's predeclared
minimum of 15. It does not support an original-model or approximate-candidate count, and the
joint-primary equivalence margin is inadmissible. All three main-profile fields therefore
remain unset; the partial exact-seed recommendation is diagnostic rather than launch authority.

The two missing precision inputs now have separate prospective protocols:
`main_original_seed_precision_v1.yaml` and `main_candidate_seed_precision_v1.yaml`. Both use
validation-only per-seed endpoint summaries, bootstrap independent model seeds, require every
frozen method-by-condition group to pass the 20% relative interval half-width rule, and select
the first adequate count from 5, 8, 10, 15, and 20. Test outcomes, rankings, winner labels,
and confirmatory outputs are structurally rejected. These protocols remain unauthorized drafts;
they define the analysis without claiming that the required twenty-seed pilot exists.

## Exploratory full-class method pilot

The draft plan is `study/preregistration/phase3_distinguishability_method_pilot_v1.yaml`.
It contains five original seeds, ten exact seeds per condition, two full-class conditions,
and three methods, for 30 unlearning runs and no new original or exact training. Reused exact
checkpoints are evaluated on test, retained, and forgotten splits before comparison.

- Statistical unit: original-model seed, not prediction example or exact/exact pair.
- Primary descriptive statistic: seed-specific median candidate/exact test JS divergence,
  averaged across five original seeds.
- Uncertainty: 5,000 deterministic bootstrap resamples of the five seed summaries.
- Exact null: all pairwise distances among ten exact replicas, reported descriptively while
  preserving their dependence.
- Secondary continuous outputs: disagreement, CKA distance, MMD, task/calibration metrics,
  membership AUROC, runtime, and exact-null percentile on all three evaluation splits.
- Decisions: no equivalence label, no confirmatory PASS/FAIL label, and no counted evaluation
  contradiction. Any observed discrepancy remains exploratory.

## Framework awaiting confirmatory freeze

- Confidence level: 95% unless a future frozen endpoint requires another level.
- Primary family: not frozen. The previous proposal crossing JS divergence and top-1
  disagreement across the entire matrix is suspended because disagreement was assay-insensitive
  in the calibrated conditions and the proposed equivalence margin failed.
- Secondary families: utility, privacy, calibration, representation, local behavior, and
  computational cost remain multidimensional. No aggregate score is authorized.
- Bootstrap: resampling must respect training-seed and evaluation-sample dependence. Pairwise
  distances are never resampled as independent observations.
- Effect sizes: absolute and relative differences, empirical percentiles, and standardized
  effects only where assumptions are defensible.
- Exclusions: only frozen integrity failures; performance-based exclusions are forbidden.
- Failed runs: retained in the failure registry and excluded from statistical observations.
- Reference count: ten exact seeds met the frozen 20% precision rule for JS in both calibrated
  full-class pilot conditions. This is a pilot-specific result, not a confirmatory guarantee.
- Stopping: fixed manifest completion, never observed significance or preferred method ranking.

This document requires a versioned Git commit, immutable seed and experiment manifests, a
frozen confirmatory family, and public prospective registration before confirmatory execution.
