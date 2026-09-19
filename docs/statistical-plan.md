# Statistical analysis plan

This plan must be frozen in Git before confirmatory experiments. Current runs are pilot
and infrastructure validation only.

## Units and estimands

The independent unit is a full training/unlearning run, not an individual test example.
Primary estimands are candidate-to-oracle differences for JS divergence, top-1
disagreement, test utility, forget-set loss-membership audit, and linear CKA distance.
Oracle-to-oracle dispersion is a required reference distribution.

## Pilot and power

The Fashion pilot used one candidate seed and five oracle seeds only to validate the
machinery. It showed that ten dependent oracle pairs cannot support a tail estimate.
The next variance-calibration stage must include repeated candidate and oracle seeds.
Run count will be chosen by confidence-interval precision for a practically meaningful
effect; post-hoc observed power will not be used. The confirmatory alpha, multiplicity
families, interval method, equivalence margins, and stopping rule will be frozen before
confirmatory conditions are inspected.

## Reporting

Report all scheduled seeds, failures, means, standard deviations, 95% confidence
intervals, and standardized plus raw effect sizes. Use paired analyses only where the
randomness is genuinely paired. Control false discovery rate within declared hypothesis
families when multiple comparisons are inferential. Emphasize practical magnitude and
uncertainty rather than binary significance.

## Exclusions

Exclude only corrupt inputs, invariant violations, or failed runs according to a frozen
rule. Preserve failed-run metadata. Never exclude a seed because its result is unfavorable.
