# Defense-style technical summary

## Why is exact retraining the reference?

It operationalizes the counterfactual in which forgotten samples were absent from training. It is
not a unique model because training is stochastic; the reference must be a population.

## Why multiple exact models?

They estimate ordinary variation due to initialization, minibatch order, optimization, and other
controlled randomness. The required population size remains unestablished and must be justified by
a convergence study.

## How do you know the metrics are valid?

Production accuracy, F1, NLL, Brier, Jensen–Shannon divergence, and total variation are checked
against an independent implementation over 50 deterministic randomized fixtures. Properties and
synthetic cases test symmetry, bounds, ordering invariance, and known behavior. This validates those
implementations, not the overall construct validity of counterfactual fidelity.

## How were methods tuned fairly?

They have not yet undergone a complete confirmatory fairness audit. Fine-tuning and naïve gradient
ascent are explicitly simple baselines; SCRUB is a pilot clean-room adaptation. This currently
blocks method-ranking claims.

## How many seeds are enough?

Unknown. Five configured legacy seeds are not accepted as a universal answer. The planned
convergence analysis must evaluate stability as k increases.

## What is the strongest alternative explanation?

Any future approximate/exact discrepancy could reflect unfair optimization, architecture mismatch,
implementation error, or an unstable exact-reference estimate rather than failed unlearning.

## Which result is most robust? Which is weakest?

No empirical result is eligible. The strongest current finding is procedural: all historical runs
fail the new evidence gate. It says nothing about method performance.

## What would falsify the central hypothesis?

Well-powered, fairly tuned approximate methods consistently falling within preregistered
exact-retraining equivalence envelopes across datasets, architectures, deletions, and fidelity
dimensions would undermine the proposed metric-disagreement concern.

## What cannot be generalized?

Nothing about real-world deletion guarantees, privacy compliance, CIFAR-scale behavior, tabular
data, sequential drift, or method ranking can be inferred from the current eligible evidence.
