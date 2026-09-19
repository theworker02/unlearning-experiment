# Research FAQ

## Why not use one exact-retrained model?

Stochastic initialization, minibatch order, optimization, and augmentation induce a distribution
of trained models. A single replica confounds unlearning discrepancy with ordinary retraining
variation.

## Why include Digits?

It provides fast deterministic CI and metric validation. It is prohibited as headline or
publication evidence.

## What is counterfactual fidelity?

The multidimensional consistency of an unlearned model with the natural population of models
trained without the forgotten samples. Benchmark 0.5 does not collapse it into one score.

## Does chance-level membership inference prove unlearning?

No. It describes one attack under one construction. The relevant comparison is also with exact
retraining privacy behavior, and failure to distinguish membership is not proof that training-data
influence has been removed.

## Why are there no empirical figures on the site?

No raw record currently passes the stricter evidence gate. Publishing legacy plots would make
invalid evidence appear current.
