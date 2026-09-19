# Counterfactual Fidelity Benchmark 0.5

Benchmark 0.5 is a **pilot research suite**, not a frozen publication benchmark. It
standardizes how an approximate unlearning result is compared with a population of models
independently trained on the retained dataset.

## Valid run

A valid run must reference versioned dataset, split, deletion, architecture, method, metric,
and statistical-protocol identifiers. It must declare a research maturity, preserve unique
sample IDs, prove that the retained and forgotten partitions reconstruct the training split,
and verify that exact retraining receives no forgotten IDs. Completed runs are append-only and
must contain hashed checkpoints, immutable predictions with labels and sample IDs, environment
metadata, runtime, configuration hash, and at least two exact-retraining seeds. Confirmatory
runs additionally require every pre-launch freeze flag enforced by the quality gate.

Passing artifact validation does not promote PILOT or VALIDATION evidence to publication
evidence. Benchmark 1.0 will freeze publication splits, deletion sets, primary metrics, and the
statistical protocol only after pilot and seed-convergence studies.

## Counterfactual comparison

For forget set F, an unlearned model U is compared to independently initialized retraining
replicas R_i trained only on D \\ F. Exact/exact pairs define empirical natural retraining
variation. Counterfactual fidelity remains a vector; Benchmark 0.5 does not define a composite
score.

## Extension contract

New datasets implement the dataset adapter, models use the model factory boundary, deletion
strategies emit immutable sample-ID manifests, and unlearning methods implement the method
contract. Extensions may add metrics but cannot redefine an existing metric identifier within
the same benchmark version.

## Status

Digits is CI-only. Fashion-MNIST is pilot-only. CIFAR and tabular entries are planned and may
not be cited as completed evidence.
