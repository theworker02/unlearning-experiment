# Methods reference

Benchmark protocol: **0.5.0**.

The training operator constructs an architecture from configuration, seeds all supported random
generators, and trains on stable sample IDs. For forget set F, exact replicas are initialized
independently and trained only on R = D \\ F. The experiment asserts set disjointness and retained
partition reconstruction before training.

Approximate methods receive an immutable context containing the original model, retained and
forgotten tensors, method seed, configuration, and device. Fine-tuning and naïve gradient ascent
are simple baselines. SCRUB is a clean-room pilot adaptation whose provenance and deviations are
recorded in the method registry; it is not yet a confirmatory reproduced baseline.

Predictions are stored per model and split with stable IDs, true labels, predicted labels, logits,
probabilities, and selected features. Exact/exact distances establish ordinary retraining variation;
approximate/exact distances are evaluated against that empirical distribution. Accuracy alone is
not treated as equivalence.

The confirmatory launch gate requires frozen dataset, split, deletion, training, statistical, and
seed manifests plus passing tests and declared outputs. `PUBLICATION` and `REPRODUCED` are promotion
states and cannot be requested by a run configuration.
