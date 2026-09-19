# Schema-v2 Fashion-MNIST counterfactual-fidelity pilot

> **PILOT evidence — not publication evidence and not a confirmatory test.**

Experiment: `fashion-cnn-random-01pct-schema2-rocm-pilot-v1`. The run used the frozen 1% random deletion manifest
and 15 exact-retraining seeds specified before execution.

## Infrastructure outcome

- Artifact quality gate: PASS.
- Frozen deletion manifest binding and hash validation: PASS.
- Forgotten/retained disjointness and exact-retraining exclusion: PASS.
- Immutable checkpoints and prediction artifacts with IDs, labels, logits, and probabilities: PASS.
- Finite metric and probability validation: PASS.
- Failed runs in this condition: none.

## Observed runtime

- Device: `cuda` through ROCm `7.2.53211-e1a6bc5663`; CUDA runtime: none.
- Original training: 9.593 seconds.
- Fifteen exact retrainings: 17.455 seconds total.
- Three approximate methods: 3.286 seconds total.

These wall-clock values describe this WSL/RX 7800 XT pilot only and are not normalized
hardware-independent compute measurements.

## Exact-retraining variability and convergence

- js_divergence: k=10→15 mean changed by 4.477%; frozen 10% pilot diagnostic PASS.
- prediction_disagreement: k=10→15 mean changed by 3.141%; frozen 10% pilot diagnostic PASS.

The complete k=3/5/8/10/15 estimates and replica-bootstrap intervals are in
`reports/exact_seed_convergence_pilot.md`.

## Approximate/exact comparison

| Method | Test accuracy | Forget accuracy | JS mean to exact | Disagreement mean to exact |
| --- | ---: | ---: | ---: | ---: |
| fine_tune | 0.8452 | 0.9000 | 0.014274 | 0.087120 |
| naive_gradient_ascent | 0.8492 | 0.9000 | 0.012908 | 0.083173 |
| scrub | 0.8496 | 0.9222 | 0.014255 | 0.087493 |

For all three methods, the mean approximate/exact JS divergence and prediction
disagreement were below the exact/exact empirical 95th percentile in this condition.
This is not an equivalence test. Forget accuracy remained high, so this condition does
not demonstrate successful forgetting under a criterion requiring reduced forget-set
accuracy. No contradiction classification was preregistered for this pilot.

## Are the metrics informative?

The exact/exact distances changed materially from k=3 to larger populations and then
met the frozen k=10→15 stability diagnostic. This supports using multiple exact seeds
and shows why a single oracle would be unstable. JS divergence and prediction
disagreement provided nonzero, distinguishable scales, but their scientific utility
across deletion regimes remains unestablished.

## What must change before Tier-2

- Repeat across original-model seeds; this pilot varies exact-retraining seeds only.
- Add a second deletion fraction and high-loss condition under schema v2.
- Freeze conventional-pass and counterfactual-fail criteria before contradiction analysis.
- Validate stronger privacy attacks and baseline tuning parity.
- Extend to CIFAR-10 and at least one independently licensed tabular dataset.
- Treat seed convergence as condition-specific until replicated.

## Scientific assessment

The pilot validates the schema-v2 machinery and makes exact-retraining variance
measurable. It does not answer the paper's central cross-condition research questions.
