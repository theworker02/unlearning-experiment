# Research questions

Benchmark protocol: **0.5.0**. One gate-passing pilot informs design, but no
question has a confirmatory or publication-eligible empirical answer.

| ID | Research question | Eligible primary evidence | Status | Uncertainty | Figure |
| --- | --- | --- | --- | --- | --- |
| RQ1 | Do approximate unlearning methods reproduce behavior within the natural distribution of exact retraining? | fashion-cnn-random-01pct-schema2-rocm-pilot-v1 | INCONCLUSIVE | Not estimable from eligible evidence. | none |
| RQ2 | Do conventional forgetting and privacy criteria agree with counterfactual fidelity? | fashion-cnn-random-01pct-schema2-rocm-pilot-v1 | INCONCLUSIVE | Not estimable from eligible evidence. | none |
| RQ3 | Does counterfactual discrepancy accumulate across sequential deletion requests? | none | INCONCLUSIVE | Not estimable from eligible evidence. | none |
| RQ4 | How do deletion size and composition affect counterfactual fidelity? | fashion-cnn-random-01pct-schema2-rocm-pilot-v1 | INCONCLUSIVE | Not estimable from eligible evidence. | none |
| RQ5 | What counterfactual fidelity is sacrificed for computational savings relative to retraining? | fashion-cnn-random-01pct-schema2-rocm-pilot-v1 | INCONCLUSIVE | Not estimable from eligible evidence. | none |

## Detailed register

### RQ1 — Do approximate unlearning methods reproduce behavior within the natural distribution of exact retraining?

- Hypothesis: Some approximate methods will lie outside at least one exact-retraining fidelity envelope.
- Status: **INCONCLUSIVE**
- Observation: One schema-v2 Fashion-MNIST pilot passes the artifact gate, but one deletion condition and original seed cannot establish general behavior.
- Statistical inference: none; the required eligible sample is absent.
- Interpretation: the hypothesis is neither supported nor rejected.
- Required evidence: Gate-passing approximate/exact and exact/exact comparisons across independent seeds.
- Planned analysis: Empirical percentiles, confidence intervals, effect sizes, and equivalence tests.
- Uncertainty: Not estimable from eligible evidence.
- Limitation: The eligible pilot varies exact-retraining seeds but not original-model seeds; datasets; architectures; or deletion regimes.

### RQ2 — Do conventional forgetting and privacy criteria agree with counterfactual fidelity?

- Hypothesis: Some conventional-pass conditions will fail at least one counterfactual-fidelity criterion.
- Status: **INCONCLUSIVE**
- Observation: The schema-v2 pilot contains conventional and counterfactual metrics, but no pass/fail criteria were frozen and no contradiction matrix is validated.
- Statistical inference: none; the required eligible sample is absent.
- Interpretation: the hypothesis is neither supported nor rejected.
- Required evidence: Paired conventional and counterfactual criteria from gate-passing runs.
- Planned analysis: Agreement rates, contradiction frequencies with uncertainty, and metric correlations.
- Uncertainty: Not estimable from eligible evidence.
- Limitation: Current membership score is a limited audit and cannot establish privacy equivalence.

### RQ3 — Does counterfactual discrepancy accumulate across sequential deletion requests?

- Hypothesis: Accumulation may be nonlinear and method-dependent.
- Status: **INCONCLUSIVE**
- Observation: Sequential deletion has interfaces and tests but no eligible empirical trajectory.
- Statistical inference: none; the required eligible sample is absent.
- Interpretation: the hypothesis is neither supported nor rejected.
- Required evidence: Stage-matched exact-retraining populations for complete deletion sequences.
- Planned analysis: Trajectory estimates, uncertainty by stage, and method-by-stage interactions.
- Uncertainty: Not estimable from eligible evidence.
- Limitation: No sequential research corpus exists.

### RQ4 — How do deletion size and composition affect counterfactual fidelity?

- Hypothesis: Model-dependent deletion regimes may produce different discrepancies from random deletion.
- Status: **INCONCLUSIVE**
- Observation: One eligible random 1% pilot exists, but no factorial comparison across size or composition exists.
- Statistical inference: none; the required eligible sample is absent.
- Interpretation: the hypothesis is neither supported nor rejected.
- Required evidence: Frozen random, loss-based, class, and other justified scenarios across fractions.
- Planned analysis: Preregistered contrasts with multiplicity control and cross-condition uncertainty.
- Uncertainty: Not estimable from eligible evidence.
- Limitation: Only random pilot deletion sets are predeclared independently of legacy runs.

### RQ5 — What counterfactual fidelity is sacrificed for computational savings relative to retraining?

- Hypothesis: Methods will occupy distinct points on a multidimensional compute-fidelity frontier.
- Status: **INCONCLUSIVE**
- Observation: One pilot records device-specific runtime and fidelity distances, but it cannot establish a general frontier.
- Statistical inference: none; the required eligible sample is absent.
- Interpretation: the hypothesis is neither supported nor rejected.
- Required evidence: Device-aware runtime and memory measurements paired with fidelity dimensions.
- Planned analysis: Pareto dominance with uncertainty and retraining-relative cost.
- Uncertainty: Not estimable from eligible evidence.
- Limitation: Legacy runtimes are not sufficient for publication comparisons.

