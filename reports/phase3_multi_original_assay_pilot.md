# Phase 3 Multi-Original-Seed Assay Pilot

**Status:** complete exploratory calibration pilot; not publication evidence.

## Design

The run trained five independent original models and fifteen independent exact-retraining
models for each of four frozen IID deletion manifests on the reduced Fashion-MNIST pilot
split. The deletion fractions were 0.5%, 2%, 5%, and 10% (45, 180, 450, and 900 training
examples). No approximate-unlearning method was evaluated. This separation prevents method
performance from influencing assay selection.

The reference metrics were test-set Jensen--Shannon divergence, prediction disagreement,
final-feature linear CKA distance, and final-feature MMD. Power simulations used seed-level
summaries rather than treating all exact/exact pairs as independent observations.

## Result

None of the sixteen fraction-by-metric conditions met its simulated minimum detectable
separation. Observed original/exact-to-exact separability ratios ranged from 0.448 to 0.864,
whereas simulated minimum ratios ranged from 1.10 to 1.55. Increasing IID deletion from
0.5% to 10% did not make the original model distinguishable from the exact-retraining
population under these probes.

| Fraction | JS ratio | Disagreement ratio | CKA-distance ratio | MMD ratio | Sensitive metrics |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 0.569 | 0.734 | 0.597 | 0.829 | 0/4 |
| 2% | 0.448 | 0.623 | 0.596 | 0.863 | 0/4 |
| 5% | 0.482 | 0.668 | 0.608 | 0.827 | 0/4 |
| 10% | 0.512 | 0.651 | 0.725 | 0.841 | 0/4 |

## Pilot questions

1. **Can exact-retraining variability be measured reliably?** The machinery produced a
   complete multi-seed distribution and seed-level power calibration. Reliability across
   independently frozen deletion manifests remains untested.
2. **Are approximate/exact differences larger than exact/exact differences?** Insufficient
   evidence: approximate methods were deliberately excluded from this blind assay pilot.
3. **Do conventional metrics disagree with counterfactual metrics?** Insufficient evidence
   in A001. P001 is retrospective only and cannot answer the confirmatory question.
4. **Is variance manageable?** Computationally yes, but scientifically it dominates the
   deletion signal under every tested IID condition.
5. **Which metrics are informative?** All four quantify retraining variability; none was
   sensitive enough to support a deletion-identifiability claim in this regime.
6. **What compute would the full study require?** This reduced 65-model calibration run took
   1,751 seconds wall time (29.2 minutes) on PyTorch 2.9.1 with ROCm 7.2 in WSL. Recorded
   model-training time was 79.9 seconds; artifact evaluation and power simulation dominated
   the remainder. Full-study cost must be re-estimated after stronger regimes are selected.
7. **Are assumptions clearly wrong?** Yes: larger random IID deletion fractions do not
   necessarily produce an identifiable counterfactual change on this reduced benchmark.
8. **Is the central question worthwhile?** It remains falsifiable and relevant, but this
   pilot shows the study must first establish identifiability using stronger deletion
   regimes, local probes, controlled memorization, or larger datasets before evaluating
   approximate methods.

## Decision

Do not freeze equivalence margins or launch the confirmatory method matrix from these data.
The next pilot should use multiple deletion manifests and stronger, prospectively specified
class-conditioned, high-influence, or controlled-memorization regimes. Any later method
comparison must preserve these calibration data and report that all tested IID conditions
were assay-insensitive.

## Evidence

- Raw result: `results/raw/phase3-fmnist-multi-original-assay-pilot-v1/result.json`
- Configuration: `configs/experiments/phase3_multi_original_assay_pilot.yaml`
- Registry ID: `A001`
