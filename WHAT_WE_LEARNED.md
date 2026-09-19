# What we learned

The project has not yet learned a defensible general answer to its machine-unlearning questions.

It established that the earlier development records are insufficient under the present evidence
standard: they lack explicit maturity and required immutable prediction artifacts, and most lack
the current per-seed exact-reference record. Consequently, no method comparison, contradiction,
sequential trend, or compute–fidelity conclusion is defensible.

One new schema-v2 Fashion-MNIST pilot passed every artifact gate and made exact-retraining variance
measurable across 15 replicas. The frozen pilot diagnostic found that exact/exact mean JS divergence
and prediction disagreement changed by less than 10% from k=10 to k=15. Approximate models fell
below exact/exact 95th-percentile distances for those two dimensions, while forget-set accuracy
remained high. These observations describe one condition; they are neither equivalence tests nor
evidence of successful forgetting.

The scientifically justified next step is replication across original seeds, deletion fractions,
and a high-loss condition before freezing the larger benchmark matrix. This conclusion concerns
research readiness, not general unlearning performance.
