# Auditable Empirical Evaluation of Machine Unlearning

**Preprint v1.0 · Research Release R001 · P001 · schema v2**

This pilot and methodology paper describes an auditable machine-unlearning evaluation
workflow based on immutable deletion manifests and distributions of independently exact-
retrained models. P001 uses one reduced Fashion-MNIST split, one frozen 1% IID deletion
manifest, one original model, and 15 exact-retraining seeds.

The frozen seed-count diagnostic changed by 4.48% for mean Jensen–Shannon divergence and
3.14% for prediction disagreement, below the predefined 10% pilot threshold. A later
retrospective analysis classified the deletion condition as assay-insensitive under both
metrics. Therefore, the execution of fine-tuning, naïve gradient ascent, and SCRUB does not
support a successful-unlearning claim in this condition.

The paper establishes infrastructure integrity and reference-ensemble estimation. It does
not establish certified removal, privacy equivalence, structured-deletion effectiveness,
cross-dataset generalization, or independent replication. The authoritative manuscript is
`paper/manuscript.tex`; all numerical paper values are generated from validated artifacts.
