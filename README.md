# Measuring Counterfactual Fidelity in Machine Unlearning


---

## License & acquisition

This project is **proprietary**. Production use, redistribution, and commercial deployment require a written commercial license or completed acquisition. See [LICENSE](./LICENSE) and [ACQUISITION.md](./ACQUISITION.md). Contact [@theworker02](https://github.com/theworker02).


*An empirical framework for testing whether approximate machine unlearning reproduces models trained without forgotten data.*

> **Benchmark v0.5.0 Ã¢â‚¬â€ Research Infrastructure Candidate. No conclusions have been established.**

## Published pilot preprint

The archival pilot manuscript and supplement are published on Zenodo:
**https://zenodo.org/records/23540870**  
DOI: [10.5281/zenodo.23540870](https://doi.org/10.5281/zenodo.23540870).

This is a preliminary pilot and methodology preprint, not a peer-reviewed or confirmatory
unlearning result. The deposit describes the surviving archive's evidence limitations.
The DOI identifies the manuscript and supplement; it does not identify a complete software
or experimental-data release.

## Research question

This repository evaluates an unlearned model against a *distribution* of independently
retrained counterfactual models, rather than treating one retraining seed as an absolute
oracle. It records predictive, distributional, calibration, privacy-proxy, representation,
parameter, and runtime measurements in immutable experiment artifacts.

## Evidence summary

Digits remains a Tier-0/Tier-1 smoke-test dataset. Historical Fashion-MNIST development
records exercise a small CNN, five configured exact-retraining seeds, two simple baselines,
and an adapted literature-backed SCRUB core. Those records predate Benchmark 0.5's mandatory
prediction and maturity schema and fail the current artifact gate. They remain legacy pilot
evidence and are not silently promoted or rewritten.

Current verified quantitative claims: **0**. Current publication-eligible runs: **0**. These
counts are generated in `release/metadata/repository_inventory.json`.

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/run_experiment.py --config configs/experiments/digits_mlp_smoke.yaml
pytest
make corpus
unlearn-bench datasets list
unlearn-bench validate
unlearn-bench status
```

Raw runs are written once beneath `results/raw/<experiment-id>/result.json`; an existing
experiment ID is never overwritten. Empirical table and figure builders accept only gate-passing
records and currently refuse to generate publication outputs. F001 is conceptual and contains no
experimental data.

Benchmark 0.5 is specified under `benchmark/`. Dataset research packages live under
`datasets/`, predeclared pilot deletion requests under `deletion_manifests/`, and the derived
query registry at `results/processed/experiments.sqlite`. Run `make setup` once before Makefile
targets so the CLI and source package are installed in the active environment.

## Benchmark components

- Exact retraining from scratch over multiple independent initialization seeds.
- Retained-data fine-tuning.
- A deliberately named `naive_gradient_ascent` diagnostic baseline, not a claim of a
  faithful implementation of every negative-gradient method in the literature.
- A clean-room implementation of SCRUB's alternating teacherÃ¢â‚¬â€œstudent max/min core,
  explicitly marked as a Fashion-MNIST adaptation without privacy rewinding.
- Random, class, and original-model high-loss deletion selection.
- Accuracy, balanced accuracy, macro-F1, agreement, JensenÃ¢â‚¬â€œShannon divergence, total
  variation, NLL, Brier score, ECE, loss-threshold membership AUC, linear CKA, and
  parameter distances.
- Normalization of unlearned-to-retrained distances by retrained-to-retrained dispersion.

See [methodology](docs/methodology.md), [reproducibility](REPRODUCIBILITY.md), and the
[preregistration draft](research/preregistration.md). No manuscript result or abstract is
written until a preregistered full suite has actually run.

## Paper and research data

No evidence-complete paper or paper-data package exists. The SQLite registry, dataset packages,
model manifest, and infrastructure release manifest support inspection; they are not a publication
result corpus. See [start here for researchers](FOR_RESEARCHERS.md).

The generated [counterfactual fidelity pilot](reports/counterfactual_fidelity_pilot.md)
and [Tier-2 plan](docs/tier2-experiment-plan.md) state the observed development evidence,
statistical limitations, and publication gates. The full study still requires CIFAR-10,
CIFAR-100, a licensed tabular dataset, more architectures and validated methods,
sequential deletion, stronger privacy attacks, and confirmatory seeds.

Current gate outcomes are recorded in the [Phase III quality evaluation](reports/phase3_quality_evaluation.md),
[research status](reports/research_status.md), and
[publication readiness](PUBLICATION_READINESS.md).

## Synthesis and presentation

The [research-question register](research/RESEARCH_QUESTIONS.md), [results portal](RESULTS.md),
and generated [research presentation](site/presentation/index.html) report every central question
as inconclusive. The presentation contains methodology and readiness evidence only; it has no
empirical result slides. [What we learned](WHAT_WE_LEARNED.md) distinguishes the artifact audit
from findings about unlearning performance.

## Integrity boundaries

The included confidence/loss membership score is a limited audit, not proof of deletion.
Parameter distance is descriptive and is not treated as functional equivalence. Digits is
a debugging dataset. Raw output is excluded from source control by default but should be
archived immutably for a research release.

## Citation

`CITATION.cff` describes the v0.5 software artifact. The archival pilot preprint is available
at https://zenodo.org/records/23540870 (DOI: 10.5281/zenodo.23540870). A complete software/data
release has not been archived under that DOI. Do not cite development pilot measurements
as established findings.

## License

Code is MIT licensed. Dataset licenses and provenance are documented separately and are
not superseded by the repository license.

## Acquisition diligence

Buyer-facing diligence materials live in [docs/acquisition/](./docs/acquisition/). Commercial licensing contact path: [COMMERCIAL.md](./COMMERCIAL.md).
