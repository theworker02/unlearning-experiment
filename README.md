# Measuring Counterfactual Fidelity in Machine Unlearning

[![Documentation checks](https://github.com/theworker02/unlearning-experiment/actions/workflows/docs.yml/badge.svg)](https://github.com/theworker02/unlearning-experiment/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.13-3776AB)
![Software version](https://img.shields.io/badge/Software-v0.5.1-4359A5)
![Benchmark protocol](https://img.shields.io/badge/Benchmark-v0.5.0-586069)
![Research maturity](https://img.shields.io/badge/Status-Preliminary%20research-9C6B08)
![License](https://img.shields.io/badge/License-Proprietary-555555)
[![Zenodo pilot DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23540870.svg)](https://doi.org/10.5281/zenodo.23540870)

**A reproducibility-oriented research framework for evaluating approximate machine unlearning against the *distribution* of models retrained without the requested data—not a single retraining seed.**

> **Research status:** Infrastructure candidate and exploratory pilot. The current repository inventory reports **zero publication-eligible runs** and **zero verified confirmatory claims**. It does not establish that any method achieves successful forgetting. The linked Zenodo pilot is a methodology preprint, **not peer reviewed** and **not** an archived software/data release.

**[Start here](#quick-start) · [Reproduction](REPRODUCE.md) · [Scientific methodology](docs/methodology.md) · [Evidence and limitations](FOR_RESEARCHERS.md) · [Changelog](CHANGELOG.md) · [Versioning](VERSIONING.md) · [Pilot preprint](https://doi.org/10.5281/zenodo.23540870)**

## Why counterfactual fidelity?

In a deletion experiment, exact retraining on retained data is the reference procedure—but retraining itself varies with initialization and other stochastic effects. Comparing an approximate unlearning result to one arbitrarily seeded reference model can therefore be misleading.

The framework builds a population of exact-retraining references and records candidate-to-reference distances alongside natural reference-to-reference dispersion. It studies where familiar utility and forgetting proxies agree or disagree with *counterfactual fidelity*, without treating any one metric as a deletion certificate.

```text
Original training data D ------> Original model
          |                            |
          +--> Delete set F            +--> Approximate unlearning
          |                                   |
          +--> Retained data D \ F            v
                    |                    Candidate model U
          +---------+---------+                |
          |         |         |                |
       Seed 1    Seed 2    Seed n              |
          |         |         |                |
       Exact R1  Exact R2  Exact Rn <-----------+
                    |
             Compare distributions
         + predictive behavior / calibration
         + privacy-proxy audits / representations
         + compute cost / model variability
```

## What the framework includes

| Component | Implemented scope | Interpretation |
| --- | --- | --- |
| Exact-retraining population | Independent reference seeds; per-reference and pairwise dispersion | Stochastic baseline, not one fixed oracle |
| Approximate methods | Retained-data fine-tuning, explicitly named naive gradient ascent, and a documented SCRUB-core adaptation | Method provenance and applicability checks are essential; no method is certified |
| Deletion requests | Random, class, and high-loss selection; immutable sample and deletion IDs | Composition affects interpretation |
| Diagnostics | Accuracy, balanced accuracy, macro-F1, disagreement, JS/TV distance, NLL, Brier score, ECE, loss-based membership AUC, linear CKA, parameter distance, runtime | Separate outcomes; no validated aggregate score |
| Evidence infrastructure | Configuration hashes, raw-artifact immutability, provenance manifests, quality gates, repeatable dataset/figure tools | A passing engineering check is **not** proof of scientific validity |
| Research assets | Digits debugging examples, legacy Fashion-MNIST pilot, benchmark definitions, planned higher-tier study | Confirmatory multi-dataset experiments remain outstanding |

See the [methodology](docs/methodology.md), [study design](docs/tier2-experiment-plan.md), [registered research questions](research/RESEARCH_QUESTIONS.md), and [scientific changelog](SCIENTIFIC_CHANGELOG.md).

## Quick start

**Prerequisites:** Python 3.11–3.13, Git, a fresh virtual environment, and sufficient disk space for Python/PyTorch dependencies. The minimal Digits experiment uses scikit-learn's local dataset and does not require a dedicated GPU. **Review the [proprietary license](LICENSE) and obtain written permission before using or running this code where required.** Public access is not a grant of reuse rights.

```bash
git clone https://github.com/theworker02/unlearning-experiment.git
cd unlearning-experiment
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux; Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_documentation.py
unlearn-bench datasets list
python scripts/run_experiment.py --config configs/experiments/digits_linear_tiny.yaml
```

The tiny Digits run is a **smoke test**, not an empirical publication result. The experiment writer creates an immutable record under `results/raw/<experiment-id>/result.json`. To rerun with a changed environment/configuration, assign a **new** `experiment_id`; do not overwrite or delete an earlier record.

Run code checks separately:

```bash
python -m ruff check .
python -m pytest
```

Additional workflows, including a *checksum-verified* Fashion-MNIST download, P001 archival verification, R1/R2/R3 meanings, and environment capture, are described in **[REPRODUCE.md](REPRODUCE.md)** and **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**. The `Makefile` is a convenience layer where `make` is installed; `make corpus` is a broader operation and may require downloaded datasets.

## Scientific evidence, without overclaiming

- **Available:** engineering tests; legacy Digits and Fashion-MNIST development records; an archival pilot manuscript and supplement on [Zenodo](https://doi.org/10.5281/zenodo.23540870).
- **Not established:** statistically confirmed equivalence or failure of unlearning; a reproducible multi-dataset flagship result; a released, validated paper-data corpus.
- **Current gates:** the existing pilot records do not pass the current publication-artifact gate. The quality gate is deliberately fail-closed, and manuscript tables and figures must not silently inherit pilot-only results.

Inspect the [pilot analysis](reports/counterfactual_fidelity_pilot.md), [repository inventory](release/metadata/repository_inventory.json), [research status](reports/research_status.md), [publication readiness](PUBLICATION_READINESS.md), and [preprint reproduction caveats](REPRODUCE_PAPER.md). Historical numbers in the pilot report are **exploratory diagnostics**, not confirmatory findings.

## Project navigation

| You want to… | Read… |
| --- | --- |
| Re-run smoke tests or inspect archived P001 | [Reproduction guide](REPRODUCE.md) |
| Understand hardware, seeds, and acceptance boundaries | [Reproducibility contract](REPRODUCIBILITY.md) |
| Audit hypotheses and planned statistical design | [Preregistration draft](research/preregistration.md) |
| Inspect protocol and registries | [Benchmark specification](benchmark/SPECIFICATION.md) |
| Extend a method or dataset | [Contributing](CONTRIBUTING.md) |
| Track software changes and scientific changes separately | [Changelog](CHANGELOG.md) / [scientific changelog](SCIENTIFIC_CHANGELOG.md) |
| Report a scientific correction | [Corrections policy](CORRECTIONS.md) |

## Citation, versioning and permissions

**Software source version:** `0.5.1` (documentation/metadata patch). **Scientific benchmark protocol:** `0.5.0`. These intentionally have different version numbers; see [VERSIONING.md](VERSIONING.md). A source version is not evidence of an externally published software release.

For the **pilot preprint**, cite [DOI:10.5281/zenodo.23540870](https://doi.org/10.5281/zenodo.23540870). For the **software**, cite the repository, version, and exact Git commit using [CITATION.cff](CITATION.cff) and [CITATION.md](CITATION.md). The Zenodo DOI **does not** cover the current software/data tree.

**License: Proprietary / all rights reserved.** Current code and materials are **not MIT-licensed**. See [LICENSE](LICENSE), [the license transition notice](LICENSE_TRANSITION_NOTICE.md), and [commercial licensing](COMMERCIAL.md). Dataset rights must be evaluated independently. This repository remains public to make its research design and limitations inspectable; public visibility alone does not permit commercial or scientific reuse beyond the written license.

Maintained by [Matthew Looney / Magnexis](https://github.com/theworker02). Research inquiries and corrections can be opened through the repository's [issue templates](https://github.com/theworker02/unlearning-experiment/issues/new/choose).
