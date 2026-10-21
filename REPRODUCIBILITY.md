# Reproducibility contract

**Scope:** Pilot research infrastructure, not confirmatory evidence. See the executable [reproduction guide](REPRODUCE.md) for prerequisites, commands, error interpretation and capture of local environments.

## Reproduction levels

- **R1 artifact verification:** Validate P001 checksum lists, forget/retain partition, independent metric fixtures and figure provenance without a new training run (`make reproduce-p001-r1` where `make` is available).
- **R2 computational reproduction:** Reconstruct the P001 inputs and rerun training in an isolated output location. The archived configuration and seeds exist, but safe, fully automated independent R2 staging is not yet available.
- **R3 independent implementation:** Independently reconstruct the experimental procedure with separate code; **not demonstrated**.

An R1 PASS is not an R2/R3 PASS, a peer review, or statistical confirmation that unlearning succeeds.

## Hardware, environment and known variability

The historical P001 development environment recorded Python 3.12.3, a PyTorch 2.9.1 ROCm 7.2 build, AMD Radeon RX 7800 XT, and WSL. The host used `HSA_ENABLE_DXG_DETECTION=1`. Small CPU verification runs are supported; GPU numerical results and performance will vary.

Archive-derived P001 time references were approximately 9.593 seconds original training, 17.455 seconds total for 15 exact retrainings, and 3.286 seconds for three approximate methods. These are **machine-specific provenance measurements**, never acceptance thresholds.

Same-stack deterministic CPU fixtures target repeatability. Cross-stack/GPU experiments require prospectively specified numerical (`R-NUM`) or scientific/statistical (`R-STAT`) acceptance criteria, not assumed byte-identical checkpoint hashes. See [KNOWN_VARIABILITY.md](KNOWN_VARIABILITY.md).

## Input integrity and immutable results

The Fashion-MNIST loader expects four checksum-verified compressed IDX files in `data/raw/fashion_mnist/raw/`. Obtain them through `python scripts/download_data.py --dataset fashion_mnist` and validate before any attempted replay. P001 must preserve the archived split, 1% deletion indices, original/model seeds, and fifteen exact-retraining seeds from [reproduction.yaml](reproduction.yaml). A new random deletion manifest is a new experiment.

Raw records are immutable and must not be overwritten or retagged as newer evidence. Derived registries, manifests, and summary outputs are separately regenerable where their dependencies are present. Record Git commit, source digests, full environment and hardware characteristics before interpreting a result.

## Publication and archive boundary

A pilot methodology manuscript and supplement **have** been archived on [Zenodo (DOI:10.5281/zenodo.23540870)](https://doi.org/10.5281/zenodo.23540870). That DOI **does not** identify a complete software, model, raw-data or R001 research release. There is no verified public R001 package, standalone software DOI, frozen confirmatory protocol, or demonstrated external R2/R3 reproduction. Publication gates are documented in [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md).

Current code is subject to the [proprietary license](LICENSE); separate written permission may be needed to execute the reproduction steps. This document records the technical contract, not a new grant of rights.
