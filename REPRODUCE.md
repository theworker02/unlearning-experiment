# Reproduce the benchmark, pilot artifacts and diagnostics

This is a **staged reproduction guide**, not a guarantee that the flagship study or published paper is independently reproducible. The [pilot preprint](https://doi.org/10.5281/zenodo.23540870) is an archived *manuscript and supplement*; it is **not** a complete deposit of the software and raw experimental outputs.

> **Rights and safety:** The current [LICENSE](LICENSE) is proprietary. Public source access does not itself authorize cloning, executing, modifying, or redistributing the material. Obtain applicable written authorization from the rightsholder before carrying out these steps. Datasets carry separate source licenses. Do not submit or replace a published experiment's immutable records.

## 1. Choose the level of reproduction

| Level | Goal | Current support |
| --- | --- | --- |
| **Smoke test** | Check a minimal Python installation, data adapter, and end-to-end runner on Digits | Small local validation example, not scientific evidence |
| **R1: artifact verification** | Check the repository's archived P001 files, hashes, partition and reference checks without retraining | Commands exist; availability of every expected payload must be verified on your checkout |
| **R2: computational reproduction** | Re-run the archived P001 plan with original data, split, deletion manifests and declared seeds | Protocol recorded, but a safe isolated R2 orchestration flow remains incomplete |
| **R3: independent reproduction** | Implement the method and analysis independently of this repository | Not demonstrated |

## 2. Prerequisites and environment

- Git and Python **3.11–3.13**; the original P001 environment used **Python 3.12.3**.
- CPU is sufficient for Digits smoke testing; PyTorch/Python and platform differences can affect results.
- For P001 provenance, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [reproduction.yaml](reproduction.yaml), [environment/P001/](environment/P001/), and [KNOWN_VARIABILITY.md](KNOWN_VARIABILITY.md).
- For Fashion-MNIST, ensure sufficient free disk space and connectivity to download files from the official project. The Python dependencies include PyTorch and may be large.

```bash
git clone https://github.com/theworker02/unlearning-experiment.git
cd unlearning-experiment
git rev-parse HEAD                       # Record this exact commit in your reproduction log
python -m venv .venv
source .venv/bin/activate                # macOS/Linux
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python --version
python -m pip freeze > local-environment-freeze.txt
```

The generated environment freeze is a **local** audit artifact, not a repository lockfile or verified lock. Capture the OS, GPU, CUDA/ROCm/MPS backend, PyTorch versions and relevant device flags too. P001 used an AMD Radeon RX 7800 XT and ROCm under WSL; unrelated hardware does not promise bitwise-identical weights.

## 3. Run the smallest smoke test first

```bash
python scripts/check_documentation.py
unlearn-bench datasets list
unlearn-bench methods list
python scripts/run_experiment.py --config configs/experiments/digits_linear_tiny.yaml
python -m ruff check .
python -m pytest
```

The tiny config in `configs/experiments/digits_linear_tiny.yaml` writes under `results/raw/<experiment-id>/`. **Experiment IDs are append-only.** A second run using an existing ID will not overwrite it; give the cloned configuration a new `experiment_id`. Do not interpret a smoke-test PASS as unlearning effectiveness.

## 4. Acquire and verify Fashion-MNIST

```bash
python scripts/download_data.py --dataset fashion_mnist
python scripts/verify_dataset.py --dataset fashion_mnist
```

The downloader verifies the four original compressed IDX files against known checksums and stores them under `data/raw/fashion_mnist/raw/`. Raw source data is not committed. If the download source is unavailable or the checksum mismatches, **stop** rather than substituting another dataset silently.

Dataset and split provenance is described in [data/README.md](data/README.md), [docs/data-provenance.md](docs/data-provenance.md), and `experiments/P001/manifests/`. P001 requires its archived deletion IDs: a fresh random 1% deletion is not the same experiment.

## 5. Inspect and verify P001 (R1)

```bash
python scripts/verify_p001_archive.py
python scripts/run_reference_checks.py
python scripts/validate_visualizations.py
```

On systems with `make`, `make reproduce-p001-r1` orchestrates those checks. The verifier checks archival SHA-256 lists under `provenance/`, the forget/retain partition, and the required archived artifacts. Missing files or hashes are **verification failures**, not reasons to fabricate replacements.

A proposed R001 package can be inspected using:

```bash
python scripts/verify_r001_release.py --release release/R001
```

This is expected to **fail** if a complete R001 payload has not been staged. It must **not** be represented as an externally published R001 release, and the pilot Zenodo DOI is not evidence of R001 publication.

## 6. Computational rerun: what is and isn't ready (R2)

The archived Fashion-MNIST pilot run uses `configs/experiments/fashion_cnn_random_01pct_schema2_rocm_pilot.yaml` and seeded exact retraining. The configuration and [reproduction.yaml](reproduction.yaml) list the reference parameters.

```bash
# Inspect before execution; this is not a safe automatic R2 replay:
cat configs/experiments/fashion_cnn_random_01pct_schema2_rocm_pilot.yaml
cat reproduction.yaml
```

To rerun, set an **isolated output root and a new experiment ID** while preserving the original source split, deletion IDs, reference seeds and documented method settings. Never run the unchanged pilot configuration into archived P001 directories. A fully validated, cross-hardware R2 runner is not yet provided.

## 7. Interpret results and report differences

- Distinguish **bitwise** (same-stack CPU fixtures), **numerical** (tolerance-bounded), and **statistical** (distributional) reproduction; neither GPU nor different PyTorch versions guarantee identical weights.
- Report every exact-retraining seed, population dispersion, candidate-to-oracle differences, calibration, MIA proxy and representation metric independently; a membership AUC near 0.5 is **not** a deletion certificate.
- Do not treat dependent oracle pairs as independent statistical replicates.
- Record commit, environment, dataset checksums, deletion manifest, method provenance, immutable output paths, failures and differences from the pilot.
- Use [reviews/reproduction/REPORT_TEMPLATE.md](reviews/reproduction/REPORT_TEMPLATE.md) for a reproduction report. A discrepancy must not be rewritten as agreement.

See [the full reproducibility contract](REPRODUCIBILITY.md), [scientific limits](FOR_RESEARCHERS.md), [publication gates](PUBLICATION_READINESS.md), and [paper reproduction status](REPRODUCE_PAPER.md).
