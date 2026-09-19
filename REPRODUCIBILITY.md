# Reproducibility contract

## Levels

- **R1 artifact reproduction:** verify archived predictions, recompute reference checks,
  and validate statistics/figure provenance with `make reproduce-p001-r1`.
- **R2 computational reproduction:** retrain from official Fashion-MNIST using the exact
  archived split, deletion request, configuration, and seeds in an isolated output root.
- **R3 independent implementation:** reproduce the protocol without this training code.

R2 is documented but its safe staging command remains pending. The immutable P001 output
directory must never be deleted or reused to make reproduction convenient.

## Environment and hardware

P001 used Python 3.12, PyTorch 2.9.1 built for ROCm 7.2, and an AMD Radeon RX 7800 XT
through WSL. `HSA_ENABLE_DXG_DETECTION=1` is required on this host. CPU execution remains
supported for small validation runs. Plan for at least 1 GB free storage and sufficient
RAM for Fashion-MNIST; the present pilot used no augmentation and batch size 128.

Measured P001 timings on that specific machine were:

| Component | Time |
| --- | ---: |
| Original training | 9.593 s |
| Fifteen exact retrainings, total | 17.455 s |
| Three approximate methods, total | 3.286 s |

These are provenance measurements, not universal performance expectations.

## R1 command

```bash
make reproduce-p001-r1
```

This checks archived hashes and split disjointness, runs independent metric fixtures, and
validates figure provenance. Dataset checks are available with `make verify-datasets`.

## Dataset acquisition

The loader expects the four official compressed IDX files under
`data/raw/fashion_mnist/raw/`. Run `python scripts/download_data.py`, then verify source
hashes and splits before training. P001 must use its archived deletion manifest; a newly
sampled 1% deletion is not P001.

## Reproduction interpretation

See `KNOWN_VARIABILITY.md`. Same-stack CPU fixtures target exact reproducibility. GPU and
cross-stack work normally targets numerical (`R-NUM`) or scientific (`R-STAT`)
reproduction within prospectively frozen tolerances, not identical checkpoint bytes.

## Current limitations

There is no Zenodo DOI, OSF registration identifier, release commit, environment lock,
or safe isolated R2 orchestration yet. Consequently Research Release R001 is prepared but
not published, and the repository must not be described as externally archived.

