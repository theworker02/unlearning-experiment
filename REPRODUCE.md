# Reproduce R001 / P001

R001 supports artifact verification without retraining and provides the inputs required for
a new P001 execution. It does not claim bitwise cross-hardware training reproducibility.

```bash
git clone <public-repository-url-after-publication>
cd unlearning-experiment
python -m pip install -e ".[dev]"
python scripts/verify_r001_release.py --release release/R001
```

To rerun P001 in a ROCm-capable environment, use the archived configuration. The experiment
writer creates a new immutable run and must not overwrite the R001 evidence:

```bash
export PYTHONPATH=src
python scripts/run_experiment.py \
  --config configs/experiments/fashion_cnn_random_01pct_schema2_rocm_pilot.yaml
```

Expected artifact verification includes dataset and manifest identity, deleted-set exclusion,
15 exact seed records, method records, prediction/checkpoint references, convergence inputs,
and F002 provenance. Runtime is descriptive and is not an acceptance tolerance.
