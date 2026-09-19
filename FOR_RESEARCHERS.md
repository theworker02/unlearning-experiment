# Start here for researchers

Benchmark protocol: **0.5.0 — Research Infrastructure Candidate**.

## Scientific question

This project studies whether approximate machine-unlearning procedures reproduce the observable
and latent behavior of models independently trained without forgotten data. It treats exact
retraining as a stochastic population and asks when conventional utility, forgetting, or privacy
criteria disagree with counterfactual fidelity.

## Present evidence boundary

No raw run currently passes the Phase III artifact gate. Existing Digits and Fashion-MNIST runs
are legacy development records. They are useful for engineering audits but do not support a
publication conclusion. See `reports/phase3_quality_evaluation.md` and `EVIDENCE.md`.

## Inspect the benchmark

```bash
python -m pip install -e ".[dev]"
unlearn-bench datasets list
unlearn-bench methods list
unlearn-bench scenarios list
unlearn-bench validate
unlearn-bench status
```

Dataset packages contain cards, checksums, exact split IDs, preprocessing, and validation reports.
The model catalog includes only checkpoints that exist. The SQLite experiment registry is derived
from immutable raw records and can be queried without parsing manuscript tables.

## Reproduce and extend

`make verify` validates code, metrics, datasets, and artifact gates. `make corpus` regenerates
derived registries and the static research site. New methods use the `UnlearningContext` contract
and must declare citation and implementation provenance. See `docs/extending_the_benchmark.md`.

## Paper and citation

There is no evidence-complete paper or archival DOI. `CITATION.cff` describes the software
artifact only. Citation verification and claim registration block manuscript release.
