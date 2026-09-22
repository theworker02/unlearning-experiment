# Buyer evaluation â€” Measuring Counterfactual Fidelity in Machine Unlearning

## Goal

In 15â€“45 minutes, verify the Product builds or runs as documented and that proprietary notices are present.

## Steps

1. Confirm root `LICENSE` is proprietary and `ACQUISITION.md` exists.
2. Skim `README.md` install/run claims.
3. Execute:

```
```bash
python -m pip install -e ".[dev]"
python scripts/run_experiment.py --config configs/experiments/digits_mlp_smoke.yaml
pytest
make corpus
unlearn-bench datasets list
unlearn-bench validate
unlearn-bench status
```
```

4. Run tests if present (`npm test`, `pytest`, `cargo test`, `go test ./...`, etc.).
5. Record README vs observed behavior gaps in workpapers.

## Pass criteria

- [ ] Clone succeeds
- [ ] Documented happy path works **or** failure is explained
- [ ] Minimal path needs no surprise secrets
- [ ] License notices intact

*Updated: 2026-09-22*
