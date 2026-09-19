# Preregistration draft

Status: **DRAFT — not frozen or committed; confirmatory experiments must not begin.**

## Research questions

The study asks whether approximate unlearning reproduces exact-retrained behavior across
predictions, probability distributions, calibration, privacy attacks, representations,
robustness, collateral effects, deletion sizes/compositions, architectures, datasets,
and sequential requests. Metric disagreement is a primary target of analysis.

## Directional hypotheses

H1–H6 follow the master protocol: preserved aggregate accuracy may coexist with
counterfactual discrepancy; sequential and difficult/influential deletion may increase
discrepancy; membership inference alone may be insufficient; collateral damage and
architecture dependence may occur. Every hypothesis may be rejected.

## Planned primary metrics

JS divergence and top-1 disagreement to multi-seed retraining; retained/test utility;
forget-set privacy audit; representation distance. Calibration, total variation,
parameter distance, robustness, cost, and class-conditional effects are secondary pending
pilot validation. No scalar aggregate is preregistered.

## Freeze blockers

- Complete literature/novelty audit.
- Add at least two publication datasets beyond debugging data and their licenses/checksums.
- Select faithful methods only after verifying papers and implementation licenses.
- Run pilot variance estimation and complete prospective power analysis.
- Fix seeds, conditions, exclusions, inferential families, and compute budget.
- Obtain an actual Git commit hash after explicit authorization to commit.

