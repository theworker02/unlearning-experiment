# Questions reviewers should ask

## Methodology

- Does exact retraining use precisely the retained IDs, preprocessing, and training protocol?
- Are model-dependent deletion sets conditioned on a frozen original checkpoint?
- Are baselines tuned under comparable budgets without inspecting confirmatory test results?

## Statistics

- Is the training run the unit of analysis?
- Does the exact-reference population size stabilize the envelope and classification?
- Were equivalence margins and multiplicity corrections frozen before final inspection?

## Privacy and construct validity

- Does the membership attack have adequate power and a valid train/test construction?
- Is privacy behavior compared with exact retraining rather than chance alone?
- Do selected metrics actually operationalize the claimed fidelity dimensions?

## Generalization

- Are conclusions robust across datasets, architectures, deletion composition, and seed subsets?
- Are small-pilot observations being generalized beyond their research maturity?

## Reproduction

- Can exact sample IDs, checkpoints, predictions, analyses, and displayed numbers be traced?
- Does a clean checkout reproduce a flagship result within a justified tolerance?

Current answer: these questions are not yet resolved by eligible empirical evidence and must block
publication claims until the corresponding audits pass.
