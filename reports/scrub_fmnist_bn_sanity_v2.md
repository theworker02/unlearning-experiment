# SCRUB BatchNorm mechanism sanity (v2)

Status: completed development-method check; not publication evidence.

Raw evidence: `results/raw/scrub-fmnist-bn-sanity-v2/result.json` (SHA-256 `947320255e23c3b790213297405d06a0b9daf8710a91cde6452b0871333d5a70`).

- Runs: 3
- Fully accepted: 2
- Failed seeds: [3102]
- Overall acceptance: `False`

## Decision

SCRUB is excluded from the comparative pilot. The BatchNorm architecture produced a non-identical initial teacher/student signal in every run, but the fixed retain-objective recovery criterion failed for one of three seeds. This is a method-validity outcome, not an equivalence, privacy, or comparative-performance result.
