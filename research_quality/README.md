# Research quality system

This directory defines the fail-closed rules separating a successful process from admissible
scientific evidence. The executable implementation is `exact_unlearning.quality`; these YAML
files are the human-reviewable policy source.

Maturity is monotonic only through explicit validation. An experiment does not become
`PUBLICATION` because it completed, and stored raw artifacts are never edited to promote them.
Promotion is recorded in the generated experiment ledger after all applicable gates pass.

States are `EXPLORATORY`, `PILOT`, `VALIDATION`, `CONFIRMATORY`, `PUBLICATION`, and
`REPRODUCED`. Confirmatory execution requires a frozen analysis plan. Publication and
reproduced states are assigned after postconditions, not requested by an ordinary run config.
