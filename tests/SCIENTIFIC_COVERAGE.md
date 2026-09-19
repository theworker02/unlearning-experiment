# Scientific test coverage

| Scientific operation | Unit | Property | Integration | Independent/reference |
| --- | :---: | :---: | :---: | :---: |
| Dataset splitting and disjointness | ✓ | ✓ | ✓ | content-hash leakage audit |
| Forget/retain construction | ✓ | ✓ | ✓ | set union and digest check |
| Exact-retraining exclusion | ✓ | ✓ | ✓ | source-ID and manifest-hash checks |
| Zero deletion | ✓ | ✓ | ✓ | metamorphic equality |
| Class deletion | ✓ | ✓ | pending runner | label-absence check |
| Accuracy/F1 | ✓ | ✓ | ✓ | scikit-learn |
| Reliability curves | ✓ | ✓ | ✓ method and exact-seed output | weighted-ECE identity |
| Class-conditioned disagreement | ✓ | ✓ | ✓ exact-ensemble output | analytic hidden-class fixture |
| JS divergence/total variation | ✓ | ✓ | ✓ | independent SciPy/manual checker |
| NLL/Brier/ECE | ✓ | ✓ | ✓ | analytic golden fixtures |
| Membership inference | ✓ | ✓ four signals and low-FPR output | ✓ exact-ensemble fidelity | benchmark reproduction pending |
| Linear CKA | ✓ | ✓ | ✓ | ✓ independent centered-Gram implementation |
| Representation artifact lineage | ✓ | ✓ | ✓ hashed per-split features | checkpoint-to-feature regeneration |
| Local forgotten-neighbor profile | ✓ | ✓ | ✓ exact-ensemble experiment output | analytic distance-band fixture |
| Artifact immutability | ✓ | ✓ | ✓ | exclusive-create fault injection |
| Experiment quality gates | ✓ | ✓ | ✓ | fault injection suite |
| Sequential deletion | ✓ | ✓ | ✓ stage-matched runner | cumulative-set reconstruction |

“Pending” entries are publication blockers, not implied coverage.
