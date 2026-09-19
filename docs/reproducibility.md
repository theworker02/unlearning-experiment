# Reproducibility design

Deterministic CPU mode seeds Python, NumPy, Torch, model initialization, and DataLoader
shuffling. The same software stack and CPU run must reproduce model parameters exactly;
the test suite enforces this for a tiny run. Cross-version and GPU bitwise identity are
not promised. Those comparisons need declared numerical tolerances.

Raw artifacts use exclusive creation and cannot be overwritten. Derived tables and
figures are rebuildable and may be replaced. Experiment IDs, canonical configuration
hashes, processed-data hashes, deletion hashes, source IDs, and environment versions
support auditing. Git commit is recorded when the directory is a committed repository.

