# Reproduction protocol

## P001-R1 — artifact reproduction

1. Verify the source archive and `provenance/*.sha256` files.
2. Verify Fashion-MNIST and the archived P001 deletion manifest.
3. Run `make verify-p001`.
4. Run `make reproduce-p001-r1` to regenerate analysis and F002 from predictions.
5. Compare generated source data and figure hashes with the archive manifest.

## P001-R2 — computational reproduction

1. Create the documented Python/PyTorch environment and run the hardware probe.
2. Acquire and validate the official IDX files.
3. Use the exact P001 split, deletion manifest, hyperparameters, and seeds.
4. Train the original model and all 15 retain-only exact models.
5. Verify deletion exclusion before every exact training.
6. Execute all approximate methods and persist predictions.
7. Run gates, analysis, and figure generation.
8. Compare numerically using frozen tolerances; do not demand cross-platform bitwise identity.

The current command performs R1 verification. A safe R2 command requires an isolated
output root and is marked unavailable until that staging workflow is implemented; it must
never overwrite P001.

