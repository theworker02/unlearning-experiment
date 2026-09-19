# Security

This repository runs local model-training code and does not require secrets or network
services for the current dataset. Treat external datasets, checkpoints, pickle files, and
third-party training scripts as untrusted. Do not load untrusted serialized Torch objects
with arbitrary object deserialization. Report security issues privately to the repository
maintainer once a disclosure channel is published.

