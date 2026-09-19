# Methodology

For training data `D`, forget positions `F`, training procedure `T`, approximate method
`U`, and seed set `S`, the implementation constructs an original model `T(D)`, candidate
`U(T(D), F, D\\F)`, and oracle ensemble `{T_s(D\\F): s in S}`.

Candidate-to-oracle distances are summarized across every oracle. Oracle dispersion is
estimated from all pairwise oracle-to-oracle distances. For each nonnegative distance
`d`, the development normalization is:

`R_d = mean_s d(U, T_s) / mean_{i<j} d(T_i, T_j)`.

`R_d` is undefined when oracle dispersion is effectively zero. It is not currently an
aggregate score, statistical test, or certificate. Metrics remain visible separately.

Dataset adapters currently cover scikit-learn Digits and checksum-verified Fashion-MNIST
IDX files. Model factories cover a linear classifier, one-hidden-layer MLP, deterministic
small CNN, and a BatchNorm-enabled small CNN used for method-validity checks.
The device resolver supports CPU, CUDA/ROCm through PyTorch's CUDA device, and MPS. Exact
training uses a declared optimizer and seed. Fine-tuning and naïve gradient ascent are
simple development baselines. SCRUB implements the NeurIPS 2023 core alternating
teacher–student objective as a documented pilot adaptation without privacy rewinding, but
remains blocked by its applicability/reproduction audit. NegGrad+ implements the same paper's
Equation 4 as a distinct literature baseline and remains blocked pending a frozen tuning budget,
behavioral reproduction, and human provenance review.
No method is certified by this framework.

Each raw record contains the canonical config hash, data hash, stable source-example IDs,
deletion hash, seeds, software/hardware metadata, runtime, oracle dispersion, and method
metrics. Exclusive file creation prevents silent overwrite.
