# Compute tiers

- Tier 0: unit, invariant, reproducibility, and tiny integration tests; CPU, minutes.
- Tier 1: Digits development configurations; CPU, minutes.
- Tier 2: preregistered Fashion-MNIST/CIFAR and tabular matrix; GPU optional, hours/days.
- Tier 3: extended architectures, attacks, and robustness; separately budgeted.

CPU support remains mandatory. A local AMD acceleration stratum is now validated on an RX
7800 XT through WSL 2, Ubuntu 24.04, ROCm 7.2.1, ROCDXG 1.2.2, and AMD's PyTorch 2.9.1
wheel. See `docs/rocm-wsl.md` and `reports/hardware_validation.md`. This does not make the
local hardware the scientific scope of Tier 2; remote accelerators remain supported through
the same device abstraction.

Runtime is measured with a monotonic wall clock; energy and peak-memory measurement are not
yet implemented and must not be inferred from wall time.
