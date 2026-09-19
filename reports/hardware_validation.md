# Hardware validation

Validated on 2026-08-30. This is infrastructure validation, not publication evidence.

## Result

- Windows device: AMD Radeon RX 7800 XT
- WSL device agent: `gfx1101`, AMD Radeon RX 7800 XT
- WSL distribution: Ubuntu 24.04.4 LTS
- WSL kernel: 6.18.33.2-microsoft-standard-WSL2
- ROCm: 7.2.1
- ROCDXG: 1.2.2
- Python: 3.12.3
- PyTorch: 2.9.1 ROCm 7.2.1 AMD wheel
- HIP runtime reported by PyTorch: 7.2.53211
- NVIDIA CUDA runtime reported by PyTorch: null
- Dependency check: passed
- Repository test suite in ROCm environment: 19 passed

`rocminfo` enumerated the discrete GPU. PyTorch allocated two 2048 by 2048 tensors on the
GPU, computed their matrix product, synchronized the device, returned finite values, and
agreed with the corresponding CPU slice within a maximum absolute difference of
`2.6702880859375e-05`. This is a smoke check, not a numerical-precision characterization.

An end-to-end Digits development experiment completed at
`results/raw/digits-mlp-rocm-random-01pct-smoke-v1/result.json`. The immutable record reports
the ROCm PyTorch build, a non-null HIP version, a null CUDA version, and successful exact-
retraining deletion exclusion. Digits remains excluded from publication evidence.

## CUDA removal

The unsupported PyTorch 2.12 CUDA 13 installation and its Python-level `cuda-*`, `nvidia-*`,
and Triton dependencies were removed from the pre-existing Ubuntu 26.04 distro. The supported
Ubuntu 24.04 research environment contains AMD ROCm PyTorch and no NVIDIA CUDA runtime.

PyTorch exposes HIP accelerators through its historical `torch.cuda` API. Consequently, the
raw experiment's device string is `cuda` even though `torch.version.cuda` is null and
`torch.version.hip` is populated.

## Current environment drift

The validation above is historical and remains the environment record for its archived runs.
It is not a claim about the machine's present default WSL Python. On 2026-09-02, a new
validation-inference benchmark preflight found `/usr/bin/python3` but `import torch` failed with
`ModuleNotFoundError`; a bounded search under `/home/mloon25` found neither `pyvenv.cfg` nor a
`torch` package directory. No training or prediction generation was attempted after that failure.

The current benchmark is therefore blocked by environment drift. The machine-readable record is
`reports/hardware_environment_drift_2026-09-02.json`. The ROCm environment must be restored and
the GPU smoke checks repeated before new timings can be compared with the archived measurements.
Falling back to Windows CPU would answer a different compute question and is not used here.

## Restored isolated execution environment (2026-09-02)

The default WSL distribution remains unsuitable for this study: it is Ubuntu 26.04 with Python
3.14 and no discoverable PyTorch environment. A supported, isolated Ubuntu 24.04 / Python 3.12
environment was restored at `/root/.venvs/unlearning-rocm721` instead. It uses the AMD
`torch-2.9.1+rocm7.2.1` wheel and matching ROCm Triton wheel, with `numpy==1.26.4`.

The non-training verifier `scripts/verify_rocm_environment.py` passed there with
`HSA_ENABLE_DXG_DETECTION=1`: it reported the RX 7800 XT, populated HIP, a null CUDA version,
and a finite device-resident dot product. This repair is an environment fact, not a new
scientific result or a replacement for the archived hardware-validation record.
