# AMD ROCm on WSL

## Validated environment

- WSL distribution: Ubuntu 24.04 LTS
- GPU: AMD Radeon RX 7800 XT (`gfx1101`)
- ROCm userspace: 7.2.1
- WSL bridge: ROCDXG 1.2.2
- Python: 3.12
- PyTorch: 2.9.1 ROCm 7.2.1 AMD wheel
- Virtual environment: `/opt/venvs/unlearning-rocm`

The pre-existing Ubuntu 26.04 Python 3.14 CUDA build was removed. It was not ABI-compatible
with AMD's Python 3.12 wheels. The research environment is isolated in Ubuntu 24.04 because
that combination is supported and reproducible.

## Run the repository

From PowerShell, enter the supported distro:

```powershell
wsl.exe -d Ubuntu-24.04 -u root
```

Then run:

```bash
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib
source /opt/venvs/unlearning-rocm/bin/activate
cd '/mnt/c/Users/matth/OneDrive/Desktop/new stuff/unlearning-experiment'
python -m pytest -q
python scripts/run_experiment.py --config configs/experiments/<config>.yaml
```

PyTorch retains the API spelling `cuda` for HIP devices. Therefore `torch.device("cuda")`
and `torch.cuda.is_available()` are expected in ROCm builds; this does not mean NVIDIA CUDA
is installed. Confirm the backend with `torch.version.hip`, which must be non-null, and
`torch.version.cuda`, which must be null.

## Verification

```bash
HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib \
  /opt/venvs/unlearning-rocm/bin/python -c \
  "import torch; print(torch.__version__, torch.version.hip, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Do not upgrade PyTorch, ROCm, ROCDXG, NumPy, or the Windows display driver during a frozen
confirmatory batch. Record any environment change as a new hardware/software stratum.
