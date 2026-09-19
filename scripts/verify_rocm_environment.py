"""Fail-closed ROCm/PyTorch environment probe for the WSL experiment runner.

This is intentionally a non-training check: it verifies the accelerator backend
before an immutable experimental record is created.
"""

from __future__ import annotations

import json


def main() -> None:
    import torch

    if torch.version.cuda is not None:
        raise RuntimeError(
            f"CUDA build detected ({torch.version.cuda}); expected an AMD ROCm build."
        )
    if torch.version.hip is None:
        raise RuntimeError("PyTorch has no HIP/ROCm runtime.")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm accelerator is unavailable to PyTorch.")

    device = torch.device("cuda")  # PyTorch retains the "cuda" device API for HIP.
    values = torch.arange(1024, device=device, dtype=torch.float32)
    result = (values @ values).item()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "hip": torch.version.hip,
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "dot_product": result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
