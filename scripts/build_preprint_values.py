from __future__ import annotations

import json
from pathlib import Path


def command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    result = json.loads(
        Path("results/raw/fashion-cnn-random-01pct-schema2-rocm-pilot-v1/result.json").read_text(
            encoding="utf-8"
        )
    )
    convergence = json.loads(
        Path("results/processed/exact_seed_convergence_fashion_pilot.json").read_text()
    )
    separability = json.loads(
        Path("results/processed/phase3/PILOT-FMNIST-IID-DELETE-001-v3-r2.json").read_text()
    )
    stability = convergence["stability"]
    metrics = separability["separability"]["metrics"]
    values = {
        "PilotForgetCount": str(result["deletion"]["count"]),
        "PilotExactSeeds": str(len(result["config"]["oracle_seeds"])),
        "PilotOriginalSeconds": f"{result['runtime']['original_training_seconds']:.2f}",
        "PilotExactSeconds": f"{sum(result['runtime']['oracle_training_seconds']):.2f}",
        "PilotJSConvergence": (
            f"{100 * stability['js_divergence']['relative_change_k10_to_k15']:.2f}\\%"
        ),
        "PilotDisagreementConvergence": (
            f"{100 * stability['prediction_disagreement']['relative_change_k10_to_k15']:.2f}\\%"
        ),
        "PilotJSSeparability": f"{metrics['js_divergence']['separability_ratio']:.3f}",
        "PilotDisagreementSeparability": (
            f"{metrics['prediction_disagreement']['separability_ratio']:.3f}"
        ),
    }
    destination = Path("paper/generated/pilot_values.tex")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(command(key, value) for key, value in values.items()))
    print(destination)


if __name__ == "__main__":
    main()
