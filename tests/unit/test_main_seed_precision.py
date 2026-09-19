from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.analyze_main_seed_precision import analyze, scalar_seed_precision_curve


def _protocol(name: str = "main_original_seed_precision_v1.yaml") -> dict[str, object]:
    protocol = yaml.safe_load((Path("study/preregistration") / name).read_text(encoding="utf-8"))
    protocol["candidate_seed_counts"] = [5, 8]
    protocol["minimum_available_pilot_seeds"] = 8
    protocol["bootstrap_repetitions"] = 200
    return protocol


def _evidence(protocol: dict[str, object]) -> dict[str, object]:
    return {
        "split": "validation",
        "publication_evidence": False,
        "groups": [
            {
                "group_id": group["group_id"],
                "seed_summaries": [
                    {"seed": 9000 + index, "value": 0.10 + index / 10000} for index in range(8)
                ],
            }
            for group in protocol["groups"]
        ],
    }


@pytest.mark.parametrize(
    "name",
    ["main_original_seed_precision_v1.yaml", "main_candidate_seed_precision_v1.yaml"],
)
def test_precision_analysis_is_model_seed_level_and_deterministic(name: str) -> None:
    protocol = _protocol(name)
    evidence = _evidence(protocol)
    first = analyze(protocol, evidence)
    second = analyze(protocol, evidence)
    assert first == second
    assert first["status"] == "PASS"
    assert first["selected_seed_count"] == 5
    assert first["validation_only"] is True
    assert first["resampling_unit"] == protocol["resampling_unit"]


def test_precision_analysis_rejects_test_or_ranking_fields() -> None:
    protocol = _protocol()
    evidence = _evidence(protocol)
    evidence["test_metrics"] = {"winner": "method-a"}
    with pytest.raises(ValueError, match="forbidden result field"):
        analyze(protocol, evidence)


def test_precision_analysis_rejects_duplicate_seed_ids() -> None:
    protocol = _protocol()
    evidence = _evidence(protocol)
    evidence["groups"][0]["seed_summaries"][1]["seed"] = 9000
    with pytest.raises(ValueError, match="unique"):
        analyze(protocol, evidence)


def test_zero_center_cannot_pass_relative_precision() -> None:
    result = scalar_seed_precision_curve(
        np.zeros(8),
        candidate_counts=(5, 8),
        repetitions=50,
        seed=1,
        confidence=0.95,
        maximum_relative_half_width=0.2,
    )
    assert result["selected_seed_count"] is None
    assert all(not item["passes"] for item in result["curve"])
