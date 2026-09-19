from pathlib import Path

import pytest
import yaml

from scripts.validate_neggrad_plus_protocol import validate


def test_neggrad_plus_protocol_is_bounded_and_execution_disabled() -> None:
    result = validate()
    assert result["candidate_count"] == 8
    assert result["candidate_runs"] == 24
    assert result["planned_total_model_updates"] == 30


def test_neggrad_plus_protocol_rejects_test_based_selection(tmp_path) -> None:
    source = Path("study/preregistration/neggrad_plus_reproduction_v1.yaml")
    protocol = yaml.safe_load(source.read_text())
    protocol["stage_2_validation_only_tuning"]["selection_data"]["test_split"] = "permitted"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="test split must be prohibited"):
        validate(path)
