from pathlib import Path

import pytest
import yaml

from scripts.validate_fine_tune_protocol import validate


def test_fine_tune_protocol_is_bounded_and_execution_disabled() -> None:
    result = validate()
    assert result["candidate_count"] == 9
    assert result["candidate_runs"] == 27
    assert result["planned_total_model_updates"] == 30


def test_fine_tune_protocol_rejects_test_based_selection(tmp_path: Path) -> None:
    protocol = yaml.safe_load(Path("study/preregistration/fine_tune_tuning_v1.yaml").read_text())
    protocol["validation_only_tuning"]["selection_data"]["test_split"] = "permitted"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="test split must be prohibited"):
        validate(path)
