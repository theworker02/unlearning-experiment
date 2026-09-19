from pathlib import Path

import pytest
import yaml

from scripts.run_scrub_sanity_reproduction import evaluate_mechanism, main


def test_scrub_sanity_runner_refuses_draft_protocol_before_reservation(tmp_path: Path) -> None:
    source = Path("study/preregistration/scrub_sanity_reproduction_v1.yaml")
    protocol = yaml.safe_load(source.read_text(encoding="utf-8"))
    protocol["status"] = "DRAFT_NOT_LOCKED"
    protocol["execution_authorized"] = False
    protocol["artifacts"]["immutable_output_root"] = str(tmp_path / "raw")
    draft = tmp_path / "draft.yaml"
    draft.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="LOCKED and authorized"):
        main(draft)
    assert not (tmp_path / "raw").exists()


def test_mechanism_acceptance_uses_directional_trace_rules() -> None:
    trace = [
        {
            "max_gradient_l2_mean": 0.2,
            "before_max": {"forget_teacher_student_kl": 0.0},
            "after_max": {"forget_teacher_student_kl": 0.3},
            "before_min": {"retain_objective": 1.2},
            "after_min": {"retain_objective": 0.8},
        }
    ]
    assert all(evaluate_mechanism(trace).values())
