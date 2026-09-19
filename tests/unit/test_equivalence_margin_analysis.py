from __future__ import annotations

import numpy as np

from scripts.analyze_equivalence_margin_calibration import js_divergence


def test_js_divergence_is_zero_for_identical_probabilities() -> None:
    values = np.asarray([[0.75, 0.25], [0.2, 0.8]])
    ids = np.asarray([4, 9])
    assert js_divergence((values, ids), (values, ids)) == 0.0


def test_js_divergence_rejects_different_sample_ids() -> None:
    values = np.asarray([[0.75, 0.25]])
    try:
        js_divergence((values, np.asarray([4])), (values, np.asarray([5])))
    except ValueError as error:
        assert "sample IDs" in str(error)
    else:
        raise AssertionError("mismatched sample IDs were accepted")
