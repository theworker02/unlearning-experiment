# Phase III quality evaluation

This evaluates existing immutable runs under the new fail-closed rules. A failed gate
does not mean the process never ran; it means the artifact cannot support publication.

| Experiment | Maturity | Gate | Publication eligible | Failures |
| --- | --- | --- | :---: | --- |
| digits-mlp-high-loss-05pct-smoke-v1 | VALIDATION | FAIL | no | recognized_maturity, raw_predictions_and_labels_present, multiple_exact_seeds |
| digits-mlp-random-05pct-smoke-v1 | VALIDATION | FAIL | no | recognized_maturity, raw_predictions_and_labels_present, multiple_exact_seeds |
| digits-mlp-rocm-random-01pct-smoke-v1 | VALIDATION | FAIL | no | recognized_maturity, raw_predictions_and_labels_present |
| fashion-cnn-high-loss-05pct-pilot-v1 | PILOT | FAIL | no | recognized_maturity, raw_predictions_and_labels_present, multiple_exact_seeds |
| fashion-cnn-random-01pct-pilot-v1 | PILOT | FAIL | no | recognized_maturity, raw_predictions_and_labels_present, multiple_exact_seeds |
| fashion-cnn-random-01pct-schema2-rocm-pilot-v1 | PILOT | PASS | no | none |
| fashion-cnn-random-05pct-pilot-v1 | PILOT | FAIL | no | recognized_maturity, raw_predictions_and_labels_present, multiple_exact_seeds |
| phase3-fmnist-class-conditioned-assay-pilot-v3 | PILOT | PASS | no | none |
| phase3-fmnist-equivalence-margin-calibration-pilot-v1 | PILOT | PASS | no | none |
| phase3-fmnist-multi-original-assay-pilot-v1 | PILOT | FAIL | no | partition_lineage_complete, raw_predictions_and_labels_present |

## Conclusion

No existing run is promoted to publication evidence. Legacy pilot artifacts predate
mandatory raw-prediction capture and explicit maturity metadata. New schema-v2 runs
can pass artifact postconditions but still remain PILOT or VALIDATION by design.
