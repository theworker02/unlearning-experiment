# Exact-retraining seed convergence pilot

Source experiment: `fashion-cnn-random-01pct-schema2-rocm-pilot-v1`. This is PILOT evidence, not a
confirmatory seed-count justification.

| k | Metric | Pairs | Mean | Pair SD | 95% replica-bootstrap interval |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | js_divergence | 3 | 0.038634 | 0.004601 | [0.033453, 0.042242] |
| 3 | prediction_disagreement | 3 | 0.158000 | 0.030883 | [0.122400, 0.177600] |
| 5 | js_divergence | 10 | 0.026173 | 0.010053 | [0.015308, 0.039565] |
| 5 | prediction_disagreement | 10 | 0.125600 | 0.033802 | [0.083257, 0.162900] |
| 8 | js_divergence | 28 | 0.020949 | 0.009435 | [0.012964, 0.030090] |
| 8 | prediction_disagreement | 28 | 0.112657 | 0.031484 | [0.083696, 0.139535] |
| 10 | js_divergence | 45 | 0.019243 | 0.008823 | [0.012691, 0.026138] |
| 10 | prediction_disagreement | 45 | 0.104427 | 0.031484 | [0.079746, 0.129014] |
| 15 | js_divergence | 105 | 0.020145 | 0.008692 | [0.015427, 0.025385] |
| 15 | prediction_disagreement | 105 | 0.107813 | 0.029790 | [0.090923, 0.125868] |

## Frozen stability diagnostic

- js_divergence: relative k=10→15 change 4.477%; diagnostic PASS.
- prediction_disagreement: relative k=10→15 change 3.141%; diagnostic PASS.

## Approximate/exact descriptive comparison at k=15

| Method | Metric | Approx/exact mean | Exact/exact mean | Exact/exact 95th percentile | Below percentile |
| --- | --- | ---: | ---: | ---: | :---: |
| fine_tune | js_divergence | 0.014274 | 0.020145 | 0.034777 | yes |
| fine_tune | prediction_disagreement | 0.087120 | 0.107813 | 0.164880 | yes |
| naive_gradient_ascent | js_divergence | 0.012908 | 0.020145 | 0.034777 | yes |
| naive_gradient_ascent | prediction_disagreement | 0.083173 | 0.107813 | 0.164880 | yes |
| scrub | js_divergence | 0.014255 | 0.020145 | 0.034777 | yes |
| scrub | prediction_disagreement | 0.087493 | 0.107813 | 0.164880 | yes |

All comparisons are descriptive for one pilot condition. Being below an empirical
pairwise percentile is not an equivalence test and does not establish successful
unlearning, privacy, or cross-condition generalization.

## Limitations

- One dataset, architecture, deletion set, and original training seed.
- Bootstrap resamples exact replicas; intervals are pilot precision diagnostics.
- The 10% stability threshold is a design diagnostic, not an equivalence margin.
