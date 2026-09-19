# Equivalence-margin v2: human SESOI review packet

Status: **superseded by approved class-stratified v2.1 decision; retained for audit**

## Decision requested

Decide whether a deterministic top-1/top-2 probability swap on 1% of the frozen 1,000-sample
Fashion-MNIST validation probe is an acceptable *smallest effect of scientific interest*
(SESOI) for calibrating excess Jensen--Shannon distance. This changes exactly 10 aggregate
top-1 predictions. The 0.5% and 2% controls change 5 and 20 predictions and are sensitivity
anchors, not alternative post-result choices.

The machine-readable decision record is
`study/amendments/equivalence_margin_v2_sesoi_review.yaml`. It is bound to the reviewed
protocol and input-plan hashes. The only valid decisions are `APPROVE_AS_SPECIFIED`,
`REVISE_BEFORE_FREEZE`, and `REJECT_EQUIVALENCE_ENDPOINT`.

## Why this is more defensible than v1

V1 used the exact/exact 95th percentile as a candidate margin. That describes stochastic
training variation but does not establish what discrepancy is scientifically negligible.
V2 instead defines a behavioral positive control in prediction units, estimates its excess-JS
effect on 15 exact models, and evaluates assay behavior on 10 disjoint exact-model seeds.
Approximate methods and test predictions are forbidden.

## Strongest objections

1. One percentage point is interpretable, but it is still a judgment rather than a universal
   property of machine unlearning. Approval must be scoped to this pilot assay.
2. The current SHA-256 sample selection is global, not class-stratified. Ten selected samples
   can therefore be class-imbalanced even though the validation split is balanced.
3. Swapping the two largest probabilities guarantees top-1 disagreement, but the induced JS
   change depends on each model's confidence margin. This is intentional for calibration but
   complicates cross-condition comparison.
4. The positive control tests metric sensitivity; it is not a realistic model of an unlearning
   failure and cannot validate any method.
5. A Fashion-MNIST/SmallCNN margin cannot be transferred to CIFAR, tabular data, another
   architecture, another probe, or another endpoint.
6. With only 10 holdout seeds, an 80% exceedance rule has coarse resolution. All per-seed
   outcomes and uncertainty must remain visible.

## Recommendation before approval

The conservative choice is `REVISE_BEFORE_FREEZE` unless global sample selection is a deliberate
scientific choice. A class-stratified control would make the primary 1% perturbation exactly one
sample per Fashion-MNIST class and reduce composition as an alternative explanation. If revised,
the protocol, plan, preflight report, and their hashes must be versioned again before execution.

That conservative alternative is now implemented as the separate
`equivalence_margin_calibration_v2_1.yaml` draft. It uses proportional largest-remainder
allocation with deterministic SHA-256 tie-breaking; on the balanced 1,000-example probe, the
primary control selects exactly one sample from each of the ten classes. V2 remains unchanged so
the reviewer can compare both designs rather than having the reviewed object silently replaced.

The v2.1 decision was approved under the author instruction to decide and proceed, frozen in
`study/amendments/equivalence_margin_calibration_v2_1_freeze_v2_2026-09-02.yaml`, and executed
only as a validation-only calibration. The v2 document remains unchanged as an audit record;
v2.1 does not retroactively rescue v1 or authorize an equivalence conclusion for a method.
