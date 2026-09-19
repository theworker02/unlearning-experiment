# Baseline fairness audit

Status: **PARTIAL — publication comparison blocked**

| Method | Provenance | Current role | Fairness status |
| --- | --- | --- | --- |
| Exact retraining | Project implementation | Counterfactual control | Validation only; training adequacy curves incomplete |
| Fine-tuning | Project implementation | Simple engineering baseline | Clearly labeled; no literature-competitive claim |
| Naive gradient ascent | Project implementation | Simple engineering baseline | Clearly labeled; not represented as state of the art |
| NegGrad+ | Clean-room Equation 4 implementation from Kurmanji et al. | Literature baseline candidate | Objective unit-tested; blocked pending tuning, reproduction, and human provenance review |
| SCRUB core | Clean-room adaptation of Kurmanji et al. | Historical interface smoke test only | Blocked: deterministic small-CNN applicability failure; omits rewinding; provenance and behavioral reproduction remain incomplete |
| SSD | Clean-room Equation 4 core from Foster et al.; author-linked MIT repository verified | Literature-backed candidate; registry-disabled | Equation core unit-tested; tuning protocol drafted but unlocked; repository version/transitive audit, reproduction, and human review remain blocked |

Before publication, every competitive method requires a frozen tuning budget, validation-only
selection procedure, source/official-code comparison where compatible, license record, and a
method reproduction check. No method may tune against final test or exact-envelope outcomes.

The machine-readable readiness contract is `research/method_readiness.yaml`. At this audit point,
fine-tuning and naive gradient ascent are eligible only for development pilots; SCRUB is blocked by
its sanity artifact and human provenance review; NegGrad+ is registered but remains tuning,
reproduction, and provenance-blocked under the execution-disabled protocol
`study/preregistration/neggrad_plus_reproduction_v1.yaml`; SCRUB+R and SSD are named candidates without registered
implementations. The main profile therefore remains mechanically non-launchable even if its
seed-count fields are later populated.

## Fine-tuning baseline selection

Fine-tuning remains a simple engineering baseline and is not presented as a competitive unlearning
method. Its nine-candidate, three-seed selection contract is now explicit in
`study/preregistration/fine_tune_tuning_v1.yaml`: test data and exact-retraining outcomes are
prohibited during tuning; candidates must satisfy a fixed five-point validation-utility constraint;
and the feasible candidate with maximum forget-training cross-entropy is selected with deterministic
tie-breaks. Only that winner may be evaluated once on the test split per original seed.

The complete runner refuses the draft protocol before reserving its immutable experiment ID. The
independent artifact validator checks the protocol hash, complete candidate-by-seed grid, exact
hyperparameters, finite metrics, checkpoint hashes, recomputed selection, and winner-only test
evaluation. The main profile now requires that artifact. The protocol is unlocked and unauthorized,
so no tuning result exists and fine-tuning remains ineligible for the main study.

## SSD implementation-license finding

The AAAI 2024 paper, author-linked implementation repository, and repository license have been
verified. The repository is pinned at commit `cdfdc0e35c1908e032a6e150d882b0fa17833f85`
(2024-09-11). Its top-level license is MIT, but that fact alone does not establish the provenance or
release eligibility of transitive code, data, pretrained checkpoints, or dependencies. The pinned
tree has 30 files, no submodules, no bundled checkpoints, and no dependency manifest or lockfile.
Its README instead names unversioned packages, an external container, W&B, a Hugging Face ViT, and
downloaded or manually supplied datasets. The adaptive SSD repository is later work and is not
treated as the reference implementation of the original method.

The pinned source attributes code to Avalanche and Bad Teaching (both MIT-licensed), plus Fast
Machine Unlearning and `pytorch-cifar100`. Exact revisions of the latter repositories were audited:
`252bc9287a511774ab857d1f682845f28584eb25` and
`11d8418f415b261e4ae3cb1ffe20d06ec95b98e4`, respectively. Neither tree contains a license,
notice, or copying file. Absence of a license is not permission to redistribute. No upstream source
is included locally, and these attributions block adopting the official repository as a
distributable reference environment without human legal review.

The paper specifies selection using the diagonal Fisher information ratio and dampening selected
parameters by `min(lambda * F_D / F_Df, 1)`. This equation-level record is preserved in
`research/method_reproduction/ssd.yaml`, including the paper's reported tuning ranges and defaults.
No official SSD code has been copied. A local clean-room core now computes per-example empirical
diagonal Fisher statistics and implements the paper's strict selection and multiplicative dampening
rule, with equation and edge-case tests. It is intentionally absent from the runtime method registry:
no official commit has been pinned and no behavioral result has been reproduced. SSD therefore
remains excluded from every executable profile. Its bounded validation-only tuning and reproduction
protocol is execution-disabled at `study/preregistration/ssd_reproduction_v1.yaml`; locking it still
requires a measured dry-run estimate and authorization. Version/transitive audit, reproduction, and
human provenance/license review also remain outstanding.

Static comparison found that the pinned official code squares minibatch-loss gradients and averages
over batches, while the local implementation squares per-example log-likelihood gradients before
averaging. The local choice follows the empirical expectation without minibatch gradient cross-terms,
but it is not behaviorally identical to the official code. The official implementation also exposes
an exponent parameter (set to one by its wrappers), while the local implementation encodes Equation 4
directly. Reference-code equivalence is prohibited until both estimators are compared under a frozen
sensitivity protocol.

The estimator comparison is now bounded prospectively in
`study/preregistration/ssd_fisher_estimator_sensitivity_v1.yaml`. It uses no test data or exact-model
outcomes, includes an analytic synthetic control, fixes three original seeds and batch sizes 1, 32,
and 128, and requires reporting all observed estimator dependence. The protocol remains unlocked,
execution-disabled, and non-publication evidence; no sensitivity result exists yet.

Both estimators and the full guarded assay path are implemented. The synthetic control verifies the
cross-term distinction exactly: on the declared two-sample linear case, the per-example estimator and
minibatch estimator at batch size one produce 0.125 for each active entry, while the two-sample
minibatch estimate produces 0.0625. This is an analytic implementation check, not empirical evidence
about SSD. The runner refuses an unlocked or unauthorized protocol before reserving its immutable
experiment ID. Its independent validator rehashes Fisher tensors, checkpoints, and prediction arrays;
recomputes tensor, selection, update, and validation-behavior comparisons; and enforces per-example
loader-batch invariance. No assay run has been executed.

The publication profile now names both SSD evidence requirements explicitly. First, the Fisher
sensitivity result must pass `scripts/validate_ssd_fisher_sensitivity_artifact.py`. Second, the fixed
sanity and validation-only tuning outputs from `ssd_reproduction_v1.yaml` must jointly pass
`scripts/validate_ssd_artifacts.py`. Passing either requirement cannot register SSD or remove its
upstream-license, reference-environment, or human provenance blockers. Missing outputs are reported
as artifact failures rather than being conflated with an unspecified tuning or reproduction task.

## SCRUB implementation-license finding

The authors' public repository identifies itself as the Python implementation of the paper and
contains an `Academic Software License` derived from SelectiveForgetting. The license permits
non-commercial research use and restricts commercial use and redistribution. The repository was
inspected at <https://github.com/meghdadk/SCRUB> and its license at
<https://github.com/meghdadk/SCRUB/blob/main/License.md> on 2026-08-31.

The local SCRUB implementation is intended as a clean-room implementation of the published
algorithm; the public release must not copy or redistribute code from the authors' repository.
Because behavioral comparison against a reference implementation does not itself settle
derivative-work or license questions, a human provenance review remains a release blocker. The
repository-wide MIT license must not be represented as licensing third-party SCRUB source.

## Paper-to-code conformance finding

The local objective has been checked against Equations 1--3 and Algorithms 1--3 in the NeurIPS
proceedings paper. The KL direction, forget-step sign, retain distillation term, cross-entropy term,
teacher freezing, student initialization, and max/min ordering are represented explicitly and covered
by unit tests. The machine-readable record is `research/method_reproduction/scrub.yaml`.

This is an objective-conformance check, not a method reproduction. The local pilot omits SCRUB+R,
the paper's validation-based stopping rule, and its learning-rate decay, and it uses a different dataset,
architecture, and development hyperparameters. No published SCRUB result has been reproduced, and
the behavioral and human provenance reviews remain release blockers.

The audit also found that the deterministic local small CNN produces identical student and teacher
logits at the initial max-step. At that equality point, the KL objective has no model-derived ascent
signal; any movement caused by floating-point residuals and adaptive-optimizer amplification is not
accepted as evidence that SCRUB's forgetting mechanism executed. Future runs now fail closed in this
case. Historical P001 SCRUB records remain immutable infrastructure artifacts but are not valid method
reproduction or competitive evidence.

A BatchNorm-enabled small CNN now passes the static train/eval signal test and is the sole candidate
for the execution-disabled sanity protocol in
`study/preregistration/scrub_sanity_reproduction_v1.yaml`. This establishes an implementation path,
not behavioral validity: the fixed-setting, three-seed mechanism check has not been authorized or run,
and SCRUB remains excluded from comparative evidence.

## SCRUB tuning gate

The missing SCRUB tuning budget is now bounded in `study/preregistration/scrub_tuning_v1.yaml`.
Eight candidates are crossed with the three original seeds from the sanity protocol, and the runner
must reuse those hash-verified original checkpoints. It cannot reserve a tuning experiment ID unless
the fixed-setting sanity artifact passes independent validation. Test data and exact-retraining
outcomes are prohibited during tuning.

Every candidate retains its mechanism trace. The validator independently recomputes the max/min
mechanism criteria, requires mechanism validity in every seed for feasibility, checks a fixed
five-point validation-utility constraint, and recomputes the forget-cross-entropy selection and
tie-breaks. Mechanism-invalid and unfavorable candidates cannot be silently excluded. Only the
selected candidate may receive one test evaluation per original seed. Both protocols remain draft,
execution-disabled, and provenance-blocked; no SCRUB tuning result exists.
