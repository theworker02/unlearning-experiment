# Tier-2 counterfactual fidelity experiment plan

Status: design draft after the Fashion-MNIST pilot. It is not frozen and must not be
executed as a confirmatory suite until the statistical and compute gates below close.

## Scientific target

The primary question is how often conventional success criteria agree with the model's
position relative to a population trained from scratch without the forgotten data. The
study will retain separate predictive, probability, calibration, privacy, representation,
local-behavior, robustness, and cost dimensions. It will not optimize or publish a single
aggregate fidelity score unless validation shows that aggregation adds information.

## Dataset and architecture blocks

- Fashion-MNIST (full official split): small CNN and MLP; development bridge and deletion-size study.
- CIFAR-10: small CNN and ResNet-18; primary vision block.
- CIFAR-100: ResNet-18; class-complexity replication on a reduced condition set.
- UCI Dry Bean (DOI 10.24432/C50S4B, CC BY 4.0): multinomial logistic regression and MLP; tabular replication.

CIFAR licensing and exact source checksums must be documented before acquisition. Dry
Bean preprocessing must be fit on retained training folds only, with categorical labels
encoded from a frozen map. The final matrix may add another modality only after these
blocks are adequately powered.

## Methods

The core comparison is exact retraining, retained-data fine-tuning, NegGrad+ as specified
in Kurmanji et al., SCRUB including a separately evaluated rewinding variant, and Selective
Synaptic Dampening (AAAI 2024, DOI 10.1609/aaai.v38i11.29092). SISA is a distinct
training-system baseline and will be evaluated in a scoped block because it changes the
original training procedure. Every implementation requires a provenance record, license
review, deviation table, and algorithm-specific regression tests before admission.

## Factor allocation

The design is intentionally not fully factorial.

1. CIFAR-10/ResNet-18 random deletion: 0.1%, 0.5%, 1%, 5%, and 10% for every core method.
2. CIFAR-10/ResNet-18 composition at 5%: random, high/low loss, high/low confidence,
   representation cluster, and controlled label corruption.
3. Fashion-MNIST/small CNN: the same deletion fractions with random and high-loss selection.
4. CIFAR-100 and Dry Bean: random 1% and 5%, plus one scientifically appropriate structured deletion.
5. Architecture replication: CIFAR-10 small CNN versus ResNet-18 at random 1% and 5%.
6. Sequential block: five disjoint 1% requests for CIFAR-10 small CNN and Dry Bean MLP;
   rebuild the multi-seed exact oracle population after every cumulative stage.

Influence-selected deletion remains optional until estimator cost and validity are
demonstrated. “Rare” deletion requires a frozen density/region definition; it will not be
implemented by an arbitrary class-frequency shortcut.

## Statistical gate

Replication counts are governed by `study/preregistration/main_count_decision_v1.yaml`.
The selector may consume validation-only precision artifacts and independently calibrated
equivalence evidence, never confirmatory method rankings or test outcomes. Its current result
is fail-closed: 15 exact replicas are supported as a floor, while original-model and candidate-
method counts remain undetermined and the equivalence margin remains inadmissible. A matching,
hash-bound `READY_FOR_FREEZE` artifact is required before numeric counts may enter the main
profile.

The Fashion pilot established that ten pairwise distances from five oracles are not an
adequate tail estimate. A variance-calibration study will use held-out seeds and repeated
candidate seeds, then choose run counts by confidence-interval precision for the primary
estimand. Pairwise oracle distances share models and are dependent; inference must use a
model-level hierarchical bootstrap or mixed-effects formulation, not treat all pairs as
independent.

The primary estimand is the difference between expected approximate-to-exact and
exact-to-exact test JS divergence. Co-primary contradiction analysis compares frozen
forget, retain, and privacy criteria with a multidimensional counterfactual decision.
Equivalence requires a two-sided confidence interval entirely inside a prespecified
practical margin. Margins will be calibrated using controlled probability perturbations
and domain-level consequences, then frozen before confirmatory seeds are opened. TOST is
permitted only for scalar endpoints satisfying its assumptions. False-discovery-rate
control will apply within declared secondary metric families.

## Required evaluation additions

- Calibrated loss and LiRA-style membership attacks, always reported relative to exact retraining.
- Reliability diagrams and class-conditional NLL/Brier/ECE.
- Layer-selected linear CKA and frozen linear probes.
- Input- and representation-neighbor collateral-effect curves around forgotten samples.
- Gaussian noise, blur, brightness, and occlusion consistency; do not call these adversarial tests.
- Peak resident/device memory and device-time measurement in addition to wall time.
- Correlation and simple regularized/interpretable prediction of counterfactual status from cheaper metrics.

## Compute gate

Benchmark one complete CIFAR-10/ResNet-18 condition, including all oracle replicas and
metrics, on the target accelerator. Record storage, wall/device time, and peak memory.
Only then expand the draft manifest and decide seed counts. CPU availability on the
development host does not constrain the scientific architecture; `device: auto` already
routes CUDA/ROCm through PyTorch CUDA semantics and MPS where available.

## Freeze and execution order

1. Complete method/license review and dataset provenance.
2. Implement and test all metrics/methods without looking at confirmatory outcomes.
3. Run variance and target-hardware calibration with development seeds.
4. Freeze equivalence margins, seed count, failure/outlier rules, stopping rule, and manifest.
5. Commit the preregistration and record its hash (requires explicit commit authorization).
6. Execute confirmatory artifacts immutably; retain every failure.
7. Validate raw schemas/checksums, derive processed statistics, then generate tables/figures.
8. Permit manuscript quantitative claims only through the evidence manifest.
