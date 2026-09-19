# Novelty audit

Status: primary-source overlap review expanded through 2026-09-01; no exclusive novelty
claim is authorized. The review is sufficient to constrain the pilot manuscript but is not
a substitute for a systematic review before confirmatory publication.

| Candidate element | Already established | Remaining defensible question |
|---|---|---|
| Retraining as counterfactual | Central to certified removal and common deep-unlearning evaluation; Lanyon et al. (2026) directly establish the importance of varying original training seeds | Whether a prespecified population-level exact-retraining envelope supports calibrated, multidimensional equivalence decisions rather than seed-conditional rankings |
| Sequential deletion | Studied by unbounded/continual work and MUSE | Whether multi-metric divergence accumulates outside natural oracle variability |
| Representation comparison | Recent representation-focused audits exist | How representation findings align or conflict with behavioral/privacy metrics under matched oracles |
| Membership inference | Widely used; known to be imperfect and protocol-sensitive | When it disagrees with exact-reference behavioral and distributional evidence |
| Aggregate distance | Related indistinguishability and benchmark scores exist | Whether a transparent normalized vector/aggregate adds information rather than hiding failures |
| Random-deletion identifiability | Wichert and Sikdar (2024) show that random forget sets can make an unlearning method difficult to distinguish from copying the original model | Whether exact-retraining population variability provides a calibrated condition-level sensitivity gate across modalities |
| Multidimensional contradictions | MUSE (Shi et al., 2025) evaluates six properties and sequential requests for language models; Gao et al. (2026) expose output/representation contradictions in vision | Whether conventional success criteria agree with distributional exact-counterfactual consistency under prespecified equivalence rules |

The working contribution is therefore an evidence-linked evaluation protocol, not a new
unlearning algorithm. The proposed retraining-dispersion normalization must be compared
against existing reference-based metrics before it is named or promoted. “For the first
time” language is prohibited unless the completed review substantiates it.

## Consequence for contribution framing

The broad claims that random deletion can be uninformative, that metrics can disagree, that
representations merit inspection, and that sequential deletion matters are already present
in prior work. Likewise, using multiple original training seeds is not a novel contribution:
Lanyon et al. (2026) show that unlearning evaluations can be sensitive to the training seed
and that repeated unlearning seeds do not generally repair a single-training-seed design.
The defensible target is narrower: a preregistered, evidence-traceable test of
whether approximate models fall within natural populations of independently exact-retrained
models, with assay sensitivity assessed before method ranking. Even this formulation remains
a candidate contribution until a broader search establishes that it is not already standard.

Primary records checked for this overlap decision include ACL Anthology DOI
10.18653/v1/2024.findings-emnlp.271, the ICLR 2025 OpenReview record TArmA033BU, PMLR
volume 300 (Gao et al., 2026), and arXiv:2510.26714v5 (Lanyon et al., 2026).
