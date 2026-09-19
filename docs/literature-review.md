# Literature review protocol and initial synthesis

Status: living review, searched 2026-08-30. This is a documented initial search, not a
completed systematic review or PRISMA study.

## Search procedure

Searches combined *machine unlearning* with *exact retraining*, *certified removal*,
*sequential/continual*, *membership inference*, *representation similarity*, *benchmark*,
and *evaluation*. Primary proceedings pages (PMLR, NeurIPS, IEEE S&P, ACM) and OpenReview
were preferred; arXiv was used for recent work without a verified proceedings version.
Works were included when they define an unlearning guarantee, introduce a representative
method, benchmark unlearning, or directly challenge an evaluation protocol. Citation
chaining and a formal duplicate/exclusion log remain required before publication.

## Established foundations

Certified removal formalizes closeness to the distribution produced by retraining and
provides a mechanism for linear classifiers [Guo et al. (ICML 2020)](https://proceedings.mlr.press/v119/guo20c.html).
Descent-to-Delete connects deletion to algorithmic stability for convex models
[Neel et al. (ALT 2021)](https://proceedings.mlr.press/v132/neel21a.html). SISA partitions
training to limit the work required by deletion, while emphasizing indistinguishability
from retraining [Bourtoule et al. (IEEE S&P 2021)](https://doi.org/10.1109/SP40001.2021.00019).
DaRE forests provide exact deletion updates for a specialized tree construction
[Brophy and Lowd (ICML 2021)](https://proceedings.mlr.press/v139/brophy21a.html).

For deep models, SCRUB evaluates application-dependent forgetting and includes a
membership-inference adaptation
[Kurmanji et al. (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/062d711fb777322e2152435459e6e9d9-Abstract.html).
The NeurIPS 2023 competition explicitly identified the lack of a standardized protocol
[competition page](https://neurips.cc/virtual/2023/competition/66581). Broad surveys now
catalogue exact, approximate, verification, privacy, and sequential settings
[Xu et al. (ACM CSUR 2024)](https://doi.org/10.1145/3603620).

## Evaluation concerns relevant to this study

Membership inference is an attack family, not a deletion certificate. Its original
black-box formulation trains an adversary to distinguish member from non-member outputs
[Shokri et al. (IEEE S&P 2017)](https://www.ieee-security.org/TC/SP2017/papers/313.pdf).
Recent work argues that ordinary attack AUC can rank retraining counterintuitively and
proposes a controlled inference game; this must be evaluated before choosing a primary
privacy statistic. Representation-level evaluation has also begun to expose failures
hidden by output metrics; a 2025 arXiv study explicitly evaluates feature representations
[Kim et al. (arXiv:2503.06991)](https://arxiv.org/abs/2503.06991).

Sequential requests are not untouched territory. SCRUB describes unbounded unlearning,
and MUSE includes sustainability under successive requests for language models
[Shi et al. (arXiv:2407.06460)](https://arxiv.org/abs/2407.06460). Accordingly, this
repository does not claim novelty for evaluating sequential deletion itself.

The Phase-II method shortlist also includes Selective Synaptic Dampening, a post-hoc
Fisher-information method evaluated with ResNet-18 and ViT
[Foster et al. (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29092), and
Approximate Data Deletion for linear/logistic models
[Izzo et al. (AISTATS 2021)](https://proceedings.mlr.press/v130/izzo21a.html). Inclusion
in the final benchmark still requires implementation and license review.

## Implication for the project

The most defensible study is a controlled synthesis: compare several behavioral,
distributional, calibration, privacy, robustness, and representation metrics to a
multi-seed exact-retraining distribution; report when those metrics disagree; and repeat
the comparison across deletion composition and sequence length. Whether the proposed
normalization is scientifically useful is an empirical question, not an assumed result.

## Gaps before a publication claim

- Complete backward/forward citation chaining and record exact query results.
- Verify peer-reviewed versions and implementation licenses for every selected method.
- Compare directly with recent comprehensive benchmarks and representation audits.
- Search 2025–2026 work on normalized oracle distributions and metric disagreement.
- Have a second reviewer independently screen the final corpus.
