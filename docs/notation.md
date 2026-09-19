# Notation index

| Symbol | Meaning |
| --- | --- |
| D | Original training dataset |
| F | Forget set requested for deletion |
| R = D \\ F | Retained training dataset |
| T | Stochastic training operator |
| U | Approximate unlearning operator |
| M = T(D) | Original trained model |
| M_u = U(M,F) | Approximately unlearned model |
| M_r = T(R) | One exact-retrained model |
| R_k = {M_r1,…,M_rk} | Exact-retraining reference population |
| d_j | Distance for fidelity dimension j |
| E_j | Empirical exact-retraining envelope for dimension j |

The symbol R denotes the retained dataset; R_k denotes a population and therefore always carries a
subscript in prose and equations. This distinction is required throughout documentation and paper.
