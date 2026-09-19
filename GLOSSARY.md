# Scholarly glossary

- **Machine unlearning:** a procedure intended to remove specified training-data influence from a
  trained system without necessarily repeating full training.
- **Forget set (F):** stable sample IDs whose deletion is requested.
- **Retained set (R):** training samples remaining after deletion, R = D \\ F.
- **Exact retraining:** training a new model from initialization using only R.
- **Exact-retraining population:** independently seeded exact retraining replicas for one condition.
- **Counterfactual model:** a model produced by training without the forgotten samples.
- **Counterfactual fidelity:** multidimensional consistency of an unlearned model with the natural
  distribution of exact-retrained models.
- **Exact-retraining envelope:** empirically estimated ordinary variability among exact replicas.
- **Sequential deletion:** multiple deletion requests applied over ordered stages.
- **Membership inference:** an attack or score estimating whether an example participated in model
  training; failure of one attack is not proof of unlearning.
- **Calibration:** agreement between predictive confidence and observed frequency.
- **Representation similarity:** comparison of learned activations using symmetry-aware methods such
  as CKA rather than neuron-by-neuron identity.
- **Publication-eligible:** passed artifact gates and subsequently promoted under the research
  governance policy; it is stricter than successful program execution.
