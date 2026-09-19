# Methods reference

P001 uses official Fashion-MNIST IDX files, normalized to `[0,1]`, with deterministic
stratified subset and validation selection. The exact dataset hash and IDs are in the raw
record. Its small CNN has two convolution/ReLU/pooling blocks, a 64-dimensional projection,
and a ten-class output. AdamW trains for four epochs with batch size 128, learning rate
0.001, and weight decay 0.0001.

The frozen IID deletion manifest selects 90 of 9,000 training samples using seed 1009.
Exact retraining asserts that none occur in the retained positions. Seeds 2001–2015 are
trained independently from scratch. Fine-tuning and naïve gradient ascent are simple
engineering baselines; SCRUB is an adapted clean-room literature-backed baseline.

For class probabilities `p` and `q`, Jensen–Shannon divergence is
`0.5 KL(p || m) + 0.5 KL(q || m)`, where `m=(p+q)/2`. Disagreement is the fraction of
different argmax predictions. ECE is the weighted absolute confidence/accuracy gap over
15 bins. Brier score is mean summed squared probability error. Linear CKA compares
centered activation Gram structure and avoids neuron-identity matching.

Training seeds initialization and minibatch order. Deterministic algorithms are requested.
ROCm still permits platform-dependent floating-point variation. Checkpoints, predictions,
metrics, runtimes, package versions, dataset identity, split IDs, deletion IDs, and hashes
are recorded. See the source configuration for every material hyperparameter.

