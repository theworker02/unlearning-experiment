# Fashion-MNIST

Role: **Low-cost counterfactual-fidelity pilot**  
Version: `official IDX files, accessed 2026-08-30`  
Source: https://github.com/zalandoresearch/fashion-mnist  
License: MIT (Zalando SE, 2017)  
Access date: 2026-08-30

## Composition and processing

Train/validation/test counts are 9000/1000/
2500. divide uint8 pixels by 255; [0, 1] fixed scaling; no fitted statistics.

## Split

stratified fixed-seed split with seed 1729. Exact IDs and hashes are
stored in `split_manifest.json`.

## Limitations and bias

Pilot subset and modest image resolution do not establish broad vision generalization. Benchmark class labels and sampling do not represent all real-world
populations or deletion requests.

## Ethical relevance

Benchmark labels are not evidence of real-world deletion guarantees or legal compliance.
