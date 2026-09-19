# Scikit-learn Digits

Role: **Infrastructure validation**  
Version: `1.9.0`  
Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html  
License: See scikit-learn and UCI dataset records  
Access date: 2026-08-30

## Composition and processing

Train/validation/test counts are 1257/180/
360. divide pixels by 16; [0, 1] fixed scaling; no fitted statistics.

## Split

stratified fixed-seed split with seed 19. Exact IDs and hashes are
stored in `split_manifest.json`.

## Limitations and bias

Development-only small grayscale dataset; forbidden as publication evidence. Benchmark class labels and sampling do not represent all real-world
populations or deletion requests.

## Ethical relevance

Benchmark labels are not evidence of real-world deletion guarantees or legal compliance.
