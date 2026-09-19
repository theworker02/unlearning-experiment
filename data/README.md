# Data

Raw datasets are not committed. The current `sklearn_digits` adapter reads the dataset
bundled with the installed scikit-learn version and records a processed checksum. See
`docs/data-provenance.md` for source and transformation details.

Future download scripts must verify authoritative checksums before processing. Local raw
and processed content belongs under ignored `data/raw/` and `data/processed/` directories.

