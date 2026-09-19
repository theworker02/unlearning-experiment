# Data provenance

The current implementation uses `sklearn.datasets.load_digits`: 1,797 handwritten digit
images, 8×8 pixels, 10 classes, derived from the UCI Optical Recognition of Handwritten
Digits dataset. Acquisition occurs through the installed scikit-learn package; no raw
copy is committed. See scikit-learn's
[dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
and the [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
for provenance and terms.

Every run records the scikit-learn version and SHA-256 of the float32 scaled feature
matrix plus int64 labels. Splits retain original row IDs. Forget IDs and their SHA-256
are recorded. A checksum mismatch is evidence of a different processed dataset and must
not be silently accepted as the same condition.

Before publication, every added dataset needs a versioned acquisition script, authoritative
license record, expected raw checksums where stable, and transformation hash.

## Fashion-MNIST pilot

Fashion-MNIST is sourced from the official Zalando Research repository under the MIT
license. The acquisition script verifies the upstream published MD5 values:

- `train-images-idx3-ubyte.gz`: `8d4fb7e6c68d591d4c3dfef9ec88bf0d`
- `train-labels-idx1-ubyte.gz`: `25c81989df183df01b3e8a0aad5dffbe`
- `t10k-images-idx3-ubyte.gz`: `bef4ecab320f06d8554ea6380940ec79`
- `t10k-labels-idx1-ubyte.gz`: `bb300cfdad3c16e7a12a480ee83cd310`

The pilot deterministically selects a stratified 10,000-example training pool, reserves
1,000 for validation, and selects 2,500 official test examples. Source IDs remain the
official IDX row positions. This subset is development evidence, not a substitute for
the full Fashion-MNIST publication block.
