# Known variability

| Source | Controlled | Seeded | Recorded | Expected effect |
| --- | :---: | :---: | :---: | --- |
| Initialization | yes | yes | yes | Between-seed model variation |
| Minibatch order | yes | yes | yes | Between-seed optimization variation |
| Augmentation | none in P001 | n/a | yes | None |
| GPU kernels and reductions | partially | no | environment | Small numeric/platform variation |
| Floating-point accumulation | partially | no | hardware/runtime | Numeric variation |
| PyTorch and ROCm versions | pinned in record | n/a | yes | Cross-stack variation |
| GPU architecture | no | n/a | yes | Timing and numeric variation |

`R-BIT` means identical bytes, `R-NUM` means values within frozen numerical tolerances,
and `R-STAT` means the preregistered scientific decision is reproduced. Cross-device GPU
reproduction is normally evaluated as R-NUM/R-STAT, not R-BIT.

