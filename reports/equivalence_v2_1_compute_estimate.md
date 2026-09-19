# Equivalence-margin v2.1 compute estimate

Status: empirical planning estimate, not a benchmark or execution authorization.

The 20 planned new exact trainings have a serial median estimate of 22.53 seconds and a 95% bootstrap interval of 22.19--22.94 seconds. The estimate resamples 30 condition-matched pilot runtimes from the recorded WSL/ROCm environment.

It excludes validation inference, data loading, process startup, parallel scheduling, peak memory, and transfer to another hardware or software environment.
