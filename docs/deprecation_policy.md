# Benchmark deprecation policy

An identifier is not silently removed when a metric, method, split, or protocol is found flawed.
Its registry entry is marked `DEPRECATED` with affected benchmark versions, reason, replacement,
known impact on results, and migration guidance. Meaning-changing replacements receive a new ID or
benchmark major version. Deprecated outputs remain readable and retain their original provenance.
