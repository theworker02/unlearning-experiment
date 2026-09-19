# Extending Benchmark 0.5

Add a registry entry before implementing an extension. Dataset adapters must return stable
sample IDs and a dataset hash through `DatasetSplit`. Architectures are created behind the model
factory. Deletion strategies must emit retained and forgotten positions and stable IDs; persist
the final IDs rather than regenerating publication sets. Methods use `UnlearningContext` and
must declare whether they are simple baselines plus citation and implementation provenance.

An extension is accepted only after unit, invariant, property/reference, integration, and fault
tests appropriate to its scientific failure modes pass. It must also produce schema-versioned,
hashed output accepted by the quality gate. Adding a new identifier is backward-compatible;
changing an existing metric, split, or scenario meaning requires a benchmark version change.
