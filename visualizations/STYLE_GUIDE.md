# Scientific visualization style guide

Benchmark protocol: **0.5.0**. This guide governs scientific communication, not branding.

## Evidence classes

Every figure is `conceptual` or `empirical`. Conceptual figures must declare
`contains_empirical_data: false`. Empirical figures require a metadata sidecar listing accepted
experiment IDs, input artifact hashes, analysis code, output hash, and generation time. A script
must refuse to generate an empirical figure when no gate-passing inputs exist.

## Typography and layout

Use a legible sans-serif face for figures and the manuscript's body face for tables. Minimum
final-size text is 8 pt. Figure titles state the measured object, not an interpretation. Panel
letters are uppercase. Legends are ordered by benchmark method registry, not result magnitude.

## Axes and uncertainty

Axes include units and begin at zero when magnitude comparison requires it; truncation must be
visibly disclosed. Report confidence intervals as intervals, not unexplained error bars. Captions
state confidence level, resampling unit, number of independent runs, and whether intervals are
descriptive or inferential. Do not connect independent categorical conditions with lines.

## Identifiers and precision

Use canonical registry IDs in source data and human-readable labels in display text. Preserve
full precision in released data. Tables normally display three significant digits, adding digits
only when resolution and uncertainty justify them. `±` must define the represented quantity.

## Color accessibility

Use colorblind-safe hues plus shape, line style, or direct labels. Critical distinctions must
remain interpretable in grayscale. Avoid red/green-only pass/fail encoding. Conceptual exact and
approximate paths use both distinct color and dash pattern.

## Prohibitions

No invented curves, rankings, matrices, error bars, sample counts, or placeholder statistics.
Dimensionality-reduction plots must name algorithm, seed, parameters, input representation, and
the limitation that a two-dimensional projection is not the full geometry.
