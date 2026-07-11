# jwst_tool: version history and physics-audit log

Historical log for the instrument selector, moved out of `README.md` (which
keeps only current reference content). The quoted fragments below are verbatim
from the pre-0.6 README, where they were written into the reference text as
each version shipped.

## v1: initial tool

The README's "Known limits" section was originally labeled "(v1)": the model
band (1.0 um H2-H2 CIA edge, MIRI LRS cut at 12 um), the literature-default
planet registry, and the no-partial-saturation limit date from the first
version.

## v4: physics audit (Rayleigh + exposed physics knobs)

Verbatim: "H2/He Rayleigh scattering (ON by default as of v4 — earlier versions
omitted it, biasing the <1.5 µm slope)".

## v5: 2026-07-11 audit response (noise, statistics, elemental abundance map)

All cached spectra were invalidated (`_VERSION = 5`). Verbatim fragments:

- "*Detect a molecule*: `σ = √Δχ²` of (full − without-X) on each mode's bins
  with a free constant depth offset profiled out (v5 — removing a molecule's
  flat continuum no longer counts as signal; matches the Fisher offset
  treatment)."
- "Floors are quoted **per R=100 bin** and anchored there: finer bins scale the
  per-bin floor by √(R/100), so the bin slider cannot manufacture floor-limited
  significance (v5). The photon and floor terms are returned separately, so
  multi-transit predictions average down only the photon term."
- "Saturated modes are excluded from BOTH the per-mode ranking and the combined
  row (v5 — previously the combined row silently included them)."
- "\"Transits → target\" is floor-aware as of v5: the photon term scales 1/N,
  the R-anchored floor is fixed, and the solver reports **never** when the
  target exceeds the floor-limited ceiling (the old 1/√N scaling was optimistic
  exactly where the floor dominated)."
- "Abundance knobs are exact **elemental** directions as of v5
  (`abundance_mode="elemental"`): lnZ and dlnCO move conserved column
  elemental ratios exactly (H/He fixed), the column sums to P/(k_B T) per
  layer, and the chemistry's conserved atom totals match the requested gas.
  v4 and earlier used species-mask scalings with documented elemental leakage
  — all cached spectra were invalidated (`_VERSION = 5`)."
- "As of v5 the on-graph T-P drives the FULL chemistry structure per evaluation
  (rates, n₀, hydrostatic geometry via the runner's in-loop refresh, Dzz/vm) —
  the one remaining baseline-T bake is the photolysis cross-section
  T-interpolation (upstream host-side step, second-order)."

The v5 statistics/noise changes are the jwst_tool arm of the repo-wide
2026-07-11 scientific-correctness pass; the full audit summary is preserved in
`../vulcan-retrieval/notes.md` and the operational consequences in the
repo-root `CLAUDE.md`.
