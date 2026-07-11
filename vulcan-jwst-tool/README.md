# vulcan-jwst-tool: JWST instrument selector (VULCAN-JAX × ExoJAX × Pandeia)

A PandExo-style planning GUI built on the vulcan-retrieval forward model
(`retrieval_framework.forward`): pick a planet and a science goal (detect
molecule X, forecast parameter constraints), run the VULCAN-JAX photochemistry
→ ExoJAX transmission model **locally**, simulate each JWST time-series mode's
transit-depth precision with the **real STScI Pandeia ETC engine**, and rank
the modes. Dist name `vulcan-jwst-tool`, import name `jwst_tool`. Version
history: `notes.md`.

## Install and launch

Local development (repo checkout, conda env `vulcan`), from the repo root:

```
pip install --no-deps -e ./vulcan-retrieval -e ./vulcan-jwst-tool
pip install streamlit pandas
jwst-tool
```

(`--no-deps` because the chemistry dependency `vulcan-jax>=0.1.17` lives on
TestPyPI, not PyPI; `streamlit` + `pandas` are the `[gui]` extra, installed
separately under `--no-deps`.) Consumer install from TestPyPI:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
```

`jwst-tool` is the console script: it preflights the `vulcan_jax` import and
the Pandeia backend env with actionable messages, then launches the GUI.
Equivalent, from the repo root:

```
streamlit run vulcan-jwst-tool/src/jwst_tool/app.py
```

First run of a new parameter set: **~2 min at the default "fast" fidelity**
(~3 min at "high"), plus ~20–60 s per freed Fisher parameter. A progress bar
tracks the stages (chemistry build → solve → spectra → Jacobians). Everything
is disk-cached at the repo-root `data/jwst_tool/` (override with
`JWST_TOOL_DATA_DIR`): repeat runs are instant.

## Planets

`planets.py` registry: **WASP-39 b** (the validated Tsai et al. 2023 baseline,
GCM T-P + Kzz), **HD 189733 b** (very bright host — saturation stress test),
**HD 209458 b**, **WASP-107 b** (low-gravity super-puff), or a **custom**
system. Every planet runs the same validated W39b SNCHO network machinery; the
identity is injected via `cfg_overrides` (gravity, Rp, R_star, orbit distance,
stellar UV spectrum from the shipped VULCAN library) for the chemistry and
`rp_cm`/`gs_cgs`/`rstar_cm` for the RT. Non-W39b planets use an isothermal
structural baseline at a representative temperature and must pick the
isothermal or Guillot T-P (the GCM baseline + Kzz-scale modes are W39b-only,
enforced loudly). All system parameters are editable in the GUI.

## What it computes

- **Forward model** (`forward.py`, subprocess): WIDE-band (1–15 µm)
  transmission spectrum, photochemistry ON by default, molecules
  H2O/CO2/CO/CH4/SO2 plus opt-in C2H2/H2S/HCN/NH3 (the SNCHO network solves
  them regardless; opting in adds their opacity + a removed spectrum each).
  Fidelity tiers: **fast** (default; nz=100, yconv 1e-2 = VULCAN master
  default, native R≈1500) and **high** (nz=150, yconv 1e-3, native R≈3000);
  both keep the full 60-layer RT grid. Knobs: metallicity (about the 10× solar
  baseline, `reanchor_atom_ini` + two-stage solve for finite steps), Δln(C/O),
  Kzz (GCM profile × factor on W39b, or constant), and the T-P
  profile — baseline+ΔT (W39b), isothermal, or Guillot (ExoJax
  `atmprof_Guillot`, the same hook the retrieval uses). Out-of-window T-P
  ([320, 2980] K) and count_max-exhausted solves **raise**, never clipped/carried.
- **Physics knobs** (all through the existing validated hooks; defaults = the
  Tsai 2023 W39b values): photochemistry on/off (off = thermochem+transport
  only; the Fisher forecast *requires* on — the validated-jvp regime),
  photolysis zenith angle (83° terminator slant default), diurnal photolysis
  factor, molecular diffusion on/off. RT side: H2/He Rayleigh scattering (ON
  by default) and an optional ExoJax power-law cloud deck (log κ₀ at 3.5 µm +
  slope α; held fixed in the Fisher forecast, i.e. no cloud marginalization).
- **Noise** (`pandeia_worker.py` in the `picaso_base` conda env — pandeia.engine
  3.0 matching the on-disk `pandeia_data-3.0rc3` refdata): per-native-pixel
  extracted flux + noise for a PHOENIX star normalized to the entered Ks mag
  (at_lambda; 2MASS zeropoint), groups auto-chosen to stay under the saturation
  limit (PandExo-style). Transit-depth error per bin:
  `var = (noise/flux)² (1/n_in + 1/n_out) / n_transits`, inverse-variance
  binned, then a **non-averaging** systematic floor in quadrature (defaults per
  mode, editable; Greene+2016-ish, in-flight performance is often better).
  Floors are quoted **per R=100 bin** and anchored there: finer bins scale the
  per-bin floor by √(R/100), so the bin slider cannot manufacture
  floor-limited significance. The photon and floor terms are returned
  separately, so multi-transit predictions average down only the photon term.
- **Science goals** (two kinds):
  - *Detect a molecule*: `σ = √Δχ²` of (full − without-X) on each mode's
    bins with a free constant depth offset profiled out (a molecule's flat
    continuum does not count as signal; matches the Fisher
    offset treatment). Still a linearized proxy: the other atmosphere
    parameters are not re-fit, so it upper-bounds a full retrieval.
  - *Constrain a parameter*: pick metallicity / C/O / Kzz (or a T-P
    parameter) + a target 1σ precision; modes ranked by the marginalized
    Fisher forecast, with transits-needed-to-target per mode and a combined
    all-modes row.
- **Fisher forecast** (`fisher.py`; automatic for the constrain goal, opt-in
  for detect): one warm-started forward-mode jvp per freed parameter through
  the full chain (the validated sensitivity pattern) + an RT-only lnR0 column.
  Per-mode rows marginalize lnR0; the combined row shares lnR0 and adds one
  absolute-depth offset nuisance per mode. Saturated modes are excluded from
  BOTH the per-mode ranking and the combined row. Model bins are d(λ)-weighted (trapezoid) means,
  matching the retrieval's exact binning-matrix convention. The GUI includes a
  "how to read this" explainer (Cramér–Rao best case, no priors, dex units).
- **Output extras**: T-P profile plot (with the [320, 2980] K opacity-window
  bounds), per-stage progress bar, reset-all button (nonce-keyed widgets).
  Default selection is 3 modes (SOSS, G395H, MIRI LRS); the ETC always
  computes all 7 per star, so adding modes later is instant.

## Modes

NIRSpec PRISM / G395H / G235H (BOTS), NIRISS SOSS order 1, NIRCam F322W2 /
F444W (grism time series), MIRI LRS slitless. A mode with no unsaturated pixels
at its shortest ramp (e.g. PRISM on WASP-39, Ks=10.2) is reported **unusable**
with the saturation numbers, matching the known PRISM brightness limit.

## Backend wiring

The Pandeia ETC engine runs in its OWN conda env; it is deliberately not a
dependency of this package. Three env vars (machine-specific defaults live in
`src/jwst_tool/instruments.py`):

- `JWST_TOOL_PANDEIA_PYTHON`: the python of an env with pandeia.engine 3.0
  (here: the `picaso_base` conda env; a base env's engine 2026.1 rejects the
  3.0rc3 refdata with an `nsuperstripe` KeyError). `noise.run_pandeia` refuses
  loudly if this python is missing.
- `JWST_TOOL_PANDEIA_REFDATA`: the matching refdata tree (here:
  `~/Documents/Important_Docs/JWST_CYCLE5/picaso_ian/data/pandeia_data-3.0rc3`).
- `JWST_TOOL_DATA_DIR`: the cache/CDBS root; default is the repo-root
  `data/jwst_tool/` (via the VULCAN_PROJECT_ROOT-aware forward config).
  `PYSYN_CDBS` is `<data dir>/cdbs` — a minimal tree: `grid/phoenix` symlinked
  from `RT-Project/picaso/reference/stellar_grids`, `comp/nonhst/johnson_j_003_syn.fits`
  fetched from ssb.stsci.edu/trds (pandeia's extinction module needs it).

## Known limits

- Model band starts at 1.0 µm (H2-H2 CIA table edge), so SOSS order 1 loses
  0.85–1.0 µm and order 2 is not offered; MIRI LRS is cut at 12 µm.
- Non-W39b planets: chemistry baseline is 10× solar FastChem EQ on an
  isothermal structural grid. The on-graph T-P drives the FULL
  chemistry structure per evaluation (rates, n₀, hydrostatic geometry via the
  runner's in-loop refresh, Dzz/vm) — the one remaining baseline-T bake is the
  photolysis cross-section T-interpolation (upstream host-side step,
  second-order). Stellar UV is the nearest shipped spectral type, shown
  explicitly in the GUI.
- Registry values are literature planning defaults; edit them for proposals.
- Default spectra are CLEAR-SKY: the cloud deck is opt-in and OFF by default,
  so feature amplitudes (and detection significances) are upper limits for
  planets with muting aerosols (W39b PRISM needed clouds). Turn the deck on
  to stress-test a goal against clouds; it is not marginalized in the Fisher
  forecast.
- "Transits → target" is floor-aware: the photon term scales 1/N,
  the R-anchored floor is fixed, and the solver reports **never** when the
  target exceeds the floor-limited ceiling.
- Fast fidelity matches High on the headline numbers (G395H SO2 3.6σ vs 3.8σ,
  Fisher σ(lnZ) 0.027 vs 0.029 dex) but mutes the weak mid-IR SO2 bands
  (MIRI LRS 0.9σ vs 1.9σ) — switch to High before quoting MIRI numbers.
- Cool columns (T ≲ 900 K, e.g. WASP-107b) converge slower: ~5 min instead of
  ~1.5 (the GUI estimate accounts for this).
- HITRAN line lists (main isotopologue), adequate for planning; not
  HITEMP/ExoMol — room-T lists under-represent hot bands at ≳1000 K, so
  absolute feature strengths lean low where hot bands matter. Broadening is
  terrestrial air by default; `config.BROADENING="h2he"` (or the profile
  override) switches to HITRAN planetary H2/He widths where available
  (`vulcan-retrieval/validation/broadening_ab.py` measures the difference).
- Abundance knobs are exact **elemental** directions
  (`abundance_mode="elemental"`): lnZ and dlnCO move conserved column
  elemental ratios exactly (H/He fixed), the column sums to P/(k_B T) per
  layer, and the chemistry's conserved atom totals match the requested gas.
- No partial-saturation strategy (pandeia group optimization only).
