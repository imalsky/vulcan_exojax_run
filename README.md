# vulcan_exojax_run

A **differentiable exoplanet forward model** and the science threads built on it.
The core chains the *live* [VULCAN-JAX](../VULCAN-JAX) photochemical-kinetics model into
the [ExoJax](https://github.com/HajimeKawahara/exojax) radiative-transfer code and
propagates gradients through the whole thing:

```
physical params ─► VULCAN-JAX ─► VMR(nz, species), T(nz), P(nz)
  (lnZ, C/O, lnKzz,     (converged column, photochemistry ON)
   T-P params)                    │  log-P bridge (differentiable interp)
                                  ▼
                        ExoJax ArtTransPure ─► transit depth (Rp/Rs)²(λ)
                        (premodit line opacity + H2-H2/H2-He CIA + Rayleigh)
                                  │  jax.jvp / adjoint
                                  ▼
                        d(spectrum)/d(param)   — sensitivity, information, retrieval
```

It only imports VULCAN-JAX and ExoJax and never modifies them.

The one thing that makes all of this possible: **VULCAN-JAX's integrator is a
`lax.while_loop`, which supports forward-mode AD (`jvp`/`jacfwd`) but not reverse-mode
through the loop.** So forward-mode is the end-to-end route (and it is the right shape:
a few physical scalars → a high-dimensional spectrum). Reverse-mode is available only
*at* the converged state (the reaction-importance adjoint in VULCAN-JAX).

---

## Installation

The two dependencies this code chains are handled differently, on purpose:

- **[VULCAN-JAX](https://github.com/imalsky/jax-vulcan)** (same author) is a pip
  dependency pinned to a commit in `requirements.txt`, so an install reproduces the
  exact chemistry this code was validated against.
- **[ExoJax](https://github.com/HajimeKawahara/exojax)** (Kawahara et al.) is a plain
  pinned pip dependency (`exojax==2.2.3`). None of its files are vendored here; it
  downloads the HITRAN line lists and CIA tables it needs into `data/` caches on first
  use (those caches are deliberately not tracked by git — see Data below).

```
git clone https://github.com/imalsky/vulcan_exojax_run
cd vulcan_exojax_run
pip install -e .                                   # editable install (pyproject.toml)
pip install -e '.[gui]'                            # + Streamlit for the jwst-tool GUI
python -m pytest retrieval_framework/tests -q     # sampler-core unit tests (fast)
jwst-tool                                          # console script: launches the GUI
```

The editable install exposes the shared library (`config`, `vulcan_chem`,
`exojax_rt`, `interp_map`) plus the `retrieval_framework` and `jwst_tool` packages,
and registers the `jwst-tool` console script. `pip install -r requirements.txt`
still works for a dependencies-only setup. A non-editable (site-packages) install
runs too, but must point `VULCAN_PROJECT_ROOT` at the project tree and
`JWST_TOOL_DATA_DIR` at a writable cache directory (the code expects the
repository's `data/` layout otherwise).

(The repo keeps the directory name the HPC scripts expect; cloning it next to a
VULCAN-JAX checkout reproduces the sibling-tree layout below unchanged.)

Scripts are run from this directory (`python -m retrieval_framework.run_smc ...`,
`python sensitivity_demo/...`); the shared library modules (`config.py`,
`vulcan_chem.py`, `exojax_rt.py`, `interp_map.py`) are imported bare from here.

**HPC / development layout.** On machines where VULCAN-JAX is developed alongside this
repo (or pip installs are impractical), the code equally supports a sibling-tree
layout: clone/copy both repos into one project root and export
`VULCAN_PROJECT_ROOT=<that root>` — `vulcan_chem.py` then imports VULCAN-JAX from
`$VULCAN_PROJECT_ROOT/VULCAN-JAX/src`. This is the layout the PBS scripts in
`runs/*/` expect (see `CLAUDE.md` for the NAS transfer rules).

---

## Shared forward-model library (top level)

Four planet-agnostic modules every thread below imports bare (`import config`, …) after
putting this directory on `sys.path`:

| file | what physics it owns |
|------|----------------------|
| `config.py` | pure constants: molecule/isotope set + masses, wavenumber band, ART pressure grid, planet defaults, cache paths. No heavy imports (safe to load before the env-sensitive VULCAN-JAX setup). |
| `vulcan_chem.py` | the VULCAN-JAX side: one warm-up convergence, then `converged_ymix(theta)` / `converged_y(...)` — re-converge the closed column as a function of (lnZ, C/O, lnKzz, T-P). Sets the network env vars + jax x64 on import. |
| `interp_map.py` | differentiable log-pressure bridge from the VULCAN grid (nz) to the ExoJax ART grid (nlayer) — `jnp.interp`, so tangents pass across it. |
| `exojax_rt.py` | ExoJax `ArtTransPure` (transmission) / `ArtEmisPure` (emission) sharing one opacity set; `transmission_depth_r(...)`, `emission_flux(...)`. |

**Cross-cutting assumptions (true for every thread):**

- **Photochemistry is ON.** Only in the photo-on regime does the warm-started
  forward-mode jvp relax to the true steady-state sensitivity (FD-validated <0.1% at
  nz=150). It is also what produces the SO₂ that anchors the WASP-39b science.
- **Closed column.** FastChem sets the equilibrium initial abundances at 10× solar and
  stays frozen/off-graph. The runner forgets the initial speciation except through the
  conserved *elemental column totals*, so metallicity and C/O are expressed as
  initial-abundance (y₀) directions on those totals.
- **Exact-elemental abundance knobs** (`abundance_mode="elemental"`, the production /
  retrieval / jwst_tool default since 2026-07-11): after the mask-scaled guess, the
  column is renormalized to Σᵢnᵢ = P/(k_BT) per layer and linearly repaired on the
  runner's own reservoir species (He/H₂O/CO/N₂/H₂S) so the column elemental ratios hit
  He/H = base, {O,N,S}/H = Z×base, C/H = Z·e^{c_o}×base **exactly**, and the conserved
  atom anchor `pv.atom_ini` is rebuilt from that column. Cold and warm continuation
  therefore share identical conserved inventories by construction. The legacy
  `"masks"` mode (published demo caches) is kept, with its elemental leakage (~0.6%
  of H per e-fold of Z; N/S leakage through the fixed-O compensation) documented in
  `vulcan_chem.py`. `chem.audit_init(theta)` and `validation/elemental_audit.py`
  verify the construction per draw.
- **Fixed-oxygen C/O guess** (`co_mode="fixed_O"`): the smooth initial guess scales
  C-bearing species and compensates O-only carriers so each layer's O total is
  invariant, valid while the compensation factor stays positive (a bound printed at
  build; the retrieval prior is capped below it). In elemental mode the projection
  makes the result exact regardless.
- **Atmospheric structure follows the proposed T-P/composition.** The runner refreshes
  the hydrostatic geometry (µ, g, H_p, dz, dzi, Hpi) in-loop from the live composition
  and the proposal's T every `update_frq` accepted steps (first firing at step 1);
  since 2026-07-11 the molecular-diffusion coefficients D_zz(T, M) (+ vm/vs where
  enabled), the convergence gate's Kzz, and the initial carry geometry are also
  rebuilt per proposal via the on-graph builders. The one remaining baseline-T bake is
  the photolysis cross-section T-interpolation (host-side upstream step; second-order)
  — and condensation, which refuses a T-varying build loudly.
- **Opacities**: CO from cached ExoMol Li2015; H₂O/CO₂/CH₄/SO₂/HCN/C₂H₂/H₂S from HITRAN
  **main isotopologue at the 296 K reference** (intensities carry the terrestrial
  isotopic abundance factor — pairing with total molecular VMR is the standard,
  slightly conservative treatment). HITRAN under-represents hot bands at the retrieved
  ~770–1540 K limb vs HITEMP/ExoMol — a known accuracy limit for real-data inference,
  swap sources per molecule in `config.MOLECULES` when the big downloads are
  acceptable. Pressure broadening defaults to **terrestrial air**; set
  `config.BROADENING="h2he"` (or the per-run profile key) for HITRAN's planetary H₂/He
  widths where available, and measure the difference with
  `validation/broadening_ab.py`. H₂-He CIA is REQUIRED physics: every depth/flux call
  takes the He profile explicitly and raises if it is missing (a silent omission
  biased the pre-2026-07-11 sensitivity-demo caches). The premodit table is baked for
  **T ∈ [300, 3000] K** — profiles outside it are not modelable (see the retrieval's
  reject-don't-clip rule).
- **ART pressure grid** spans **1×10⁻⁸ – 7 bar**. The top sits one decade *above*
  VULCAN's 1×10⁻⁷ bar chemistry top on purpose: the interpolation CLAMPS the topmost
  chemistry values over that decade (constant-abundance + isothermal extension, a
  common transmission convention — not chemistry). Without it the strong bands (CO₂
  4.3, CO 4.7 µm) saturate into a flat wall. This is an explicit, logged modeling
  choice; `validation/top_pressure_ladder.py` measures it against chemistry actually
  solved to 1×10⁻⁸ bar.
- **T-P is ExoJax's own Guillot** (`atmprof_Guillot`, a plain `jnp.exp`, forward-mode
  clean — it bypasses the Heng+14 exponential-integral pathology in VULCAN's own
  `build_atm`). The same analytic T(P) drives both the chemistry (VULCAN grid) and the
  RT (ART grid); the precise scope of that consistency (and the f=1/4 shape-parameter
  caveat) is documented in `retrieval_framework/tp_profile.py`.
- **Native spectral resolution** defaults to nu_pts=1652 (R~1000 over the production
  band) — a GPU-gradient-memory bound, not a demonstrated convergence point.
  `validation/resolution_ladder.py` is the convergence test (binned depths + Jacobian
  columns vs a nu_pts ladder, optional LSF); run it before quoting few-ppm numbers.

---

## Scientific-correctness pass (2026-07-11, external-audit response)

An external scientific audit of this bundle was answered in full; the load-bearing
changes (details in each module's docstring):

1. **Abundance knobs are exact elemental directions** (`abundance_mode="elemental"`,
   default for retrieval + jwst_tool): exact conserved ratios, Σn=P/k_BT, exact
   `atom_ini`, path-independent inventories. Legacy `"masks"` kept for the published
   demo caches.
2. **Atmospheric structure rebuilt per proposal**: D_zz(T,M), vm/vs, convergence-gate
   Kzz, and the initial carry geometry now follow the retrieved T-P (the in-loop
   hydrostatic refresh already did µ/g/H_p/dz from step 1 — the audit's "frozen
   structure" claim was narrower than stated, but real for these pieces).
3. **H₂-He CIA is required** everywhere (was silently skippable and WAS skipped by the
   sensitivity demo / zco / Fisher-forecast caches — all those caches are stale and
   must be regenerated); emission shares the transmission opacity terms (CIA + cloud;
   Rayleigh deliberately transmission-only) and is labeled emergent flux.
4. **Broadening knob** (`air`/`h2he`) + A/B script; hot-line-list limits documented as
   accuracy caveats, per-molecule source swap points marked.
5. **Evidence semantics fixed**: `logZ` is reported as evidence under the OPERATIONAL
   prior (T-P window × converged support, renormalized); the measured support fraction
   (persisted from init through checkpoints) and `logZ_box = logZ + ln f` ride with
   every output. Warm-cap rejections are counted separately per sweep/stage
   (`warmcap=`), and `validation/mala_reversibility.py` probes cap symmetry.
   Tempered (β<1) draws are labeled on every figure/export path.
6. **validate_warm now gates on three axes**: Δ logL (<0.1), binned-spectrum ppm
   (<5), and elemental-inventory agreement — not logL alone.
7. **jwst_tool v5**: floor-aware transits-to-target (photon term averages down, the
   R=100-anchored floor does not — "never" is a possible answer), offset-marginalized
   detection Δχ², d(λ)-weighted model binning, saturated modes excluded from all
   forecasts consistently.
8. **New validation suite** (`validation/`): `elemental_audit.py`,
   `resolution_ladder.py`, `top_pressure_ladder.py`, `broadening_ab.py`,
   `mala_reversibility.py` — the numerical-convergence and statistical checks the
   audit required before interpreting a real-data posterior. **Run them on the GPU
   node before the next production retrieval**; every pre-existing chemistry/spectrum
   cache (demo npz, zco/Fisher caches, jwst_tool model cache, SMC checkpoints)
   predates the physics fixes and is stale.

---

## The science threads

### `sensitivity_demo/` — where does the spectrum's information live?
**Question:** which wavelengths best constrain each physical parameter (an
observation-planning "which instrument/band?" view).
**Method:** forward-mode `jvp` of the WASP-39b transit depth w.r.t. (lnZ, C/O, lnKzz,
T_int), coloring the spectrum by each `d(depth)/d(param)`.
**Assumptions:** the four y₀/profile knobs above; scalar-`T_int` temperature shift (a
proxy; the retrieval upgrades this to a full Guillot T-P via `vulcan_chem`'s `tp_eval`
hook). nz=150, 2.9–5.2 µm for the headline, 1–15 µm for the wide-band figure.
**Outputs:** `../jax_paper/figures/exojax_sensitivity.png` (the manuscript figure) +
`data/{sensitivity,wide_sensitivity}.npz`. Key result: metallicity is best measured in
the 4.0–4.3 µm SO₂/CO₂ band (the same window JWST uses).

### `validation/` — is the gradient right?
**Question:** does the end-to-end forward-mode derivative equal a re-converged finite
difference, and does the C/O continuation actually hold C/O fixed?
**Method:** offline, CO-only (fully cached) FD checks of `d/dlnZ`, `d/dT_int`, and the
`c_o_ref` continuation; a full-resolution (nz=150) chemistry-jvp check.
**Assumptions:** none beyond the shared library; deliberately offline so it runs on a
laptop as the pre-flight before trusting any figure.
**Outputs:** pass/fail to stdout (jvp vs FD to ~2% on responding levels, machine-precision
on T), `data/smoke.npz`.

### `zco_information/` — how much *independent* info about Z vs C/O, and from where?
**Question:** the JWST spectrum constrains a Z–C/O combination; how much *unique*
information does it carry about each, and how much of it comes from disequilibrium
(quench + photochemistry) rather than equilibrium?
**Method:** Fisher/Laplace analysis on the autodiff Jacobian of the real Carter & May
2024 combined spectrum, with a true fixed-O C/O knob, marginalizing lnKzz, T_int, a
reference-radius nuisance, and per-instrument offsets. Compares equilibrium → quench →
photochem tiers.
**Assumptions (documented toy limits):** local-linear (Gaussian) Fisher; **no clouds,
no free T-P, no stellar contamination** — so *absolute* σ are best-case lower bounds,
but the *relative* statements (which wavelengths, which chemistry tier, which parameter
combination is degenerate) are robust. Equilibrium tier drifts ~3% (moldiff off).
**Outputs:** `../jax_paper/figures/zco_{information,disequilibrium,geometry}.png`,
`data/{zco_jacobians,zco_walk}.npz`. Guide: `../jax_paper/docs/ZCO_Guide.md`.

### `fisher_forecast/` — which instrument mode measures metallicity best?
**Question:** Cramér-Rao forecast of the achievable precision on metallicity and carbon
enrichment across JWST modes (NIRISS/NIRCam/G395H/PRISM/MIRI).
**Method:** Fisher matrix on the cached sensitivity Jacobian with real or PandExo noise,
marginalizing Kzz and T_int.
**Assumptions:** one calibrated photon-noise model across instruments, so bar differences
are wavelength coverage + resolution only. **Superseded** for the science by
`zco_information/` (kept as a reusable forecasting tool). Guide:
`../jax_paper/docs/Fischer_Guide.md`.

### `retrieval_framework/` + `runs/` — the full Bayesian retrieval
**Question:** given the *real* WASP-39b spectrum, what are the posterior composition and
T-P, using a differentiable **photochemical** forward model (not free-chemistry)?
**Method:** a reusable, planet-agnostic framework — adaptive-tempered SMC with a MALA
mutation kernel driven by forward-mode-jvp gradients through the whole chemistry→spectrum
chain (no BlackJAX; the SMC core is ~200 lines of pure JAX validated on an analytic
Gaussian). `retrieval_framework/` holds the machinery; each concrete run is a minimal
**case directory** in `runs/` (`case.py` = planet identity + priors + presets, plus the
PBS script, overrides, and outputs).
**Firsts** (to our knowledge; literature review 2026-07): this is the first

1. **Bayesian retrieval with gradients through a full photochemical-kinetics forward
   model.** No published kinetics retrieval uses gradients: the FRECKLL line — the only
   prior full-kinetics retrievals (Al-Refaie et al. 2024; Bardet et al. 2025) — is
   gradient-free nested sampling at 5–10 parameters on 180 CPU cores × 24 h and
   ~874,000 CPU-hours respectively, and Khorshid et al. 2024 call retrieval with
   photochemistry computationally "impractical". Published gradient-based retrievals
   (ExoJAX HMC-NUTS) use free-chemistry forward models that run in ~ms, not a stiff
   kinetics solver.
2. **Atmospheric retrieval sampled with Sequential Monte Carlo** (adaptive-tempered
   SMC with preconditioned-MALA mutations; no published SMC atmospheric retrieval
   found, 2022–2026).
3. **Full-kinetics retrieval on a single GPU inside 24 hours** (10 parameters, one
   GH200), versus the CPU-cluster scale of all prior kinetics retrievals.

Statements are as of the 2026-07 review; the WASP-39b production run is the
demonstration case.
**Assumptions (WASP-39b case):** Guillot T-P retrieved jointly with (lnZ, C/O, lnKzz) +
radius + cloud + instrument offset; two-stage solve (converge baseline composition at the
retrieved T, then apply the composition scaling and re-converge — the T-transient
otherwise erases the inventory perturbation); convergence uses the **VULCAN-master
canonical criteria** (`yconv_cri=0.01`); `count_max=5000` accepted steps, and a solve
that doesn't converge is a **failed/rejected draw, not clipped or extended**; T-P
profiles are drawn **raw and rejected if any layer leaves [300,3000] K** (never clipped);
literature-anchored priors (see `runs/w39b_smc_retrieval/case.py`).
**Outputs:** posterior + diagnostics npz and plots under
`runs/w39b_smc_retrieval/data/<preset>/`. See `retrieval_framework/README.md` (algorithm,
GPU architecture, and full development log) and `CLAUDE.md` (operational critical notes).

---

## Data (`data/`)

Tracked in git: the real Carter & May (2024) WASP-39b product CSVs (`cm24_wasp39b/`,
published measurements, small), the Rustamkulov et al. (2023) PRISM + benchmark
spectra, and the small generated sensitivity/Fisher/Z-C-O npz caches the figure
scripts read.

NOT tracked (see `.gitignore`): the HITRAN line-list caches (`exojax_linelists/`,
~190 MB) and the offline opacity cache (`opacity_cache/`: CO ExoMol + H2-H2/H2-He CIA,
~170 MB). ExoJax regenerates both on first use (on NAS this goes through the proxy the
PBS script exports). After the first populated run the tree is fully offline, and the
populated tree is what gets `scp -r`'d to HPC as one unit (never rsync — `CLAUDE.md`).

## Quick start (local, `vulcan` conda env)

```
cd vulcan_exojax_run
python validation/smoke_test.py                          # offline gradient FD check (pre-flight)
python sensitivity_demo/run_demo.py                      # headline sensitivity figure
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval   # retrieval gradient FD checks
```

The retrieval's production entry points and HPC workflow are in
`retrieval_framework/README.md` and `runs/w39b_smc_retrieval/README.md`; the transfer +
run rules are in `CLAUDE.md`.
