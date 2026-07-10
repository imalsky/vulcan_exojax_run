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
pip install -r requirements.txt
python -m pytest retrieval_framework/tests -q     # sampler-core unit tests (fast)
```

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
- **Fixed-oxygen C/O knob** (`co_mode="fixed_O"`): C/O is changed by scaling C-bearing
  species and compensating the O-only carriers so each layer's O total is invariant —
  an exact, layer-by-layer C/O change, valid while the compensation factor stays
  positive (a bound printed at build; the retrieval prior is capped below it).
- **Opacities**: CO from cached ExoMol Li2015; H₂O/CO₂/CH₄/SO₂/HCN/C₂H₂/H₂S from HITRAN
  **main isotopologue at the 296 K reference**. HITRAN's 296 K reference under-represents
  the hottest bands of a ~1100 K atmosphere vs HITEMP/ExoMol; adequate for the
  methodology (the *pattern* of sensitivity, not absolute line fidelity, is the point).
  The premodit table is baked for **T ∈ [300, 3000] K** — profiles outside it are not
  modelable (see the retrieval's reject-don't-clip rule).
- **ART pressure grid** spans **1×10⁻⁸ – 7 bar**. The top is set *below* VULCAN's 1×10⁻⁷
  bar chemistry top on purpose: without it the strong bands (CO₂ 4.3, CO 4.7 µm) go
  optically thick to the model top and the transit radius saturates into a flat wall at
  4.2–5.2 µm; extending to 1×10⁻⁸ bar removes it (constant-abundance + isothermal
  upper-atmosphere extension, standard transmission practice).
- **T-P is ExoJax's own Guillot** (`atmprof_Guillot`, a plain `jnp.exp`, forward-mode
  clean — it bypasses the Heng+14 exponential-integral pathology in VULCAN's own
  `build_atm`). The same T(P) drives both the chemistry (VULCAN grid) and the RT (ART
  grid), so one self-consistent profile.

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
