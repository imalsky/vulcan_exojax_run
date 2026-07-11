"""Forward model runner: VULCAN-JAX photochemistry -> ExoJax transmission spectrum.

Two faces:

* Imported by the GUI (light): ``params_key`` / ``cache_path`` / ``load_result``
  touch only the disk cache -- no JAX, no VULCAN, no ExoJax imports.
* Run as a script (heavy):  ``python jwst_tool/forward.py params.json``
  runs the live pipeline and writes the npz cache entry. Progress goes to
  stdout as "[fwd] ..." lines; machine-parsable "[fwd] PROG <frac> <label>"
  lines drive the GUI progress bar.

Planets: every system in ``planets.PLANETS`` (plus "custom") runs on the same
validated W39b SNCHO machinery -- the planet identity (gravity, radius, star,
orbit, UV spectrum) is injected via cfg_overrides for the chemistry and via
profile rp_cm/gs_cgs/rstar_cm for the RT. Only WASP-39b has the GCM T-P + Kzz
baseline; other planets get an isothermal structural baseline at a
representative temperature and MUST use the isothermal / Guillot T-P modes
(canonical_params rejects "baseline" and GCM-scaled Kzz for them, loudly).

Quality tiers (the GUI's fidelity switch; "fast" is the default):
    fast  nz=100, yconv 1e-2 (VULCAN master default), nu_pts=4000, 40 layers
    high  nz=150, yconv 1e-3, nu_pts=8000, 60 layers (the original WIDE setup)

Atmosphere-structure knobs (all consumed by the same validated pipeline hooks the
retrieval framework uses):

    T-P profile (tp_mode):
      "baseline"    WASP-39b GCM T(P) + uniform shift dT          (tp_eval=None)
      "isothermal"  T(P) = T_iso                                  (tp_eval hook)
      "guillot"     ExoJax atmprof_Guillot(Tirr, Tint, log10 kappa, log10 gamma)
                    with f=0.25 and the planet's surface gravity  (tp_eval hook)
    Kzz:
      kzz_mode "scale"  baseline GCM Kzz profile x kzz_x
      kzz_mode "const"  constant Kzz = kzz_const cm^2/s (cfg_overrides
                        Kzz_prof="const"), further x kzz_x if given
    Composition:
      met_x_solar  metallicity in x solar -> lnZ = ln(met/10) about the 10x baseline
      dco          Delta ln(C/O) carbon-enrichment proxy
    Physics (cfg_overrides / RT flags; defaults = the previous hard-coded values):
      use_photo     photochemistry on/off (off = thermochem + transport only;
                    Fisher REQUIRES on -- the validated-jvp regime)
      sl_angle_deg  photolysis zenith angle (deg; 83 = Tsai 2023 terminator slant)
      f_diurnal     diurnal photolysis factor (1.0 = permanent dayside)
      use_moldiff   molecular diffusion on/off (homopause)
      use_rayleigh  H2/He Rayleigh scattering (ON by default from v4; v3 lacked it)
      cloud_on + log_kappa_cloud + alpha_cloud
                    ExoJax power-law cloud deck (fixed in the Fisher forecast)
      extra_mols    opt-in RT molecules beyond the base 5 (C2H2/H2S/HCN/NH3)

Any T-P that leaves the modelable premodit window [320, 2980] K on either grid is
REJECTED with a clear error (never clipped) -- same rule as the retrieval.

Fisher machinery: with ``fisher_params`` set, the runner also computes the
spectrum Jacobian d(depth)/d(param) with one warm-started forward-mode jvp per
parameter (the validated sensitivity pattern: continuation from the converged
column, photochemistry ON), plus an RT-only lnR0 (reference-radius nuisance)
column. ``fisher.py`` turns that + the Pandeia noise into parameter forecasts.

Per-molecule "removed" spectra (for detection significance) zero that molecule's
VMR in the RT only -- atmospheric structure (T, mmw) is kept, the standard
nested-model comparison used in observation planning.
"""
from __future__ import annotations

import hashlib
import json
import sys
import os
import time
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = TOOL_DIR.parent
# keep in sync with instruments.DATA_DIR (env-overridable for non-editable installs)
MODEL_CACHE = Path(os.environ.get(
    "JWST_TOOL_DATA_DIR", str(BUNDLE_DIR / "data" / "jwst_tool"))) / "model_cache"

try:
    from . import planets            # imported as jwst_tool.forward (the GUI)
except ImportError:                  # script mode: python jwst_tool/forward.py
    sys.path.insert(0, str(TOOL_DIR))
    import planets

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]   # always-on WIDE-profile set
# Opt-in RT additions: the SNCHO network already solves these; adding one costs a
# premodit build (~10-15 s, HITRAN lines downloaded on first use) + one removed
# spectrum. C2H2/HCN carry the high-C/O signal, H2S the 3.8-4.6 um reduced-sulfur
# feature, NH3 the cool (<~900 K) nitrogen chemistry.
EXTRA_MOLECULES = ["C2H2", "H2S", "HCN", "NH3"]
_VERSION = 5   # bump to invalidate all cached spectra (v5: exact-elemental
               # abundance map, on-graph Dzz/geometry rebuild, He CIA required,
               # broadening knob in the RT -- all pre-v5 spectra are stale)


def active_molecules(cp: dict) -> list[str]:
    """RT molecule set for canonical params: base set + selected extras."""
    return MOLECULES + [m for m in EXTRA_MOLECULES if m in cp["extra_mols"]]

# Model fidelity tiers layered on config.WIDE (1-15 um band unchanged).
# Both keep the full 60-layer ART grid (RT evals are ~1-2 s, so thinning it buys
# nothing). Measured fast-vs-high agreement (W39b defaults): G395H SO2 3.6 vs
# 3.8 sigma, F444W 2.8 vs 3.0, Fisher sigma(lnZ) 0.027 vs 0.029 dex; the weak
# mid-IR SO2 bands are the one real casualty (MIRI LRS 0.9 vs 1.9 sigma -- the
# nz=100 / yconv 1e-2 chemistry mutes the high-altitude SO2 shell). Use "high"
# for final numbers.
QUALITY = {
    "fast": dict(nz=100, yconv_cri=1.0e-2, nu_pts=4000, art_nlayer=60),
    "high": dict(nz=150, yconv_cri=1.0e-3, nu_pts=8000, art_nlayer=60),
}

# Modelable temperature window (premodit table range, 20 K inset) -- reject, never clip.
T_WINDOW = (320.0, 2980.0)

# Parameters that can be freed in the Fisher forecast, per tp_mode.
CHEM_PARAM_NAMES = ["lnZ", "dlnCO", "lnKzz"]
TP_PARAM_NAMES = {
    "baseline": ["dT"],
    "isothermal": ["T_iso"],
    "guillot": ["Tirr", "Tint", "log_kappa", "log_gamma"],
}
# Display units + friendly names for the GUI's constraint table / science goals.
PARAM_UNITS = {"lnZ": "dex(Z)", "dlnCO": "ln(C/O)", "lnKzz": "dex(Kzz)",
               "dT": "K", "T_iso": "K", "Tirr": "K", "Tint": "K",
               "log_kappa": "dex", "log_gamma": "dex"}
PARAM_LABELS = {"lnZ": "Metallicity", "dlnCO": "C/O ratio",
                "lnKzz": "Vertical mixing (Kzz)", "dT": "T-P shift ΔT",
                "T_iso": "Isothermal T", "Tirr": "Guillot T_irr",
                "Tint": "Guillot T_int", "log_kappa": "Guillot log κ_IR",
                "log_gamma": "Guillot log γ"}


def canonical_params(params: dict) -> dict:
    tp_mode = str(params.get("tp_mode", "baseline"))
    planet = str(params.get("planet", "wasp39b"))
    if planet not in planets.PLANETS and planet != "custom":
        raise ValueError(f"unknown planet {planet!r}")
    sysd = planets.system_fields(planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS))
    quality = str(params.get("quality", "fast"))
    if quality not in QUALITY:
        raise ValueError(f"unknown quality {quality!r} (choose from {list(QUALITY)})")
    sflux = str(params.get("sflux", sysd["sflux"]))
    if sflux not in planets.SFLUX_CHOICES:
        raise ValueError(f"unknown stellar UV spectrum {sflux!r} "
                         f"(choose from {list(planets.SFLUX_CHOICES)})")
    cp = {
        "planet": planet,
        "quality": quality,
        "rp_rjup": round(float(params.get("rp_rjup", sysd["rp_rjup"])), 4),
        "gs_cgs": round(float(params.get("gs_cgs", sysd["gs_cgs"])), 1),
        "rstar_rsun": round(float(params.get("rstar_rsun", sysd["rstar_rsun"])), 4),
        "orbit_au": round(float(params.get("orbit_au", sysd["orbit_au"])), 5),
        "sflux": sflux,
        "met_x_solar": round(float(params.get("met_x_solar", 10.0)), 4),
        "dco": round(float(params.get("dco", 0.0)), 4),
        "kzz_mode": str(params.get("kzz_mode", "scale")),
        "kzz_x": round(float(params.get("kzz_x", 1.0)), 4),
        "kzz_const": round(float(params.get("kzz_const", 1.0e9)), 1),
        "tp_mode": tp_mode,
        "dT": round(float(params.get("dT", 0.0)), 2),
        "T_iso": round(float(params.get("T_iso", 1100.0)), 2),
        "Tirr": round(float(params.get("Tirr", 1560.0)), 2),
        "Tint": round(float(params.get("Tint", 100.0)), 2),
        "log_kappa": round(float(params.get("log_kappa", -2.3)), 3),
        "log_gamma": round(float(params.get("log_gamma", -1.0)), 3),
        # physical VULCAN knobs (all flow through the validated cfg_overrides hook;
        # defaults reproduce the previous hard-coded behavior = the W39b cfg values)
        "use_photo": bool(params.get("use_photo", True)),
        "sl_angle_deg": round(float(params.get("sl_angle_deg", 83.0)), 1),
        "f_diurnal": round(float(params.get("f_diurnal", 1.0)), 3),
        "use_moldiff": bool(params.get("use_moldiff", True)),
        # RT physics: Rayleigh is known zero-parameter physics, ON by default
        # (v3 and earlier ran without it -- that biased the <1.5 um slope);
        # the cloud deck is the ExoJax power-law retrieval cloud, OFF by default.
        "use_rayleigh": bool(params.get("use_rayleigh", True)),
        "cloud_on": bool(params.get("cloud_on", False)),
        "log_kappa_cloud": round(float(params.get("log_kappa_cloud", -1.0)), 3),
        "alpha_cloud": round(float(params.get("alpha_cloud", 0.0)), 2),
        "extra_mols": sorted(str(m) for m in (params.get("extra_mols") or [])),
        "fisher_params": sorted(str(p) for p in (params.get("fisher_params") or [])),
        "version": _VERSION,
    }
    if not 0.0 <= cp["sl_angle_deg"] <= 89.0:
        raise ValueError(f"sl_angle_deg={cp['sl_angle_deg']} outside [0, 89] deg")
    if not 0.0 < cp["f_diurnal"] <= 1.0:
        raise ValueError(f"f_diurnal={cp['f_diurnal']} outside (0, 1]")
    bad_mols = set(cp["extra_mols"]) - set(EXTRA_MOLECULES)
    if bad_mols:
        raise ValueError(f"unknown extra molecules {sorted(bad_mols)} "
                         f"(choose from {EXTRA_MOLECULES})")
    if cp["fisher_params"] and not cp["use_photo"]:
        raise ValueError(
            "Fisher forecast requires photochemistry ON: the warm-started "
            "steady-state jvp is validated only in the photo-on regime "
            "(config.py run-profile notes). Enable photochemistry or clear "
            "the Fisher parameter list.")
    if not cp["use_photo"]:            # photolysis knobs are inert without photo
        cp["sl_angle_deg"] = 0.0
        cp["f_diurnal"] = 1.0
    if not cp["cloud_on"]:             # cloud knobs are inert when the deck is off
        cp["log_kappa_cloud"] = 0.0
        cp["alpha_cloud"] = 0.0
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "baseline":
        cp["dT"] = 0.0
    if tp_mode != "isothermal":
        cp["T_iso"] = 0.0
    if tp_mode != "guillot":
        cp["Tirr"] = cp["Tint"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
    if cp["kzz_mode"] != "const":
        cp["kzz_const"] = 0.0
    # GCM-tied modes exist only for the planet whose profiles are baked in (W39b).
    gcm = planets.PLANETS.get(planet, {}).get("has_gcm_baseline", False)
    if tp_mode == "baseline" and not gcm:
        raise ValueError(f"tp_mode='baseline' (GCM T-P profile) is only available "
                         f"for WASP-39b, not {planet!r} -- use isothermal or guillot")
    if cp["kzz_mode"] == "scale" and not gcm:
        raise ValueError(f"kzz_mode='scale' (GCM Kzz profile) is only available "
                         f"for WASP-39b, not {planet!r} -- use kzz_mode='const'")
    bad = set(cp["fisher_params"]) - set(CHEM_PARAM_NAMES + TP_PARAM_NAMES[tp_mode])
    if bad:
        raise ValueError(f"fisher_params {sorted(bad)} not available for tp_mode={tp_mode}")
    return cp


def params_key(params: dict) -> str:
    s = json.dumps(canonical_params(params), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict) -> Path:
    return MODEL_CACHE / f"{params_key(params)}.npz"


def load_result(params: dict):
    """Cached spectrum dict or None. Keys: wl_um, depth, mols, depth_wo (nmol, n_nu),
    and (if Fisher was requested) jac (n_par, n_nu) + jac_names."""
    p = cache_path(params)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Heavy path (script mode only below this line)
# ---------------------------------------------------------------------------

def _build_tp(cp: dict, gs_cgs: float):
    """(tp_eval, n_tp, tp_values, theta_names) for the chosen T-P mode.

    tp_eval(tp_params, p_bar) is pure JAX (differentiable); None for baseline mode
    (where the runner's own validated uniform-shift knob theta[3]=dT is used).
    """
    import jax.numpy as jnp

    mode = cp["tp_mode"]
    if mode == "baseline":
        return None, 0, [cp["dT"]], CHEM_PARAM_NAMES + ["dT"]
    if mode == "isothermal":
        def tp_eval(tp, p_bar):
            return jnp.zeros_like(jnp.asarray(p_bar)) + tp[0]
        return tp_eval, 1, [cp["T_iso"]], CHEM_PARAM_NAMES + ["T_iso"]
    if mode == "guillot":
        from exojax.atm.atmprof import atmprof_Guillot

        def tp_eval(tp, p_bar):
            p = jnp.asarray(p_bar)
            Tirr, Tint = tp[0], tp[1]
            kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
            return atmprof_Guillot(p, gs_cgs, kappa, gamma, Tint, Tirr, 0.25)
        vals = [cp["Tirr"], cp["Tint"], cp["log_kappa"], cp["log_gamma"]]
        return tp_eval, 4, vals, CHEM_PARAM_NAMES + TP_PARAM_NAMES["guillot"]
    raise ValueError(f"unknown tp_mode {mode!r}")


def _make_progress(cp: dict, log):
    """Sequential stage tracker: emits "[fwd] PROG <frac> <label>" lines.

    The stage list MUST mirror run_model's actual stage order (same
    conditionals); weights are rough wall-clock seconds so the GUI bar moves
    honestly. advance() is called at the START of each stage.
    """
    two_stage = not (cp["met_x_solar"] == 10.0 and cp["dco"] == 0.0)
    mols = active_molecules(cp)
    stages = [("building chemistry model (compile + warm-up)", 45.0),
              ("building radiative transfer (opacities + CIA)",
               10.0 + 3.0 * len(cp["extra_mols"]))]
    if two_stage:
        stages += [("chemistry stage 1/2: T + Kzz relaxation", 35.0),
                   ("chemistry stage 2/2: composition continuation", 30.0)]
    else:
        stages += [("solving photochemistry", 35.0)]
    stages += [("full transmission spectrum", 8.0)]
    stages += [(f"spectrum without {m}", 4.0) for m in mols]
    stages += [(f"Jacobian d/d({n})", 40.0) for n in cp["fisher_params"]]
    if cp["fisher_params"]:
        stages += [("Jacobian d/d(lnR0)", 5.0)]
    total = sum(w for _, w in stages)
    state = {"i": 0, "done": 0.0}

    def advance():
        label, w = stages[state["i"]]
        log(f"[fwd] PROG {state['done'] / total:.3f} {label}")
        state["i"] += 1
        state["done"] += w

    def finish():
        log("[fwd] PROG 1.000 done")

    return advance, finish


def run_model(params: dict, log=print) -> Path:
    sys.path.insert(0, str(BUNDLE_DIR))
    import config                                    # noqa: E402
    import vulcan_chem                               # noqa: E402  (env + x64 first)
    import jax                                       # noqa: E402
    import jax.numpy as jnp                          # noqa: E402
    import exojax_rt                                 # noqa: E402
    import interp_map                                # noqa: E402

    cp = canonical_params(params)
    advance, finish = _make_progress(cp, log)
    tp_eval, n_tp, tp_vals, theta_names = _build_tp(cp, cp["gs_cgs"])
    theta = np.array([np.log(cp["met_x_solar"] / 10.0), cp["dco"],
                      np.log(cp["kzz_x"])] + tp_vals, dtype=np.float64)
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4)))}")

    profile = dict(config.WIDE)
    profile.update(QUALITY[cp["quality"]])
    # exact-elemental abundance map (lnZ / dlnCO are true column elemental
    # directions; conserved totals rebuilt per theta -- see vulcan_chem docstring).
    # reanchor_atom_ini is moot in this mode but kept for a masks-mode fallback.
    profile["abundance_mode"] = "elemental"
    profile["co_mode"] = "fixed_O"
    profile["broadening"] = str(cp.get("broadening", "air"))
    profile["reanchor_atom_ini"] = True   # finite-Z steps must re-anchor atom totals
    # step-size cap, validated state-preserving (retrieval case.py): prevents the
    # adaptive-dt ballooning non-convergence at high Kzz the GUI sliders can reach
    profile["dt_max"] = 1.0e11
    mols_active = active_molecules(cp)
    profile["molecules"] = mols_active
    profile["use_photo"] = cp["use_photo"]        # build_chem_model reads this key
    profile["use_rayleigh"] = cp["use_rayleigh"]  # exojax_rt reads this flag

    # --- planet identity ------------------------------------------------------
    rp_cm = cp["rp_rjup"] * planets.R_JUP_CM
    rstar_cm = cp["rstar_rsun"] * planets.R_SUN_CM
    profile["rp_cm"] = rp_cm            # RT geometry (exojax_rt reads these)
    profile["gs_cgs"] = cp["gs_cgs"]
    profile["rstar_cm"] = rstar_cm
    ovr = {                              # chemistry side (applied pre-pre-loop)
        "gs": cp["gs_cgs"], "Rp": rp_cm, "r_star": cp["rstar_rsun"],
        "orbit_radius": cp["orbit_au"],
        "sflux_file": f"atm/stellar_flux/{cp['sflux']}",
        "use_moldiff": cp["use_moldiff"],
    }
    if cp["use_photo"]:                  # photolysis geometry/averaging knobs
        ovr["sl_angle"] = float(np.deg2rad(cp["sl_angle_deg"]))
        ovr["f_diurnal"] = cp["f_diurnal"]
    gcm = planets.PLANETS.get(cp["planet"], {}).get("has_gcm_baseline", False)
    if not gcm:
        # No baked GCM T-P/Kzz for this planet: isothermal structural baseline
        # at a representative temperature (the on-graph tp_eval replaces T for
        # the chemistry+RT; the structure sets the hydrostatic grid + EQ init),
        # constant Kzz. canonical_params already rejected baseline/scale modes.
        T_struct = (cp["T_iso"] if cp["tp_mode"] == "isothermal"
                    else cp["Tirr"] / np.sqrt(2.0))   # ~equilibrium T at f=0.25
        ovr.update({"atm_type": "isothermal", "Tiso": float(T_struct),
                    "Kzz_prof": "const", "const_Kzz": cp["kzz_const"]})
        log(f"[fwd] planet {cp['planet']}: isothermal structural baseline "
            f"{T_struct:.0f} K, const Kzz {cp['kzz_const']:.1e} cm2/s, "
            f"UV = {cp['sflux']}")
    elif cp["kzz_mode"] == "const":
        # constant eddy-diffusion profile via the same cfg_overrides hook the
        # fisher_zco tiers use; lnKzz (theta[2]) still multiplies it on-graph.
        ovr.update({"Kzz_prof": "const", "const_Kzz": cp["kzz_const"]})
    profile["cfg_overrides"] = ovr

    t0 = time.time()
    advance()
    log("[fwd] building chemistry model (VULCAN-JAX warm-up ~40 s) ...")
    chem = vulcan_chem.build_chem_model(profile, tp_eval=tp_eval, n_tp_params=n_tp)
    log(f"[fwd] chemistry ready in {time.time()-t0:.0f} s")

    # --- T-P validity: REJECT (never clip) out-of-window profiles ------------
    if tp_eval is not None:
        T_check = np.asarray(tp_eval(jnp.asarray(theta[3:]), jnp.asarray(chem.p_bar)))
    else:
        T_check = np.asarray(chem.T_base) + cp["dT"]
    tmin, tmax = float(T_check.min()), float(T_check.max())
    if tmin < T_WINDOW[0] or tmax > T_WINDOW[1]:
        raise RuntimeError(
            f"T-P profile leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K (min {tmin:.0f} K, max {tmax:.0f} K). "
            "Adjust the profile parameters -- out-of-window layers are rejected, "
            "not clipped (opacity tables end there).")
    log(f"[fwd] T-P in window: [{tmin:.0f}, {tmax:.0f}] K")

    t0 = time.time()
    advance()
    log("[fwd] building ExoJax RT (opacities + CIA) ...")
    rt = exojax_rt.build_rt_model(profile)
    log(f"[fwd] RT ready in {time.time()-t0:.0f} s")

    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
    mol_cols = {k: chem.sidx[config.MOLECULES[k]["vulcan"]] for k in rt.molecules}
    h2 = chem.sidx["H2"]
    he = chem.sidx["He"]
    p_art_j = jnp.asarray(rt.p_art_bar)

    def art_T(th):
        if tp_eval is None:
            return to_art(jnp.asarray(chem.T_base) + th[3])
        return tp_eval(th[3:], p_art_j)

    # ExoJax power-law retrieval cloud [log10 kappac0 (cm^2/g at 3.5 um), alphac];
    # held FIXED in the Fisher forecast (no cloud marginalization -- documented).
    cloud_vec = (jnp.asarray([cp["log_kappa_cloud"], cp["alpha_cloud"]])
                 if cp["cloud_on"] else None)

    def depth_from_y(y, th, lnR0=0.0, drop_mol=None):
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        T_art = art_T(th)
        mmw_art = to_art(ymix @ chem.species_masses)
        vmr = {k: to_art(ymix[:, c]) for k, c in mol_cols.items()}
        if drop_mol is not None:
            vmr[drop_mol] = jnp.zeros_like(vmr[drop_mol])
        return rt.transmission_depth_r(
            vmr, to_art(ymix[:, h2]), T_art, mmw_art, jnp.asarray(lnR0),
            vmr_he=to_art(ymix[:, he]), cloud=cloud_vec)

    # --- chemistry: two-stage for composition steps (validated pattern) ------
    t0 = time.time()
    th0 = jnp.asarray(theta)
    def _check_converged(ac, stage):
        ac = int(ac)
        if ac >= int(chem.count_max):
            raise RuntimeError(
                f"chemistry did NOT converge ({stage}: {ac} accepted steps hit the "
                f"count_max={chem.count_max} cap). This parameter corner has no "
                "certified steady state -- adjust T-P / Kzz / composition rather "
                "than trusting an unconverged spectrum.")

    if cp["met_x_solar"] == 10.0 and cp["dco"] == 0.0:
        advance()
        log("[fwd] solving chemistry (single stage, baseline composition) ...")
        y_sol, ac = chem.converged_y(th0, return_diag=True)
        _check_converged(ac, "single stage")
    else:
        advance()
        log("[fwd] solving chemistry stage 1/2 (T/Kzz relaxation) ...")
        th_relax = th0.at[0].set(0.0).at[1].set(0.0)
        y_relaxed, ac1 = chem.converged_y(th_relax, return_diag=True)
        _check_converged(ac1, "stage 1, T/Kzz relaxation")
        advance()
        log(f"[fwd] stage 1 done ({time.time()-t0:.0f} s); "
            "stage 2/2 (composition, warm continuation) ...")
        y_sol, ac2 = chem.converged_y(th0, warm_y=y_relaxed, lnZ_ref=0.0,
                                      c_o_ref=0.0, return_diag=True)
        _check_converged(ac2, "stage 2, composition continuation")
    y_np = np.asarray(y_sol)
    if not np.all(np.isfinite(y_np)):
        raise RuntimeError("chemistry solve returned non-finite abundances -- "
                           "parameter set outside the modelable range")
    log(f"[fwd] chemistry solved in {time.time()-t0:.0f} s total")

    # --- RT: full spectrum + one spectrum per removed molecule ---------------
    t0 = time.time()
    advance()
    log("[fwd] radiative transfer: full spectrum (jit compile on first call) ...")
    depth = np.asarray(depth_from_y(y_sol, th0))
    log(f"[fwd] full spectrum in {time.time()-t0:.0f} s")

    depth_wo = np.zeros((len(mols_active), depth.shape[0]))
    for i, mol in enumerate(mols_active):
        t1 = time.time()
        advance()
        depth_wo[i] = np.asarray(depth_from_y(y_sol, th0, drop_mol=mol))
        log(f"[fwd] spectrum without {mol} in {time.time()-t1:.0f} s")

    # --- Fisher Jacobian: warm-started jvp per free parameter + lnR0 ---------
    jac_names = list(cp["fisher_params"])
    jac = np.zeros((len(jac_names) + 1, depth.shape[0])) if jac_names else None
    if jac_names:
        lnZ0, co0 = float(theta[0]), float(theta[1])

        def f_theta(th):
            # continuation from the converged column: primal is a no-op re-converge,
            # the jvp is the validated warm-started steady-state tangent
            y = chem.converged_y(th, warm_y=y_sol, lnZ_ref=lnZ0, c_o_ref=co0)
            return depth_from_y(y, th)

        for j, name in enumerate(jac_names):
            t1 = time.time()
            advance()
            i_par = theta_names.index(name)
            e = np.zeros_like(theta)
            e[i_par] = 1.0
            _, dd = jax.jvp(f_theta, (th0,), (jnp.asarray(e),))
            jac[j] = np.asarray(dd)
            log(f"[fwd] Jacobian d(depth)/d({name}) in {time.time()-t1:.0f} s")

        t1 = time.time()
        advance()
        _, dd = jax.jvp(lambda r: depth_from_y(y_sol, th0, lnR0=r),
                        (jnp.asarray(0.0),), (jnp.asarray(1.0),))
        jac[-1] = np.asarray(dd)
        jac_names.append("lnR0")
        log(f"[fwd] Jacobian d(depth)/d(lnR0) [RT-only nuisance] in {time.time()-t1:.0f} s")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(params)
    ymix_np = y_np / y_np.sum(axis=1, keepdims=True)
    arrays = dict(
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        depth=depth, depth_wo=depth_wo,
        mols=np.array(mols_active, dtype="U8"),
        ymix=ymix_np, p_bar=np.asarray(chem.p_bar),
        T=np.asarray(T_check), theta=theta,
        theta_names=np.array(theta_names, dtype="U16"),
        params_json=np.array(json.dumps(cp), dtype="U2048"),
    )
    if jac is not None:
        arrays["jac"] = jac
        arrays["jac_names"] = np.array(jac_names, dtype="U16")
    np.savez_compressed(out, **arrays)
    finish()
    log(f"[fwd] cached -> {out.name}")
    return out


def main():
    params = json.load(open(sys.argv[1]))
    run_model(params, log=lambda *a: print(*a, flush=True))
    print("[fwd] DONE", flush=True)


if __name__ == "__main__":
    main()
