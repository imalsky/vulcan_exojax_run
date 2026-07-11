"""JWST instrument selector -- Streamlit GUI.

    cd vulcan_exojax_run
    streamlit run jwst_tool/app.py

Pipeline per run: VULCAN-JAX photochemistry -> ExoJax transmission spectrum
(local subprocess, disk-cached; ~1.5-2 min at the default "fast" fidelity) ->
Pandeia ETC noise per instrument mode (picaso_base subprocess, disk-cached) ->
science-goal scoring per mode. Two goal types: DETECT a molecule (nested-model
delta-chi2 significance) or CONSTRAIN a parameter (Fisher forecast from the
autodiff Jacobian, vs a target precision). Planets beyond WASP-39b come from
the registry in planets.py (or a fully custom system).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR.parent))

from jwst_tool import detect, fisher as fisher_mod, forward, noise as noise_mod
from jwst_tool import instruments as ins
from jwst_tool import planets

st.set_page_config(page_title="JWST Instrument Selector",
                   layout="wide")

st.title("JWST instrument selector")
st.caption(
    "VULCAN-JAX photochemistry → ExoJAX transmission spectrum → Pandeia ETC noise. "
    "Pick a planet and a science goal, run the model locally, and see which "
    "instrument mode achieves it best."
)

_PROG_RE = re.compile(r"\[fwd\] PROG ([0-9.]+) (.*)")

# default target precision per parameter (display units: dex / K / ln-units)
_TARGET_DEFAULT = {"lnZ": 0.10, "dlnCO": 0.10, "lnKzz": 0.30, "dT": 50.0,
                   "T_iso": 50.0, "Tirr": 50.0, "Tint": 50.0,
                   "log_kappa": 0.30, "log_gamma": 0.30}


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
# Reset = bump a nonce that namespaces EVERY widget key: all widgets are
# re-created at their defaults (session_state.clear() alone does not reset
# keyless widgets, whose state lives outside the exposed dict).
_NONCE = st.session_state.setdefault("reset_nonce", 0)


def _reset_all():
    n = st.session_state.get("reset_nonce", 0) + 1
    st.session_state.clear()
    st.session_state["reset_nonce"] = n


def K(name: str) -> str:
    return f"n{_NONCE}_{name}"


with st.sidebar:
    st.header("Planet & star")
    planet_key = st.selectbox(
        "Planet", list(planets.PLANETS) + ["custom"], key=K("planet"),
        format_func=lambda k: planets.PLANETS[k]["label"] if k in planets.PLANETS
        else "Custom planet …",
        help="Every planet runs the same validated chemistry+RT machinery; the "
             "system identity (gravity, radii, star, orbit, UV spectrum) is "
             "swapped in. Only WASP-39b carries a GCM T-P/Kzz baseline.")
    pdef = planets.PLANETS.get(planet_key, planets.CUSTOM_DEFAULTS)
    has_gcm = planets.PLANETS.get(planet_key, {}).get("has_gcm_baseline", False)
    st.caption(pdef["note"] if planet_key in planets.PLANETS
               else "Starts from WASP-39 b values — edit everything below.")

    def _k(name: str) -> str:            # per-planet widget state
        return K(f"{planet_key}_{name}")

    with st.expander("System parameters", expanded=(planet_key == "custom")):
        teff = st.number_input("Star T_eff (K)", 3000.0, 7000.0,
                               pdef["star"]["teff"], 50.0, key=_k("teff"),
                               help="PHOENIX SED for the ETC (with log g, [Fe/H]).")
        logg = st.number_input("Star log g", 3.5, 5.5, pdef["star"]["log_g"], 0.1,
                               key=_k("logg"))
        feh = st.number_input("Star [Fe/H]", -2.0, 0.5,
                              pdef["star"]["metallicity"], 0.1, key=_k("feh"))
        ks_mag = st.number_input("Ks mag (2MASS)", 4.0, 16.0,
                                 pdef["star"]["ks_mag"], 0.1, key=_k("ks"),
                                 help="Sets absolute count rates → saturation & "
                                      "photon noise. Brighter = more saturation.")
        rstar = st.number_input("R_star (R_sun)", 0.2, 3.0, pdef["rstar_rsun"],
                                0.01, key=_k("rstar"), format="%.3f",
                                help="Transit-depth normalization + UV flux at "
                                     "the planet.")
        rp = st.number_input("R_p (R_Jup)", 0.1, 2.5, pdef["rp_rjup"], 0.01,
                             key=_k("rp"), format="%.3f")
        g_ms2 = st.number_input("Surface gravity (m/s²)", 1.0, 100.0,
                                pdef["gs_cgs"] / 100.0, 0.5, key=_k("g"),
                                help="Sets the scale height: lower gravity = "
                                     "bigger spectral features.")
        orbit_au = st.number_input("Semi-major axis (au)", 0.005, 1.0,
                                   pdef["orbit_au"], 0.001, key=_k("a"),
                                   format="%.4f",
                                   help="Scales the stellar UV reaching the "
                                        "planet (photochemistry).")
        t14 = st.number_input("Transit duration T14 (hr)", 0.5, 10.0,
                              pdef["t14_hr"], 0.1, key=_k("t14"))
        sflux = st.selectbox("Stellar UV spectrum (photochemistry)",
                             list(planets.SFLUX_CHOICES),
                             index=list(planets.SFLUX_CHOICES).index(pdef["sflux"]),
                             format_func=planets.SFLUX_CHOICES.get, key=_k("sflux"),
                             help="Shipped VULCAN spectra — pick the nearest "
                                  "spectral type. Drives photolysis (SO2, CH4 …).")

    teq = float(pdef["teq_k"])
    with st.expander("Atmosphere structure"):
        tp_options = (["baseline", "isothermal", "guillot"] if has_gcm
                      else ["isothermal", "guillot"])
        tp_mode = st.selectbox(
            "T-P profile", tp_options, index=0, key=_k("tp"),
            format_func={"baseline": "WASP-39b GCM profile + ΔT",
                         "isothermal": "Isothermal",
                         "guillot": "Guillot (2010)"}.get,
            help=None if has_gcm else
            "No GCM profile is baked in for this planet — defaults are set from "
            "its equilibrium temperature.")
        tp_kwargs = {}
        if tp_mode == "baseline":
            tp_kwargs["dT"] = st.slider("ΔT (K, uniform shift)", -200.0, 200.0,
                                        0.0, 25.0, key=K("dT"))
        elif tp_mode == "isothermal":
            tp_kwargs["T_iso"] = st.slider("T_iso (K)", 400.0, 2500.0,
                                           float(np.clip(teq, 400.0, 2500.0)),
                                           25.0, key=_k("tiso"))
        else:
            tirr0 = float(np.clip(round(teq * np.sqrt(2.0) / 10) * 10,
                                  800.0, 2500.0))
            tp_kwargs["Tirr"] = st.slider("T_irr (K)", 800.0, 2500.0, tirr0, 20.0,
                                          key=_k("tirr"),
                                          help="≈ √2 × equilibrium temperature.")
            tp_kwargs["Tint"] = st.slider("T_int (K)", 50.0, 500.0, 100.0, 25.0,
                                          key=_k("tint"))
            tp_kwargs["log_kappa"] = st.slider("log₁₀ κ_IR (cm²/g)", -4.0, 0.0,
                                               -2.3, 0.1, key=_k("lk"))
            tp_kwargs["log_gamma"] = st.slider("log₁₀ γ (κ_vis/κ_IR)", -2.0, 0.3,
                                               -1.0, 0.05, key=_k("lg"))

        if has_gcm:
            kzz_mode = st.radio("K_zz profile", ["scale", "const"],
                                horizontal=True, key=_k("kzzmode"),
                                format_func={"scale": "GCM profile × factor",
                                             "const": "constant"}.get)
        else:
            kzz_mode = "const"
            st.caption("K_zz: constant profile (no GCM K_zz for this planet).")
        if kzz_mode == "scale":
            kzz_x = st.select_slider("K_zz multiplier",
                                     options=[0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
                                     value=1.0, key=K("kzzx"))
            kzz_const = 1.0e9
        else:
            log_kzz = st.slider("log₁₀ K_zz (cm²/s)", 6.0, 12.0, 9.0, 0.25,
                                key=_k("kzz"),
                                help="Eddy diffusion: stronger mixing quenches "
                                     "photochemical gradients.")
            kzz_const, kzz_x = 10.0 ** log_kzz, 1.0

    with st.expander("Composition"):
        st.caption("Element totals re-anchored from the 10× solar FastChem baseline.")
        met = st.select_slider(
            "Metallicity (× solar)",
            options=[1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0],
            value=10.0, key=K("met"))
        dco = st.slider("Δ ln(C/O) (carbon enrichment)", -0.5, 0.5, 0.0, 0.05,
                        key=K("dco"))

    with st.expander("Physics"):
        st.caption("Chemistry knobs flow through the validated cfg_overrides "
                   "hook; defaults reproduce the Tsai et al. 2023 W39b setup.")
        use_photo = st.checkbox(
            "Photochemistry (UV photolysis)", value=True, key=K("photo"),
            help="Off = thermochemistry + transport only (no SO2 story, no "
                 "photolysis products). The Fisher forecast requires ON — the "
                 "warm-started jvp is only validated in the photo-on regime.")
        sl_angle_deg = st.slider(
            "Photolysis zenith angle (°)", 0.0, 89.0, 83.0, 1.0, key=K("sza"),
            disabled=not use_photo,
            help="Slant path of the stellar UV. 83° = terminator slant "
                 "(Tsai et al. 2023 W39b); ~57° ≈ dayside average.")
        f_diurnal = st.slider(
            "Diurnal photolysis factor", 0.1, 1.0, 1.0, 0.05, key=K("fdiur"),
            disabled=not use_photo,
            help="Multiplies every photolysis rate. 1.0 = permanent dayside "
                 "(tidally locked); 0.5 mimics day–night averaging.")
        use_moldiff = st.checkbox(
            "Molecular diffusion", value=True, key=K("moldiff"),
            help="Species-dependent molecular diffusion competing with Kzz "
                 "(sets the homopause; matters high up).")
        extra_mols = st.multiselect(
            "Extra RT molecules", forward.EXTRA_MOLECULES, default=[],
            key=K("xmols"),
            help="Added to the base H2O/CO2/CO/CH4/SO2 opacity set (the "
                 "chemistry always solves them). C2H2/HCN matter at high C/O, "
                 "H2S at 3.8–4.6 µm, NH3 on cool (≲900 K) planets. First use "
                 "downloads the HITRAN lines (~10–15 s each per run).")
        st.markdown("**Scattering & clouds (RT)**")
        use_rayleigh = st.checkbox(
            "H₂/He Rayleigh scattering", value=True, key=K("rayl"),
            help="Zero-free-parameter known physics; matters shortward of "
                 "~1.5 µm (the SOSS blue end). Leave ON except for comparisons.")
        cloud_on = st.checkbox(
            "Cloud deck (power-law opacity)", value=False, key=K("cloud"),
            help="ExoJax power-law retrieval cloud, uniformly mixed: "
                 "κ(ν) = κ₀·(ν/ν₀)^α per gram of atmosphere. Held FIXED in "
                 "the Fisher forecast (no cloud marginalization), so forecasts "
                 "with a thick deck are best-case.")
        if cloud_on:
            log_kappa_cloud = st.slider(
                "log₁₀ κ_cloud (cm²/g at 3.5 µm)", -4.0, 2.0, -1.0, 0.1,
                key=K("ck"),
                help="Gray amplitude. τ=1 pressure ≈ g/(κ·10⁶) bar: at WASP-39b "
                     "gravity, −1 → ~4 mbar deck, −3 → ~0.4 bar.")
            alpha_cloud = st.slider(
                "Cloud spectral slope α (κ ∝ ν^α)", 0.0, 4.0, 0.0, 0.25,
                key=K("ca"),
                help="0 = gray deck; 4 ≈ Rayleigh-like small-particle haze.")
        else:
            log_kappa_cloud, alpha_cloud = -1.0, 0.0

    avail_free = forward.CHEM_PARAM_NAMES + forward.TP_PARAM_NAMES[tp_mode]
    mol_options = forward.MOLECULES + [m for m in forward.EXTRA_MOLECULES
                                       if m in extra_mols]

    with st.expander("Science goal", expanded=True):
        goal = st.radio(
            "Goal", ["detect", "constrain"], horizontal=True, key=K("goal"),
            format_func={"detect": "Detect a molecule",
                         "constrain": "Constrain a parameter"}.get,
            help="Detect: significance of molecule X being present (Δχ² between "
                 "the spectrum with and without it). Constrain: how tightly a "
                 "parameter (metallicity, C/O, Kzz, …) could be measured — a "
                 "Fisher forecast from the autodiff Jacobian.")
        goal_param, target_prec = None, None
        if goal == "detect":
            target_mol = st.selectbox("Detect molecule", mol_options,
                                      index=mol_options.index("SO2"),
                                      key=K("mol_" + "_".join(sorted(extra_mols))))
        else:
            target_mol = None
            if not use_photo:
                st.warning("Parameter constraints use the Fisher forecast, "
                           "which needs photochemistry ON (Physics section).")
            goal_param = st.selectbox(
                "Constrain parameter", avail_free, key=K(f"gp_{tp_mode}"),
                format_func=lambda n: f"{forward.PARAM_LABELS[n]} ({n})",
                help="Constraint is marginalized over the other free parameters "
                     "(Fisher forecast section) and a reference-radius nuisance.")
            unit = forward.PARAM_UNITS[goal_param]
            if unit == "K":
                target_prec = st.number_input(f"Target precision (±{unit})",
                                              5.0, 500.0,
                                              _TARGET_DEFAULT[goal_param], 5.0,
                                              key=K(f"tgt_{goal_param}"))
            else:
                target_prec = st.number_input(f"Target precision (±{unit})",
                                              0.01, 3.0,
                                              _TARGET_DEFAULT[goal_param], 0.01,
                                              key=K(f"tgt_{goal_param}"))
        target_sig = st.number_input(
            "Significance level (σ)", 1.0, 10.0, 3.0, 0.5, key=K("tsig"),
            help="3σ is the standard evidence threshold; 5σ a firm detection. "
                 "Detection verdicts require this significance; a precision "
                 "target must be met AT it (so the 1σ error must be "
                 "precision/level); Fisher intervals are quoted at it.")
        n_transits = st.slider("Number of transits", 1, 10, 1, key=K("ntr"))
        t_base = st.number_input("Out-of-transit baseline (hr)", 0.5, 10.0,
                                 float(t14), 0.1, key=_k("tbase"),
                                 help="Sets how well the stellar flux is "
                                      "anchored; PandExo convention is ≈ T14.")
        r_bin = st.select_slider("Binned resolving power R",
                                 options=[50, 100, 200], value=100, key=K("rbin"))
        mode_keys = st.multiselect(
            "Instrument modes",
            options=list(ins.MODES),
            default=ins.DEFAULT_MODES, key=K("modes"),
            help="The ETC computes every mode once per star, so adding modes "
                 "later is instant.",
            format_func=lambda k: (f"{ins.MODES[k]['label']}  "
                                   f"({ins.MODES[k]['wl_min']:g}–"
                                   f"{ins.MODES[k]['wl_max']:g} µm)"))

    with st.expander("Model fidelity"):
        quality = st.radio(
            "Fidelity", ["fast", "high"], index=0, key=K("quality"),
            format_func={"fast": "Fast (default)", "high": "High"}.get,
            captions=["100 chemistry layers (60 RT layers), native R≈1500 — "
                      "≈1.5 min per new setup",
                      "150 chemistry layers (60 RT layers), native R≈3000 — "
                      "≈3 min per new setup"],
            help="Same physics, coarser grids. Fast matches High on the headline "
                 "numbers (G395H SO2 3.6σ vs 3.8σ) but mutes the weak MIRI "
                 "mid-IR SO2 bands — switch to High for final numbers.")

    with st.expander("Fisher forecast"):
        st.caption(
            "Expected 1σ constraints on atmosphere parameters from this "
            "observation — a linearized retrieval (Cramér–Rao bound) built "
            "from d(spectrum)/d(parameter), computed by autodiff through the "
            "full chemistry+RT chain. No MCMC, no priors.")
        if goal == "constrain":
            fisher_extra = st.multiselect(
                "Jointly free parameters", avail_free,
                default=[p for p in ("lnZ", "dlnCO", "lnKzz")],
                key=K(f"fx_{tp_mode}"),
                help="The goal parameter is always included. More free "
                     "parameters = a more honest (wider) forecast; each adds "
                     "~20–60 s of Jacobian time.")
            fisher_params = sorted(set(fisher_extra) | {goal_param})
        else:
            do_fisher = st.checkbox(
                "Compute parameter constraints too", value=False,
                key=K("dofish"), disabled=not use_photo,
                help="One warm-started forward-mode jvp per parameter "
                     "(~20–60 s each). Needs photochemistry ON.")
            fisher_params = st.multiselect(
                "Free parameters", avail_free, key=K(f"fp_{tp_mode}"),
                default=["lnZ", "dlnCO", "lnKzz"]) if (do_fisher and use_photo) else []

    with st.expander("Advanced"):
        sat_limit = st.slider("Saturation limit (full-well fraction)",
                              0.5, 0.95, 0.80, 0.05, key=K("sat"))
        show_noise = st.checkbox("Show simulated noise realization", value=False,
                                 key=K("shownoise"))
        seed = st.number_input("Realization seed", 0, 9999, 0, key=K("seed"))
        st.markdown("**Systematic noise floors (ppm, per R=100 bin)**")
        st.caption("Anchored at R=100: finer binning scales the per-bin floor "
                   "by √(R/100) so slicing the band into more bins cannot "
                   "manufacture floor-limited significance; coarser bins keep "
                   "the full floor (systematics do not average down).")
        floors = {k: st.number_input(ins.MODES[k]["label"], 0.0, 200.0,
                                     ins.MODES[k]["floor_ppm"], 5.0,
                                     key=K(f"floor_{k}"))
                  for k in mode_keys}

    st.button("Reset all settings", on_click=_reset_all,
              help="Back to the defaults (also clears the current results).")

params = dict(planet=planet_key, quality=quality,
              rp_rjup=rp, gs_cgs=g_ms2 * 100.0, rstar_rsun=rstar,
              orbit_au=orbit_au, sflux=sflux,
              met_x_solar=met, dco=dco,
              kzz_mode=kzz_mode, kzz_x=kzz_x, kzz_const=kzz_const,
              tp_mode=tp_mode, fisher_params=fisher_params,
              use_photo=use_photo, sl_angle_deg=sl_angle_deg,
              f_diurnal=f_diurnal, use_moldiff=use_moldiff,
              use_rayleigh=use_rayleigh, cloud_on=cloud_on,
              log_kappa_cloud=log_kappa_cloud, alpha_cloud=alpha_cloud,
              extra_mols=extra_mols, **tp_kwargs)
star = dict(teff=teff, log_g=logg, metallicity=feh, ks_mag=ks_mag)
planet_label = (planets.PLANETS[planet_key]["label"]
                if planet_key in planets.PLANETS else "custom planet")

try:
    cached = forward.load_result(params) is not None
    params_error = None
except ValueError as e:          # e.g. stale widget combo mid-rerun
    cached, params_error = False, str(e)

n_jvp = len(fisher_params)
base_min = 1.8 if quality == "fast" else 2.8
if met != 10.0 or dco != 0.0:
    base_min += 0.6 if quality == "fast" else 0.9
base_min += 0.25 * len(extra_mols)   # opa build + removed spectrum per extra
# cool columns (<~900 K) converge much more slowly (WASP-107b: ~5 min measured)
t_char = {"isothermal": tp_kwargs.get("T_iso", 1100.0),
          "guillot": tp_kwargs.get("Tirr", 1560.0) / np.sqrt(2.0)}.get(tp_mode, 1100.0)
if t_char < 900.0:
    base_min += 2.5
per_jvp = 0.5 if quality == "fast" else 0.8
est = "instant (cached)" if cached else (
    f"~{base_min + per_jvp * n_jvp:.0f} min (local {quality}-fidelity run"
    + (f" + {n_jvp} Jacobian directions" if n_jvp else "") + ")")
col_btn, col_note = st.columns([1, 3])
run_clicked = col_btn.button("Run", type="primary", width="stretch")
col_note.caption(f"**{planet_label}**, {quality} fidelity — model spectrum: "
                 f"**{est}**. ETC noise is cached per star.")


# ---------------------------------------------------------------------------
# Compute on click
# ---------------------------------------------------------------------------
def compute():
    if params_error:
        st.error(f"Invalid parameter combination: {params_error}")
        return None
    if not mode_keys:
        st.error("Select at least one instrument mode.")
        return None

    model = forward.load_result(params)
    if model is None:
        with st.status("Running VULCAN-JAX + ExoJAX forward model locally …",
                       expanded=True) as status:
            bar = st.progress(0.0, text="starting …")
            pfile = forward.MODEL_CACHE / f"{forward.params_key(params)}.params.json"
            forward.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(forward.canonical_params(params)))
            proc = subprocess.Popen(
                [sys.executable, str(TOOL_DIR / "forward.py"), str(pfile)],
                cwd=str(TOOL_DIR.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            box = st.empty()
            lines = []
            for line in proc.stdout:
                line = line.rstrip()
                m = _PROG_RE.match(line)
                if m:
                    bar.progress(min(1.0, float(m.group(1))), text=m.group(2))
                else:
                    lines.append(line)
                    box.code("\n".join(lines[-10:]))
            proc.wait()
            if proc.returncode != 0:
                status.update(label="Forward model failed", state="error")
                st.error("Forward model failed:\n\n```\n"
                         + "\n".join(lines[-25:]) + "\n```")
                return None
            bar.progress(1.0, text="done")
            status.update(label="Forward model done", state="complete")
        model = forward.load_result(params)
        if model is None:
            st.error("Forward model finished but produced no cache file.")
            return None

    # ETC: always ALL modes (one cache per star; selection changes stay instant)
    all_modes = list(ins.MODES)
    job = noise_mod.noise_job(star, all_modes, sat_limit=sat_limit)
    have_cache = (ins.NOISE_CACHE / f"{noise_mod.job_key(job)}.json").exists()
    if have_cache:
        etc = noise_mod.run_pandeia(job)
    else:
        with st.status("Running Pandeia ETC (STScI engine, picaso_base env) …",
                       expanded=True) as status:
            bar = st.progress(0.0, text="starting the ETC …")
            box = st.empty()
            lines = []
            n_started = [0]

            def _cb(s):
                if s.startswith("[pandeia] ") and s.endswith("..."):
                    bar.progress(n_started[0] / len(all_modes),
                                 text=s.removeprefix("[pandeia] ")
                                 .removesuffix("...")
                                 + f" ({n_started[0] + 1}/{len(all_modes)})")
                    n_started[0] += 1
                else:
                    lines.append(s)
                    box.code("\n".join(lines[-8:]))

            etc = noise_mod.run_pandeia(job, progress=_cb)
            bar.progress(1.0, text="done")
            status.update(label="Pandeia ETC done", state="complete")

    t_in_s, t_out_s = t14 * 3600.0, t_base * 3600.0
    results, failed, unusable = [], [], []
    for k in mode_keys:
        if "error" in etc[k]:
            failed.append((k, etc[k]["error"]))
        elif etc[k].get("unusable") or not etc[k].get("wl"):
            unusable.append((k, etc[k].get("reason", "no usable pixels")))
        else:
            results.append(detect.evaluate_mode(
                k, etc[k], model, target_mol, r_bin, t_in_s, t_out_s,
                n_transits, floors[k]))
    return dict(model=model, results=results, failed=failed, unusable=unusable,
                fisher_names=list(fisher_params))


if run_clicked:
    out = compute()
    if out is not None:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(
            goal=goal, target=target_mol, goal_param=goal_param,
            target_prec=target_prec, target_sig=target_sig,
            n_transits=n_transits, show_noise=show_noise, seed=seed,
            r_bin=r_bin, planet=planet_label)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
if "out" not in st.session_state:
    st.info("Pick a planet and a science goal in the sidebar, then press **Run**.")
    st.stop()

out = st.session_state["out"]
meta = st.session_state["out_meta"]
model, results = out["model"], out["results"]
goal_r = meta.get("goal", "detect")

for k, err in out["failed"]:
    st.error(f"{ins.MODES[k]['label']}: Pandeia calculation failed — see details.")
    with st.expander(f"{ins.MODES[k]['label']} traceback"):
        st.code(err[-2500:])
for k, reason in out["unusable"]:
    st.warning(f"**{ins.MODES[k]['label']}: unusable on this star** — {reason}.")

if not results:
    st.stop()

fisher_names = ([str(x) for x in model["jac_names"][:-1]]
                if "jac_names" in model else [])
ok = [r for r in results if not r["saturated"]]

# --- verdict ---------------------------------------------------------------
if goal_r == "detect":
    tsig = float(meta.get("target_sig") or 3.0)
    ranked = sorted(ok or results, key=lambda r: -r["sigma_detect"])
    best = ranked[0]
    bsig = best["sigma_detect"]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for detecting {meta['target']} on "
               f"{meta.get('planet', '?')}: {best['label']}** — "
               f"{bsig:.1f}σ in {ntr} transit{'s' if ntr > 1 else ''} "
               f"(target {tsig:g}σ; median precision "
               f"{best['median_sigma_ppm']:.0f} ppm per R={meta['r_bin']} bin).")
    if bsig >= tsig:
        st.success(verdict + "  Meets the target.")
    elif bsig > 0:
        # floor-aware transit solver: the photon term averages down with N, the
        # (R-anchored) systematic floor does not -- a plain 1/sqrt(N) law was
        # optimistic exactly where it mattered (floor-dominated bright stars)
        tt = detect.transits_to_target(best, tsig)
        if tt["reachable"]:
            st.error(verdict + f"  Missing the target — {tt['n']} transits of "
                     f"{best['label']} would reach it (floor-aware estimate).")
        else:
            st.error(verdict + f"  Missing the target — and NO number of transits "
                     f"reaches it: the systematic floor caps this mode at "
                     f"{tt['sig_inf']:.1f}σ. Lower the floor, choose other modes, "
                     "or relax the target.")
    else:
        st.error(verdict + "  No signal in the selected bands — try other "
                 "modes or a different goal.")
else:
    gp = meta["goal_param"]
    unit = forward.PARAM_UNITS[gp]
    glabel = forward.PARAM_LABELS[gp]
    target = float(meta["target_prec"])
    tsig = float(meta.get("target_sig") or 3.0)
    with_jac = [r for r in results if r.get("jac_bins") is not None]
    # one saturation policy everywhere: a saturated mode is unusable data, so it
    # is excluded from BOTH the per-mode ranking and the combined forecast (the
    # combined row used to silently include modes the per-mode view dropped)
    usable_jac = [r for r in with_jac if not r["saturated"]]
    per_mode = {}          # tsig-sigma half-widths, display units
    for r in usable_jac:
        s = fisher_mod.display_sigma(gp, fisher_mod.mode_forecast(r, fisher_names)[gp])
        if np.isfinite(s):
            per_mode[r["mode_key"]] = tsig * s
    comb = (tsig * fisher_mod.display_sigma(
        gp, fisher_mod.combined_forecast(usable_jac, fisher_names)[gp])
        if len(usable_jac) >= 2 else np.inf)
    if not per_mode:
        st.error(f"No selected mode constrains {glabel} — its Jacobian has no "
                 "signal in these bands. Try other modes or a different goal.")
        st.stop()
    bk = min(per_mode, key=per_mode.get)
    bs = per_mode[bk]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for constraining {glabel} on "
               f"{meta.get('planet', '?')}: {ins.MODES[bk]['label']}** — "
               f"±{bs:.3g} {unit} at {tsig:g}σ in {ntr} transit"
               f"{'s' if ntr > 1 else ''} (target ±{target:g} {unit} "
               f"at {tsig:g}σ).")
    if bs <= target:
        st.success(verdict + "  Meets the target.")
    elif np.isfinite(comb) and comb <= target:
        st.warning(verdict + f"  No single mode reaches the target, but the "
                   f"combination of all selected modes does "
                   f"(±{comb:.3g} {unit} at {tsig:g}σ).")
    else:
        best_r = next(r for r in usable_jac if r["mode_key"] == bk)
        tt = fisher_mod.transits_to_target(best_r, fisher_names, gp,
                                           target / tsig, detect.sigma_at_transits)
        if tt["reachable"]:
            st.error(verdict + f"  Missing the target — {tt['n']} transits of "
                     f"{ins.MODES[bk]['label']} would reach it (floor-aware "
                     "estimate).")
        else:
            st.error(verdict + f"  Missing the target — and NO number of transits "
                     f"reaches it: the systematic floor caps this mode at "
                     f"±{tsig * tt['sig_inf']:.3g} {unit} at {tsig:g}σ. Lower the "
                     "floor, combine modes, or relax the target.")

# --- spectrum figure -------------------------------------------------------
wl = model["wl_um"]
order = np.argsort(wl)
wl_s, d_s = wl[order], model["depth"][order] * 1e6

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
ax.plot(wl_s, d_s, color="#555555", lw=0.7, alpha=0.8, zorder=2,
        label="model (native)")
if goal_r == "detect":
    mols = [str(x) for x in model["mols"]]
    d_wo_s = model["depth_wo"][mols.index(meta["target"])][order] * 1e6
    ax.plot(wl_s, d_wo_s, color="#999999", lw=0.9, ls="--", zorder=1,
            label=f"model without {meta['target']}")
rng = np.random.default_rng(int(meta["seed"]))
for r in results:
    c = ins.MODE_COLOR[r["mode_key"]]
    y = r["depth"] * 1e6
    if meta["show_noise"]:
        y = y + rng.normal(0.0, r["sigma"] * 1e6)
    label = r["label"] + (" (saturated!)" if r["saturated"] else "")
    ax.errorbar(r["wl"], y, yerr=r["sigma"] * 1e6, fmt="o", ms=3.0, lw=1.0,
                color=c, ecolor=c, elinewidth=0.8, capsize=0, zorder=3, label=label)
ax.set_xscale("log")
ticks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
lo = min(min(r["wl"].min() for r in results), 1.0)
hi = max(r["wl"].max() for r in results)
ax.set_xlim(lo * 0.97, hi * 1.03)
sel = (wl_s >= lo * 0.97) & (wl_s <= hi * 1.03)
pad = 0.06 * (d_s[sel].max() - d_s[sel].min())
ax.set_ylim(d_s[sel].min() - pad, d_s[sel].max() + 3 * pad)
ax.set_xlabel("wavelength (μm)")
ax.set_ylabel("transit depth (ppm)")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
st.pyplot(fig, width="stretch")
plt.close(fig)

# --- goal chart + T-P profile ----------------------------------------------
col1, col2 = st.columns([2.6, 1.4])

with col1:
    if goal_r == "detect":
        st.subheader(f"{meta['target']} detection significance")
        rs = sorted(results, key=lambda r: r["sigma_detect"])
        names = [r["label"] + (" (sat)" if r["saturated"] else "") for r in rs]
        vals = [r["sigma_detect"] for r in rs]
        cols = [ins.MODE_COLOR[r["mode_key"]] for r in rs]
        xrefs, xlabel = (3.0, 5.0), (f"detection significance "
                                     f"({meta['n_transits']} transit"
                                     f"{'s' if meta['n_transits'] > 1 else ''})")
        fmt_v = lambda v: f"{v:.1f}σ"
        vline_target = float(meta.get("target_sig") or 3.0)
    else:
        st.subheader(f"Expected precision on {glabel}")
        items = sorted(per_mode.items(), key=lambda kv: -kv[1])   # best at top
        names = [ins.MODES[k]["label"] for k, _ in items]
        vals = [v for _, v in items]
        cols = [ins.MODE_COLOR[k] for k, _ in items]
        if np.isfinite(comb):
            names.append("ALL SELECTED (combined)")
            vals.append(comb)
            cols.append("#555555")
        xrefs, xlabel = (), (f"expected ±{gp} at {tsig:g}σ [{unit}] "
                             f"({meta['n_transits']} transit"
                             f"{'s' if meta['n_transits'] > 1 else ''}; "
                             "lower is better)")
        fmt_v = lambda v: f"{v:.3g}"
        vline_target = target
    fig2, ax2 = plt.subplots(figsize=(6.4, 0.55 * len(names) + 1.2), dpi=150)
    bars = ax2.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_width() + max(vals) * 0.02,
                 b.get_y() + b.get_height() / 2, fmt_v(v),
                 va="center", fontsize=9, color="#333333")
    for ref in xrefs:
        if ref < max(vals) * 1.15:
            ax2.axvline(ref, color="#bbbbbb", lw=0.8, ls=":")
            ax2.text(ref, len(names) - 0.3, f"{ref:.0f}σ", fontsize=7,
                     color="#888888", ha="center", va="bottom")
    if vline_target is not None:
        ax2.axvline(vline_target, color="#e34948", lw=1.0, ls="--")
        ax2.text(vline_target, len(names) - 0.28, " target", fontsize=7,
                 color="#e34948", ha="left", va="bottom")
    ax2.set_xlim(0, max(max(vals), vline_target or 0) * 1.18 + 1e-12)
    ax2.set_xlabel(xlabel)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

with col2:
    st.subheader("T-P profile")
    cpj = json.loads(str(model["params_json"]))
    fig3, ax3 = plt.subplots(figsize=(3.4, 3.6), dpi=150)
    ax3.plot(model["T"], model["p_bar"], color="#2a78d6", lw=1.6)
    for tlim in (320.0, 2980.0):
        ax3.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
    ax3.set_yscale("log")
    ax3.invert_yaxis()
    ax3.set_xlabel("temperature (K)")
    ax3.set_ylabel("pressure (bar)")
    ax3.grid(alpha=0.25, lw=0.5)
    fig3.tight_layout()
    st.pyplot(fig3, width="stretch")
    plt.close(fig3)
    st.caption(f"As modeled ({cpj.get('tp_mode', '?')} mode). Dotted lines: "
               "the [320, 2980] K opacity window — profiles outside it are "
               "rejected, never clipped.")

# --- mode details table ------------------------------------------------------
st.subheader("Mode details")
rows = []
key_order = (lambda r: -r["sigma_detect"]) if goal_r == "detect" else (
    lambda r: per_mode.get(r["mode_key"], np.inf))
for r in sorted(results, key=key_order):
    notes = []
    if r["saturated"]:
        notes.append(f"saturates (full-well {r['sat_frac']:.2f} at min groups)")
    if r["warnings"]:
        notes.append("; ".join(list(r["warnings"])[:2]))
    row = {"mode": r["label"],
           "band (μm)": f"{r['wl'].min():.2f}–{r['wl'].max():.2f}"}
    if goal_r == "detect":
        row["σ_detect"] = round(r["sigma_detect"], 1)
        _t = float(meta.get("target_sig") or 3.0)
        if r["sigma_detect"] > 0:
            _tt = detect.transits_to_target(r, _t)
            row["transits → target"] = (_tt["n"] if _tt["reachable"] else
                                        f"never (floor caps at {_tt['sig_inf']:.1f}σ)")
        else:
            row["transits → target"] = "—"
    else:
        s = per_mode.get(r["mode_key"], np.inf)
        row[f"±{gp} at {tsig:g}σ [{unit}]"] = (f"{s:.3g}" if np.isfinite(s)
                                               else ("saturated" if r["saturated"]
                                                     else "unconstrained"))
        if np.isfinite(s) and not r["saturated"]:
            _tt = fisher_mod.transits_to_target(r, fisher_names, gp,
                                                target / tsig,
                                                detect.sigma_at_transits)
            row["transits → target"] = (_tt["n"] if _tt["reachable"] else
                                        f"never (floor caps at ±{tsig * _tt['sig_inf']:.3g})")
        else:
            row["transits → target"] = "—"
    row.update({"median σ (ppm)": round(r["median_sigma_ppm"]),
                "bins": r["n_bins"], "ngroup": r["ngroup"],
                "cadence (s)": round(r["t_cycle_s"], 1),
                "notes": "; ".join(notes)})
    rows.append(row)
st.dataframe(rows, width="stretch", hide_index=True)
if goal_r == "detect":
    st.caption(
        "σ_detect = √Δχ² of (full − without-molecule) over the mode's bins, with a "
        "free constant depth offset profiled out (removing a molecule's flat "
        "continuum no longer counts as signal). σ_bin combines Pandeia "
        "photon+detector noise for in/out-of-transit integrations with the "
        "systematic floor, anchored at R=100 bins (finer binning cannot "
        "manufacture floor-limited significance). 'transits → target' averages "
        "down the photon term only — the floor does not integrate out, so it can "
        "honestly read 'never'. Groups are chosen to stay under the saturation "
        "limit, PandExo-style."
    )
else:
    st.caption(
        f"± per mode is the marginalized Fisher forecast scaled to {tsig:g}σ "
        "(see the table below); 'transits → target' re-solves the Fisher forecast "
        "at each transit count with the photon term scaled 1/N and the R-anchored "
        "systematic floor held fixed — floor-limited targets read 'never' instead "
        "of an optimistic 1/√N estimate. Saturated modes are excluded from all "
        "forecasts."
    )

# --- Fisher forecast -------------------------------------------------------
# authoritative parameter order = the Jacobian rows as cached (canonical/sorted),
# NOT the multiselect order
if fisher_names and "jac" in model:
    tsig_f = float(meta.get("target_sig") or 3.0)
    st.subheader("Fisher parameter forecast")
    with_jac = [r for r in results if r.get("jac_bins") is not None]

    def _cell(n, s):
        v = tsig_f * fisher_mod.display_sigma(n, s)
        return "unconstrained" if not np.isfinite(v) or v > 1e4 else f"{v:.3g}"

    frows = []
    usable_f = [r for r in with_jac if not r["saturated"]]
    for r in with_jac:
        if r["saturated"]:
            # shown for completeness, but a saturated mode contributes no usable
            # data -- same exclusion policy as the verdict + combined row
            frows.append({"mode": r["label"] + "  [saturated — excluded]",
                          **{f"±{n} at {tsig_f:g}σ [{forward.PARAM_UNITS[n]}]": "—"
                             for n in fisher_names}})
            continue
        sig = fisher_mod.mode_forecast(r, fisher_names)
        frows.append({"mode": r["label"],
                      **{f"±{n} at {tsig_f:g}σ [{forward.PARAM_UNITS[n]}]":
                         _cell(n, sig[n]) for n in fisher_names}})
    if len(usable_f) >= 2:
        sig = fisher_mod.combined_forecast(usable_f, fisher_names)
        frows.append({"mode": "ALL SELECTED (combined, non-saturated)",
                      **{f"±{n} at {tsig_f:g}σ [{forward.PARAM_UNITS[n]}]":
                         _cell(n, sig[n]) for n in fisher_names}})
    st.dataframe(frows, width="stretch", hide_index=True)
    with st.expander("How to read this table"):
        st.markdown(
            f"- Each cell is the **expected ±uncertainty at {tsig_f:g}σ** "
            f"(= {tsig_f:g} × the Fisher 1σ) on that parameter if you fitted "
            "all listed parameters *simultaneously* to that mode's simulated "
            "data — a linearized best case (Cramér–Rao bound), so real "
            "retrieval posteriors can only be wider.\n"
            "- The sensitivities d(spectrum)/d(parameter) come from **automatic "
            "differentiation through the full VULCAN-JAX chemistry + ExoJAX RT "
            "chain** (photochemistry on), not from finite-difference re-runs.\n"
            "- Each per-mode row also fits (and marginalizes over) a reference-"
            "radius nuisance **lnR0**; the combined row shares lnR0 across modes "
            "and adds one absolute-depth **offset per mode** — that's what keeps "
            "multi-instrument combinations honest.\n"
            "- **No priors** are applied: a parameter with no spectral response "
            "in a mode's band reads *unconstrained* rather than a fake number.\n"
            "- lnZ and lnKzz are reported in **dex** (factors of 10); dlnCO in "
            "natural-log units (0.1 ≈ 10%).\n"
            "- σ is evaluated at the transit count you set. Only the "
            "photon/detector term averages down with more transits; the "
            "systematic floor does not — use the 'transits → target' column, "
            "not a 1/√N extrapolation."
        )
elif out.get("fisher_names"):
    st.info("Fisher forecast requested but the cached model has no Jacobian — "
            "press Run.")
