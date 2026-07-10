"""Central configuration for the VULCAN-JAX -> ExoJax transmission-sensitivity demo.

Pure constants only: NO heavy imports here (no jax, no vulcan_jax, no exojax), so
this module is safe to import before the env-order-sensitive VULCAN-JAX setup runs.

The demo chains the *live* VULCAN-JAX chemistry forward model into an ExoJax
``ArtTransPure`` transmission model and propagates forward-mode tangents from four
physical parameters -- (ln Z, C/O, ln Kzz, T_int) -- all the way to the transit
spectrum, so every wavelength can be colored by d(transit_depth)/d(parameter).

Planet: WASP-39b (matches the validated jax_paper sensitivity scripts + the JWST
SO2/CO2 metallicity story).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# VULCAN_PROJECT_ROOT env var makes the bundle portable to HPC checkouts (e.g.
# /nobackup/$USER/VULCAN_Project on NAS); the default is the local tree.
PROJECT_ROOT = Path(os.environ.get(
    "VULCAN_PROJECT_ROOT", "/Users/imalsky/Desktop/Emulators/VULCAN_Project"))
JAXROOT = PROJECT_ROOT / "VULCAN-JAX"
JP = PROJECT_ROOT / "jax_paper"  # for _common.apply_style (house figure style)
DEMO_DIR = PROJECT_ROOT / "vulcan_exojax_run"                   # this bundle (moved out of jax_paper 2026-07-08)
OUTPUTS = DEMO_DIR / "data"                                     # npz caches + observed spectra live with the bundle
FIGS = JP / "figures"                                           # manuscript figures stay in jax_paper/figures
DEMO_DATABASE = DEMO_DIR / "data" / "exojax_linelists"          # HITRAN line lists (with the bundle)

# Offline opacity cache (CO ExoMol Li2015 + H2-H2/H2-He CIA), lives IN the bundle
# (data/opacity_cache/) so the bundle has no dependency on any sibling project --
# copied in 2026-07-07 from what was previously a reused emulator-demo/ cache.
_CACHE = DEMO_DIR / "data" / "opacity_cache"
CO_CACHED_DIR = _CACHE / "CO" / "12C-16O" / "Li2015"
CIA_H2H2_FILE = _CACHE / "H2-H2_2011.cia"
# H2-He CIA (He is ~16% by number at 10x solar; real continuum contribution).
# Download once: https://hitran.org/data/CIA/H2-He_2011.cia -> this path.
# exojax_rt degrades gracefully (warns, skips the term) if the file is absent.
CIA_H2HE_FILE = _CACHE / "H2-He_2011.cia"

# Reference wavenumber (cm^-1) for the ExoJax powerlaw_clouds retrieval cloud:
# kappa(nu) = kappac0 * (nu/CLOUD_NUC0)^alphac, kappac0 in cm^2 per gram of
# atmosphere (pRT convention; alphac = 0 is a gray cloud). 2857 cm^-1 = 3.5 um,
# mid-band for the 2.0-5.3 um retrieval window.
CLOUD_NUC0 = 2857.0

# ---------------------------------------------------------------------------
# VULCAN-JAX network selection (must be set as env vars BEFORE importing vulcan_jax)
# ---------------------------------------------------------------------------
VULCAN_NETWORK = "thermo/SNCHO_photo_network.txt"
VULCAN_ATOM_LIST = "H,O,C,N,S"
W39B_CFG_MODULE = "vulcan_jax.cfg_examples.vulcan_cfg_W39b"

# atom_list column order inside composition.compo_array (probed from the package).
# Index of each element column we touch when building the Z / C/O knobs.
ATOM_COLS = {"H": 0, "O": 1, "C": 2, "He": 3, "N": 4, "S": 5}

# Molar masses (g/mol) for every compo_array column, in column order. Used to turn
# a VMR profile into a mean-molecular-weight profile (compo_array @ this vector).
# atom_list = (H,O,C,He,N,S,P,Na,K,Si,Fe,Ar,Ti,V,Mg,Ca,e)
ATOMIC_MASSES = [
    1.008, 15.999, 12.011, 4.0026, 14.007, 32.06, 30.974, 22.990,
    39.098, 28.085, 55.845, 39.948, 47.867, 50.942, 24.305, 40.078, 5.4858e-4,
]

# ---------------------------------------------------------------------------
# WASP-39b physical constants (from cfg_examples/vulcan_cfg_W39b.py)
# ---------------------------------------------------------------------------
R_SUN_CM = 6.957e10
RP_CM = 1.279 * 7.1492e9   # planet radius (cm) at the bottom pressure P_b
GS_CGS = 422.0             # surface gravity (cm/s^2)
RSTAR_CM = 0.932 * R_SUN_CM

# ---------------------------------------------------------------------------
# Opacity / radiative-transfer grid
# ---------------------------------------------------------------------------
# ART pressure bounds (bar). The bottom stays inside VULCAN's envelope; the TOP is
# set BELOW VULCAN's 1e-7 bar chemistry top on purpose -- the log-P interpolation
# clamps the topmost VULCAN VMR/T there, i.e. a constant-abundance + isothermal upper-
# atmosphere extension (standard transmission-modeling practice). Without it, strong
# bands (CO2 4.3, CO 4.7 um) go optically thick to the model top and the transit radius
# saturates into a flat "wall" at 4.2-5.2 um (saturated fraction 4.8% at 1e-6 bar);
# extending to 1e-8 bar removes it (0.1%), letting the bands rise to real peaks.
ART_PTOP_BAR = 1.0e-8
ART_PBTM_BAR = 7.0
T_OPA_MIN_K = 300.0
T_OPA_MAX_K = 3000.0

# Each molecule: VULCAN species name, molar mass (g/mol), and opacity source.
# CO is fully offline (cached ExoMol Li2015). H2O/CO2/CH4/SO2 use HITRAN, downloaded
# on first run into .database/<db>/ (small, public, no login -- only the main
# isotopologue, isotope=1). HITRAN's 296 K reference under-represents the hottest
# bands of a ~1100 K atmosphere relative to HITEMP/ExoMol, but it is fully adequate for
# this methodology demo (the sensitivity pattern, not absolute line fidelity, is the
# point) and avoids the multi-GB ExoMol downloads / credentialed HITEMP fetch.
# "source" is one of {"exomol_cached", "exomol", "hitran"}; "db" is the ExoMol
# "<iso>/<list>" path suffix or the per-molecule download dir name under .database.
# molmass is set explicitly (exojax isotope_molmass returns None for CH4).
MOLECULES = {
    "CO":  {"vulcan": "CO",  "molmass": 28.010, "source": "exomol_cached", "db": str(CO_CACHED_DIR)},
    "H2O": {"vulcan": "H2O", "molmass": 18.015, "source": "hitran", "db": "H2O"},
    "CO2": {"vulcan": "CO2", "molmass": 43.990, "source": "hitran", "db": "CO2"},
    "CH4": {"vulcan": "CH4", "molmass": 16.043, "source": "hitran", "db": "CH4"},
    "SO2": {"vulcan": "SO2", "molmass": 64.066, "source": "hitran", "db": "SO2"},
    # High-C/O + sulfur discriminators for the SMC retrieval prior box
    # (runs/w39b_smc_retrieval): the retrieval explores C/O up to ~1 where C2H2/HCN
    # carry the signal; H2S is the reduced-S reservoir. Same HITRAN path as above.
    "HCN":  {"vulcan": "HCN",  "molmass": 27.025, "source": "hitran", "db": "HCN"},
    "C2H2": {"vulcan": "C2H2", "molmass": 26.037, "source": "hitran", "db": "C2H2"},
    "H2S":  {"vulcan": "H2S",  "molmass": 34.081, "source": "hitran", "db": "H2S"},
}

# Bulk gas used for CIA + the dominant background (H2).
BULK_H2_VULCAN = "H2"

# ---------------------------------------------------------------------------
# Run profiles
# ---------------------------------------------------------------------------
# Wavenumbers in cm^-1. wavelength(um) = 1e4 / nu.
#
# Two non-obvious requirements, both about keeping the forward-mode tangent valid:
#   * Photochemistry must be ON. Only in the photo-on regime does the warm-started jvp
#     relax to the true steady-state sensitivity (validated: jvp vs re-converged FD <0.1%
#     at nz=150). With photo OFF the W39b column lands in a regime where the tangent is
#     under-relaxed/unstable.
#   * Let convergence happen naturally (default count_min/count_max). Do NOT pin a fixed
#     step count -- forcing dt to dt_max drives the Ros2 step's forward tangent singular.
SMOKE = {
    "use_photo": True,
    "nz": 40,                  # coarse column -> cheaper warm-up + jvps
    "yconv_cri": 1.0e-3,
    "molecules": ["CO"],       # fully offline
    "nu_min": 4280.0,          # ~2.31-2.34 um, the cached CO 2-0 band (matches smc.py)
    "nu_max": 4360.0,
    "nu_pts": 600,
    "art_nlayer": 20,
}
FULL = {
    "use_photo": True,         # photo ON -> SO2 chemistry (WASP-39b story)
    "nz": 150,                 # canonical W39b grid
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": 1923.0,          # ~5.2 um
    "nu_max": 3450.0,          # ~2.9 um  (NIRSpec G395H/PRISM red: CH4 3.3, SO2 4.0, CO2 4.3, CO 4.7)
    "nu_pts": 6000,
    "art_nlayer": 60,
}
# Wide-band overview: 1-15 um (the supported window -- H2-H2 CIA stops at 1 um / 10000
# cm-1 on the short side, line lists reach ~20 um). Computed on a finer native grid and
# displayed at R=100. Used for BOTH the transmission and emission figures.
WIDE = {
    "use_photo": True,
    "nz": 150,
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": 667.0,           # 15 um
    "nu_max": 10000.0,         # 1 um  (H2-H2 CIA upper edge)
    "nu_pts": 8000,            # native R ~ 2950; binned to display_R for the figure
    "art_nlayer": 60,
    "display_R": 100,
}

# Parameter vector order: theta = [lnZ, c_o_pert, lnKzz, T_int_K]
THETA_LABELS = ["lnZ", "C/O", "lnKzz", "T_int"]
THETA0 = [0.0, 0.0, 0.0, 0.0]   # baseline (no perturbation)
