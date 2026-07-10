"""(wavelength, sigma) noise providers for the Fisher / Cramer-Rao transmission forecast.

A *noise provider* is a callable

    sigma(wl_um_centers, dwl_um) -> sigma_depth

returning the per-bin 1-sigma uncertainty on transit depth, in the SAME units as
the demo's ``depth`` (fractional, i.e. (R_p/R_star)^2), on whatever binned
wavelength grid the Fisher tool hands it. Two providers ship here:

    make_parametric(instrument, n_transits, star)
        Self-contained photon-noise model: stellar blackbody x JWST collecting
        area x system throughput x in-transit integration, plus a per-bin
        systematic floor added in quadrature. No external dependencies.

    make_pandexo(npz_path)
        Reads a (wl, sigma) table produced offline by ``noise_pandexo.py``
        (PandExo in its own environment) and interpolates onto the requested grid.

The Fisher tool (``fig_fisher_forecast.py``) only ever calls the provider; it is
agnostic to which one it got. So PandExo is a drop-in upgrade behind the same
contract -- start with the parametric model (zero deps), swap in PandExo for a
publication-grade number without touching the forecast code.

Caveats (documented, not hidden): the parametric normalization is a representative
JWST ETC-style estimate, not an official Pandeia number; the systematic floor is
treated as a fixed per-bin term that does NOT average down with more transits
(conservative -- shows realistic diminishing returns); throughput/floor values are
round representative numbers. Calibrate against PandExo for real proposals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --- physical constants (SI) ------------------------------------------------
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT = 2.99792458e8      # m / s
K_BOLTZ = 1.380649e-23      # J / K
R_SUN_M = 6.957e8           # m
PC_M = 3.085677581e16       # m

# --- WASP-39 system ---------------------------------------------------------
# Star params: Teff/R* from the discovery + Gaia DR3 distance; geometry consistent
# with vulcan_exojax_run/config.py (R_star = 0.932 R_sun). Transit duration T14 ~ 2.8 h.
WASP39 = dict(
    Teff_K=5485.0,
    Rstar_m=0.932 * R_SUN_M,
    dist_pc=213.0,
    transit_dur_s=2.80 * 3600.0,
    A_tel_m2=25.4,              # JWST unobstructed collecting area
)

# --- instrument presets -----------------------------------------------------
# R          resolving power used to build the constant-R bin grid
# eta        representative end-to-end system throughput (optics * QE)
# floor_ppm  per-bin systematic noise floor, added in quadrature
# wl_lo/hi   nominal wavelength coverage (um)
# These are representative, NOT official ETC values. They exist so the relative
# "which instrument / which wavelengths" comparison is meaningful; for absolute
# numbers, replace with a PandExo table via make_pandexo().
INSTRUMENTS = {
    "NIRSpec PRISM": dict(R=100, eta=0.40, floor_ppm=18.0, wl_lo=0.60, wl_hi=5.30),
    "NIRSpec G395M": dict(R=1000, eta=0.35, floor_ppm=16.0, wl_lo=2.87, wl_hi=5.18),
    "NIRSpec G395H": dict(R=2700, eta=0.30, floor_ppm=16.0, wl_lo=2.87, wl_hi=5.18),
    "NIRISS SOSS": dict(R=700, eta=0.35, floor_ppm=20.0, wl_lo=0.85, wl_hi=2.85),
    "NIRCam F322W2": dict(R=1000, eta=0.40, floor_ppm=20.0, wl_lo=2.42, wl_hi=4.02),
    "MIRI LRS": dict(R=100, eta=0.30, floor_ppm=30.0, wl_lo=5.00, wl_hi=12.00),
}


def constant_R_grid(wl_lo, wl_hi, R):
    """Constant resolving-power grid over [wl_lo, wl_hi] um.

    Returns (centers, edges, dwl) with d(ln lambda) = 1/R per bin.
    """
    n = int(np.ceil(np.log(wl_hi / wl_lo) * R))
    edges = wl_lo * np.exp(np.arange(n + 1) / R)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, edges, np.diff(edges)


def blackbody_photon_irradiance(wl_m, Teff_K, Rstar_m, dist_pc):
    """Stellar photon irradiance at the telescope.

    Returns photons s^-1 m^-2 per metre of wavelength (spectral photon irradiance).
    """
    wl = np.asarray(wl_m, dtype=float)
    expo = H_PLANCK * C_LIGHT / (wl * K_BOLTZ * Teff_K)
    B_lambda = (2.0 * H_PLANCK * C_LIGHT ** 2 / wl ** 5) / np.expm1(expo)   # W m^-2 m^-1 sr^-1
    F_lambda = np.pi * B_lambda * (Rstar_m / (dist_pc * PC_M)) ** 2          # W m^-2 m^-1
    E_photon = H_PLANCK * C_LIGHT / wl                                       # J / photon
    return F_lambda / E_photon


def make_parametric(instrument, n_transits=1, star=WASP39):
    """Photon-noise + floor provider for one instrument over ``n_transits`` transits.

    Transit-depth precision per bin = sqrt(2 / N_photons)  (differencing the
    in-transit and out-of-transit stellar flux, equal time), combined in
    quadrature with a fixed systematic floor.
    """
    if instrument not in INSTRUMENTS:
        raise KeyError(f"unknown instrument {instrument!r}; have {list(INSTRUMENTS)}")
    inst = INSTRUMENTS[instrument]

    def sigma(wl_um_centers, dwl_um):
        wl_m = np.asarray(wl_um_centers, dtype=float) * 1e-6
        dwl_m = np.asarray(dwl_um, dtype=float) * 1e-6
        rate = blackbody_photon_irradiance(wl_m, star["Teff_K"], star["Rstar_m"], star["dist_pc"])
        # photons collected in-transit per bin, summed over all transits
        n_phot = rate * star["A_tel_m2"] * inst["eta"] * dwl_m * star["transit_dur_s"] * n_transits
        sig_phot = np.sqrt(2.0 / np.maximum(n_phot, 1.0))     # fractional transit-depth units
        floor = inst["floor_ppm"] * 1e-6                       # fixed per-bin systematic
        return np.sqrt(sig_phot ** 2 + floor ** 2)

    sigma.label = f"{instrument} x{n_transits}"
    return sigma


def make_pandexo(npz_path):
    """Provider that reads a PandExo (wl, sigma) table and interpolates onto a grid.

    The npz must contain ``wl`` (um) and ``sigma`` (fractional transit depth), as
    written by ``noise_pandexo.py``. Note: PandExo already integrated over its own
    bins, so this interpolates sigma; for a maximally faithful forecast, bin the
    Jacobian onto PandExo's native grid (see pandexo_grid()).
    """
    d = np.load(npz_path)
    wl_tab = np.asarray(d["wl"], dtype=float)
    sig_tab = np.asarray(d["sigma"], dtype=float)
    order = np.argsort(wl_tab)
    wl_tab, sig_tab = wl_tab[order], sig_tab[order]

    def sigma(wl_um_centers, dwl_um):
        return np.interp(np.asarray(wl_um_centers, dtype=float), wl_tab, sig_tab)

    sigma.label = f"pandexo:{Path(npz_path).stem}"
    return sigma


def pandexo_grid(npz_path):
    """Return PandExo's native (centers, sigma) so the Jacobian can be binned onto it."""
    d = np.load(npz_path)
    wl = np.asarray(d["wl"], dtype=float)
    order = np.argsort(wl)
    return wl[order], np.asarray(d["sigma"], dtype=float)[order]


# --- REAL published spectra (most realistic noise: the actually-achieved error bars) ---
# ASCII columns are wavelength_um, transit_depth_ppm, err_ppm, optionally followed by
# wavelength_low_um, wavelength_high_um. Lines beginning with '#' are comments.
OBS_W39B_PRISM = ("/Users/imalsky/Desktop/Emulators/VULCAN_Project/"
                  "vulcan_exojax_run/data/wasp39b_prism.txt")  # Rustamkulov+2023 FIREFLy, Zenodo 7388032

# Per-mode real WASP-39b spectra. The benchmark set (Carter+2024, uniform Eureka!
# reduction, Zenodo 10161743) is apples-to-apples across modes -> use it for the
# cross-instrument comparison. The FIREFLy PRISM stays the headline anchor.
# Written by fetch_benchmark_spectra.py.
_OBSDIR = "/Users/imalsky/Desktop/Emulators/VULCAN_Project/vulcan_exojax_run/data"
OBS_W39B = {
    "PRISM_firefly": OBS_W39B_PRISM,
    "PRISM":  _OBSDIR + "/wasp39b_bench_PRISM.txt",
    "G395H":  _OBSDIR + "/wasp39b_bench_G395H.txt",
    "NIRISS": _OBSDIR + "/wasp39b_bench_NIRISS.txt",
    "NIRCam": _OBSDIR + "/wasp39b_bench_NIRCam.txt",
}


def _edges_from_centers(wl_um):
    """Infer contiguous bin edges from wavelength centers."""
    wl = np.asarray(wl_um, dtype=float)
    mid = 0.5 * (wl[:-1] + wl[1:])
    edges = np.concatenate([[2.0 * wl[0] - mid[0]], mid, [2.0 * wl[-1] - mid[-1]]])
    return edges[:-1], edges[1:]


def load_observed(path):
    """Load a published transmission spectrum -> (wl_um, depth_ppm, sigma_frac).

    sigma_frac is the per-bin 1-sigma on transit depth in FRACTIONAL units (err_ppm*1e-6),
    matching the Fisher tool's `depth`. These are the published per-bin uncertainties,
    which include the achieved photon/statistical precision and reduction-specific
    systematics in the reported diagonal errors, but not an off-diagonal covariance.
    """
    a = np.atleast_2d(np.genfromtxt(path, comments="#"))
    wl, depth_ppm, err_ppm = a[:, 0], a[:, 1], a[:, 2]
    good = np.isfinite(wl) & np.isfinite(err_ppm) & (err_ppm > 0)
    o = np.argsort(wl[good])
    return wl[good][o], depth_ppm[good][o], err_ppm[good][o] * 1e-6


def load_observed_bins(path):
    """Load a spectrum and return wavelength bin bounds.

    Five-column files are interpreted as
    wavelength_um, depth_ppm, err_ppm, wavelength_low_um, wavelength_high_um.
    Older three-column files fall back to edges inferred from adjacent centers.
    """
    a = np.atleast_2d(np.genfromtxt(path, comments="#"))
    wl, depth_ppm, err_ppm = a[:, 0], a[:, 1], a[:, 2]
    if a.shape[1] >= 5:
        wl_lo = np.minimum(a[:, 3], a[:, 4])
        wl_hi = np.maximum(a[:, 3], a[:, 4])
    else:
        wl_lo, wl_hi = _edges_from_centers(wl)
    good = (
        np.isfinite(wl)
        & np.isfinite(depth_ppm)
        & np.isfinite(err_ppm)
        & np.isfinite(wl_lo)
        & np.isfinite(wl_hi)
        & (err_ppm > 0)
        & (wl_hi > wl_lo)
    )
    o = np.argsort(wl[good])
    return (
        wl[good][o],
        depth_ppm[good][o],
        err_ppm[good][o] * 1e-6,
        wl_lo[good][o],
        wl_hi[good][o],
    )


def calibrate_scale(instrument, obs_wl, obs_sigma, star=WASP39):
    """Multiplier that makes the parametric model's sigma match an observed spectrum's
    sigma (median ratio on the observed grid). Anchors the parametric ABSOLUTE scale to
    real achieved performance, so forecasts for other modes inherit a realistic
    normalization instead of a guessed throughput. Returns a single positive float.
    """
    dwl = np.abs(np.gradient(np.asarray(obs_wl, dtype=float)))
    par = make_parametric(instrument, n_transits=1, star=star)(obs_wl, dwl)
    return float(np.median(np.asarray(obs_sigma) / par))


def make_parametric_calibrated(instrument, scale, n_transits=1, star=WASP39):
    """Parametric provider scaled by `scale` (from calibrate_scale) -> real-anchored forecast."""
    base = make_parametric(instrument, n_transits=n_transits, star=star)

    def sigma(wl_um_centers, dwl_um):
        return scale * base(wl_um_centers, dwl_um)

    sigma.label = f"{instrument} x{n_transits} (cal x{scale:.2f})"
    return sigma
