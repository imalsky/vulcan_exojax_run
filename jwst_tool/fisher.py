"""Fisher-information parameter forecast from the autodiff spectrum Jacobian.

Given each instrument mode's binned Jacobian J (n_par, n_bins) and per-bin depth
sigma, the Fisher matrix is

    F_ij = sum_b J_ib J_jb / sigma_b^2

and the marginalized 1-sigma forecast on parameter i is sqrt((F^-1)_ii).

Nuisance handling (mirrors the zco_information campaign):
  * per mode: the free parameters + lnR0 (reference-radius) are jointly fit;
    lnR0 is marginalized out of the report.
  * combined (all selected modes): one SHARED lnR0 plus one constant depth
    OFFSET per mode (absolute-calibration nuisance between visits), all
    marginalized. Offsets are what make multi-instrument combinations honest --
    within a single band an offset and lnR0 are nearly degenerate.

A parameter with (numerically) no spectral response comes back as inf, shown as
"unconstrained" by the GUI rather than a fake number.
"""
from __future__ import annotations

import numpy as np

_LN10 = np.log(10.0)

# report-unit conversion: sigma in ln-units -> display units
_TO_DISPLAY = {"lnZ": 1.0 / _LN10, "lnKzz": 1.0 / _LN10}


def _marg_sigmas(F: np.ndarray, n_report: int) -> np.ndarray:
    """Marginalized sigmas for the first n_report parameters of F (inf if singular)."""
    try:
        cov = np.linalg.inv(F)
        d = np.diag(cov)[:n_report]
        return np.where(d > 0, np.sqrt(np.abs(d)), np.inf)
    except np.linalg.LinAlgError:
        # singular: eigen-decompose, invert the supported subspace, flag the rest
        w, V = np.linalg.eigh(F)
        wmax = float(np.max(np.abs(w))) if len(w) else 0.0
        good = np.abs(w) > 1e-12 * max(wmax, 1e-300)
        cov = (V[:, good] / w[good]) @ V[:, good].T
        out = np.sqrt(np.abs(np.diag(cov)[:n_report]))
        null = ~good
        if null.any():
            # any reported parameter with weight in the null space is unconstrained
            bad = (np.abs(V[:, null]).max(axis=1)[:n_report] > 1e-6)
            out[bad] = np.inf
        return out


def display_sigma(name: str, sigma: float) -> float:
    return sigma * _TO_DISPLAY.get(name, 1.0)


def mode_forecast(result: dict, free_names: list[str]) -> dict:
    """Per-mode marginalized sigmas. result needs jac_bins (n_par, n_bins) whose rows
    are [free..., lnR0] and sigma (n_bins,)."""
    J = np.asarray(result["jac_bins"])
    s = np.asarray(result["sigma"])
    F = (J / s[None, :] ** 2) @ J.T
    sig = _marg_sigmas(F, len(free_names))
    return dict(zip(free_names, sig))


def combined_forecast(results: list[dict], free_names: list[str]) -> dict:
    """All modes jointly: shared free params + shared lnR0 + one offset per mode."""
    n_f = len(free_names)
    n_modes = len(results)
    n_tot = n_f + 1 + n_modes                     # free + lnR0 + offsets
    F = np.zeros((n_tot, n_tot))
    for m, r in enumerate(results):
        J = np.asarray(r["jac_bins"])             # rows: free..., lnR0
        s = np.asarray(r["sigma"])
        nb = J.shape[1]
        Jg = np.zeros((n_tot, nb))
        Jg[:n_f] = J[:n_f]
        Jg[n_f] = J[n_f]                          # shared lnR0
        Jg[n_f + 1 + m] = 1.0                     # this mode's depth offset
        F += (Jg / s[None, :] ** 2) @ Jg.T
    sig = _marg_sigmas(F, n_f)
    return dict(zip(free_names, sig))


def transits_to_target(result: dict, free_names: list[str], gp: str,
                       target_display: float, sigma_at_transits) -> dict:
    """Smallest transit count at which the marginalized (display-unit) forecast on
    ``gp`` reaches ``target_display`` -- with the systematic floor respected.

    ``sigma_at_transits(result, n) -> per-bin sigma`` comes from detect.py (photon
    variance scales 1/N, R-anchored floor does not). Returns
    dict(n=int|None, reachable=bool, sig_inf=float): ``sig_inf`` is the
    floor-limited best case (display units); ``n`` is None when the target beats
    it -- no transit count reaches the target, which the old 1/sqrt(N)
    extrapolation could never say.
    """
    from . import detect as _detect  # local import: fisher stays numpy-only otherwise

    def _sig_with(sigma):
        r2 = dict(result); r2["sigma"] = sigma
        return display_sigma(gp, mode_forecast(r2, free_names)[gp])

    sig_inf = _sig_with(np.maximum(np.asarray(result["floor"]), 1e-30))
    if not np.isfinite(sig_inf) or target_display < sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, _detect.N_TRANSITS_CAP + 1):
        if _sig_with(sigma_at_transits(result, n)) <= target_display:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)
