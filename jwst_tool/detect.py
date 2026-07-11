"""Detection-significance math: bin the model per instrument, combine with the
per-bin depth uncertainty, and score the science goal.

Significance of "molecule X is present" is the nested-model chi-square distance
between the full spectrum and the spectrum with X's opacity removed, evaluated
on the instrument's bins -- WITH a free constant depth offset profiled out:

    chi2 = sum_b (s_b/sigma_b)^2 - (sum_b s_b/sigma_b^2)^2 / sum_b sigma_b^-2
    sigma_detect = sqrt(chi2),   s_b = d_full - d_without_X

The offset marginalization removes the common-mode (absolute-depth) part of the
molecule's contribution -- the part a real fit reabsorbs into the continuum /
reference radius -- so removing a molecule's flat continuum no longer counts as
signal. This matches how the Fisher combined forecast treats per-mode offsets.
It is still a linearized proxy (no re-fit of the other atmosphere parameters),
i.e. an upper bound on the detection significance of a full retrieval.

Multi-transit extrapolation uses the noise-model components (photon term scales
as 1/N, the R-anchored floor does not), so "transits to target" saturates
honestly instead of promising 1/sqrt(N) forever.
"""
from __future__ import annotations

import numpy as np

from . import instruments as ins
from . import noise as noise_mod

# hard cap for the transits-to-target search: beyond this the answer is
# "effectively unreachable" for any real proposal anyway
N_TRANSITS_CAP = 500


def bin_model(wl_model: np.ndarray, depth: np.ndarray, edges: np.ndarray):
    """Wavelength-weighted mean model depth per bin (NaN where no model points).

    Uses local trapezoid weights (half the neighbor spacing per native point)
    rather than a plain mean, so an uneven native grid cannot skew the bin
    average -- the same d(lambda)-weighted convention as the retrieval's exact
    binning matrix (retrieval_framework.observations.build_binning_matrix).
    """
    wl_model = np.asarray(wl_model, float)
    w_local = np.empty_like(wl_model)
    w_local[1:-1] = 0.5 * (wl_model[2:] - wl_model[:-2])
    w_local[0] = wl_model[1] - wl_model[0]
    w_local[-1] = wl_model[-1] - wl_model[-2]
    idx = np.digitize(wl_model, edges) - 1
    nb = len(edges) - 1
    out = np.full(nb, np.nan)
    for b in range(nb):
        sel = idx == b
        if sel.any():
            w = w_local[sel]
            out[b] = float(np.sum(w * depth[sel]) / np.sum(w))
    return out


def detection_significance(signal: np.ndarray, sigma: np.ndarray,
                           marginalize_offset: bool = True) -> float:
    """sqrt(Delta chi^2) of a binned signal against noise, offset-profiled.

    ``marginalize_offset=True`` (default) projects out a constant depth offset
    (see module docstring); False reproduces the raw nested-model quadrature sum.
    """
    signal = np.asarray(signal, float)
    sigma = np.asarray(sigma, float)
    chi2 = float(np.sum((signal / sigma) ** 2))
    if marginalize_offset and signal.size > 1:
        w = 1.0 / sigma ** 2
        chi2 -= float(np.sum(signal * w) ** 2 / np.sum(w))
    return float(np.sqrt(max(chi2, 0.0)))


def sigma_at_transits(result: dict, n_transits: int) -> np.ndarray:
    """Per-bin depth sigma of an evaluated mode re-scaled to ``n_transits``.

    Photon/detector variance scales as 1/N from the evaluated count; the
    R-anchored floor is N-independent.
    """
    n0 = int(result["n_transits_eval"])
    scale = n0 / float(max(1, int(n_transits)))
    return np.sqrt(np.asarray(result["var_phot"]) * scale
                   + np.asarray(result["floor"]) ** 2)


def transits_to_target(result: dict, target_sig: float) -> dict:
    """Smallest transit count reaching ``target_sig`` for the detect goal.

    Returns dict(n=int|None, reachable=bool, sig_inf=float): ``sig_inf`` is the
    floor-limited ceiling (infinite transits); ``n`` is None when the target
    exceeds it (no number of transits reaches the target -- say so instead of
    quoting an optimistic 1/sqrt(N) number).
    """
    if result.get("depth_wo") is None:
        return dict(n=None, reachable=False, sig_inf=float("nan"))
    signal = np.asarray(result["depth"]) - np.asarray(result["depth_wo"])
    floor = np.asarray(result["floor"])
    sig_inf = detection_significance(signal, np.maximum(floor, 1e-30))
    if target_sig > sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, N_TRANSITS_CAP + 1):
        if detection_significance(signal, sigma_at_transits(result, n)) >= target_sig:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)


def evaluate_mode(mode_key: str, mode_result: dict, model: dict, target_mol,
                  R_bin: float, t_in_s: float, t_out_s: float, n_transits: int,
                  floor_ppm: float) -> dict:
    """One instrument mode -> binned model, sigmas, and detection significance.

    Bins cover the intersection of the mode's science band, the model's coverage,
    and the pixels pandeia actually returned. ``target_mol=None`` (the
    parameter-constraint science goal) skips the molecule-removed comparison:
    ``sigma_detect`` comes back NaN and ``depth_wo`` None.
    """
    m = ins.MODES[mode_key]
    wl_model = model["wl_um"]
    order = np.argsort(wl_model)
    wl_model = wl_model[order]
    depth = model["depth"][order]
    mols = [str(x) for x in model["mols"]]
    depth_wo = (model["depth_wo"][mols.index(target_mol)][order]
                if target_mol is not None else None)

    wl_pix = np.asarray(mode_result["wl"])
    lo = max(m["wl_min"], float(wl_model.min()), float(wl_pix.min()))
    hi = min(m["wl_max"], float(wl_model.max()), float(wl_pix.max()))
    if hi <= lo:
        raise ValueError(f"{mode_key}: no overlap between instrument band and model")

    edges = noise_mod.make_bins(lo, hi, R_bin)
    nz = noise_mod.depth_error_bins(mode_result, edges, t_in_s, t_out_s,
                                    n_transits, floor_ppm)
    d_full = bin_model(wl_model, depth, edges)
    d_wo = bin_model(wl_model, depth_wo, edges) if depth_wo is not None else None

    # keep bins that have noise pixels AND model coverage
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep_noise = np.isin(np.round(centers, 12), np.round(nz["wl_center"], 12))
    keep = keep_noise & np.isfinite(d_full)
    if d_wo is not None:
        keep &= np.isfinite(d_wo)
    kc = np.round(centers[keep], 12)
    sig_map = dict(zip(np.round(nz["wl_center"], 12), nz["sigma"]))
    vp_map = dict(zip(np.round(nz["wl_center"], 12), nz["var_phot"]))
    fl_map = dict(zip(np.round(nz["wl_center"], 12), nz["floor"]))
    sigma = np.array([sig_map[c] for c in kc])
    var_phot = np.array([vp_map[c] for c in kc])
    floor = np.array([fl_map[c] for c in kc])

    wl_c = centers[keep]
    d_full_b = d_full[keep]
    if d_wo is not None:
        d_wo_b = d_wo[keep]
        sigma_detect = detection_significance(d_full_b - d_wo_b, sigma)
    else:
        d_wo_b, sigma_detect = None, float("nan")

    # Fisher Jacobian, binned on the same bins (rows: free params ..., lnR0)
    jac_bins = None
    if "jac" in model:
        jac_bins = np.stack([bin_model(wl_model, row[order], edges)[keep]
                             for row in model["jac"]])

    return dict(
        jac_bins=jac_bins,
        mode_key=mode_key, label=m["label"],
        wl=wl_c, depth=d_full_b, depth_wo=d_wo_b, sigma=sigma,
        var_phot=var_phot, floor=floor, n_transits_eval=int(nz["n_transits"]),
        sigma_detect=sigma_detect,
        median_sigma_ppm=float(np.median(sigma) * 1e6),
        n_bins=int(keep.sum()),
        ngroup=int(mode_result["ngroup"]),
        sat_frac=float(mode_result["sat_frac"]),
        saturated=bool(mode_result.get("saturated", False)),
        t_cycle_s=float(mode_result["t_cycle_s"]),
        warnings=mode_result.get("warnings", {}),
    )
