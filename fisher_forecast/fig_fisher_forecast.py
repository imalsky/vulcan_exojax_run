"""Fisher / Cramer-Rao observation-planning tool, built on the cached VULCAN-JAX -> ExoJax
transmission-spectrum Jacobian and REAL JWST error bars.

The demo (vulcan_exojax_run/) produces J = d(transit depth)/d(theta) at every
wavelength by forward-mode AD straight through the chemistry and the radiative
transfer. The cache contains theta = [lnZ, carbon proxy, lnKzz, dT]. We REPORT the
two abundance directions (lnZ, carbon) but MARGINALIZE over lnKzz and dT:

    F = J^T diag(1/sigma^2) J            (the full 4x4 Fisher)
    C = F^-1                             (4x4 covariance)
    report the (lnZ, carbon) 2x2 sub-block of C   -> Kzz, dT marginalized

From that, with no sampling and no retrieval: marginal error bars (sqrt of the sub-block
diagonal); the metallicity/carbon degeneracy; which wavelengths UNIQUELY constrain each
parameter (per-bin info with the other parameters projected out); and an instrument
comparison. Marginalizing Kzz/dT (vs holding them fixed) inflates sigma(log Z) ~2.8x.

REALISM: the default noise is the ACTUAL achieved JWST error bars on WASP-39b --
the Rustamkulov et al. (2023) NIRSpec PRISM spectrum (FIREFLy reduction, Zenodo
7388032), which reports the achieved per-bin precision from the real reduction,
including the realized photon/statistical errors and reduction-specific systematics
in the diagonal error bars. It does not provide or use an off-diagonal wavelength
covariance. For instruments without a local real spectrum, the parametric photon
model is CALIBRATED to the real PRISM sigma (so its absolute scale is anchored to
reality) and used as a labelled forecast.

What this forecast does and does NOT capture (be honest):
  * captured: real per-bin sigma(lambda), realistic wavelength structure, the model's
    own nonlinear chemistry->spectrum gradients.
  * NOT captured: off-diagonal (wavelength-correlated) noise covariance is neglected
    (diagonal N -- standard in published Fisher/information-content work but still an
    approximation); only the four cached parameters are varied (no clouds, T-P, reference
    radius, individual abundances, stellar contamination), so absolute CRB are LOWER BOUNDS /
    best-case. Relative instrument & wavelength comparisons are more robust than the
    absolute numbers, but still inherit these assumptions.

Usage:
    python fig_fisher_forecast.py            # generate the demo figures (real PRISM noise)
    python fig_fisher_forecast.py --verify   # Monte-Carlo + algebraic self-checks

Reads cached Jacobians + the real spectrum read-only; touches no VULCAN-JAX library code.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2.0 (and trapz removed); support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

ROOT = Path(os.environ.get("VULCAN_PROJECT_ROOT",
                            "/Users/imalsky/Desktop/Emulators/VULCAN_Project"))
RUN = ROOT / "vulcan_exojax_run"
DEMO_OUT = RUN / "data"   # exojax demo npz consolidated into the run bundle
DATA = RUN / "data"
FIGS = ROOT / "jax_paper" / "figures"   # manuscript figures stay in jax_paper/figures

sys.path.insert(0, str(ROOT / "jax_paper" / "scripts"))   # shared _common
sys.path.insert(0, str(RUN / "fisher_forecast"))          # noise_model is a sibling in fisher_forecast/
from noise_model import (INSTRUMENTS, constant_R_grid,                         # noqa: E402
                         make_parametric_calibrated, calibrate_scale,
                         load_observed, OBS_W39B_PRISM, OBS_W39B)

BANDS = {"H2O": 2.70, "CH4": 3.30, "SO2": 4.00, "CO2": 4.30, "CO": 4.66}
INTEREST = (0, 1)          # lnZ, carbon enrichment -- the science directions we report
NUISANCE = (2, 3)          # lnKzz, dT -- MARGINALIZED (audit issue #2), not held fixed
THETA_TEX = [r"$\ln Z$", "carbon enrichment"]
THETA_NAMES = ["lnZ", "carbon_enrichment"]
# The Jacobian over the full PRISM reach (1-15 um file, clipped to PRISM 1-5.3 um);
# 1 um is the demo's H2-H2 CIA floor (PRISM formally starts at 0.5 um).
PRISM_LO, PRISM_HI = 1.0, 5.35


# --------------------------------------------------------------------------- #
#  Core                                                                        #
# --------------------------------------------------------------------------- #
def load_jacobian(npz, jkey="J"):
    """Return (wl_um, depth, J (n,4), labels) with ALL theta columns
    [lnZ, carbon, lnKzz, dT]. Kzz/dT are kept so they can be MARGINALIZED, not fixed."""
    d = np.load(npz, allow_pickle=True)
    labels = ([str(x) for x in d["theta_labels"]] if "theta_labels" in d.files
              else ["lnZ", "carbon_proxy", "lnKzz", "T_int"])
    return np.asarray(d["wl_um"]), np.asarray(d["depth"]), np.asarray(d[jkey]), labels


def edges_from_centers(c):
    c = np.asarray(c, dtype=float)
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def _as_bin_bounds(centers, edges):
    """Return per-bin lower/upper bounds from either edge array or (lo, hi) pairs."""
    centers = np.asarray(centers, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim == 1 and edges.size == centers.size + 1:
        lo, hi = edges[:-1], edges[1:]
    elif edges.ndim == 2 and edges.shape == (centers.size, 2):
        lo, hi = edges[:, 0], edges[:, 1]
    else:
        raise ValueError("edges must be length n+1 or shape (n, 2)")
    return np.minimum(lo, hi), np.maximum(lo, hi)


def bin_to_grid(centers, edges, wl, depth, J):
    """Integrate model depth + J columns into observed wavelength bins.

    The cached model wavelength grid is log-spaced and descending, so an arithmetic
    mean of samples would implicitly weight by dln(lambda). This function sorts the
    model, interpolates each bin edge, and computes a wavelength-average over dlambda.
    """
    centers = np.asarray(centers, dtype=float)
    bin_lo, bin_hi = _as_bin_bounds(centers, edges)
    order = np.argsort(wl)
    wl = np.asarray(wl, dtype=float)[order]
    depth = np.asarray(depth, dtype=float)[order]
    J = np.asarray(J, dtype=float)[order]
    Y = np.column_stack([depth, J])

    db = np.full(len(centers), np.nan)
    Jb = np.full((len(centers), J.shape[1]), np.nan)
    for b, (lo, hi) in enumerate(zip(bin_lo, bin_hi)):
        if not (np.isfinite(lo) and np.isfinite(hi) and wl[0] <= lo < hi <= wl[-1]):
            continue
        inside = (wl > lo) & (wl < hi)
        x = np.concatenate([[lo], wl[inside], [hi]])
        y = np.column_stack([np.interp(x, wl, Y[:, k]) for k in range(Y.shape[1])])
        avg = _trapezoid(y, x, axis=0) / (hi - lo)
        db[b] = avg[0]
        Jb[b] = avg[1:]
    keep = np.isfinite(db)
    return keep, db, Jb


def fisher(Jb, sigma):
    Ninv = 1.0 / np.asarray(sigma) ** 2
    F = Jb.T @ (Ninv[:, None] * Jb)
    return F, np.linalg.inv(F)


def marginal_info_per_bin(Jb, sigma, p, others):
    """Per-bin Fisher information for parameter `p` AFTER marginalizing the `others`
    columns: the noise-whitened p-sensitivity with its projection onto the others removed
    (OLS residual). The residuals sum to 1/C_pp, the marginal precision on p -- so this is
    the honest 'where does the spectrum UNIQUELY constrain p' map, degeneracies removed."""
    Jw = np.asarray(Jb) / np.asarray(sigma)[:, None]
    a = Jw[:, p].copy()
    if len(others):
        B = Jw[:, list(others)]
        coeff, *_ = np.linalg.lstsq(B, a, rcond=None)
        a = a - B @ coeff
    return a ** 2


def profiled_direction_contribution(Jb, sigma, interest=(0, 1), nuisance=()):
    """Per-bin information along the marginalized least-constrained direction.

    In the figures this is called with interest=(lnZ, carbon) and nuisance=(lnKzz, T_int),
    so the target direction comes from the Schur-complement Fisher matrix after profiling
    over Kzz/T_int. With no nuisance columns it reduces to the least-constrained direction
    of the bare interest-block Fisher.
    """
    Jw = np.asarray(Jb, dtype=float) / np.asarray(sigma, dtype=float)[:, None]
    A = Jw[:, interest]
    if nuisance:
        B = Jw[:, nuisance]
        Faa = A.T @ A
        Fab = A.T @ B
        Fbb = B.T @ B
        Feff = Faa - Fab @ np.linalg.solve(Fbb, Fab.T)
    else:
        B = None
        Feff = A.T @ A
    _, evecs = np.linalg.eigh(Feff)
    v = evecs[:, 0]
    y = A @ v
    if B is not None:
        coeff, *_ = np.linalg.lstsq(B, y, rcond=None)
        y = y - B @ coeff
    return y ** 2, v, Feff


def real_prism(wl_lo=PRISM_LO, wl_hi=PRISM_HI):
    """Real WASP-39b PRISM spectrum clipped to the model's reach: (wl, depth_ppm, sigma_frac)."""
    ow, od, os = load_observed(OBS_W39B_PRISM)
    sel = (ow >= wl_lo) & (ow <= wl_hi)
    return ow[sel], od[sel], os[sel]


def forecast_observed(wlM, depthM, JM, obs_wl, obs_sigma, n_transits=1, obs_edges=None):
    """Bin the model Jacobian onto the OBSERVED grid and use the real per-bin sigma
    scaled as 1/sqrt(n_transits), i.e. treating the reported errors as random."""
    edges = edges_from_centers(obs_wl) if obs_edges is None else obs_edges
    keep, dbM, Jb = bin_to_grid(obs_wl, edges, wlM, depthM, JM)
    sig = obs_sigma[keep] / np.sqrt(n_transits)
    F, C = fisher(Jb[keep], sig)
    return dict(centers=obs_wl[keep], keep=keep, depthM=dbM[keep], J=Jb[keep], sigma=sig, F=F, C=C)


def forecast_param(wlM, depthM, JM, grid_instrument, provider):
    """Constant-R instrument grid + a parametric/calibrated provider (for modes without
    a local real spectrum)."""
    inst = INSTRUMENTS[grid_instrument]
    lo = max(inst["wl_lo"], wlM.min())
    hi = min(inst["wl_hi"], wlM.max())
    centers, edges, dwl = constant_R_grid(lo, hi, inst["R"])
    keep, db, Jb = bin_to_grid(centers, edges, wlM, depthM, JM)
    sig = np.asarray(provider(centers[keep], dwl[keep]))
    F, C = fisher(Jb[keep], sig)
    return dict(centers=centers[keep], depthM=db[keep], J=Jb[keep], sigma=sig, F=F, C=C)


def ellipse_xy(C2, center=(0.0, 0.0), dchi2=2.30, n=200):
    """Joint confidence ellipse for a 2x2 covariance (default 68% / 2 dof)."""
    vals, vecs = np.linalg.eigh(C2)
    t = np.linspace(0, 2 * np.pi, n)
    pts = (vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0) * dchi2))) @ np.stack([np.cos(t), np.sin(t)])
    return center[0] + pts[0], center[1] + pts[1]


# --------------------------------------------------------------------------- #
#  Verification (on the REAL PRISM noise)                                      #
# --------------------------------------------------------------------------- #
def verify():
    print("=== Fisher self-verification (4-param Fisher, Kzz/T_int MARGINALIZED; noise = real PRISM) ===")
    wlM, depthM, JM, labels = load_jacobian(DEMO_OUT / "wide_sensitivity.npz", jkey="J_trans")
    ow, od, os = real_prism()
    print(f" real PRISM: {len(ow)} bins in [{ow.min():.2f},{ow.max():.2f}] um, "
          f"sigma median {np.median(os)*1e6:.1f} ppm [{os.min()*1e6:.1f},{os.max()*1e6:.1f}] (1 transit)")
    fc = forecast_observed(wlM, depthM, JM, ow, os, n_transits=4)
    Jb, sig, F, C = fc["J"], fc["sigma"], fc["F"], fc["C"]
    p = Jb.shape[1]

    sym = np.max(np.abs(F - F.T)) / np.max(np.abs(F))
    evals = np.linalg.eigvalsh(F)
    print(f" (1) F symmetric={sym:.1e}  min eig={evals.min():.3e}  cond={evals.max()/evals.min():.2e}")

    ratio = np.diag(C) / (1.0 / np.diag(F))
    print(" (2) marginal/conditional variance ratio = "
          + ", ".join(f"{labels[i]}:{ratio[i]:.1f}" for i in range(p)))
    marg_ok = np.all(ratio >= 1.0 - 1e-9)

    rng = np.random.default_rng(0)
    M = C @ (Jb.T * (1.0 / sig ** 2))
    n_mc = 60000
    theta_true = np.array([0.30, 0.20, 0.50, 50.0])[:p]
    data = (Jb @ theta_true)[None, :] + rng.normal(size=(n_mc, sig.size)) * sig
    Cemp = np.cov((data @ M.T).T)
    relerr = np.max(np.abs(Cemp - C) / np.sqrt(np.outer(np.diag(C), np.diag(C))))
    print(f" (3) Monte-Carlo cov vs F^-1: max rel err {relerr:.4f} "
          f"(expect ~{1/np.sqrt(n_mc):.4f})")

    sigZ = np.sqrt(C[0, 0]) / np.log(10.0)
    sigZ_cond = (1.0 / np.sqrt(F[0, 0])) / np.log(10.0)
    rho = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
    print(f" forecast (PRISM x4): sigma(log10 Z) = {sigZ:.3f} dex MARGINALIZED over Kzz,T_int "
          f"(vs {sigZ_cond:.3f} dex if held fixed -> {sigZ/sigZ_cond:.1f}x); rho(lnZ,carbon) = {rho:+.3f}")
    ok = (sym < 1e-10) and (evals.min() > 0) and marg_ok and (relerr < 0.05)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib
    matplotlib.use("Agg")
    try:
        from _common import apply_style
        apply_style()
    except Exception:
        pass


def _bands(ax, y_text, only=None):
    """Light vertical guides + labels for the main molecular bands."""
    x0, x1 = ax.get_xlim()
    for name, lam in BANDS.items():
        if (only is None or name in only) and x0 <= lam <= x1:
            ax.axvline(lam, color="0.72", lw=0.7, ls=":", zorder=0)
            ax.text(lam, y_text, name, fontsize=8.5, color="0.4", ha="center", va="bottom")


def fig_information(out_png):
    """Figure 1 -- WHERE the information lives. Top: the real WASP-39b PRISM spectrum (the
    data). Bottom: the marginalized 'unique information' for metallicity (peaks at the SO2
    feature ~4 um) and for carbon (peaks at CO/CH4 ~2.3 um). They sit in DIFFERENT
    molecules, which is why both are measurable -- they are not degenerate."""
    import matplotlib.pyplot as plt

    wlM, depthM, JM, _ = load_jacobian(DEMO_OUT / "wide_sensitivity.npz", jkey="J_trans")
    ow, od, os = real_prism()
    fc = forecast_observed(wlM, depthM, JM, ow, os, n_transits=1)
    wl = fc["centers"]
    depth_ppm = od[fc["keep"]]
    err_ppm = os[fc["keep"]] * 1e6
    iZ = marginal_info_per_bin(fc["J"], fc["sigma"], 0, (1, 2, 3)); iZ = iZ / iZ.max()
    iC = marginal_info_per_bin(fc["J"], fc["sigma"], 1, (0, 2, 3)); iC = iC / iC.max()

    fig, (axs, axi) = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.07))
    # top: the measured spectrum
    axs.errorbar(wl, depth_ppm, yerr=err_ppm, fmt="o", ms=2.6, lw=0, elinewidth=0.6,
                 color="0.2", ecolor="0.72", zorder=3)
    axs.set_xlim(wl.min(), wl.max())
    pad = 0.08 * (depth_ppm.max() - depth_ppm.min())
    axs.set_ylim(depth_ppm.min() - pad, depth_ppm.max() + 2.0 * pad)
    _bands(axs, depth_ppm.max() + 1.05 * pad)
    axs.set_ylabel("Transit depth [ppm]")
    axs.set_title("WASP-39b transmission spectrum  (real JWST NIRSpec/PRISM, Rustamkulov+2023)",
                  fontsize=12.5)

    # bottom: where each parameter is UNIQUELY measured
    axi.fill_between(wl, 0, iZ, color="#cc3311", alpha=0.50, lw=0, zorder=2)
    axi.plot(wl, iZ, color="#cc3311", lw=1.5, label="metallicity (Z)", zorder=3)
    axi.fill_between(wl, 0, iC, color="#3b6fb6", alpha=0.36, lw=0, zorder=1)
    axi.plot(wl, iC, color="#3b6fb6", lw=1.5, label="carbon", zorder=3)
    axi.set_ylim(0, 1.30)
    _bands(axi, 1.04)
    axi.annotate("SO$_2$ — only metallicity\ncan explain this bump", xy=(wl[np.argmax(iZ)], 1.0),
                 xytext=(4.25, 0.58), fontsize=9, color="#992211", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#992211", lw=1.0))
    axi.annotate("CO / CH$_4$\n(carbon)", xy=(wl[np.argmax(iC)], 1.0),
                 xytext=(1.18, 0.55), fontsize=9, color="#234e7d", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#234e7d", lw=1.0))
    axi.set_xlabel(r"Wavelength ($\mu$m)")
    axi.set_ylabel("unique information\n(relative)")
    axi.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Metallicity and carbon are read from DIFFERENT molecules — so both are measurable",
                 fontsize=13.0, y=0.975)
    fig.text(0.5, 0.005, "'Unique information' = each parameter's signal after the others "
             "(the other abundance, Kzz, T_int) are marginalized out. Real PRISM errors, 1 transit.",
             ha="center", fontsize=8.3, style="italic", color="0.35")
    fig.subplots_adjust(left=0.115, right=0.965, top=0.905, bottom=0.12)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"wrote {out_png}")


def fig_forecast(out_png):
    """Figure 2 -- HOW WELL, and WITH WHAT. (a) the joint Z-carbon error ellipse for 1/2/4
    transits (real PRISM errors, marginalized); (b) what each JWST mode sees; (c) the
    metallicity precision per mode. Panels (b,c) use ONE calibrated noise model, so the
    comparison is coverage + resolution only -- a level playing field."""
    import matplotlib.pyplot as plt

    wlM, depthM, JM, _ = load_jacobian(DEMO_OUT / "wide_sensitivity.npz", jkey="J_trans")
    ow, od, os = real_prism()
    ln10 = np.log(10.0)

    fig = plt.figure(figsize=(15.2, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.34)

    # (a) how precise: marginalized Z-carbon ellipse, 1/2/4 transits
    axA = fig.add_subplot(gs[0, 0])
    for n, col in [(1, "#4477aa"), (2, "#228833"), (4, "#cc3311")]:
        fc = forecast_observed(wlM, depthM, JM, ow, os, n_transits=n)
        ex, ey = ellipse_xy(fc["C"][np.ix_([0, 1], [0, 1])])
        sZ = np.sqrt(fc["C"][0, 0]) / ln10
        axA.plot(ex / ln10, ey, color=col, lw=2.0,
                 label=f"{n} transit{'s' if n > 1 else ''}:  $\\sigma_{{\\log Z}}$={sZ:.2f}")
    axA.axhline(0, color="0.85", lw=0.6); axA.axvline(0, color="0.85", lw=0.6)
    axA.plot(0, 0, "k+", ms=8)
    axA.set_xlabel(r"$\Delta\,\log_{10} Z$  [dex]"); axA.set_ylabel(r"$\Delta$ carbon")
    axA.set_title("(a) how precise?  (joint 68%)", fontsize=12.5)
    axA.legend(fontsize=8.3, loc="upper right")

    # one calibrated noise model for all instruments (level playing field)
    SHORT = {"NIRISS SOSS": "NIRISS", "NIRCam F322W2": "NIRCam", "NIRSpec G395H": "G395H",
             "NIRSpec PRISM": "PRISM", "MIRI LRS": "MIRI"}
    pw, pd, ps = load_observed(OBS_W39B["PRISM"])
    scale = calibrate_scale("NIRSpec PRISM", pw, ps)
    insts = list(SHORT)
    cols = plt.cm.viridis(np.linspace(0.10, 0.85, len(insts)))
    rows = []
    for name, col in zip(insts, cols):
        fc = forecast_param(wlM, depthM, JM, name, make_parametric_calibrated(name, scale, n_transits=4))
        lo = max(INSTRUMENTS[name]["wl_lo"], wlM.min())
        hi = min(INSTRUMENTS[name]["wl_hi"], wlM.max())
        rows.append((SHORT[name], np.sqrt(fc["C"][0, 0]) / ln10, lo, hi, col))

    # (b) what each instrument sees
    axB = fig.add_subplot(gs[0, 1])
    o = np.argsort(wlM)
    axB.plot(wlM[o], depthM[o] * 1e6, color="0.55", lw=0.9, zorder=1)
    lo_y, hi_y = depthM.min() * 1e6, depthM.max() * 1e6
    step = 0.10 * (hi_y - lo_y)
    for i, (disp, _s, lo, hi, col) in enumerate(rows):
        y = lo_y - step * (i + 1)
        axB.plot([lo, hi], [y, y], lw=7, color=col, solid_capstyle="butt", label=disp)
    axB.set_xlim(1, 12.5); axB.set_ylim(lo_y - step * (len(rows) + 0.8), hi_y + 0.5 * step)
    axB.set_xlabel(r"Wavelength ($\mu$m)"); axB.set_ylabel("Transit depth [ppm]")
    axB.set_title("(b) what each instrument sees", fontsize=12.5)
    axB.legend(fontsize=8, loc="upper right", ncol=2)

    # (c) which is best for metallicity
    axC = fig.add_subplot(gs[0, 2])
    x = np.arange(len(rows))
    for xi, (disp, sZ, lo, hi, col) in zip(x, rows):
        axC.bar(xi, sZ, 0.72, color=col)
        axC.text(xi, sZ, f"{sZ:.2f}", ha="center", va="bottom", fontsize=8.5)
    axC.set_xticks(x); axC.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    axC.set_ylabel(r"$\sigma(\log_{10} Z)$ [dex]")
    axC.set_title("(c) best for metallicity?  (4 transits)", fontsize=12.5)
    axC.margins(y=0.15)

    fig.suptitle("Forecasting a WASP-39b metallicity measurement   "
                 "(real PRISM errors; Kzz & T_int marginalized)", fontsize=13.2, y=1.02)
    fig.text(0.5, -0.04, "Panels (b,c): ONE calibrated noise model for all instruments, so differences are "
             "coverage + resolution only. Absolute $\\sigma$ are best-case (no clouds / T-P / radius).",
             ha="center", fontsize=8.6, style="italic", color="0.33")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    if "--verify" in sys.argv:
        return verify()
    FIGS.mkdir(parents=True, exist_ok=True)
    _style()
    fig_information(FIGS / "fisher_information.png")
    fig_forecast(FIGS / "fisher_forecast.png")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
