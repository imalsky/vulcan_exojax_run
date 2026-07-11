"""Host-side noise interface: subprocess to the Pandeia worker + transit-depth error math.

The worker gives, per instrument mode, the per-native-pixel extracted stellar
flux and its 1-integration sigma. This module turns that into a transit-depth
uncertainty per spectral bin, PandExo-style:

    depth = 1 - F_in/F_out
    var(depth)_pixel = (sigma_1int/flux)^2 * (1/n_int_in + 1/n_int_out) / n_transits
    var(depth)_bin   = 1 / sum_pixels(1/var_pixel)          (inverse-variance)
    sigma_bin        = sqrt(var_bin + floor^2)               (floor does NOT average down)

Results are cached by a hash of (star, modes, sat_limit) so the ETC runs once
per star/instrument set.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import instruments as ins


def noise_job(star: dict, mode_keys: list[str], sat_limit: float = 0.80) -> dict:
    modes = []
    for key in mode_keys:
        m = dict(ins.MODES[key])
        modes.append({
            "key": key, "instrument": m["instrument"], "mode": m["mode"],
            "config": m.get("config", {}), "strategy": m.get("strategy", {}),
            "ngroup_min": m["ngroup_min"], "ngroup_max": m["ngroup_max"],
        })
    return {
        "refdata": ins.PANDEIA_REFDATA, "cdbs": ins.PYSYN_CDBS,
        "star": {k: float(star[k]) for k in ("teff", "log_g", "metallicity", "ks_mag")},
        "sat_limit": float(sat_limit),
        "modes": modes,
        "worker_version": 2,   # cache-buster: bump when pandeia_worker output changes
    }


def job_key(job: dict) -> str:
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def run_pandeia(job: dict, progress=None, force: bool = False) -> dict:
    """Run the worker in picaso_base (or return the cached result).

    ``progress``: optional callable(str) receiving worker stdout lines live.
    Raises RuntimeError (loudly, with stderr) if the worker process itself dies;
    per-mode pandeia failures come back as {"error": traceback} entries.
    """
    ins.NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ins.NOISE_CACHE / f"{job_key(job)}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    py = Path(ins.PICASO_PYTHON)
    if not py.exists():
        raise RuntimeError(
            f"Pandeia backend python not found at {py} (the picaso_base conda env "
            "with pandeia.engine 3.0). The noise model cannot run without it.")

    in_json = ins.NOISE_CACHE / f"{job_key(job)}.job.json"
    out_json = ins.NOISE_CACHE / f"{job_key(job)}.out.json"
    in_json.write_text(json.dumps(job))
    worker = ins.TOOL_DIR / "pandeia_worker.py"

    proc = subprocess.Popen([str(py), str(worker), str(in_json), str(out_json)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if progress:
            progress(line.rstrip())
    proc.wait()
    if proc.returncode != 0 or not out_json.exists():
        err = proc.stderr.read()
        raise RuntimeError(f"pandeia worker failed (rc={proc.returncode}):\n{err[-3000:]}")

    result = json.loads(out_json.read_text())
    cache.write_text(json.dumps(result))
    return result


def make_bins(wl_lo: float, wl_hi: float, R: float) -> np.ndarray:
    """Log-spaced bin EDGES at resolving power R over [wl_lo, wl_hi]."""
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


# The quoted per-mode systematic floors (instruments.MODES / Greene+2016-style
# numbers) are per-bin values AT R=100 bins. A real systematic is spectrally
# correlated, so it cannot be made to shrink by slicing the band into more bins:
# treating a fixed per-bin floor as white noise inflated a floor-dominated
# detection significance by ~sqrt(R_bin/100) when the bin slider moved 100->200.
# Anchoring here keeps the floor-limited information content of the band
# R-independent: per-bin floor = floor_ppm * sqrt(R_bin/R_REF) for finer bins.
# Coarser-than-reference bins KEEP the full floor (no sqrt averaging-down --
# conservative, systematics don't integrate out).
FLOOR_REF_R = 100.0


def depth_error_bins(mode_result: dict, edges: np.ndarray,
                     t_in_s: float, t_out_s: float, n_transits: int,
                     floor_ppm: float) -> dict:
    """Per-bin transit-depth sigma from a worker mode result.

    Returns dict(wl_center, sigma, n_pix, var_phot, floor, n_transits) with
    empty-pixel bins dropped. ``var_phot`` is the photon/detector bin variance AT
    the evaluated ``n_transits`` (it scales as 1/N); ``floor`` is the per-bin
    R-anchored systematic (N-independent; see FLOOR_REF_R). sigma =
    sqrt(var_phot + floor^2). Returning the two components separately is what
    lets callers extrapolate to other transit counts CORRECTLY -- a plain
    1/sqrt(N) scaling of sigma is optimistic wherever the floor contributes.
    """
    wl = np.asarray(mode_result["wl"])
    flux = np.asarray(mode_result["flux"])
    noise = np.asarray(mode_result["noise_1int"])
    t_cycle = float(mode_result["t_cycle_s"])

    n_in = max(1, int(t_in_s / t_cycle))
    n_out = max(1, int(t_out_s / t_cycle))
    var_pix = (noise / flux) ** 2 * (1.0 / n_in + 1.0 / n_out) / max(1, int(n_transits))

    idx = np.digitize(wl, edges) - 1
    nb = len(edges) - 1
    inv_var = np.zeros(nb)
    npx = np.zeros(nb, dtype=int)
    for b in range(nb):
        sel = idx == b
        if sel.any():
            inv_var[b] = np.sum(1.0 / var_pix[sel])
            npx[b] = int(sel.sum())
    keep = npx > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    var_phot = 1.0 / inv_var[keep]
    r_bin = (centers / np.diff(edges))[keep]
    floor = (floor_ppm * 1e-6) * np.sqrt(np.maximum(r_bin, FLOOR_REF_R) / FLOOR_REF_R)
    sigma = np.sqrt(var_phot + floor ** 2)
    return dict(wl_center=centers[keep], sigma=sigma, n_pix=npx[keep],
                var_phot=var_phot, floor=floor,
                n_transits=int(max(1, int(n_transits))))
