"""Fetch the real WASP-39b JWST transmission spectra (per instrument mode) from the
Carter et al. (2024) benchmark Zenodo record, WITHOUT downloading the 1.7 GB archive.

A zip's central directory lives at the end of the file, so an HTTP range-request reader
lets zipfile list the archive and extract only the small per-mode CSVs (a few KB each).
We read the uniform-reduction (Eureka!, Fit_LimbDarkening) transmission spectra and write
clean files into outputs/ with wavelength centers, transit depths, uncertainties, and
the published wavelength-bin bounds.

CSV columns: wave, wave_low, wave_hig, rp/rs, rp/rs_err_low, rp/rs_err_hih, ...
  depth = (rp/rs)^2 ;  sigma_depth = 2*(rp/rs)*sigma_rprs  (sigma_rprs = mean of the
  asymmetric low/high errors).

Source: Carter et al. 2024, Nature Astronomy; Zenodo 10.5281/zenodo.10161743.

Run:  python fetch_benchmark_spectra.py
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
import os
from pathlib import Path

import numpy as np

URL = "https://zenodo.org/records/10161743/files/ERS_DataSynthesis_Zenodo.zip"
OUT = Path(os.environ.get("VULCAN_PROJECT_ROOT",
                           "/Users/imalsky/Desktop/Emulators/VULCAN_Project")) / "vulcan_exojax_run" / "data"
BASE = "ZENODO/4_TRANSMISSION_SPECTRA/Fit_LimbDarkening/"

# Mode -> list of CSV members to concatenate (multi-detector / multi-order modes).
# PRISM has no R100 reduction (it is intrinsically R~100) -> use its native bins_scale1.
MODES = {
    "PRISM":  ["NIRSpec_PRISM/bins_scale1.csv"],
    "G395H":  ["NIRSpec_G395H_NRS1/R100.csv", "NIRSpec_G395H_NRS2/R100.csv"],
    "NIRISS": ["NIRISS_SOSS_Order1/R100.csv", "NIRISS_SOSS_Order2/R100.csv"],
    "NIRCam": ["NIRCam_F322W2/R100.csv"],
}


def _remote_zip(url):
    n = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
            .headers["Content-Length"])

    class HF:
        def __init__(s): s.p = 0
        def seek(s, o, w=0): s.p = o if w == 0 else (s.p + o if w == 1 else n + o); return s.p
        def tell(s): return s.p
        def seekable(s): return True
        def read(s, k=-1):
            if k is None or k < 0:
                k = n - s.p
            if k <= 0:
                return b""
            a, b = s.p, min(n, s.p + k) - 1
            d = urllib.request.urlopen(
                urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()
            s.p = a + len(d)
            return d

    return zipfile.ZipFile(HF()), n / 1e9


def _parse_csv(raw):
    """Return (wl_um, depth_ppm, err_ppm, wl_lo_um, wl_hi_um) from one CSV blob."""
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    wl, dep, err, wl_lo, wl_hi = [], [], [], [], []
    for r in rows:
        try:
            w = float(r["wave"])
            lo = float(r["wave_low"])
            hi = float(r["wave_hig"])
            rprs = float(r["rp/rs"])
            elo = abs(float(r["rp/rs_err_low"]))
            ehi = abs(float(r["rp/rs_err_hih"]))
        except (KeyError, ValueError, TypeError):
            continue
        s = 0.5 * (elo + ehi)
        if not (
            np.isfinite(w)
            and np.isfinite(lo)
            and np.isfinite(hi)
            and np.isfinite(rprs)
            and np.isfinite(s)
            and s > 0
            and hi > lo
        ):
            continue
        wl.append(w)
        dep.append(rprs ** 2 * 1e6)              # transit depth [ppm]
        err.append(2.0 * rprs * s * 1e6)         # propagated depth error [ppm]
        wl_lo.append(lo)
        wl_hi.append(hi)
    return np.array(wl), np.array(dep), np.array(err), np.array(wl_lo), np.array(wl_hi)


def main():
    z, gb = _remote_zip(URL)
    print(f"opened remote zip ({gb:.2f} GB) via range requests; extracting {sum(len(v) for v in MODES.values())} CSVs")
    for mode, members in MODES.items():
        wl, dep, err, wl_lo, wl_hi = [], [], [], [], []
        for m in members:
            w, d, e, lo, hi = _parse_csv(z.read(BASE + m))
            wl.append(w); dep.append(d); err.append(e); wl_lo.append(lo); wl_hi.append(hi)
            print(f"  {mode:7s} <- {m:34s} {len(w):4d} pts")
        wl = np.concatenate(wl); dep = np.concatenate(dep); err = np.concatenate(err)
        wl_lo = np.concatenate(wl_lo); wl_hi = np.concatenate(wl_hi)
        o = np.argsort(wl)
        wl, dep, err, wl_lo, wl_hi = wl[o], dep[o], err[o], wl_lo[o], wl_hi[o]
        out = OUT / f"wasp39b_bench_{mode}.txt"
        with open(out, "w") as f:
            f.write(f"# WASP-39b JWST {mode} transmission spectrum (Eureka!, Fit_LimbDarkening, ~R100)\n")
            f.write("# Carter et al. 2024, Nature Astronomy; Zenodo 10.5281/zenodo.10161743\n")
            f.write("# depth=(rp/rs)^2; err=2*(rp/rs)*sigma_rprs (asym errors averaged)\n")
            f.write("# columns: wavelength_um  transit_depth_ppm  err_ppm  wavelength_low_um  wavelength_high_um\n")
            for a, b, c, lo, hi in zip(wl, dep, err, wl_lo, wl_hi):
                f.write(f"{a:.6f} {b:.3f} {c:.3f} {lo:.6f} {hi:.6f}\n")
        print(f"  -> {out.name}: {len(wl)} pts, [{wl.min():.2f},{wl.max():.2f}] um, "
              f"median err {np.median(err):.1f} ppm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
