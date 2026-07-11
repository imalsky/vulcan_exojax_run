"""Offline producer of PandExo (wl, sigma) noise tables for the Fisher forecast.

PandExo (Batalha et al. 2017) is heavyweight -- it pulls Pandeia reference data and
stellar grids and wants its own environment -- so it stays DECOUPLED from the
gradient/Fisher code. This script runs PandExo once per instrument mode and writes a
small npz that fig_fisher_forecast.py consumes through noise_model.make_pandexo():

    outputs/pandexo_<planet>_<mode>.npz   with keys:
        wl     : (n,) wavelength bin centers [um]
        sigma  : (n,) 1-sigma per-bin uncertainty on transit depth [fractional]
        meta   : 0-d str  (provenance)

Run (in a PandExo environment):
    python noise_pandexo.py "NIRSpec Prism" --transits 1
    python noise_pandexo.py "NIRSpec G395H" --transits 5

Without PandExo installed, use --mock to emit a clearly-labeled synthetic table (from
the parametric model) so the (wl, sigma) contract and the downstream tool can be
exercised end-to-end:
    python noise_pandexo.py "NIRSpec Prism" --mock
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("VULCAN_PROJECT_ROOT",
                            "/Users/imalsky/Desktop/Emulators/VULCAN_Project"))
OUT = ROOT / "vulcan_exojax_run" / "data"
sys.path.insert(0, str(ROOT / "vulcan_exojax_run" / "fisher_forecast"))   # noise_model sibling

# Map our preset names <-> PandExo's instrument strings.
PANDEXO_MODE = {
    "NIRSpec Prism": "NIRSpec PRISM",
    "NIRSpec G395H": "NIRSpec G395H",
    "NIRSpec G395M": "NIRSpec G395M",
    "NIRISS SOSS": "NIRISS SOSS",
    "MIRI LRS": "MIRI LRS",
}

# WASP-39 inputs for PandExo's exo dict.
STAR = dict(temp=5485.0, metal=0.0, logg=4.45, mag=10.20, ref_wave=1.25,
            Rs=0.932, type="phoenix")     # mag is approx K-band; ref_wave in um
PLANET = dict(transit_duration=2.80 * 3600.0, period=4.055)


def _outpath(mode, planet="W39b"):
    tag = mode.replace(" ", "").replace("NIRSpec", "NRS").replace("NIRISS", "NIS")
    return OUT / f"pandexo_{planet}_{tag}.npz"


def run_real(mode, n_transits):
    """Run PandExo for one instrument mode; return (wl_um, sigma_depth)."""
    from pandexo.engine import justdoit as jdi   # noqa: F401  (heavy, optional)

    exo = jdi.load_exo_dict()
    exo["observation"]["sat_level"] = 80
    exo["observation"]["sat_unit"] = "%"
    exo["observation"]["noccultations"] = n_transits
    exo["observation"]["R"] = None
    exo["observation"]["baseline"] = 2.0 * PLANET["transit_duration"]
    exo["observation"]["baseline_unit"] = "total"
    exo["observation"]["noise_floor"] = 0
    exo["star"].update(type=STAR["type"], mag=STAR["mag"], ref_wave=STAR["ref_wave"],
                       temp=STAR["temp"], metal=STAR["metal"], logg=STAR["logg"])
    exo["planet"]["type"] = "constant"
    exo["planet"]["transit_duration"] = PLANET["transit_duration"]
    exo["planet"]["td_unit"] = "s"
    exo["planet"]["f_unit"] = "rp^2/r*^2"
    exo["planet"]["radius"] = 1.279
    exo["planet"]["r_unit"] = "R_jup"

    res = jdi.run_pandexo(exo, [PANDEXO_MODE[mode]], save_file=False)
    fin = res["FinalSpectrum"]
    wl = np.asarray(fin["wave"], dtype=float)
    sigma = np.asarray(fin["error_w"], dtype=float)     # 1-sigma on (Rp/R*)^2 per bin
    good = np.isfinite(wl) & np.isfinite(sigma) & (sigma > 0)
    return wl[good], sigma[good]


def run_mock(mode, n_transits):
    """Synthetic (wl, sigma) from the parametric model -- contract test only."""
    from noise_model import INSTRUMENTS, constant_R_grid, make_parametric
    inst = INSTRUMENTS[PANDEXO_MODE[mode]]
    centers, edges, dwl = constant_R_grid(inst["wl_lo"], inst["wl_hi"], inst["R"])
    sigma = make_parametric(PANDEXO_MODE[mode], n_transits=n_transits)(centers, dwl)
    return centers, sigma


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mode = argv[0]
    if mode not in PANDEXO_MODE:
        print(f"unknown mode {mode!r}; have {list(PANDEXO_MODE)}")
        return 2
    n_transits = int(argv[argv.index("--transits") + 1]) if "--transits" in argv else 1
    mock = "--mock" in argv

    if mock:
        wl, sigma = run_mock(mode, n_transits)
        prov = f"MOCK parametric (NOT real PandExo); {mode}; {n_transits} transit(s)"
    else:
        try:
            wl, sigma = run_real(mode, n_transits)
        except ImportError:
            print("PandExo not importable. Install pandexo.engine in a dedicated env, "
                  "or re-run with --mock to emit a synthetic table.")
            return 3
        prov = f"PandExo {mode}; {n_transits} transit(s); WASP-39"

    OUT.mkdir(parents=True, exist_ok=True)
    out = _outpath(mode)
    np.savez(out, wl=wl, sigma=sigma, meta=np.array(prov))
    print(f"wrote {out}  ({len(wl)} bins, median sigma {np.median(sigma)*1e6:.1f} ppm)")
    print(f"  provenance: {prov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
