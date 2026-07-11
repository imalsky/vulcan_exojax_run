"""JWST time-series instrument-mode registry + paths for the noise backend.

Each mode entry carries the Pandeia configuration used by ``pandeia_worker.py``
(running inside the ``picaso_base`` conda env: pandeia.engine 3.0 +
pandeia_data-3.0rc3) plus display metadata and a default systematic noise floor.

Noise floors: pre-flight planning convention (Greene et al. 2016 assumed
20/30/50 ppm for NIRISS/NIRCam/MIRI); in-flight results are often better
(e.g. Schlawin et al. 2021 find ~<10 ppm for NIRCam grism), so the defaults
here sit between the two and every floor is editable in the GUI. Floors are
per-bin values AT R=100 (noise.FLOOR_REF_R): the noise model scales the per-bin
floor as sqrt(R_bin/100) for finer bins so binning choices cannot manufacture
floor-limited significance.
"""
from __future__ import annotations

import os
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = TOOL_DIR.parent                      # vulcan_exojax_run/
# JWST_TOOL_DATA_DIR overrides the cache/CDBS root for non-editable installs
# (default: the bundle's data/jwst_tool/, correct for the editable install).
DATA_DIR = Path(os.environ.get("JWST_TOOL_DATA_DIR",
                               str(BUNDLE_DIR / "data" / "jwst_tool")))
MODEL_CACHE = DATA_DIR / "model_cache"
NOISE_CACHE = DATA_DIR / "noise_cache"

# Pandeia backend environment (the real STScI ETC engine, same as PandExo's core).
# pandeia.engine 2026.1 in the base env rejects the on-disk 3.0rc3 refdata, so the
# worker runs in picaso_base (pandeia 3.0), which matches it. Both paths are
# machine-specific; override via env vars on any other machine / install
# (noise.run_pandeia refuses loudly if the python is missing).
PICASO_PYTHON = os.environ.get(
    "JWST_TOOL_PANDEIA_PYTHON",
    "/opt/homebrew/Caskroom/miniforge/base/envs/picaso_base/bin/python")
PANDEIA_REFDATA = os.environ.get(
    "JWST_TOOL_PANDEIA_REFDATA",
    "/Users/imalsky/Documents/Important_Docs/JWST_CYCLE5/picaso_ian/data/pandeia_data-3.0rc3")
# Minimal synphot CDBS assembled for this tool: phoenix grid symlinked from
# RT-Project/picaso, johnson_j bandpass fetched from ssb.stsci.edu/trds.
PYSYN_CDBS = str(DATA_DIR / "cdbs")

# 2MASS Ks zeropoint (Cohen et al. 2003): 666.7 Jy at 2.159 um. Used to convert the
# star's Ks magnitude to an absolute at_lambda normalization (avoids needing the
# full CDBS comp/ tree for photsys normalization).
KS_ZEROPOINT_JY = 666.7
KS_LAMBDA_UM = 2.159

# Fixed categorical color order (validated dataviz palette) -- one color per mode,
# never re-assigned when the user's selection changes.
_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# wl_min/wl_max: the usable science bandpass we bin over (intersected with the
# forward model's 1-15 um coverage; NIRISS SOSS order 1 nominally reaches 0.85 um
# but the model band starts at 1.0 um -- the H2-H2 CIA table's short edge).
MODES = {
    "nirspec_prism": dict(
        label="NIRSpec PRISM",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="prism", filter="clear"),
                    detector=dict(subarray="sub512")),
        wl_min=0.6, wl_max=5.25,
        floor_ppm=20.0, ngroup_min=2, ngroup_max=90,
    ),
    "nirspec_g395h": dict(
        label="NIRSpec G395H",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g395h", filter="f290lp"),
                    detector=dict(subarray="sub2048")),
        wl_min=2.87, wl_max=5.18,
        floor_ppm=15.0, ngroup_min=2, ngroup_max=90,
    ),
    "nirspec_g235h": dict(
        label="NIRSpec G235H",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g235h", filter="f170lp"),
                    detector=dict(subarray="sub2048")),
        wl_min=1.66, wl_max=3.07,
        floor_ppm=15.0, ngroup_min=2, ngroup_max=90,
    ),
    "niriss_soss": dict(
        label="NIRISS SOSS (ord 1)",
        instrument="niriss", mode="soss",
        config=dict(instrument=dict(filter="clear", disperser="gr700xd"),
                    detector=dict(subarray="substrip256")),
        strategy=dict(order=1),
        wl_min=0.85, wl_max=2.8,
        floor_ppm=20.0, ngroup_min=2, ngroup_max=30,
    ),
    "nircam_f322w2": dict(
        label="NIRCam F322W2",
        instrument="nircam", mode="ssgrism",
        config=dict(instrument=dict(filter="f322w2", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        wl_min=2.45, wl_max=3.95,
        floor_ppm=25.0, ngroup_min=2, ngroup_max=180,
    ),
    "nircam_f444w": dict(
        label="NIRCam F444W",
        instrument="nircam", mode="ssgrism",
        config=dict(instrument=dict(filter="f444w", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        wl_min=3.9, wl_max=4.95,
        floor_ppm=25.0, ngroup_min=2, ngroup_max=180,
    ),
    "miri_lrs": dict(
        label="MIRI LRS (slitless)",
        instrument="miri", mode="lrsslitless",
        config=dict(detector=dict(subarray="slitlessprism")),
        wl_min=5.0, wl_max=12.0,
        floor_ppm=40.0, ngroup_min=5, ngroup_max=300,
    ),
}

MODE_COLOR = {key: _COLORS[i % len(_COLORS)] for i, key in enumerate(MODES)}

# GUI default selection: blue-to-red coverage with the three workhorses.
# (The ETC always computes ALL modes per star, so changing the selection is free.)
DEFAULT_MODES = ["niriss_soss", "nirspec_g395h", "miri_lrs"]

# Per-planet system defaults (star, geometry, T14, UV spectrum) live in planets.py.
