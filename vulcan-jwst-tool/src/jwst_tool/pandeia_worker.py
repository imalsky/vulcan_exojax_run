"""Pandeia ETC worker -- runs INSIDE the picaso_base conda env (pandeia.engine 3.0).

Standalone on purpose: no imports from the rest of the tool, stdlib + pandeia only.

    python pandeia_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata path>, "cdbs": <PYSYN_CDBS path>,
     "star": {"teff":.., "log_g":.., "metallicity":.., "ks_mag":..},
     "sat_limit": 0.80,
     "modes": [{"key":.., "instrument":.., "mode":.., "config": {...},
                "strategy": {...}, "ngroup_min":.., "ngroup_max":..}, ...]}

result.json, per mode key:
    {"wl": [...um], "flux": [...e-/s], "noise_1int": [...e-/s, sigma for 1 integ],
     "t_cycle_s": .., "ngroup": .., "sat_frac": .., "saturated": bool,
     "warnings": {...}}      -- or {"error": "..."} if that mode failed.

Group selection mirrors PandExo: probe at ngroup_min, assume the brightest-pixel
full-well fraction grows linearly with groups, take the largest ngroup that stays
under sat_limit. If even ngroup_min busts the limit the mode is flagged saturated
(kept, with its degraded ngroup_min numbers, so the GUI can say WHY it's bad).
"""
import copy
import json
import math
import os
import sys
import traceback

import numpy as np


def _make_calc(build_default_calc, m, star):
    calc = build_default_calc("jwst", m["instrument"], m["mode"])
    for section, kv in (m.get("config") or {}).items():
        calc["configuration"][section].update(kv)
    calc["configuration"]["detector"]["nint"] = 1
    calc["configuration"]["detector"]["nexp"] = 1
    for k, v in (m.get("strategy") or {}).items():
        calc["strategy"][k] = v
    calc["scene"][0]["spectrum"]["sed"] = {
        "sed_type": "phoenix", "teff": float(star["teff"]),
        "log_g": float(star["log_g"]), "metallicity": float(star["metallicity"])}
    # Ks mag -> absolute flux at 2.159 um (2MASS zeropoint 666.7 Jy, Cohen 2003)
    f_mjy = 666.7e3 * 10.0 ** (-0.4 * float(star["ks_mag"]))
    calc["scene"][0]["spectrum"]["normalization"] = {
        "type": "at_lambda", "norm_wave": 2.159, "norm_waveunit": "um",
        "norm_flux": f_mjy, "norm_fluxunit": "mjy"}
    return calc


def _run(perform_calculation, calc, ngroup):
    c = copy.deepcopy(calc)
    c["configuration"]["detector"]["ngroup"] = int(ngroup)
    return perform_calculation(c)


def _one_mode(build_default_calc, perform_calculation, m, star, sat_limit):
    calc = _make_calc(build_default_calc, m, star)
    ng_min, ng_max = int(m["ngroup_min"]), int(m["ngroup_max"])

    probe = _run(perform_calculation, calc, ng_min)
    sat_probe = float(probe["scalar"]["fraction_saturation"])
    saturated = False
    if sat_probe > 0:
        # full-well fraction ~ linear in ngroup (constant flux up the ramp)
        ng_best = int(math.floor(ng_min * sat_limit / sat_probe))
    else:
        ng_best = ng_max
    if ng_best < ng_min:
        saturated = True          # even the shortest ramp busts the limit
        ng_best = ng_min
    ng_best = max(ng_min, min(ng_max, ng_best))

    rpt = probe if ng_best == ng_min else _run(perform_calculation, calc, ng_best)

    wl, _sn = rpt["1d"]["sn"]
    flux = np.asarray(rpt["1d"]["extracted_flux"][1], dtype=float)
    noise = np.asarray(rpt["1d"]["extracted_noise"][1], dtype=float)
    wl = np.asarray(wl, dtype=float)
    good = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(noise) & (flux > 0) & (noise > 0)

    if not good.any():
        # pandeia returned no usable pixels (e.g. NIRSpec PRISM on a bright star:
        # saturation within the shortest ramp NaNs the extracted noise everywhere)
        return {
            "unusable": True,
            "reason": (f"no unsaturated pixels at the shortest ramp "
                       f"(ngroup={ng_best}, max full-well fraction "
                       f"{float(rpt['scalar']['fraction_saturation']):.1f}) -- "
                       "target too bright for this mode"),
            "ngroup": int(ng_best),
            "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
            "saturated": True,
            "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
        }

    return {
        "wl": wl[good].tolist(),
        "flux": flux[good].tolist(),
        "noise_1int": noise[good].tolist(),
        "t_cycle_s": float(rpt["scalar"]["total_exposure_time"]),
        "ngroup": int(ng_best),
        "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
        "saturated": bool(saturated),
        "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
    }


def main():
    job = json.load(open(sys.argv[1]))
    os.environ["pandeia_refdata"] = job["refdata"]
    os.environ["PYSYN_CDBS"] = job["cdbs"]
    import warnings as _w
    _w.filterwarnings("ignore")
    from pandeia.engine.calc_utils import build_default_calc
    from pandeia.engine.perform_calculation import perform_calculation

    out = {}
    for m in job["modes"]:
        key = m["key"]
        print(f"[pandeia] {key} ...", flush=True)
        try:
            out[key] = _one_mode(build_default_calc, perform_calculation,
                                 m, job["star"], float(job.get("sat_limit", 0.8)))
            if out[key].get("unusable"):
                print(f"[pandeia] {key}: UNUSABLE ({out[key]['reason']})", flush=True)
            else:
                print(f"[pandeia] {key}: ngroup={out[key]['ngroup']} "
                      f"sat={out[key]['sat_frac']:.2f} npix={len(out[key]['wl'])}",
                      flush=True)
        except Exception:
            out[key] = {"error": traceback.format_exc()}
            print(f"[pandeia] {key}: FAILED", flush=True)

    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
    print("[pandeia] done", flush=True)


if __name__ == "__main__":
    main()
