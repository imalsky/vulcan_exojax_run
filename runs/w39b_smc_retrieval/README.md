# w39b_smc_retrieval — WASP-39b retrieval case

Fits the **real Carter & May (2024) combined JWST transmission spectrum of
WASP-39b** (NIRISS SOSS + NIRSpec G395H, `../../data/cm24_wasp39b/`) with the
reusable differentiable VULCAN-JAX → ExoJax SMC retrieval framework in
`../../retrieval_framework/` (see its README for the algorithm, the staged GPU
architecture, memory/count_max engineering notes, and validation history).

This directory holds ONLY what is specific to this run:

- `case.py` — planet identity (gravity, radii, VULCAN cfg module, C&M product
  table) + the `smoke` / `gpu` / `prod` presets (`PRESETS` dict).
- `run_nas_w39b.pbs` — the NAS GH200 submit script (all modes: run, SYNTH,
  CALIBRATE_ONLY, CALIBRATE_COUNT_MAX, PROBE_MEMORY, NSYS profiling).
- `overrides/*.json` — optional Config-override files
  (`SMC_RETRIEVAL_OVERRIDES_FILE=overrides/<f>.json`, resolved against this dir).
- `data/<preset>/` — run outputs (posterior npz, config.json, run.log, plots/).
- `logs/` — PBS live logs + GPU monitor + nsys reports.

## Run

Local smoke (offline, CPU, ~minutes; always do this after framework changes):

```
cd ../..    # vulcan_exojax_run/
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval   # gradient FD checks
```

NAS GH200 (from this directory):

```
qsub run_nas_w39b.pbs                          # real-data production (gpu preset)
qsub -v SYNTH=1 run_nas_w39b.pbs               # synthetic recovery test first
qsub -v CALIBRATE_COUNT_MAX=1,CALIBRATE_COUNT_MAX_PROBE=60000,CALIBRATE_N_DRAWS=96 run_nas_w39b.pbs
```

## Status / open items (2026-07-08)

- **`count_max=5000`** (lowered from 10000, Isaac 2026-07-08; do NOT raise it). The earlier "≥21% of draws exceed
  the cap" finding (calib job 64437/64523) was traced to a **`dt_max` ballooning**
  numerical artifact, not slow chemistry: VULCAN's default `dt_max=1e17 s` let the
  step balloon to ~1e16 s on high-Kzz columns so the solver spun without settling.
  **Fixed** by `dt_max=1e11` in `case.py` (converges the ballooning draws in ~1000
  steps; the truth is untouched). See the root-level `../../CLAUDE.md` "dt_max
  ballooning" section for the full diagnosis and the VULCAN-publication check.
- **Re-run the calibration** (now cheap at native R=100 by default) to measure the
  residual non-convergence with `dt_max=1e11`:
  `qsub -v CALIBRATE_COUNT_MAX=1,CALIBRATE_COUNT_MAX_PROBE=5000,CALIBRATE_N_DRAWS=96 run_nas_w39b.pbs`
  A genuine residual (marginal-`longdy` and photochemical-limit-cycle columns) will
  remain; those get **rejected at init** (count_max=5k, failures accepted) — wire
  the init-reject once the residual fraction is known.
- Do one `SYNTH=1` recovery run at gpu fidelity before trusting the real-data
  posterior.
