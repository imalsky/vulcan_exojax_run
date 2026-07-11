# data: provenance and regeneration policy

All data for both packages lives here, at the repo root. Code resolves this
tree via `VULCAN_PROJECT_ROOT` (the directory containing the
`vulcan_exojax_run/` checkout) or, in an editable checkout, by inferring the
repo root from the installed package location;
`vulcan-retrieval/src/retrieval_framework/forward/config.py` raises loudly when
neither resolves. Data does not travel with pip installs on purpose.

## Tracked in git

- `cm24_wasp39b/`: the real Carter & May (2024) WASP-39b transmission products
  (Zenodo 10161743, Fixed_LimbDarkening CSVs: NIRISS SOSS orders 1/2, NIRSpec
  G395H NRS1/NRS2, NIRCam, all R=100), plus `PRISM_native.csv` (the
  Rustamkulov et al. 2023 PRISM spectrum). Published measurements, small, the
  retrieval's observation source. Never regenerate; replace only with a newer
  published reduction.
- `jwst_tool/cdbs/`: the minimal synphot CDBS tree the Pandeia backend needs
  (`grid/phoenix` symlinked from an external stellar-grid tree,
  `comp/nonhst/johnson_j_003_syn.fits` fetched from ssb.stsci.edu/trds).

## Gitignored regenerable caches

- `exojax_linelists/` (~190 MB): HITRAN line-list caches. ExoJax re-downloads
  them on first use (through the NAS proxy on HPC). The `h2he` broadening knob
  adds separate `<db>_h2he` cache dirs on its first use.
- `opacity_cache/` (~170 MB): offline CO ExoMol Li2015 plus H2-H2 and H2-He CIA
  tables. Regenerable by download; the H2-He CIA file's canonical URL is
  `https://hitran.org/data/CIA/main/H2-He_2011.cia` (147 MB; note the `/main/`
  path segment, the bare `/data/CIA/` URL 404s).
- `jwst_tool/model_cache/`, `jwst_tool/noise_cache/`: jwst-tool model spectra
  and Pandeia results, regenerated per run and versioned by `_VERSION` in the
  code (v5 invalidated all earlier cached spectra).
- `*.npz` at this level: generated sensitivity, zco-jacobian, and zco-walk
  caches. The previously tracked set was deleted as stale on 2026-07-11 (every
  chemistry/spectrum cache predating the scientific-correctness pass is
  invalid). Regenerate with `vulcan-retrieval/examples/run_demo.py` /
  `run_figs.py` and `vulcan-retrieval/scripts/zco_information/
  build_zco_jacobians.py` / `build_zco_walk.py`; never re-track them.

## HPC seeding

Code deploys to the NAS by `git pull`; data does not ride along. Seed the big
caches ONCE into a fresh clone, either by copying from a parked tree on
/nobackup (`cp -r <old tree>/data/opacity_cache vulcan_exojax_run/data/`, same
for `exojax_linelists`) or by a one-time scp from local (see the repo-root
`CLAUDE.md` for the exact scp proxy command; never rsync, never tarballs). The
PBS preflight errors without `opacity_cache/`; a missing `exojax_linelists/`
just re-downloads through the proxy.
