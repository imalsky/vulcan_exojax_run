# vulcan_exojax_run

Differentiable VULCAN photochemistry to JWST spectra. The repo chains the live
[VULCAN-JAX](https://github.com/imalsky/jax-vulcan) photochemical-kinetics
model into [ExoJax](https://github.com/HajimeKawahara/exojax) radiative
transfer and propagates gradients through the whole chain: a shared forward
model, a gradient-informed SMC/MALA Bayesian retrieval (WASP-39b on real
Carter & May 2024 data), and a Pandeia-backed JWST instrument-selection tool.
It only imports VULCAN-JAX and ExoJax and never modifies them.

## Packages

| directory | dist name | import name | purpose |
|-----------|-----------|-------------|---------|
| `vulcan-retrieval/` | `vulcan-retrieval` | `retrieval_framework` | the shared differentiable forward model (`retrieval_framework.forward`) plus the SMC / forward-mode-MALA retrieval framework, WASP-39b case, examples, validation, Z vs C/O analysis |
| `vulcan-jwst-tool/` | `vulcan-jwst-tool` | `jwst_tool` | JWST instrument selector: forward model to Pandeia ETC noise, detection significance + Fisher forecasts (Streamlit GUI, console script `jwst-tool`) |

The chemistry backend is the sibling
[VULCAN-JAX](https://github.com/imalsky/jax-vulcan) repo (dist `vulcan-jax` on
TestPyPI), developed and released separately.

## Install

Local development (conda env `vulcan`), from this directory:

```
pip install --no-deps -e ./vulcan-retrieval -e ./vulcan-jwst-tool
```

`--no-deps` because `vulcan-jax>=0.1.17` lives on TestPyPI, not PyPI. Consumer
installs from TestPyPI:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-retrieval
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
```

HPC (NAS GH200): code deploys by `git pull`, the PBS preflight editable-installs
the synced trees, and `requirements-hpc.txt` pins the exact validated chemistry
commit for a standalone reproduction env. Rules and setup: `CLAUDE.md`.

Entry points (run from this directory):

```
python -m retrieval_framework.run_smc vulcan-retrieval/runs/w39b_smc_retrieval
python -m pytest vulcan-retrieval/tests -q
jwst-tool
```

## Data policy

ALL data stays at the repo-root `data/` tree, shared by both packages and never
shipped inside a pip package. Tracked: `data/cm24_wasp39b/` (published
observation CSVs) and the minimal `data/jwst_tool/cdbs/`. Gitignored,
regenerable by download: `data/exojax_linelists/`, `data/opacity_cache/`, the
generated `data/*.npz` caches, and the jwst-tool model/noise caches; on HPC the
two big caches are seeded once per clone. Paths resolve via
`VULCAN_PROJECT_ROOT` (the directory containing this checkout) or are inferred
from an editable checkout; `retrieval_framework.forward.config` raises loudly
otherwise. Provenance and regeneration details: `data/notes.md`.

## Repo map

```
vulcan_exojax_run/
├── README.md                  this index
├── CLAUDE.md                  operational rules (HPC deploy, retrieval decisions)
├── requirements-hpc.txt       HPC reproducibility pins (exact vulcan-jax commit)
├── data/                      shared data tree (see Data policy + data/notes.md)
├── vulcan-retrieval/
│   ├── README.md              package docs (forward engine + retrieval framework)
│   ├── notes.md               historical development log
│   ├── src/retrieval_framework/          the framework
│   │   └── forward/                      the shared forward-model engine
│   ├── tests/                 fast unit tests
│   ├── runs/w39b_smc_retrieval/          the WASP-39b case (case.py + PBS + notes.md)
│   ├── examples/              sensitivity demo (+ notes.md)
│   ├── validation/            9 validation scripts (+ notes.md)
│   └── scripts/zco_information/          Z vs C/O information analysis (+ notes.md)
└── vulcan-jwst-tool/
    ├── README.md              tool docs (physics, modes, backend wiring)
    ├── notes.md               version history (v1 to v5)
    └── src/jwst_tool/         the Streamlit GUI + forward/noise/Fisher backends
```

## Docs convention

Exactly one README per package plus this index; current usage and reference
content only. Historical design logs and dev diaries live in per-directory
`notes.md` files, moved verbatim from the pre-0.6 docs (they carry load-bearing
measured numbers).

## License

GPL-3.0-only (see `LICENSE`; both packages carry the same license).
