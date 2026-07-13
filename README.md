# vulcan_exojax_run — RETIRED

**This repository is retired and archived (read-only). Do not install
anything from it.** The two packages it used to vendor moved to standalone
repositories on 2026-07-11 and all development continues there:

| package | authoritative repository | superseding commit at retirement |
|---------|--------------------------|----------------------------------|
| `vulcan-retrieval` (import `retrieval_framework`) | <https://github.com/imalsky/vulcan-retrieval> | `d2543a3219c049a5f2c5451bafbc645718db653f` |
| `vulcan-jwst-tool` (import `jwst_tool`) | <https://github.com/imalsky/vulcan-jwst-tool> | `96582fa3e4195eca402358e3d1aea45d128b1673` |

The chemistry backend remains
[VULCAN-JAX](https://github.com/imalsky/jax-vulcan) (dist `vulcan-jax`),
developed and released separately.

## Why this matters

The `vulcan-retrieval/` and `vulcan-jwst-tool/` directories in this
repository are **frozen, stale copies**. They predate correctness fixes made
in the standalone repositories (among others: the count-space measurement
operator, exact piecewise-linear cell integration, PandExo hard-minimum
noise-floor semantics, unit-invariant Fisher rank, basis-invariant nuisance
projection, and the flux-weighted LSF count ratio). Installing from this
repository can produce scientifically different answers under the same
package names. A 2026-07-12 external audit flagged exactly this divergence;
retiring this install path is the fix.

## What to do instead

Follow the installation instructions in the standalone repositories'
READMEs. Consumer installs come from TestPyPI:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-retrieval
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
```

This repository stays online only so that history and old links resolve.
