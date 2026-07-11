"""Console entry point: ``jwst-tool`` launches the Streamlit GUI.

Equivalent to ``streamlit run jwst_tool/app.py`` from the bundle directory, but
works from anywhere once the package is installed (editable install
recommended). Preflight checks catch the two external requirements with
actionable messages instead of a mid-run stack trace: the VULCAN-JAX project
tree (``$VULCAN_PROJECT_ROOT``) and the Pandeia backend env
(``$JWST_TOOL_PANDEIA_PYTHON``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("jwst-tool: streamlit is not installed in this environment.\n"
              "Install the GUI extra:  pip install -e '.[gui]'", file=sys.stderr)
        return 2

    import config as shared_config          # resolves VULCAN_PROJECT_ROOT
    jaxroot = Path(shared_config.JAXROOT)
    if not (jaxroot / "src" / "vulcan_jax").exists():
        print(f"jwst-tool: VULCAN-JAX checkout not found at {jaxroot}.\n"
              "Set VULCAN_PROJECT_ROOT to the project directory that contains "
              "the VULCAN-JAX/ and vulcan_exojax_run/ trees.", file=sys.stderr)
        return 2

    from jwst_tool import instruments as ins
    if not Path(ins.PICASO_PYTHON).exists():
        print(f"jwst-tool: Pandeia backend python not found at {ins.PICASO_PYTHON}.\n"
              "Point JWST_TOOL_PANDEIA_PYTHON at a python with pandeia.engine 3.0 "
              "(and JWST_TOOL_PANDEIA_REFDATA at the matching refdata). The GUI "
              "still starts, but every noise calculation will refuse to run.",
              file=sys.stderr)

    app = Path(__file__).resolve().parent / "app.py"
    # run from the bundle dir so the app's relative-path expectations hold
    cwd = str(app.parent.parent)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:]
    return subprocess.call(cmd, cwd=cwd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
