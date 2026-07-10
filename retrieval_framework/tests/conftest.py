"""Path setup: make the vulcan_exojax_run/ bundle (and thus the `retrieval` package
and the shared forward-model lib) importable when pytest runs from anywhere."""
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent.parent   # vulcan_exojax_run/
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))
