"""VMC and fixed-node DMC for the beryllium atom.

Which radial freedom of a compact multi-determinant ansatz moves the nodal
surface, and by how much. See README.md for the result and for what this
package does not claim.

LAYOUT

The modules are a library; the two entry points live in scripts/:

    wavefunction   VMC and the wave-function definition (reference energies)
    dmc            fixed-node diffusion Monte Carlo
    nodes          nodal diagnostics (no DMC needed)
    stats          blocking, chi2 and group fits
    figures        the figures of the manuscript

OUTPUT DIRECTORIES

Results are written outside the source tree, so that the repository stays
free of generated files:

    data/      .npz written by the pipeline (trial functions, DMC runs,
               nodal analyses)
    results/   figures and text tables

Both are resolved here, once, and can be redirected with the environment
variables BE_NQS_DATA and BE_NQS_RESULTS -- which is how `--quick` runs are
kept from overwriting production output, since quick outputs carry the same
file names:

    BE_NQS_DATA=/tmp/quick BE_NQS_RESULTS=/tmp/quick \
        python scripts/run_study.py validate --quick

data_path() and results_path() take a BARE file name and return a path inside
the corresponding directory, creating it on demand. A name that already has a
directory component is returned unchanged, so the helpers are idempotent and
safe to apply to a path that has already been resolved.
"""

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "DATA_DIR", "RESULTS_DIR",
           "data_path", "results_path"]


def _project_root():
    """The repository root in a development checkout, else the current dir.

    The package lives at <root>/src/be_nqs/. When it is imported from an
    installed copy instead, there is no such root and the working directory
    is the only sensible default.
    """
    here = Path(__file__).resolve()
    if here.parent.parent.name == "src":
        return here.parent.parent.parent
    return Path.cwd()


PROJECT_ROOT = _project_root()
DATA_DIR = Path(os.environ.get("BE_NQS_DATA", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("BE_NQS_RESULTS", PROJECT_ROOT / "results"))


def _resolve(base, name):
    p = Path(name)
    if p.is_absolute() or p.parent != Path("."):
        return str(p)                     # already placed: leave it alone
    base.mkdir(parents=True, exist_ok=True)
    return str(base / p)


def data_path(name):
    """Path of a pipeline .npz inside DATA_DIR."""
    return _resolve(DATA_DIR, name)


def results_path(name):
    """Path of a figure or table inside RESULTS_DIR."""
    return _resolve(RESULTS_DIR, name)
