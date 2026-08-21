"""Shared helpers for the firedrake_ts test suite."""

import numpy as np
from firedrake import Function, FunctionSpace, TestFunction, UnitIntervalMesh

#: The ARKSSP stepper's fully-qualified name, as PETSc resolves it from
#: ``-ts_python_type``. Spelled once so a rename does not have to be found in
#: ten string literals across four files.
PYTHON_STEPPER = "firedrake_ts.ark_ssp.ARKSSP"

#: Selects the ARKSSP stepper. Merge a tableau and step size on top.
ARK_SSP = {"ts_type": "python", "ts_python_type": PYTHON_STEPPER}

#: ARKSSP on the production tableau -- the one nearly every stepper test wants.
#: Five files were each re-deriving or re-spelling this.
ARK_SSP_G5 = {**ARK_SSP, "ts_ark_ssp_type": "esdirk_gamma5"}

#: PETSc's own second-order ARK-IMEX, the reference stepper for every
#: cross-check and negative control. Also spelled in five places before.
ARKIMEX_2C = {"ts_type": "arkimex", "ts_arkimex_type": "2c"}

#: Solution of ``u' = -u`` at ``t = 1``, the oracle for every decay test.
EXACT_DECAY = np.exp(-1.0)


def scalar_problem(u0=1.0, cells=4, degree=1):
    """A fresh ``(u, u_t, v)`` triple on a P-``degree`` space, ``u`` set to ``u0``."""
    mesh = UnitIntervalMesh(cells)
    V = FunctionSpace(mesh, "P", degree)
    u = Function(V)
    u_t = Function(V)
    v = TestFunction(V)
    u.assign(u0)
    return u, u_t, v
