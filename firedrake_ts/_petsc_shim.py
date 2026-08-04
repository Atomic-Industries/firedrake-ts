"""ctypes access to a PETSc ``TS`` entry point that petsc4py does not expose.

Only ``TSSetSNES`` is needed, to repair the SNES residual callback after
Firedrake's RHS projection solver displaces it; see
:func:`repair_ts_snes_callbacks`.
"""

import ctypes
import glob
import os

from firedrake.petsc import PETSc

__all__ = ["repair_ts_snes_callbacks"]

_lib = None


def _dlopen_petsc():
    """Return a ctypes handle whose symbol table includes libpetsc's TS symbols.

    petsc4py loads libpetsc with ``RTLD_LOCAL``, so ``CDLL(None)`` does not see
    it. Prefer the versioned libpetsc under ``PETSC_DIR``/``PETSC_ARCH``; fall
    back to re-``dlopen``-ing petsc4py's own extension module with
    ``RTLD_GLOBAL``, which promotes its already-loaded dependency to the global
    namespace.
    """
    import petsc4py

    cfg = petsc4py.get_config()
    pattern = os.path.join(cfg["PETSC_DIR"], cfg["PETSC_ARCH"], "lib", "libpetsc.so*")
    candidates = [*sorted(glob.glob(pattern)), PETSc.__file__]
    for path in candidates:
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if hasattr(lib, "TSSetSNES"):
            return lib
    raise RuntimeError(
        "could not resolve libpetsc's TS symbols; tried " + ", ".join(candidates)
    )


def _load():
    global _lib
    if _lib is None:
        lib = _dlopen_petsc()
        lib.TSSetSNES.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.TSSetSNES.restype = ctypes.c_int
        _lib = lib
    return _lib


def repair_ts_snes_callbacks(ts):
    """Reinstate ``SNESTSFormFunction`` on ``ts``'s SNES if it has been displaced.

    ``SNESSetFunction`` is DM-scoped in PETSc -- it records the callback on the
    DM's ``DMSNES``. The RHS projection solver built by
    ``_TSContext._rhs_projection_solver`` is a Firedrake ``LinearSolver``, hence
    a ``NonlinearVariationalSolver``, whose constructor installs
    ``_SNESContext.form_function``. Because that solver is built on the same
    function space, it shares the TS's DM, so the install lands on the TS's SNES
    as well.

    The consequence is severe and silent: ``_SNESContext.form_function``
    assembles ``problem.F`` against a ``udot`` that the TS never updates, which
    for a mass-matrix-only ``F`` is identically zero. Every stage solve then
    "converges" in zero iterations at its initial guess, no explicit
    contribution ever enters the state, and the solution does not advance at
    all -- ``u' = -u`` integrates to ``u(1) = 1``.

    ``TSSetSNES`` reinstalls ``SNESTSFormFunction`` (and ``SNESTSFormJacobian``,
    when the Jacobian is still PETSc's). It is cheap and is what PETSc itself
    does after its ARKIMEX startup step.
    """
    snes = ts.getSNES()
    if snes.getFunction()[1] is None:
        # Still PETSc's C callback -- nothing was displaced.
        return
    ierr = _load().TSSetSNES(ts.handle, snes.handle)
    if ierr:
        raise RuntimeError(f"TSSetSNES failed with ierr={ierr}")
    if snes.getFunction()[1] is not None:
        raise RuntimeError(
            "TSSetSNES did not restore SNESTSFormFunction on the TS's SNES; "
            "the DAE residual would be evaluated against a stale time derivative"
        )
