"""ctypes access to PETSc ``TS`` entry points that petsc4py does not expose.

Two are needed.

``TSSetSNES`` -- to repair the SNES residual callback after Firedrake's RHS
projection solver displaces it; see :func:`repair_ts_snes_callbacks`.

``TSAdaptSetCheckStage`` -- the only hook in current PETSc that fires between a
stage solve and the formation of that stage's derivatives, which is where a
stage-local modification has to run to have any effect.
"""

import ctypes
import glob
import os

from firedrake.petsc import PETSc

__all__ = [
    "repair_ts_snes_callbacks",
    "set_stage_hook",
    "stage_hook_propagates",
]


# PetscErrorCode (*)(TSAdapt, TS, PetscReal, Vec, PetscBool *)
_CHECKSTAGE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
)

_lib = None
_vec_from_pointer = None


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
        if hasattr(lib, "TSGetAdapt"):
            return lib
    raise RuntimeError(
        "could not resolve libpetsc's TS symbols; tried " + ", ".join(candidates)
    )


def _load():
    global _lib, _vec_from_pointer
    if _lib is not None:
        return _lib, _vec_from_pointer

    lib = _dlopen_petsc()
    lib.TSGetAdapt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.TSGetAdapt.restype = ctypes.c_int
    lib.TSAdaptSetCheckStage.argtypes = [ctypes.c_void_p, _CHECKSTAGE]
    lib.TSAdaptSetCheckStage.restype = ctypes.c_int
    lib.TSSetSNES.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.TSSetSNES.restype = ctypes.c_int

    # Wrap a raw Vec in a petsc4py Vec through petsc4py's exported Cython C-API.
    # PYFUNCTYPE, not CFUNCTYPE: this call creates a Python object, and
    # CFUNCTYPE releases the GIL.
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
        ctypes.py_object,
        ctypes.c_char_p,
    ]
    addr = ctypes.pythonapi.PyCapsule_GetPointer(
        PETSc.__pyx_capi__["PyPetscVec_New"], b"PyObject *(Vec)"
    )
    _lib = lib
    _vec_from_pointer = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p)(addr)
    return _lib, _vec_from_pointer


def _check(ierr, what):
    if ierr:
        raise RuntimeError(f"{what} failed with ierr={ierr}")


def repair_ts_snes_callbacks(ts):
    """Reinstate ``SNESTSFormFunction`` on ``ts``'s SNES if it has been displaced.

    ``SNESSetFunction`` is DM-scoped in PETSc -- it records the callback on the
    DM's ``DMSNES``. The RHS projection solver built by
    ``_TSContext._rhs_projection_solver`` is a Firedrake ``LinearSolver``, hence
    a ``NonlinearVariationalSolver``, whose constructor installs
    ``_SNESContext.form_function``. Because that solver is built on the same
    function space, it shares the TS's DM, so the install lands on the TS's SNES
    as well.

    ``TSSetSNES`` reinstalls ``SNESTSFormFunction`` (and ``SNESTSFormJacobian``,
    when the Jacobian is still PETSc's). It is cheap and is what PETSc itself
    does after its ARKIMEX startup step.
    """
    snes = ts.getSNES()
    if snes.getFunction()[1] is None:
        # Still PETSc's C callback -- nothing was displaced.
        return
    lib, _ = _load()
    _check(lib.TSSetSNES(ts.handle, snes.handle), "TSSetSNES")
    if snes.getFunction()[1] is not None:
        raise RuntimeError(
            "TSSetSNES did not restore SNESTSFormFunction on the TS's SNES; "
            "the DAE residual would be evaluated against a stale time derivative"
        )


def set_stage_hook(ts, hook):
    """Install ``hook(ts, t, Y)`` to run on every implicitly-solved stage.

    ``Y`` is a petsc4py ``Vec`` aliasing PETSc's stage vector; mutate it in
    place. Stages that PETSc treats as explicit (those with a zero diagonal
    entry in the implicit tableau) do not call checkstage and so are not seen;
    for a tableau with an explicit first stage that stage is a plain copy of the
    incoming solution, so there is nothing there to act on.

    Returns the ctypes trampoline. The caller **must** keep it alive for as long
    as the ``TS``: PETSc holds only a raw function pointer, so if the trampoline
    is collected PETSc calls freed memory.
    """
    lib, vec_from_pointer = _load()

    def trampoline(_adapt, _ts, t, y_ptr, accept):
        try:
            hook(ts, float(t), vec_from_pointer(y_ptr))
        except Exception:
            # An exception must not unwind into C. Report the stage as
            # unacceptable so PETSc retries with a smaller step rather than
            # continuing with a half-modified stage.
            import traceback

            traceback.print_exc()
            accept[0] = 0  # PETSC_FALSE
            return 0
        accept[0] = 1  # PETSC_TRUE
        return 0

    fn = _CHECKSTAGE(trampoline)
    adapt = ctypes.c_void_p()
    _check(lib.TSGetAdapt(ts.handle, ctypes.byref(adapt)), "TSGetAdapt")
    _check(lib.TSAdaptSetCheckStage(adapt, fn), "TSAdaptSetCheckStage")
    return fn


_propagates = None


def stage_hook_propagates():
    """Does a stage mutation actually change the accepted step? (cached)

    Guards against the shim degrading to a silent no-op -- the failure mode of
    ``TSSetPostStage`` -- if a future PETSc moves where checkstage is called.
    Integrates ``u' = -u`` twice under ARKIMEX, once with a hook that shifts
    every stage, and checks the two results differ.
    """
    global _propagates
    if _propagates is not None:
        return _propagates

    prefix = "firedrake_ts_stage_probe_"
    opts = PETSc.Options()
    opts[prefix + "ts_adapt_type"] = "none"
    try:
        results = []
        for mutate in (False, True):
            ts = PETSc.TS().create(comm=PETSc.COMM_SELF)
            ts.setOptionsPrefix(prefix)
            ts.setType("arkimex")
            u = PETSc.Vec().createSeq(1, comm=PETSc.COMM_SELF)
            u.set(1.0)
            mat = PETSc.Mat().createAIJ([1, 1], comm=PETSc.COMM_SELF)
            mat.setUp()

            def ifunction(_ts, _t, x, xdot, f):
                xdot.copy(f)
                f.axpy(1.0, x)

            def ijacobian(_ts, _t, _x, _xdot, shift, jac, _pre):
                jac.zeroEntries()
                jac.setValue(0, 0, shift + 1.0)
                jac.assemble()

            ts.setIFunction(ifunction, u.duplicate())
            ts.setIJacobian(ijacobian, mat, mat)
            ts.setRHSFunction(lambda _ts, _t, _x, g: g.set(0.0), u.duplicate())
            ts.setTime(0.0)
            ts.setMaxTime(1.0)
            ts.setTimeStep(0.25)
            ts.setMaxSteps(4)
            ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)
            ts.setFromOptions()
            keep = None
            if mutate:
                keep = set_stage_hook(ts, lambda _ts, _t, y: y.shift(1e-3))
            ts.solve(u)
            results.append(u.getArray()[0])
            del keep
        _propagates = bool(abs(results[0] - results[1]) > 1e-12)
        return _propagates
    finally:
        del opts[prefix + "ts_adapt_type"]
